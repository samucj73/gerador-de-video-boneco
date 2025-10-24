import streamlit as st
from datetime import datetime, timedelta
import requests
import json
import os
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

# =============================
# Configurações e Segurança
# =============================

# Mover para variáveis de ambiente (CRÍTICO)
API_KEY = os.getenv("FOOTBALL_API_KEY", "9058de85e3324bdb969adc005b5d918a")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003073115320")
TELEGRAM_CHAT_ID_ALT2 = os.getenv("TELEGRAM_CHAT_ID_ALT2", "-1002754276285")

HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL_FD = "https://api.football-data.org/v4"
BASE_URL_TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# Constantes
ALERTAS_PATH = "alertas.json"
CACHE_JOGOS = "cache_jogos.json"
CACHE_CLASSIFICACAO = "cache_classificacao.json"
CACHE_TIMEOUT = 3600  # 1 hora em segundos

# =============================
# Dicionário de Ligas
# =============================
LIGA_DICT = {
    "FIFA World Cup": "WC",
    "UEFA Champions League": "CL", 
    "Bundesliga": "BL1",
    "Eredivisie": "DED",
    "Campeonato Brasileiro Série A": "BSA",
    "Primera Division": "PD",
    "Ligue 1": "FL1",
    "Championship (Inglaterra)": "ELC",
    "Primeira Liga (Portugal)": "PPL",
    "European Championship": "EC",
    "Serie A (Itália)": "SA",
    "Premier League (Inglaterra)": "PL"
}

# =============================
# Dicionário de Perfis de Equipes (Atualizado)
# =============================
PERFIS_EQUIPES = {
    # Equipes Ofensivas (Over 2.5 natural)
    "Bayern Munich": "OFENSIVA", "Borussia Dortmund": "OFENSIVA", 
    "Manchester City": "OFENSIVA", "Liverpool": "OFENSIVA",
    "Paris Saint-Germain": "OFENSIVA", "Barcelona": "OFENSIVA",
    "Real Madrid": "OFENSIVA", "Ajax": "OFENSIVA",
    "RB Leipzig": "OFENSIVA", "Atalanta": "OFENSIVA",
    
    # Equipes Defensivas (Under 2.5 natural)
    "Union Berlin": "DEFENSIVA", "Getafe": "DEFENSIVA",
    "Atlético Madrid": "DEFENSIVA", "Wolverhampton Wanderers": "DEFENSIVA",
    "Burnley": "DEFENSIVA", "Cádiz": "DEFENSIVA",
    "Stuttgart": "DEFENSIVA", "Mainz 05": "DEFENSIVA",
    
    # Equipes Equilibradas
    "default": "EQUILIBRADA"
}

# =============================
# Utilitários de Cache e Persistência
# =============================
def carregar_json(caminho: str) -> dict:
    """Carrega dados JSON com verificação de timeout."""
    try:
        if os.path.exists(caminho):
            with open(caminho, "r", encoding='utf-8') as f:
                dados = json.load(f)
            
            # Verificar se o cache é muito antigo
            if caminho in [CACHE_JOGOS, CACHE_CLASSIFICACAO]:
                agora = datetime.now().timestamp()
                for key in list(dados.keys()):
                    if isinstance(dados[key], dict) and '_timestamp' in dados[key]:
                        if agora - dados[key]['_timestamp'] > CACHE_TIMEOUT:
                            del dados[key]
                    elif agora - os.path.getmtime(caminho) > CACHE_TIMEOUT:
                        return {}
            return dados
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"Erro ao carregar {caminho}: {e}")
    return {}

def salvar_json(caminho: str, dados: dict):
    """Salva dados JSON com timestamp."""
    try:
        # Adicionar timestamp para caches temporais
        if caminho in [CACHE_JOGOS, CACHE_CLASSIFICACAO]:
            dados['_timestamp'] = datetime.now().timestamp()
        
        with open(caminho, "w", encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
    except IOError as e:
        st.error(f"Erro ao salvar {caminho}: {e}")

def carregar_alertas() -> dict:
    return carregar_json(ALERTAS_PATH)

def salvar_alertas(alertas: dict):
    salvar_json(ALERTAS_PATH, alertas)

def carregar_cache_jogos() -> dict:
    return carregar_json(CACHE_JOGOS)

def salvar_cache_jogos(dados: dict):
    salvar_json(CACHE_JOGOS, dados)

def carregar_cache_classificacao() -> dict:
    return carregar_json(CACHE_CLASSIFICACAO)

def salvar_cache_classificacao(dados: dict):
    salvar_json(CACHE_CLASSIFICACAO, dados)

# =============================
# Utilitários de Data e Formatação
# =============================
def formatar_data_iso(data_iso: str) -> tuple[str, str]:
    """Formata data ISO para data e hora brasileira."""
    try:
        data_jogo = datetime.fromisoformat(data_iso.replace("Z", "+00:00")) - timedelta(hours=3)
        return data_jogo.strftime("%d/%m/%Y"), data_jogo.strftime("%H:%M")
    except ValueError:
        return "Data inválida", "Hora inválida"

def abreviar_nome(nome: str, max_len: int = 15) -> str:
    """Abrevia nomes longos para exibição."""
    if len(nome) <= max_len:
        return nome
    palavras = nome.split()
    abreviado = " ".join([p[0] + "." if len(p) > 2 else p for p in palavras])
    return abreviado[:max_len-3] + "..." if len(abreviado) > max_len else abreviado

# =============================
# Comunicação com APIs
# =============================
def enviar_telegram(msg: str, chat_id: str = TELEGRAM_CHAT_ID) -> bool:
    """Envia mensagem para o Telegram com tratamento de erro."""
    try:
        response = requests.get(
            BASE_URL_TG, 
            params={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
        return response.status_code == 200
    except requests.RequestException as e:
        st.error(f"Erro ao enviar para Telegram: {e}")
        return False

def obter_dados_api(url: str, timeout: int = 10) -> dict | None:
    """Faz requisição genérica à API com tratamento de erro."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Erro na requisição API: {e}")
        return None

def obter_classificacao(liga_id: str) -> dict:
    """Obtém dados de classificação da liga."""
    cache = carregar_cache_classificacao()
    
    if liga_id in cache:
        return cache[liga_id]

    url = f"{BASE_URL_FD}/competitions/{liga_id}/standings"
    data = obter_dados_api(url)
    
    if not data:
        return {}

    standings = {}
    for s in data.get("standings", []):
        if s["type"] != "TOTAL":
            continue
        for t in s["table"]:
            name = t["team"]["name"]
            standings[name] = {
                "scored": t.get("goalsFor", 0),
                "against": t.get("goalsAgainst", 0),
                "played": t.get("playedGames", 1),
                "position": t.get("position", 0),
                "points": t.get("points", 0)
            }

    cache[liga_id] = standings
    salvar_cache_classificacao(cache)
    return standings

def obter_jogos(liga_id: str, data: str) -> list:
    """Obtém jogos da liga para uma data específica."""
    cache = carregar_cache_jogos()
    key = f"{liga_id}_{data}"
    
    if key in cache:
        return cache[key]

    url = f"{BASE_URL_FD}/competitions/{liga_id}/matches?dateFrom={data}&dateTo={data}"
    data = obter_dados_api(url)
    
    jogos = data.get("matches", []) if data else []
    cache[key] = jogos
    salvar_cache_jogos(cache)
    
    return jogos

# =============================
# Lógica de Análise e Alertas - MELHORADA
# =============================
def determinar_perfil_equipe(nome_equipe: str) -> str:
    """Determina o perfil ofensivo/defensivo da equipe."""
    return PERFIS_EQUIPES.get(nome_equipe, PERFIS_EQUIPES["default"])

def calcular_fator_motivacao(dados_home: dict, dados_away: dict) -> float:
    """Calcula fator de motivação baseado na posição na tabela."""
    pos_home = dados_home.get("position", 10)
    pos_away = dados_away.get("position", 10)
    
    # Equipes no topo (luta por título) ou fundo (luta contra rebaixamento) têm mais motivação
    if pos_home <= 3 or pos_home >= 15 or pos_away <= 3 or pos_away >= 15:
        return 1.15
    elif pos_home <= 6 or pos_away <= 6:
        return 1.10
    return 1.0

def calcular_fator_estilo_jogo(perfil_home: str, perfil_away: str) -> float:
    """Calcula fator baseado no estilo de jogo das equipes."""
    fatores = {
        ("OFENSIVA", "OFENSIVA"): 1.35,
        ("OFENSIVA", "DEFENSIVA"): 1.10,
        ("DEFENSIVA", "OFENSIVA"): 1.10,
        ("DEFENSIVA", "DEFENSIVA"): 0.70,
        ("OFENSIVA", "EQUILIBRADA"): 1.20,
        ("EQUILIBRADA", "OFENSIVA"): 1.20,
        ("DEFENSIVA", "EQUILIBRADA"): 0.85,
        ("EQUILIBRADA", "DEFENSIVA"): 0.85,
        ("EQUILIBRADA", "EQUILIBRADA"): 1.0
    }
    return fatores.get((perfil_home, perfil_away), 1.0)

def aplicar_margem_seguranca(estimativa: float, confianca: float) -> tuple[float, float]:
    """Aplica margem de segurança conservadora nas estimativas."""
    if confianca < 70:
        # Redução adicional para confiança baixa
        estimativa_ajustada = estimativa * 0.85
        confianca_ajustada = confianca * 0.9
    elif confianca < 80:
        estimativa_ajustada = estimativa * 0.90
        confianca_ajustada = confianca * 0.95
    else:
        estimativa_ajustada = estimativa * 0.92
        confianca_ajustada = confianca * 0.98
    
    return max(estimativa_ajustada, 0.5), min(confianca_ajustada, 95)

def calcular_tendencia_melhorada(home: str, away: str, classificacao: dict) -> tuple[float, float, str]:
    """Calcula tendência de gols com algoritmo melhorado."""
    dados_home = classificacao.get(home, {"scored": 0, "against": 0, "played": 1, "position": 10, "points": 0})
    dados_away = classificacao.get(away, {"scored": 0, "against": 0, "played": 1, "position": 10, "points": 0})

    # Evitar divisão por zero
    played_home = max(dados_home["played"], 1)
    played_away = max(dados_away["played"], 1)

    # 1. Cálculo das médias base (pesos ajustados)
    media_home_feitos = dados_home["scored"] / played_home
    media_home_sofridos = dados_home["against"] / played_home
    media_away_feitos = dados_away["scored"] / played_away
    media_away_sofridos = dados_away["against"] / played_away

    # 2. Fatores contextuais
    perfil_home = determinar_perfil_equipe(home)
    perfil_away = determinar_perfil_equipe(away)
    fator_estilo = calcular_fator_estilo_jogo(perfil_home, perfil_away)
    fator_motivacao = calcular_fator_motivacao(dados_home, dados_away)

    # 3. Cálculo da estimativa bruta com pesos
    estimativa_bruta = (
        (media_home_feitos * 0.25) +      # Ataque da casa
        (media_away_sofridos * 0.25) +    # Defesa visitante
        (media_away_feitos * 0.20) +      # Ataque visitante  
        (media_home_sofridos * 0.20) +    # Defesa da casa
        ((media_home_feitos + media_away_feitos) * 0.10)  # Potencial ofensivo total
    )

    # 4. Aplicar fatores contextuais
    estimativa_contextual = estimativa_bruta * fator_estilo * fator_motivacao

    # 5. Determinar tendência base mais conservadora
    if estimativa_contextual >= 2.8:
        tendencia = "Mais 2.5"
        confianca_base = min(90, 65 + (estimativa_contextual - 2.8) * 12)
    elif estimativa_contextual >= 1.8:
        tendencia = "Mais 1.5"
        confianca_base = min(85, 55 + (estimativa_contextual - 1.8) * 15)
    else:
        tendencia = "Menos 2.5"
        confianca_base = min(80, 50 + (1.8 - estimativa_contextual) * 12)

    # 6. Aplicar margem de segurança
    estimativa_final, confianca_final = aplicar_margem_seguranca(estimativa_contextual, confianca_base)

    return estimativa_final, confianca_final, tendencia

def enviar_alerta_telegram(fixture: dict, tendencia: str, estimativa: float, confianca: float):
    """Envia alerta formatado para o Telegram."""
    home = fixture["homeTeam"]["name"]
    away = fixture["awayTeam"]["name"]
    data_formatada, hora_formatada = formatar_data_iso(fixture["utcDate"])
    competicao = fixture.get("competition", {}).get("name", "Desconhecido")

    status = fixture.get("status", "DESCONHECIDO")
    gols_home = fixture.get("score", {}).get("fullTime", {}).get("home")
    gols_away = fixture.get("score", {}).get("fullTime", {}).get("away")
    
    placar = f"{gols_home} x {gols_away}" if gols_home is not None and gols_away is not None else None

    # Adicionar análise contextual
    perfil_home = determinar_perfil_equipe(home)
    perfil_away = determinar_perfil_equipe(away)
    analise_contexto = f"🏠 {perfil_home} | 🚌 {perfil_away}"

    msg = (
        f"⚽ <b>Alerta de Gols!</b>\n"
        f"🏟️ {home} vs {away}\n"
        f"📅 {data_formatada} ⏰ {hora_formatada} (BRT)\n"
        f"📌 Status: {status}\n"
        f"🔍 {analise_contexto}\n"
    )
    
    if placar:
        msg += f"📊 Placar: <b>{placar}</b>\n"
        
    msg += (
        f"📈 Tendência: <b>{tendencia}</b>\n"
        f"🎯 Estimativa: <b>{estimativa:.2f} gols</b>\n"
        f"💯 Confiança: <b>{confianca:.0f}%</b>\n"
        f"🏆 Liga: {competicao}"
    )
    
    enviar_telegram(msg)

def verificar_enviar_alerta(fixture: dict, tendencia: str, estimativa: float, confianca: float):
    """Verifica e envia alerta se necessário."""
    alertas = carregar_alertas()
    fixture_id = str(fixture["id"])
    
    if fixture_id not in alertas:
        alertas[fixture_id] = {
            "tendencia": tendencia,
            "estimativa": estimativa,
            "confianca": confianca,
            "conferido": False
        }
        enviar_alerta_telegram(fixture, tendencia, estimativa, confianca)
        salvar_alertas(alertas)

# =============================
# Geração de Relatórios
# =============================
def gerar_relatorio_pdf(jogos_conferidos: list) -> io.BytesIO:
    """Gera relatório PDF dos jogos conferidos."""
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter, 
                          rightMargin=20, leftMargin=20, 
                          topMargin=20, bottomMargin=20)

    data = [["Jogo", "Tendência", "Estimativa", "Confiança", 
             "Placar", "Status", "Resultado", "Hora"]] + jogos_conferidos

    table = Table(data, repeatRows=1, 
                 colWidths=[120, 70, 60, 60, 50, 70, 60, 70])
    
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4B4B4B")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F5F5F5")),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ])

    # Alternar cores das linhas
    for i in range(1, len(data)):
        bg_color = colors.HexColor("#E0E0E0") if i % 2 == 0 else colors.HexColor("#F5F5F5")
        style.add('BACKGROUND', (0,i), (-1,i), bg_color)

    table.setStyle(style)
    pdf.build([table])
    buffer.seek(0)
    return buffer

# =============================
# Interface Streamlit
# =============================
def main():
    st.set_page_config(page_title="⚽ Alerta de Gols", layout="wide")
    st.title("⚽ Sistema de Alertas Automáticos de Gols - Versão Melhorada")

    # Sidebar para configurações
    with st.sidebar:
        st.header("Configurações")
        top_n = st.selectbox("📊 Jogos no Top", [3, 5, 10], index=0)
        
        st.header("⚙️ Configurações de Análise")
        usar_analise_melhorada = st.checkbox("🧠 Usar Análise Melhorada", value=True)
        mostrar_perfis = st.checkbox("👥 Mostrar Perfis das Equipes", value=False)
        
        st.info("A análise melhorada inclui:\n- Perfis ofensivos/defensivos\n- Fatores de motivação\n- Margens de segurança")

    # Controles principais
    col1, col2 = st.columns([2, 1])
    
    with col1:
        data_selecionada = st.date_input(
            "📅 Data para análise:", 
            value=datetime.today()
        )
    
    with col2:
        todas_ligas = st.checkbox(
            "🌍 Todas as ligas", 
            value=True,
            help="Buscar jogos de todas as ligas disponíveis"
        )

    liga_selecionada = None
    if not todas_ligas:
        liga_selecionada = st.selectbox(
            "📌 Liga específica:", 
            list(LIGA_DICT.keys())
        )

    # Processamento de jogos
    if st.button("🔍 Buscar Partidas", type="primary"):
        processar_jogos(data_selecionada, todas_ligas, liga_selecionada, top_n, usar_analise_melhorada, mostrar_perfis)

    # Botões de ação
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Atualizar Status"):
            atualizar_status_partidas()
    
    with col2:
        if st.button("📊 Conferir Resultados"):
            conferir_resultados()
    
    with col3:
        if st.button("🧹 Limpar Cache"):
            limpar_caches()

def processar_jogos(data_selecionada, todas_ligas, liga_selecionada, top_n, usar_analise_melhorada=True, mostrar_perfis=False):
    """Processa e analisa os jogos do dia."""
    hoje = data_selecionada.strftime("%Y-%m-%d")
    ligas_busca = LIGA_DICT.values() if todas_ligas else [LIGA_DICT[liga_selecionada]]
    
    st.write(f"⏳ Buscando jogos para {data_selecionada.strftime('%d/%m/%Y')}...")
    
    top_jogos = []
    progress_bar = st.progress(0)
    total_ligas = len(ligas_busca)

    for i, liga_id in enumerate(ligas_busca):
        classificacao = obter_classificacao(liga_id)
        jogos = obter_jogos(liga_id, hoje)

        for match in jogos:
            home = match["homeTeam"]["name"]
            away = match["awayTeam"]["name"]
            
            # Usar análise melhorada ou antiga
            if usar_analise_melhorada:
                estimativa, confianca, tendencia = calcular_tendencia_melhorada(home, away, classificacao)
            else:
                estimativa, confianca, tendencia = calcular_tendencia_antiga(home, away, classificacao)

            verificar_enviar_alerta(match, tendencia, estimativa, confianca)

            # Adicionar informações de perfil se solicitado
            info_extra = ""
            if mostrar_perfis:
                perfil_home = determinar_perfil_equipe(home)
                perfil_away = determinar_perfil_equipe(away)
                info_extra = f" ({perfil_home[0]}/{perfil_away[0]})"

            top_jogos.append({
                "id": match["id"],
                "home": home + info_extra,
                "away": away,
                "tendencia": tendencia,
                "estimativa": estimativa,
                "confianca": confianca,
                "liga": match.get("competition", {}).get("name", "Desconhecido"),
                "hora": datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")) - timedelta(hours=3),
                "status": match.get("status", "DESCONHECIDO"),
            })

        progress_bar.progress((i + 1) / total_ligas)

    # Enviar top jogos
    if top_jogos:
        enviar_top_jogos(top_jogos, top_n)
        st.success(f"✅ Análise concluída! {len(top_jogos)} jogos processados.")
        
        # Mostrar estatísticas da análise
        if usar_analise_melhorada:
            mostrar_estatisticas_analise(top_jogos)
    else:
        st.warning("⚠️ Nenhum jogo encontrado para a data selecionada.")

def calcular_tendencia_antiga(home: str, away: str, classificacao: dict) -> tuple[float, float, str]:
    """Função antiga mantida para comparação."""
    dados_home = classificacao.get(home, {"scored": 0, "against": 0, "played": 1})
    dados_away = classificacao.get(away, {"scored": 0, "against": 0, "played": 1})

    played_home = max(dados_home["played"], 1)
    played_away = max(dados_away["played"], 1)

    media_home_feitos = dados_home["scored"] / played_home
    media_home_sofridos = dados_home["against"] / played_home
    media_away_feitos = dados_away["scored"] / played_away
    media_away_sofridos = dados_away["against"] / played_away

    estimativa = ((media_home_feitos + media_away_sofridos) / 2 +
                  (media_away_feitos + media_home_sofridos) / 2)

    # Determinar tendência e confiança (lógica antiga)
    if estimativa >= 3.0:
        tendencia = "Mais 2.5"
        confianca = min(95, 70 + (estimativa - 3.0) * 10)
    elif estimativa >= 2.0:
        tendencia = "Mais 1.5"
        confianca = min(90, 60 + (estimativa - 2.0) * 10)
    else:
        tendencia = "Menos 2.5"
        confianca = min(85, 55 + (2.0 - estimativa) * 10)

    return estimativa, confianca, tendencia

def mostrar_estatisticas_analise(jogos: list):
    """Mostra estatísticas da análise realizada."""
    if not jogos:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_estimativa = sum(j['estimativa'] for j in jogos) / len(jogos)
        st.metric("📊 Média Estimativa", f"{avg_estimativa:.2f}")
    
    with col2:
        avg_confianca = sum(j['confianca'] for j in jogos) / len(jogos)
        st.metric("💯 Confiança Média", f"{avg_confianca:.1f}%")
    
    with col3:
        over_25 = sum(1 for j in jogos if j['tendencia'] == "Mais 2.5")
        st.metric("⚡ Over 2.5", f"{over_25}")
    
    with col4:
        under_25 = sum(1 for j in jogos if j['tendencia'] == "Menos 2.5")
        st.metric("🛡️ Under 2.5", f"{under_25}")

def enviar_top_jogos(jogos: list, top_n: int):
    """Envia os top N jogos para o Telegram (somente jogos não finalizados)."""
    # 🔎 Filtrar apenas jogos que ainda não começaram
    jogos_filtrados = [j for j in jogos if j["status"] not in ["FINISHED", "IN_PLAY", "POSTPONED", "SUSPENDED"]]

    if not jogos_filtrados:
        st.warning("⚠️ Nenhum jogo elegível para o Top Jogos (todos já iniciados ou finalizados).")
        return

    # Ordenar por confiança e pegar top N
    top_jogos_sorted = sorted(jogos_filtrados, key=lambda x: x["confianca"], reverse=True)[:top_n]

    msg = f"📢 TOP {top_n} Jogos do Dia\n\n"
    for j in top_jogos_sorted:
        hora_format = j["hora"].strftime("%H:%M")
        msg += (
            f"🏟️ {j['home']} vs {j['away']}\n"
            f"🕒 {hora_format} BRT | Liga: {j['liga']} | Status: {j['status']}\n"
            f"📈 Tendência: {j['tendencia']} | Estimativa: {j['estimativa']:.2f} | "
            f"💯 Confiança: {j['confianca']:.0f}%\n\n"
        )

    # Envio ao Telegram
    if enviar_telegram(msg, TELEGRAM_CHAT_ID_ALT2):
        st.success(f"🚀 Top {top_n} jogos (sem finalizados) enviados para o canal!")
    else:
        st.error("❌ Erro ao enviar top jogos para o Telegram")

def atualizar_status_partidas():
    """Atualiza o status das partidas em cache."""
    cache_jogos = carregar_cache_jogos()
    mudou = False

    for key in cache_jogos.keys():
        if key == "_timestamp":
            continue
            
        liga_id, data = key.split("_")
        try:
            url = f"{BASE_URL_FD}/competitions/{liga_id}/matches?dateFrom={data}&dateTo={data}"
            data_api = obter_dados_api(url)
            
            if data_api and "matches" in data_api:
                cache_jogos[key] = data_api["matches"]
                mudou = True
                
        except Exception as e:
            st.error(f"Erro ao atualizar liga {liga_id}: {e}")

    if mudou:
        salvar_cache_jogos(cache_jogos)
        st.success("✅ Status das partidas atualizado!")
    else:
        st.info("ℹ️ Nenhuma atualização disponível.")

def conferir_resultados():
    """Conferência de resultados dos jogos alertados."""
    alertas = carregar_alertas()
    jogos_cache = carregar_cache_jogos()
    
    if not alertas:
        st.info("ℹ️ Nenhum alerta para conferir.")
        return

    jogos_conferidos = []
    mudou = False

    for fixture_id, info in alertas.items():
        if info.get("conferido"):
            continue

        # Encontrar jogo no cache
        jogo_dado = None
        for key, jogos in jogos_cache.items():
            if key == "_timestamp":
                continue
            for match in jogos:
                if str(match["id"]) == fixture_id:
                    jogo_dado = match
                    break
            if jogo_dado:
                break

        if not jogo_dado:
            continue

        # Processar resultado
        resultado_info = processar_resultado_jogo(jogo_dado, info)
        if resultado_info:
            exibir_resultado_streamlit(resultado_info)
            
            if resultado_info["status"] == "FINISHED":
                enviar_resultado_telegram(resultado_info)
                info["conferido"] = True
                mudou = True

        # Coletar dados para PDF
        jogos_conferidos.append(preparar_dados_pdf(jogo_dado, info, resultado_info))

    if mudou:
        salvar_alertas(alertas)
        st.success("✅ Resultados conferidos e atualizados!")

    # Gerar PDF se houver jogos
    if jogos_conferidos:
        gerar_pdf_jogos(jogos_conferidos)

def processar_resultado_jogo(jogo: dict, info: dict) -> dict | None:
    """Processa o resultado de um jogo."""
    home = jogo["homeTeam"]["name"]
    away = jogo["awayTeam"]["name"]
    status = jogo.get("status", "DESCONHECIDO")
    gols_home = jogo.get("score", {}).get("fullTime", {}).get("home")
    gols_away = jogo.get("score", {}).get("fullTime", {}).get("away")
    
    placar = f"{gols_home} x {gols_away}" if gols_home is not None and gols_away is not None else "-"
    total_gols = (gols_home or 0) + (gols_away or 0)

    # Determinar resultado da aposta
    resultado = "⏳ Aguardando"
    if status == "FINISHED":
        tendencia = info["tendencia"]
        if "Mais 2.5" in tendencia:
            resultado = "🟢 GREEN" if total_gols > 2 else "🔴 RED"
        elif "Mais 1.5" in tendencia:
            resultado = "🟢 GREEN" if total_gols > 1 else "🔴 RED"
        elif "Menos 2.5" in tendencia:
            resultado = "🟢 GREEN" if total_gols < 3 else "🔴 RED"
        else:
            resultado = "⚪ INDEFINIDO"

    return {
        "home": home,
        "away": away,
        "status": status,
        "placar": placar,
        "tendencia": info["tendencia"],
        "estimativa": info["estimativa"],
        "confianca": info["confianca"],
        "resultado": resultado,
        "total_gols": total_gols
    }

def exibir_resultado_streamlit(resultado: dict):
    """Exibe resultado formatado no Streamlit."""
    bg_color = "#1e4620" if resultado["resultado"] == "🟢 GREEN" else \
               "#5a1e1e" if resultado["resultado"] == "🔴 RED" else "#2c2c2c"
    
    st.markdown(f"""
    <div style="border:1px solid #444; border-radius:10px; padding:12px; margin-bottom:10px;
                background-color:{bg_color}; font-size:15px; color:#f1f1f1;">
        <b>🏟️ {resultado['home']} vs {resultado['away']}</b><br>
        📌 Status: <b>{resultado['status']}</b><br>
        ⚽ Tendência: <b>{resultado['tendencia']}</b> | Estim.: {resultado['estimativa']:.2f} | Conf.: {resultado['confianca']:.0f}%<br>
        📊 Placar: <b>{resultado['placar']}</b><br>
        ✅ Resultado: {resultado['resultado']}
    </div>
    """, unsafe_allow_html=True)

def enviar_resultado_telegram(resultado: dict):
    """Envia resultado para o Telegram."""
    msg = (
        f"📊 <b>Resultado Conferido</b>\n"
        f"🏟️ {resultado['home']} vs {resultado['away']}\n"
        f"⚽ Tendência: {resultado['tendencia']} | Estim.: {resultado['estimativa']:.2f} | Conf.: {resultado['confianca']:.0f}%\n"
        f"📊 Placar Final: <b>{resultado['placar']}</b>\n"
        f"✅ Resultado: <b>{resultado['resultado']}</b>"
    )
    enviar_telegram(msg, TELEGRAM_CHAT_ID_ALT2)

def preparar_dados_pdf(jogo: dict, info: dict, resultado: dict) -> list:
    """Prepara dados para geração do PDF."""
    home = abreviar_nome(jogo["homeTeam"]["name"])
    away = abreviar_nome(jogo["awayTeam"]["name"])
    hora = datetime.fromisoformat(jogo["utcDate"].replace("Z", "+00:00")) - timedelta(hours=3)
    
    return [
        f"{home} vs {away}",
        info["tendencia"],
        f"{info['estimativa']:.2f}",
        f"{info['confianca']:.0f}%",
        resultado["placar"] if resultado else "-",
        jogo.get("status", "DESCONHECIDO"),
        resultado["resultado"] if resultado else "⏳ Aguardando",
        hora.strftime("%d/%m %H:%M")
    ]

def gerar_pdf_jogos(jogos_conferidos: list):
    """Gera e disponibiliza PDF dos jogos conferidos."""
    df_conferidos = pd.DataFrame(jogos_conferidos, columns=[
        "Jogo", "Tendência", "Estimativa", "Confiança", 
        "Placar", "Status", "Resultado", "Hora"
    ])

    buffer = gerar_relatorio_pdf(jogos_conferidos)
    
    st.download_button(
        label="📄 Baixar Relatório PDF",
        data=buffer,
        file_name=f"jogos_conferidos_{datetime.today().strftime('%Y-%m-%d')}.pdf",
        mime="application/pdf"
    )

def limpar_caches():
    """Limpa todos os caches do sistema."""
    try:
        for cache_file in [CACHE_JOGOS, CACHE_CLASSIFICACAO, ALERTAS_PATH]:
            if os.path.exists(cache_file):
                os.remove(cache_file)
        st.success("✅ Caches limpos com sucesso!")
    except Exception as e:
        st.error(f"❌ Erro ao limpar caches: {e}")

if __name__ == "__main__":
    main()
