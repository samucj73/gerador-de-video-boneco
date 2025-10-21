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
# Dicionário de Ligas (expandido)
# =============================
LIGA_DICT = {
    # Europa
    "UEFA Champions League": "CL",
    "Premier League (Inglaterra)": "PL",
    "Bundesliga": "BL1",
    "Ligue 1": "FL1",
    "Serie A (Itália)": "SA",
    "Eredivisie": "DED",
    "Championship (Inglaterra)": "ELC",
    "Primeira Liga (Portugal)": "PPL",
    # América
    "Campeonato Brasileiro Série A": "BSA",
    "Copa Libertadores": "COPA_LIB",
    "Copa Sul-Americana": "COPA_SULAM",
    "MLS": "MLS",
    "Liga MX": "LIGAMX",
    "Argentina Primera División": "ARG_PRIMERA",
    # Mundial / Outros
    "FIFA World Cup": "WC",
    "European Championship": "EC",
    "Saudi Pro League": "SAUDI",
    "Primera Division": "PD",
}

# Inverso rápido por nome da competição (algumas APIs usam 'name' ou 'code')
def map_competition_name_to_id(name: str) -> str | None:
    if not name:
        return None
    for k, v in LIGA_DICT.items():
        if k.lower() in name.lower() or (name.lower() in k.lower()):
            return v
    return None

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
                # Se estrutura com timestamps por chave
                for key in list(dados.keys()):
                    if isinstance(dados[key], dict) and '_timestamp' in dados[key]:
                        if agora - dados[key]['_timestamp'] > CACHE_TIMEOUT:
                            del dados[key]
                # Se o arquivo em si for antigo, considerar vazio
                if agora - os.path.getmtime(caminho) > CACHE_TIMEOUT:
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
            # manter estrutura por chave, mas adicionar timestamp geral
            dados['_timestamp'] = datetime.now().timestamp()
        
        with open(caminho, "w", encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
    except IOError as e:
        st.error(f"Erro ao salvar {caminho}: {e}")

def carregar_alertas() -> dict:
    return carregar_json(ALERTAS_PATH) or {}

def salvar_alertas(alertas: dict):
    salvar_json(ALERTAS_PATH, alertas)

def carregar_cache_jogos() -> dict:
    return carregar_json(CACHE_JOGOS) or {}

def salvar_cache_jogos(dados: dict):
    salvar_json(CACHE_JOGOS, dados)

def carregar_cache_classificacao() -> dict:
    return carregar_json(CACHE_CLASSIFICACAO) or {}

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
    except Exception:
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
    """Obtém dados de classificação da liga (cacheado)."""
    if not liga_id:
        return {}
    cache = carregar_cache_classificacao()
    
    if liga_id in cache:
        return cache[liga_id]

    # Se liga_id aparenta ser custom (ex: COPA_LIB), não chamar endpoint oficial
    # Tentar usar mapping conhecido para endpoints reais; por enquanto, somente chamadas diretas
    try:
        url = f"{BASE_URL_FD}/competitions/{liga_id}/standings"
        data = obter_dados_api(url)
    except Exception:
        data = None

    if not data:
        return {}

    standings = {}
    for s in data.get("standings", []):
        if s.get("type") != "TOTAL":
            continue
        for t in s.get("table", []):
            name = t["team"]["name"]
            standings[name] = {
                "scored": t.get("goalsFor", 0),
                "against": t.get("goalsAgainst", 0),
                "played": t.get("playedGames", 1)
            }

    cache[liga_id] = standings
    salvar_cache_classificacao(cache)
    return standings

def obter_jogos(liga_id: str, data: str) -> list:
    """Obtém jogos da liga para uma data específica (cacheado)."""
    cache = carregar_cache_jogos()
    key = f"{liga_id}_{data}"
    
    if key in cache:
        return cache[key]

    url = f"{BASE_URL_FD}/competitions/{liga_id}/matches?dateFrom={data}&dateTo={data}"
    data_api = obter_dados_api(url)
    
    jogos = data_api.get("matches", []) if data_api else []
    cache[key] = jogos
    salvar_cache_jogos(cache)
    
    return jogos

# =============================
# Lógica de Análise e Alertas
# =============================
def calcular_tendencia(home: str, away: str, classificacao: dict) -> tuple[float, float, str]:
    """Calcula tendência de gols baseada em estatísticas históricas."""
    dados_home = classificacao.get(home, {"scored": 0, "against": 0, "played": 1})
    dados_away = classificacao.get(away, {"scored": 0, "against": 0, "played": 1})

    # Evitar divisão por zero
    played_home = max(dados_home["played"], 1)
    played_away = max(dados_away["played"], 1)

    media_home_feitos = dados_home["scored"] / played_home
    media_home_sofridos = dados_home["against"] / played_home
    media_away_feitos = dados_away["scored"] / played_away
    media_away_sofridos = dados_away["against"] / played_away

    estimativa = ((media_home_feitos + media_away_sofridos) / 2 +
                  (media_away_feitos + media_home_sofridos) / 2)

    # Determinar tendência e confiança
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

def calcular_favorito_vitoria(home: str, away: str, classificacao: dict) -> tuple[str, float]:
    """
    Calcula o time favorito para vencer com base em desempenho ofensivo/defensivo.
    Retorna o time favorito e a confiança estimada (0-100).
    Fórmula simples: considera diferença de (gols marcados - gols sofridos) por jogo.
    """
    dados_home = classificacao.get(home, {"scored": 0, "against": 0, "played": 1})
    dados_away = classificacao.get(away, {"scored": 0, "against": 0, "played": 1})

    played_home = max(dados_home.get("played", 1), 1)
    played_away = max(dados_away.get("played", 1), 1)

    saldo_home = (dados_home.get("scored", 0) - dados_home.get("against", 0)) / played_home
    saldo_away = (dados_away.get("scored", 0) - dados_away.get("against", 0)) / played_away

    diff = saldo_home - saldo_away

    # Se ambos os times não tiverem dados, sem favorito
    if played_home == 0 and played_away == 0:
        return "Empate", 50.0

    # Regras simples para favorito
    if abs(diff) < 0.15:
        favorito = "Empate"
        confianca = 50.0
    elif diff > 0:
        favorito = f"{home} vencer"
        confianca = min(95.0, 60.0 + diff * 20.0)  # escala para confiança
    else:
        favorito = f"{away} vencer"
        confianca = min(95.0, 60.0 + abs(diff) * 20.0)

    # Normalizar para valores plausíveis
    confianca = max(50.0, min(95.0, confianca))
    return favorito, round(confianca, 1)

def get_liga_id_from_fixture(fixture: dict) -> str | None:
    """Tenta extrair um liga_id utilizável a partir do fixture (code ou nome)."""
    comp = fixture.get("competition", {}) if fixture else {}
    code = comp.get("code") or comp.get("id") or None
    name = comp.get("name")
    if code:
        return code
    # mapear pelo nome
    mapped = map_competition_name_to_id(name)
    return mapped

def enviar_alerta_telegram(fixture: dict, tendencia: str, estimativa: float, confianca: float):
    """Envia alerta formatado para o Telegram, agora incluindo favorito."""
    home = fixture["homeTeam"]["name"]
    away = fixture["awayTeam"]["name"]
    data_formatada, hora_formatada = formatar_data_iso(fixture["utcDate"])
    competicao = fixture.get("competition", {}).get("name", "Desconhecido")

    status = fixture.get("status", "DESCONHECIDO")
    gols_home = fixture.get("score", {}).get("fullTime", {}).get("home")
    gols_away = fixture.get("score", {}).get("fullTime", {}).get("away")
    
    placar = f"{gols_home} x {gols_away}" if gols_home is not None and gols_away is not None else None

    # Obter classificação pela competição (tentativa)
    liga_id = get_liga_id_from_fixture(fixture)
    classificacao = obter_classificacao(liga_id) if liga_id else {}

    favorito, conf_fav = calcular_favorito_vitoria(home, away, classificacao)

    msg = (
        f"⚽ <b>Alerta de Gols!</b>\n"
        f"🏟️ {home} vs {away}\n"
        f"📅 {data_formatada} ⏰ {hora_formatada} (BRT)\n"
        f"📌 Status: {status}\n"
    )
    
    if placar:
        msg += f"📊 Placar: <b>{placar}</b>\n"
        
    msg += (
        f"📈 Tendência: <b>{tendencia}</b>\n"
        f"🎯 Estimativa: <b>{estimativa:.2f} gols</b>\n"
        f"💯 Confiança: <b>{confianca:.0f}%</b>\n"
        f"🏆 Liga: {competicao}\n"
        f"🏆 Favorito: <b>{favorito}</b> ({conf_fav:.0f}%)"
    )
    
    enviar_telegram(msg)

def verificar_enviar_alerta(fixture: dict, tendencia: str, estimativa: float, confianca: float):
    """Verifica e envia alerta se necessário. Agora salva favorito/conf_fav no cache."""
    alertas = carregar_alertas()
    fixture_id = str(fixture["id"])
    
    if fixture_id not in alertas:
        # calcular favorito no momento do alerta
        liga_id = get_liga_id_from_fixture(fixture)
        classificacao = obter_classificacao(liga_id) if liga_id else {}
        favorito, conf_fav = calcular_favorito_vitoria(
            fixture["homeTeam"]["name"], fixture["awayTeam"]["name"], classificacao
        )

        alertas[fixture_id] = {
            "tendencia": tendencia,
            "estimativa": estimativa,
            "confianca": confianca,
            "favorito": favorito,
            "conf_fav": conf_fav,
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
    st.title("⚽ Sistema de Alertas Automáticos de Gols")

    # Sidebar para configurações
    with st.sidebar:
        st.header("Configurações")
        top_n = st.selectbox("📊 Jogos no Top", [3, 5, 10], index=0)
        st.info("Configure as opções de análise")

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
        processar_jogos(data_selecionada, todas_ligas, liga_selecionada, top_n)

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

def processar_jogos(data_selecionada, todas_ligas, liga_selecionada, top_n):
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
            estimativa, confianca, tendencia = calcular_tendencia(home, away, classificacao)

            verificar_enviar_alerta(match, tendencia, estimativa, confianca)

            top_jogos.append({
                "id": match["id"],
                "home": home,
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
    else:
        st.warning("⚠️ Nenhum jogo encontrado para a data selecionada.")

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

    for key in list(cache_jogos.keys()):
        if key == "_timestamp":
            continue
            
        try:
            liga_id, data = key.split("_", 1)
        except ValueError:
            continue

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
        tendencia = info.get("tendencia", "")
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
        "tendencia": info.get("tendencia"),
        "estimativa": info.get("estimativa"),
        "confianca": info.get("confianca"),
        "favorito": info.get("favorito", "N/A"),
        "conf_fav": info.get("conf_fav", 0),
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
        🏆 Favorito: <b>{resultado.get('favorito', 'N/A')}</b> ({resultado.get('conf_fav', 0):.0f}%)<br>
        ⚽ Tendência: <b>{resultado['tendencia']}</b> | Estim.: {resultado['estimativa']:.2f} | Conf.: {resultado['confianca']:.0f}%<br>
        📊 Placar: <b>{resultado['placar']}</b><br>
        ✅ Resultado: {resultado['resultado']}
    </div>
    """, unsafe_allow_html=True)

def enviar_resultado_telegram(resultado: dict):
    """Envia resultado para o Telegram (inclui favorito)."""
    msg = (
        f"📊 <b>Resultado Conferido</b>\n"
        f"🏟️ {resultado['home']} vs {resultado['away']}\n"
        f"🏆 Favorito: <b>{resultado.get('favorito','N/A')}</b> ({resultado.get('conf_fav',0):.0f}%)\n"
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

    favorito_str = ""
    if info.get("favorito"):
        favorito_str = f" ({info.get('favorito')})"

    return [
        f"{home} vs {away}{favorito_str}",
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
