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
import time
import hashlib

# =============================
# Configurações e Segurança
# =============================

# Apenas TELEGRAM precisa de token (ESPN é pública)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003073115320")
TELEGRAM_CHAT_ID_ALT2 = os.getenv("TELEGRAM_CHAT_ID_ALT2", "-1002754276285")

BASE_URL_TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# Constantes
ALERTAS_PATH = "alertas.json"
CACHE_JOGOS = "cache_jogos.json"
CACHE_CLASSIFICACAO = "cache_classificacao.json"
CACHE_TIMEOUT = 3600  # 1 hora em segundos

# =============================
# Mapeamento de Ligas (ESPN slugs)
# =============================
LIGAS_ESPN = {
    "UEFA Champions League": "uefa.champions",
    "Copa Libertadores": "conmebol.libertadores",
    "Copa Sul-Americana": "conmebol.sudamericana"
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
            # Se for cache global e tiver timestamp de arquivo, checar idade do arquivo
            if caminho in [CACHE_JOGOS, CACHE_CLASSIFICACAO]:
                agora = datetime.now().timestamp()
                # se dados tiverem chaves com _timestamp, remover entradas antigas
                if isinstance(dados, dict):
                    # Se o arquivo inteiro possuir apenas um _timestamp, checar modificação do arquivo
                    if '_timestamp' in dados:
                        if agora - dados['_timestamp'] > CACHE_TIMEOUT:
                            return {}
                    # Também checar cada entrada que seja dict e tenha _timestamp
                    for key in list(dados.keys()):
                        if key == '_timestamp':
                            continue
                        val = dados.get(key)
                        if isinstance(val, dict) and '_timestamp' in val:
                            if agora - val['_timestamp'] > CACHE_TIMEOUT:
                                del dados[key]
            return dados
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"Erro ao carregar {caminho}: {e}")
    return {}

def salvar_json(caminho: str, dados: dict):
    """Salva dados JSON com timestamp."""
    try:
        # Adicionar timestamp para caches temporais
        if caminho in [CACHE_JOGOS, CACHE_CLASSIFICACAO]:
            # Se for cache por chave, já pode ter sido fornecido; caso contrário, colocamos timestamp geral
            if isinstance(dados, dict):
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
    """Formata data ISO para data e hora brasileira (BRT)."""
    try:
        data_jogo = datetime.fromisoformat(data_iso.replace("Z", "+00:00")) - timedelta(hours=3)
        return data_jogo.strftime("%d/%m/%Y"), data_jogo.strftime("%H:%M")
    except Exception:
        return "Data inválida", "Hora inválida"

def abreviar_nome(nome: str, max_len: int = 15) -> str:
    """Abrevia nomes longos para exibição."""
    if not nome:
        return ""
    if len(nome) <= max_len:
        return nome
    palavras = nome.split()
    abreviado = " ".join([p[0] + "." if len(p) > 2 else p for p in palavras])
    return abreviado[:max_len-3] + "..." if len(abreviado) > max_len else abreviado

# =============================
# Comunicação com Telegram
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

# =============================
# API ESPN - Buscar jogos
# =============================
def buscar_jogos_espn(liga_slug: str, data_str: str = None) -> list:
    """
    Busca jogos na API pública da ESPN.
    liga_slug: slug ESPN, ex: "uefa.champions"
    data_str: "YYYY-MM-DD" (opcional) - a ESPN aceita param 'dates' no formato YYYYMMDD em alguns endpoints
    """
    try:
        # Monta URL (scoreboard)
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{liga_slug}/scoreboard"
        params = {}
        if data_str:
            # ESPN costuma aceitar dates no formato YYYYMMDD
            try:
                d = datetime.strptime(data_str, "%Y-%m-%d")
                params['dates'] = d.strftime("%Y%m%d")
            except Exception:
                pass
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        dados = response.json()
        partidas = []

        for evento in dados.get("events", []):
            # Extrair dados principais
            evt_id = evento.get("id") or hashlib.sha1(json.dumps(evento, sort_keys=True).encode()).hexdigest()
            hora_iso = evento.get("date")  # já em ISO
            # Pegar competição (pode vir em competitions[0])
            comp = evento.get("competitions", [])
            comp0 = comp[0] if comp else {}
            league_name = comp0.get("league", {}).get("name") or evento.get("league", {}).get("name") or liga_slug

            competitors = comp0.get("competitors", [])
            # ESPN geralmente coloca home como competitor com 'homeAway' == 'home'
            home_name = away_name = None
            home_score = away_score = None
            for c in competitors:
                if c.get("homeAway") == "home":
                    home_name = c.get("team", {}).get("displayName")
                    home_score = c.get("score")
                elif c.get("homeAway") == "away":
                    away_name = c.get("team", {}).get("displayName")
                    away_score = c.get("score")
            # fallback: se len == 2 e não setados por homeAway
            if not home_name or not away_name:
                if len(competitors) >= 2:
                    home_name = competitors[0].get("team", {}).get("displayName")
                    away_name = competitors[1].get("team", {}).get("displayName")
                    home_score = competitors[0].get("score")
                    away_score = competitors[1].get("score")

            status_desc = evento.get("status", {}).get("type", {}).get("description") or evento.get("status", {}).get("type", {}).get("state") or "-"
            status_state = evento.get("status", {}).get("type", {}).get("state") or "SCHEDULED"

            partidas.append({
                # Normalizar formato para o resto do app
                "id": evt_id,
                "utcDate": hora_iso,
                "competition": {"name": league_name},
                "homeTeam": {"name": home_name or "-"},
                "awayTeam": {"name": away_name or "-"},
                "status": status_state,
                "statusDesc": status_desc,
                "score": {
                    "fullTime": {
                        "home": int(home_score) if (home_score not in (None, "") and str(home_score).isdigit()) else None,
                        "away": int(away_score) if (away_score not in (None, "") and str(away_score).isdigit()) else None
                    }
                }
            })
        return partidas
    except requests.RequestException as e:
        st.error(f"Erro ao buscar jogos da ESPN: {e}")
        return []
    except Exception as e:
        st.error(f"Erro ao parsear resposta ESPN: {e}")
        return []

# =============================
# Funções que substituem obter_jogos / obter_classificacao
# =============================
def obter_classificacao(liga_slug: str) -> dict:
    """
    Por enquanto retorna um dicionário vazio (placeholder).
    Mantive a interface para compatibilidade com calcular_tendencia.
    Se quiser, depois posso implementar scraping/extração de ranking da ESPN.
    """
    # Poderíamos carregar um cache local se já tivéssemos estatísticas por time.
    return {}

def obter_jogos(liga_slug: str, data: str) -> list:
    """
    Retorna lista de matches no formato esperado pelo app, usando ESPN.
    Usa cache por liga_slug + data.
    """
    cache = carregar_cache_jogos()
    key = f"{liga_slug}_{data}"
    agora = datetime.now().timestamp()

    # Se cache existe e não expirou, devolver
    if key in cache:
        entry = cache[key]
        # cada entrada pode ter '_timestamp'
        if isinstance(entry, dict) and '_timestamp' in entry:
            if agora - entry['_timestamp'] <= CACHE_TIMEOUT:
                return entry.get('matches', [])
        else:
            # fallback: se for lista, assumimos válido
            return entry

    # Buscar na ESPN
    partidos = buscar_jogos_espn(liga_slug, data)
    # Salvar no cache com timestamp
    cache[key] = {
        "matches": partidos,
        "_timestamp": agora
    }
    salvar_cache_jogos(cache)
    return partidos

# =============================
# Lógica de Análise e Alertas
# =============================
def calcular_tendencia(home: str, away: str, classificacao: dict) -> tuple[float, float, str]:
    """Calcula tendência de gols baseada em estatísticas históricas (se houver)."""
    # Se não houver classificação ou times, usar valores default
    dados_home = classificacao.get(home, {"scored": 0, "against": 0, "played": 1})
    dados_away = classificacao.get(away, {"scored": 0, "against": 0, "played": 1})

    played_home = max(dados_home.get("played", 1), 1)
    played_away = max(dados_away.get("played", 1), 1)

    media_home_feitos = dados_home.get("scored", 0) / played_home
    media_home_sofridos = dados_home.get("against", 0) / played_home
    media_away_feitos = dados_away.get("scored", 0) / played_away
    media_away_sofridos = dados_away.get("against", 0) / played_away

    estimativa = ((media_home_feitos + media_away_sofridos) / 2 +
                  (media_away_feitos + media_home_sofridos) / 2)

    # Se estimativa zero (sem dados), usar heurística simples:
    if estimativa == 0:
        # heurística simples: 2.5 média
        estimativa = 2.2

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

def enviar_alerta_telegram(fixture: dict, tendencia: str, estimativa: float, confianca: float):
    """Envia alerta formatado para o Telegram."""
    home = fixture["homeTeam"]["name"]
    away = fixture["awayTeam"]["name"]
    data_formatada, hora_formatada = formatar_data_iso(fixture.get("utcDate") or "")
    competicao = fixture.get("competition", {}).get("name", "Desconhecido")

    status = fixture.get("status", "DESCONHECIDO")
    gols_home = fixture.get("score", {}).get("fullTime", {}).get("home")
    gols_away = fixture.get("score", {}).get("fullTime", {}).get("away")

    placar = f"{gols_home} x {gols_away}" if gols_home is not None and gols_away is not None else None

    msg = (
        f"⚽ <b>Alerta de Gols!</b>\n"
        f"🏟️ {home} vs {away}\n"
        f"📅 {data_formatada} ⏰ {hora_formatada} (BRT)\n"
        f"📌 Status: {fixture.get('statusDesc', status)}\n"
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
    """Verifica e envia alerta se necessário (evitar duplicação)."""
    alertas = carregar_alertas()
    fixture_id = str(fixture["id"])

    # Se não existe, criar e enviar
    if fixture_id not in alertas:
        alertas[fixture_id] = {
            "tendencia": tendencia,
            "estimativa": estimativa,
            "confianca": confianca,
            "conferido": False,
            "last_sent": time.time()
        }
        enviar_alerta_telegram(fixture, tendencia, estimativa, confianca)
        salvar_alertas(alertas)
    else:
        # Evitar reenvio: se já enviado nas últimas 3 horas, não reenviar
        last = alertas[fixture_id].get("last_sent", 0)
        if time.time() - last > 3 * 3600:
            alertas[fixture_id].update({
                "tendencia": tendencia,
                "estimativa": estimativa,
                "confianca": confianca,
                "last_sent": time.time()
            })
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
    st.set_page_config(page_title="⚽ Alerta de Gols (ESPN)", layout="wide")
    st.title("⚽ Sistema de Alertas Automáticos de Gols (Fonte: ESPN)")

    # Sidebar para configurações
    with st.sidebar:
        st.header("Configurações")
        top_n = st.selectbox("📊 Jogos no Top", [3, 5, 10], index=0)
        st.info("Busca apenas: Champions, Libertadores e Sul-Americana (ESPN)")

    # Controles principais
    col1, col2 = st.columns([2, 1])

    with col1:
        data_selecionada = st.date_input(
            "📅 Data para análise:",
            value=datetime.today()
        )

    with col2:
        todas_ligas = st.checkbox(
            "🌍 Todas as ligas (as 3 selecionadas)",
            value=True,
            help="Buscar jogos das 3 competições configuradas"
        )

    liga_selecionada = None
    if not todas_ligas:
        liga_selecionada = st.selectbox(
            "📌 Liga específica:",
            list(LIGAS_ESPN.keys())
        )

    # Processamento de jogos
    if st.button("🔍 Buscar Partidas", type="primary"):
        processar_jogos(data_selecionada, todas_ligas, liga_selecionada, top_n)

    # Botões de ação
    col1b, col2b, col3b = st.columns(3)

    with col1b:
        if st.button("🔄 Atualizar Status"):
            atualizar_status_partidas()

    with col2b:
        if st.button("📊 Conferir Resultados"):
            conferir_resultados()

    with col3b:
        if st.button("🧹 Limpar Cache"):
            limpar_caches()

def processar_jogos(data_selecionada, todas_ligas, liga_selecionada, top_n):
    """Processa e analisa os jogos do dia (usando ESPN)."""
    hoje = data_selecionada.strftime("%Y-%m-%d")
    if todas_ligas:
        ligas_busca = list(LIGAS_ESPN.values())
    else:
        ligas_busca = [LIGAS_ESPN[liga_selecionada]]

    st.write(f"⏳ Buscando jogos para {data_selecionada.strftime('%d/%m/%Y')}...")
    top_jogos = []
    progress_bar = st.progress(0)
    total_ligas = len(ligas_busca)

    for i, liga_slug in enumerate(ligas_busca):
        classificacao = obter_classificacao(liga_slug)
        jogos = obter_jogos(liga_slug, hoje)

        for match in jogos:
            home = match["homeTeam"]["name"]
            away = match["awayTeam"]["name"]
            estimativa, confianca, tendencia = calcular_tendencia(home, away, classificacao)

            verificar_enviar_alerta(match, tendencia, estimativa, confianca)

            # normalizar hora para objeto datetime (p/ exibição)
            hora_dt = None
            try:
                hora_dt = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")) - timedelta(hours=3)
            except Exception:
                hora_dt = None

            top_jogos.append({
                "id": match["id"],
                "home": home,
                "away": away,
                "tendencia": tendencia,
                "estimativa": estimativa,
                "confianca": confianca,
                "liga": match.get("competition", {}).get("name", "Desconhecido"),
                "hora": hora_dt,
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
    jogos_filtrados = [j for j in jogos if j["status"] not in ["FINISHED", "IN_PLAY", "POSTPONED", "SUSPENDED"]]

    if not jogos_filtrados:
        st.warning("⚠️ Nenhum jogo elegível para o Top Jogos (todos já iniciados ou finalizados).")
        return

    top_jogos_sorted = sorted(jogos_filtrados, key=lambda x: x["confianca"], reverse=True)[:top_n]

    msg = f"📢 TOP {top_n} Jogos do Dia\n\n"
    for j in top_jogos_sorted:
        hora_format = j["hora"].strftime("%H:%M") if j["hora"] else "-"
        msg += (
            f"🏟️ {j['home']} vs {j['away']}\n"
            f"🕒 {hora_format} BRT | Liga: {j['liga']} | Status: {j['status']}\n"
            f"📈 Tendência: {j['tendencia']} | Estimativa: {j['estimativa']:.2f} | "
            f"💯 Confiança: {j['confianca']:.0f}%\n\n"
        )

    # Envio ao Telegram (canal alternativo por padrão)
    if enviar_telegram(msg, TELEGRAM_CHAT_ID_ALT2):
        st.success(f"🚀 Top {top_n} jogos (sem finalizados) enviados para o canal!")
    else:
        st.error("❌ Erro ao enviar top jogos para o Telegram")

def atualizar_status_partidas():
    """Atualiza o status das partidas em cache (re-busca cada chave)."""
    cache_jogos = carregar_cache_jogos()
    mudou = False

    for key in list(cache_jogos.keys()):
        if key == "_timestamp":
            continue

        try:
            liga_id, data = key.split("_", 1)
            # Re-buscar via ESPN
            partidas = buscar_jogos_espn(liga_id, data)
            cache_jogos[key] = {
                "matches": partidas,
                "_timestamp": datetime.now().timestamp()
            }
            mudou = True
        except Exception as e:
            st.error(f"Erro ao atualizar liga {key}: {e}")

    if mudou:
        salvar_cache_jogos(cache_jogos)
        st.success("✅ Status das partidas atualizado!")
    else:
        st.info("ℹ️ Nenhuma atualização disponível.")

# =============================
# Conferência de resultados
# =============================
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
        for key, entry in jogos_cache.items():
            if key == "_timestamp":
                continue
            partidas = entry.get("matches") if isinstance(entry, dict) and entry.get("matches") is not None else entry
            for match in partidas:
                if str(match.get("id")) == str(fixture_id):
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
    hora = None
    try:
        hora = datetime.fromisoformat(jogo["utcDate"].replace("Z", "+00:00")) - timedelta(hours=3)
    except Exception:
        hora = None

    return [
        f"{home} vs {away}",
        info["tendencia"],
        f"{info['estimativa']:.2f}",
        f"{info['confianca']:.0f}%",
        resultado["placar"] if resultado else "-",
        jogo.get("status", "DESCONHECIDO"),
        resultado["resultado"] if resultado else "⏳ Aguardando",
        hora.strftime("%d/%m %H:%M") if hora else "-"
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
