# app_nba_streamlit.py
import streamlit as st
from datetime import datetime, timedelta, date
import requests
import json
import os
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import time

# =============================
# Configurações e Segurança
# =============================
# BallDontLie não exige API key pública (gratuita), mas mantenha variáveis de ambiente
BALLDONTLIE_BASE = os.getenv("BALLDONTLIE_BASE", "https://www.balldontlie.io/api/v1")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003073115320")
TELEGRAM_CHAT_ID_ALT2 = os.getenv("TELEGRAM_CHAT_ID_ALT2", "-1002754276285")

BASE_URL_TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# Constantes de cache
ALERTAS_PATH = "alertas_nba.json"
CACHE_JOGOS = "cache_jogos_nba.json"
CACHE_MEDIAS = "cache_medias_nba.json"
CACHE_TIMEOUT = 3600  # 1 hora em segundos

# =============================
# Utilitários de Cache e Persistência
# =============================
def carregar_json(caminho: str) -> dict:
    try:
        if os.path.exists(caminho):
            with open(caminho, "r", encoding='utf-8') as f:
                dados = json.load(f)
            # Verificar timeout global do arquivo
            agora = datetime.now().timestamp()
            if agora - os.path.getmtime(caminho) > CACHE_TIMEOUT:
                return {}
            return dados
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"Erro ao carregar {caminho}: {e}")
    return {}

def salvar_json(caminho: str, dados: dict):
    try:
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

def carregar_cache_medias() -> dict:
    return carregar_json(CACHE_MEDIAS) or {}

def salvar_cache_medias(dados: dict):
    salvar_json(CACHE_MEDIAS, dados)

# =============================
# Utilitários de Data e Formatação
# =============================
def formatar_data_iso_brt(data_iso: str) -> tuple[str, str]:
    """Formata data ISO para data e hora em BRT (-3)."""
    try:
        data_jogo = datetime.fromisoformat(data_iso.replace("Z", "+00:00")) - timedelta(hours=3)
        return data_jogo.strftime("%d/%m/%Y"), data_jogo.strftime("%H:%M")
    except Exception:
        return "Data inválida", "Hora inválida"

def abreviar_nome(nome: str, max_len: int = 20) -> str:
    if len(nome) <= max_len:
        return nome
    return nome[:max_len-3] + "..."

# =============================
# Comunicação com APIs (BallDontLie)
# =============================
def enviar_telegram(msg: str, chat_id: str = TELEGRAM_CHAT_ID) -> bool:
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

def request_balldontlie(path: str, params: dict = None, timeout: int = 10) -> dict | None:
    try:
        url = f"{BALLDONTLIE_BASE.rstrip('/')}/{path.lstrip('/')}"
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Erro na requisição BallDontLie ({path}): {e}")
        return None

# =============================
# Obter jogos e médias de times
# =============================
def obter_jogos_nbA(data_str: str) -> list:
    """
    Obtém jogos da BallDontLie para uma data (YYYY-MM-DD).
    Cache simples por data.
    """
    cache = carregar_cache_jogos()
    key = f"games_{data_str}"
    if key in cache:
        return cache[key]

    jogos = []
    page = 1
    per_page = 100
    params = {"dates[]": data_str, "per_page": per_page, "page": page}
    while True:
        data = request_balldontlie("games", params=params)
        if not data or "data" not in data:
            break
        jogos.extend(data["data"])
        if not data.get("meta") or page >= data["meta"].get("total_pages", 1):
            break
        page += 1
        params["page"] = page
        time.sleep(0.1)  # respeitar limites

    cache[key] = jogos
    salvar_cache_jogos(cache)
    return jogos

def obter_medias_time(team_id: int, season: int = None, window_games: int = 20) -> dict:
    """
    Calcula médias recentes do time (pontos marcados, sofridos, pontos por jogo, diff médio)
    usando endpoint /stats filtrado por team_ids[] (paginação).
    Retorna dict com keys: pts_for, pts_against, games, pts_diff_avg, pace_est (opcional).
    """
    cache = carregar_cache_medias()
    key = f"{team_id}_{season or 'current'}_{window_games}"
    if key in cache:
        return cache[key]

    # Buscar jogos recentes do time (últimos window_games)
    # BallDontLie: vamos buscar stats por team_id ordenando por data (não há sort, então recolher páginas)
    stats = []
    page = 1
    per_page = 100
    params = {"team_ids[]": team_id, "per_page": per_page, "page": page}
    while len(stats) < window_games:
        data = request_balldontlie("stats", params=params)
        if not data or "data" not in data:
            break
        # cada item tem 'pts' e 'game' com date and home_team_id / visitor_team_id
        stats.extend(data["data"])
        if not data.get("meta") or page >= data["meta"].get("total_pages", 1):
            break
        page += 1
        params["page"] = page
        time.sleep(0.1)

    # Usar apenas últimos window_games
    stats = stats[:window_games]

    if not stats:
        medias = {"pts_for": 0.0, "pts_against": 0.0, "games": 0, "pts_diff_avg": 0.0}
        cache[key] = medias
        salvar_cache_medias(cache)
        return medias

    # Precisamos conhecer, para cada stat, se o time foi home ou visitor para computar pts_for e pts_against
    pts_for = 0
    pts_against = 0
    jogos_count = 0

    for s in stats:
        jogo = s.get("game") or {}
        # Identificar se a stat pertence ao team as player-team? In BallDontLie each stat entry has 'team' object id?
        # A estrutura tem 'team' indicando o time do jogador — mas /stats retorna por jogador; ao filtrar team_ids[],
        # teremos estatísticas de todos os jogadores daquele time por jogo (muitos registros por jogo).
        # Para simplificar: somar pontos por jogo por team agregando 'pts' por game.id.
        # => Construir agregação por game.id
        pass

    # --- agregação por jogo (reprocessar) ---
    games_agg = {}
    for s in stats:
        g = s.get("game")
        if not g:
            continue
        gid = g.get("id")
        if gid not in games_agg:
            games_agg[gid] = {"pts_team": 0, "opp_pts": None, "date": g.get("date"), "home_team_id": g.get("home_team_id"), "visitor_team_id": g.get("visitor_team_id")}
        # 'pts' do jogador é parte do total do time; somamos todos jogadores daquele team naquele jogo
        pts = s.get("pts") or 0
        games_agg[gid]["pts_team"] += pts

    # Agora, para cada jogo agregado, precisamos saber o placar final (para pegar pts_opp).
    # Buscar game details via /games/<id> para obter score.
    jogos_agregados = list(games_agg.items())[:window_games]
    pts_for = 0
    pts_against = 0
    jogos_count = 0
    for gid, info in jogos_agregados:
        game_data = request_balldontlie(f"games/{gid}")
        if not game_data:
            continue
        # Scores: 'home_team_score', 'visitor_team_score'
        home_score = game_data.get("home_team_score")
        visitor_score = game_data.get("visitor_team_score")
        if home_score is None or visitor_score is None:
            # pular jogos sem placar definido
            continue
        # descobrir se nosso team_id é home ou visitor
        if info["home_team_id"] == team_id:
            team_pts = home_score
            opp_pts = visitor_score
        else:
            team_pts = visitor_score
            opp_pts = home_score

        pts_for += team_pts
        pts_against += opp_pts
        jogos_count += 1
        # respeitar limites
        time.sleep(0.05)

    if jogos_count == 0:
        medias = {"pts_for": 0.0, "pts_against": 0.0, "games": 0, "pts_diff_avg": 0.0}
    else:
        medias = {
            "pts_for": pts_for / jogos_count,
            "pts_against": pts_against / jogos_count,
            "games": jogos_count,
            "pts_diff_avg": (pts_for - pts_against) / jogos_count
        }

    cache[key] = medias
    salvar_cache_medias(cache)
    return medias

# =============================
# Lógica de Análise e Alertas (NBA)
# =============================
def calcular_tendencia_nba(home_team_id: int, away_team_id: int, season: int = None) -> tuple[float, float, str]:
    """
    Estima o total de pontos esperado entre as equipes baseado nas médias recentes.
    Retorna: estimativa_total (float), confianca (0-100), tendencia_str (e.g. 'Mais 210.5').
    """
    season = season or datetime.today().year
    w = 20  # janela de jogos usadas para média
    home_med = obter_medias_time(home_team_id, season, window_games=w)
    away_med = obter_medias_time(away_team_id, season, window_games=w)

    # Se não houver dados, fallback neutro
    if home_med["games"] == 0 or away_med["games"] == 0:
        estimativa = 215.0  # fallback médio
        confianca = 55.0
        tendencia = "Mais 210.5"
        return estimativa, confianca, tendencia

    # Estimativa simples: média de pontos-for de casa + média de pontos-for de visitante
    estimativa = (home_med["pts_for"] + away_med["pts_for"])
    # Ajuste por ritmo/diferencial pode ser adicionado; por enquanto, direta
    # Determinar confiança: maior quando ambos times têm muitos jogos e médias semelhantes
    base_conf = 50.0
    jogos_medios = min(home_med["games"], away_med["games"])
    base_conf += min(30, jogos_medios)  # mais jogos -> mais confiança
    # Ajustar com base na magnitude do diff médio
    diff_abs = abs(home_med["pts_diff_avg"] - away_med["pts_diff_avg"])
    base_conf += max(0, min(15, 5 - diff_abs))  # times muito diferentes reduzem um pouco a confiança

    # Normalizar
    confianca = max(40.0, min(95.0, base_conf))

    # Determinar tendência por thresholds (ajustáveis)
    # Sugestão: thresholds típicos NBA: 210.5, 220.5, 230.5
    if estimativa >= 230:
        tendencia = "Mais 230.5"
    elif estimativa >= 220:
        tendencia = "Mais 220.5"
    elif estimativa >= 215:
        tendencia = "Mais 215.5"
    elif estimativa >= 210:
        tendencia = "Mais 210.5"
    else:
        tendencia = "Menos 210.5"

    return round(estimativa, 1), round(confianca, 1), tendencia

def calcular_favorito_vitoria_nba(home_team_id: int, away_team_id: int, season: int = None) -> tuple[str, float]:
    """
    Calcula favorito baseado na diferença média de pontos (pts_diff_avg).
    Retorna string com favorito e confiança.
    """
    season = season or datetime.today().year
    home_med = obter_medias_time(home_team_id, season, window_games=20)
    away_med = obter_medias_time(away_team_id, season, window_games=20)

    if home_med["games"] == 0 and away_med["games"] == 0:
        return "Empate", 50.0

    diff = home_med["pts_diff_avg"] - away_med["pts_diff_avg"]
    # diff positivo -> home favorito
    if abs(diff) < 2.0:  # diferença pequena (poucos pontos)
        favorito = "Empate"
        confianca = 50.0
    elif diff > 0:
        favorito = "Casa vencer"
        confianca = min(95.0, 55.0 + diff * 4.0)
    else:
        favorito = "Fora vencer"
        confianca = min(95.0, 55.0 + abs(diff) * 4.0)

    confianca = max(50.0, min(95.0, confianca))
    return favorito, round(confianca, 1)

# =============================
# Envio de alertas adaptado para NBA
# =============================
def enviar_alerta_telegram_nba(fixture: dict, tendencia: str, estimativa: float, confianca: float):
    """Formata e envia alerta para Telegram (NBA)."""
    home = fixture["home_team"]["full_name"]
    away = fixture["visitor_team"]["full_name"]
    data_formatada, hora_formatada = formatar_data_iso_brt(fixture["date"])
    status = fixture.get("status", "SCHEDULED")
    # Determinar favorito
    favorito, conf_fav = calcular_favorito_vitoria_nba(fixture["home_team"]["id"], fixture["visitor_team"]["id"])

    msg = (
        f"🏀 <b>Alerta NBA - Total de Pontos</b>\n"
        f"🏟️ {home} vs {away}\n"
        f"📅 {data_formatada} ⏰ {hora_formatada} (BRT)\n"
        f"📌 Status: {status}\n"
        f"📈 Tendência: <b>{tendencia}</b>\n"
        f"🎯 Estimativa: <b>{estimativa:.1f} pontos</b>\n"
        f"💯 Confiança: <b>{confianca:.0f}%</b>\n"
        f"🏆 Favorito: <b>{favorito}</b> ({conf_fav:.0f}%)"
    )
    enviar_telegram(msg)

def verificar_enviar_alerta_nba(fixture: dict, tendencia: str, estimativa: float, confianca: float):
    alertas = carregar_alertas()
    fixture_id = str(fixture["id"])
    if fixture_id not in alertas:
        # calcular favorito no momento do alerta
        favorito, conf_fav = calcular_favorito_vitoria_nba(fixture["home_team"]["id"], fixture["visitor_team"]["id"])
        alertas[fixture_id] = {
            "tendencia": tendencia,
            "estimativa": estimativa,
            "confianca": confianca,
            "favorito": favorito,
            "conf_fav": conf_fav,
            "conferido": False
        }
        enviar_alerta_telegram_nba(fixture, tendencia, estimativa, confianca)
        salvar_alertas(alertas)

# =============================
# Conferência de resultados / PDF (NBA)
# =============================
def processar_resultado_jogo_nba(jogo: dict, info: dict) -> dict | None:
    """Processa resultado de um jogo NBA (conferência)."""
    home = jogo["home_team"]["full_name"]
    away = jogo["visitor_team"]["full_name"]
    status = jogo.get("status", "SCHEDULED")
    home_score = jogo.get("home_team_score")
    visitor_score = jogo.get("visitor_team_score")
    placar = f"{home_score} x {visitor_score}" if home_score is not None and visitor_score is not None else "-"
    total_pts = (home_score or 0) + (visitor_score or 0)

    resultado = "⏳ Aguardando"
    if status == "Final" or status == "final" or status == "FINAL":
        tendencia = info.get("tendencia", "")
        # interpretar tendência
        if "Mais" in tendencia:
            # extrair threshold numérico
            try:
                th = float(tendencia.split()[-1])
            except Exception:
                th = 210.5
            resultado = "🟢 GREEN" if total_pts > th else "🔴 RED"
        elif "Menos" in tendencia:
            try:
                th = float(tendencia.split()[-1])
            except Exception:
                th = 210.5
            resultado = "🟢 GREEN" if total_pts < th else "🔴 RED"
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
        "total_pts": total_pts
    }

def enviar_resultado_telegram_nba(resultado: dict):
    msg = (
        f"📊 <b>Resultado Conferido (NBA)</b>\n"
        f"🏟️ {resultado['home']} vs {resultado['away']}\n"
        f"🏆 Favorito: <b>{resultado.get('favorito','N/A')}</b> ({resultado.get('conf_fav',0):.0f}%)\n"
        f"📈 Tendência: {resultado['tendencia']} | Estim.: {resultado['estimativa']:.1f} | Conf.: {resultado['confianca']:.0f}%\n"
        f"📊 Placar Final: <b>{resultado['placar']}</b>\n"
        f"✅ Resultado: <b>{resultado['resultado']}</b>"
    )
    enviar_telegram(msg, TELEGRAM_CHAT_ID_ALT2)

# =============================
# PDF e Interface Streamlit (manutenção)
# =============================
def gerar_relatorio_pdf(jogos_conferidos: list) -> io.BytesIO:
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter,
                          rightMargin=20, leftMargin=20,
                          topMargin=20, bottomMargin=20)

    data = [["Jogo", "Tendência", "Estimativa", "Confiança",
             "Placar", "Status", "Resultado", "Hora"]] + jogos_conferidos

    table = Table(data, repeatRows=1,
                 colWidths=[140, 70, 60, 60, 60, 70, 60, 70])

    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C3E50")),
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

    for i in range(1, len(data)):
        bg_color = colors.HexColor("#EAECEE") if i % 2 == 0 else colors.HexColor("#FFFFFF")
        style.add('BACKGROUND', (0,i), (-1,i), bg_color)

    table.setStyle(style)
    pdf.build([table])
    buffer.seek(0)
    return buffer

# =============================
# Streamlit UI
# =============================
def main():
    st.set_page_config(page_title="🏀 Alerta NBA", layout="wide")
    st.title("🏀 Sistema de Alertas Automáticos - NBA (BallDontLie)")

    with st.sidebar:
        st.header("Configurações")
        top_n = st.selectbox("📊 Jogos no Top", [3, 5, 10], index=0)
        janela = st.slider("Janela de jogos para médias", 5, 40, 20)
        st.info("Ajuste a janela para balancear sensibilidade / estabilidade")

    col1, col2 = st.columns([2, 1])
    with col1:
        data_selecionada = st.date_input("📅 Data para análise:", value=date.today())
    with col2:
        enviar_top_automatico = st.checkbox("📤 Enviar Top automaticamente?", value=False)

    if st.button("🔍 Buscar Partidas (NBA)", type="primary"):
        processar_jogos_nba(data_selecionada, top_n, janela, enviar_top_automatico)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Atualizar Status / Cache"):
            atualizar_status_partidas_nba()
    with col2:
        if st.button("📊 Conferir Resultados"):
            conferir_resultados_nba()
    with col3:
        if st.button("🧹 Limpar Cache"):
            limpar_caches_nba()

def processar_jogos_nba(data_selecionada, top_n, janela, enviar_top_automatico):
    hoje = data_selecionada.strftime("%Y-%m-%d")
    st.write(f"⏳ Buscando jogos NBA para {data_selecionada.strftime('%d/%m/%Y')}...")
    jogos = obter_jogos_nbA(hoje)
    if not jogos:
        st.warning("⚠️ Nenhum jogo encontrado para a data selecionada.")
        return

    top_jogos = []
    total = len(jogos)
    progress = st.progress(0)
    for i, g in enumerate(jogos):
        # calcular estimativa e favorito
        estimativa, confianca, tendencia = calcular_tendencia_nba(g["home_team"]["id"], g["visitor_team"]["id"])
        verificar_enviar_alerta_nba(g, tendencia, estimativa, confianca)

        hora = datetime.fromisoformat(g["date"].replace("Z", "+00:00")) - timedelta(hours=3)
        top_jogos.append({
            "id": g["id"],
            "home": g["home_team"]["full_name"],
            "away": g["visitor_team"]["full_name"],
            "tendencia": tendencia,
            "estimativa": estimativa,
            "confianca": confianca,
            "hora": hora,
            "status": g.get("status", "SCHEDULED"),
        })
        progress.progress((i+1)/total)

    # enviar top jogos (não-iniciados)
    enviar_top_jogos_nba(top_jogos, top_n, enviar_top_automatico)
    st.success(f"✅ Análise concluída! {len(top_jogos)} jogos processados.")

def enviar_top_jogos_nba(jogos: list, top_n: int, enviar_automatico: bool = False):
    jogos_filtrados = [j for j in jogos if j["status"].upper() not in ["FINAL", "IN_PLAY", "POSTPONED", "SUSPENDED"]]
    if not jogos_filtrados:
        st.warning("⚠️ Nenhum jogo elegível para o Top (todos já iniciados ou finalizados).")
        return

    top_jogos_sorted = sorted(jogos_filtrados, key=lambda x: x["confianca"], reverse=True)[:top_n]
    msg = f"📢 TOP {top_n} Jogos NBA - Hoje\n\n"
    for j in top_jogos_sorted:
        hora_format = j["hora"].strftime("%H:%M")
        msg += (
            f"🏟️ {j['home']} vs {j['away']}\n"
            f"🕒 {hora_format} BRT | Status: {j['status']}\n"
            f"📈 Tendência: {j['tendencia']} | Estimativa: {j['estimativa']:.1f} | "
            f"💯 Confiança: {j['confianca']:.0f}%\n\n"
        )

    if enviar_automatico:
        if enviar_telegram(msg, TELEGRAM_CHAT_ID_ALT2):
            st.success(f"🚀 Top {top_n} jogos (sem finalizados) enviados para o canal!")
        else:
            st.error("❌ Erro ao enviar top jogos para o Telegram")
    else:
        st.code(msg)

def atualizar_status_partidas_nba():
    # limpando cache de jogos para forçar re-fetch
    try:
        if os.path.exists(CACHE_JOGOS):
            os.remove(CACHE_JOGOS)
        if os.path.exists(CACHE_MEDIAS):
            os.remove(CACHE_MEDIAS)
        st.success("✅ Cache de jogos e médias limpo.")
    except Exception as e:
        st.error(f"Erro ao limpar cache: {e}")

def conferir_resultados_nba():
    alertas = carregar_alertas()
    jogos_cache = carregar_cache_jogos()
    if not alertas:
        st.info("ℹ️ Nenhum alerta para conferir.")
        return

    jogos_conferidos = []
    mudou = False

    for fixture_id, info in list(alertas.items()):
        if info.get("conferido"):
            continue

        # procurar o jogo no cache por data (percorrer)
        jogo_dado = None
        for key, jogos in jogos_cache.items():
            if not key.startswith("games_"):
                continue
            for match in jogos:
                if str(match["id"]) == fixture_id:
                    jogo_dado = match
                    break
            if jogo_dado:
                break

        if not jogo_dado:
            # tentar buscar game direto via API
            game_data = request_balldontlie(f"games/{fixture_id}")
            if game_data:
                jogo_dado = game_data

        if not jogo_dado:
            continue

        resultado_info = processar_resultado_jogo_nba(jogo_dado, info)
        if resultado_info:
            exibir_resultado_streamlit_nba(resultado_info)
            if resultado_info["status"] and resultado_info["status"].upper() in ["FINAL", "FINALIZED"]:
                enviar_resultado_telegram_nba(resultado_info)
                info["conferido"] = True
                mudou = True

        jogos_conferidos.append(preparar_dados_pdf_nba(jogo_dado, info, resultado_info))

    if mudou:
        salvar_alertas(alertas)
        st.success("✅ Resultados conferidos e atualizados!")

    if jogos_conferidos:
        gerar_pdf_jogos_nba(jogos_conferidos)

def exibir_resultado_streamlit_nba(resultado: dict):
    bg_color = "#1e4620" if resultado["resultado"] == "🟢 GREEN" else \
               "#5a1e1e" if resultado["resultado"] == "🔴 RED" else "#2c2c2c"
    st.markdown(f"""
    <div style="border:1px solid #444; border-radius:10px; padding:12px; margin-bottom:10px;
                background-color:{bg_color}; font-size:15px; color:#f1f1f1;">
        <b>🏟️ {resultado['home']} vs {resultado['away']}</b><br>
        📌 Status: <b>{resultado['status']}</b><br>
        🏆 Favorito: <b>{resultado.get('favorito', 'N/A')}</b> ({resultado.get('conf_fav', 0):.0f}%)<br>
        🏀 Tendência: <b>{resultado['tendencia']}</b> | Estim.: {resultado['estimativa']:.1f} | Conf.: {resultado['confianca']:.0f}%<br>
        📊 Placar: <b>{resultado['placar']}</b><br>
        ✅ Resultado: {resultado['resultado']}
    </div>
    """, unsafe_allow_html=True)

def preparar_dados_pdf_nba(jogo: dict, info: dict, resultado: dict) -> list:
    home = abreviar_nome(jogo.get("home_team", {}).get("full_name", ""))
    away = abreviar_nome(jogo.get("visitor_team", {}).get("full_name", ""))
    # pegar hora se existir
    data_iso = jogo.get("date")
    try:
        hora = datetime.fromisoformat(data_iso.replace("Z", "+00:00")) - timedelta(hours=3)
        hora_str = hora.strftime("%d/%m %H:%M")
    except Exception:
        hora_str = "-"

    favorito_str = f" ({info.get('favorito')})" if info.get("favorito") else ""
    return [
        f"{home} vs {away}{favorito_str}",
        info.get("tendencia", ""),
        f"{info.get('estimativa', 0):.1f}",
        f"{info.get('confianca', 0):.0f}%",
        resultado.get("placar", "-") if resultado else "-",
        jogo.get("status", "SCHEDULED"),
        resultado.get("resultado", "⏳ Aguardando") if resultado else "⏳ Aguardando",
        hora_str
    ]

def gerar_pdf_jogos_nba(jogos_conferidos: list):
    df_conferidos = pd.DataFrame(jogos_conferidos, columns=[
        "Jogo", "Tendência", "Estimativa", "Confiança",
        "Placar", "Status", "Resultado", "Hora"
    ])
    buffer = gerar_relatorio_pdf(jogos_conferidos)
    st.download_button(
        label="📄 Baixar Relatório PDF",
        data=buffer,
        file_name=f"jogos_conferidos_nba_{datetime.today().strftime('%Y-%m-%d')}.pdf",
        mime="application/pdf"
    )

def limpar_caches_nba():
    try:
        for cache_file in [CACHE_JOGOS, CACHE_MEDIAS, ALERTAS_PATH]:
            if os.path.exists(cache_file):
                os.remove(cache_file)
        st.success("✅ Caches limpos com sucesso!")
    except Exception as e:
        st.error(f"❌ Erro ao limpar caches: {e}")

if __name__ == "__main__":
    main()
