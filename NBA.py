# app_nba_elite_master.py
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
# CONFIGURAÇÕES E SEGURANÇA
# =============================
# Coloque sua API key do BallDontLie em BALLDONTLIE_API_KEY (ou será usado o default abaixo)
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", "7da89f74-317a-45a0-88f9-57cccfef5a00")
BALLDONTLIE_BASE = os.getenv("BALLDONTLIE_BASE", "https://api.balldontlie.io/v1")  # v1 compatível
# Telegram (mantive os defaults que você tinha)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003073115320")
TELEGRAM_CHAT_ID_ALT2 = os.getenv("TELEGRAM_CHAT_ID_ALT2", "-1002754276285")
BASE_URL_TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# Arquivos de cache/estado
ALERTAS_PATH = "alertas_nba.json"
CACHE_GAMES = "cache_games_nba.json"
CACHE_TEAMS = "cache_teams_nba.json"
CACHE_STATS = "cache_stats_nba.json"
CACHE_TIMEOUT = 3600  # 1h

# Headers com auth
HEADERS_BDL = {"Authorization": BALLDONTLIE_API_KEY}

# =============================
# UTILITÁRIOS DE CACHE E IO
# =============================
def carregar_json(caminho: str) -> dict:
    try:
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)
            # invalida cache antigo por tempo
            if datetime.now().timestamp() - os.path.getmtime(caminho) > CACHE_TIMEOUT:
                return {}
            return dados
    except Exception as e:
        st.error(f"Erro ao carregar {caminho}: {e}")
    return {}

def salvar_json(caminho: str, dados: dict):
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Erro ao salvar {caminho}: {e}")

def carregar_alertas():
    return carregar_json(ALERTAS_PATH) or {}

def salvar_alertas(dados):
    salvar_json(ALERTAS_PATH, dados)

def carregar_cache_games():
    return carregar_json(CACHE_GAMES) or {}

def salvar_cache_games(dados):
    salvar_json(CACHE_GAMES, dados)

def carregar_cache_teams():
    return carregar_json(CACHE_TEAMS) or {}

def salvar_cache_teams(dados):
    salvar_json(CACHE_TEAMS, dados)

def carregar_cache_stats():
    return carregar_json(CACHE_STATS) or {}

def salvar_cache_stats(dados):
    salvar_json(CACHE_STATS, dados)

# =============================
# FORMATAÇÃO E UTILS
# =============================
def formatar_data_brt(date_iso: str) -> tuple[str, str]:
    try:
        # BallDontLie may provide "datetime" or "date" fields; handle both
        if not date_iso:
            return "Data inválida", "Hora inválida"
        dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00")) - timedelta(hours=3)
        return dt.strftime("%d/%m/%Y"), dt.strftime("%H:%M")
    except Exception:
        return "Data inválida", "Hora inválida"

def abreviar(nome: str, l=20):
    if not nome:
        return ""
    return nome if len(nome) <= l else nome[:l-3] + "..."

# =============================
# REQUISIÇÕES À BALLDONTLIE
# =============================
def balldontlie_get(path: str, params: dict | None = None, timeout: int = 10) -> dict | None:
    try:
        url = BALLDONTLIE_BASE.rstrip("/") + "/" + path.lstrip("/")
        resp = requests.get(url, headers=HEADERS_BDL, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Erro BallDontLie {path}: {e}")
        return None

# =============================
# TEAMS (cache)
# =============================
def obter_times():
    cache = carregar_cache_teams()
    if "teams" in cache and cache["teams"]:
        return cache["teams"]
    data = balldontlie_get("teams")
    if not data:
        return {}
    teams = {t["id"]: t for t in data.get("data", [])}
    cache["teams"] = teams
    salvar_cache_teams(cache)
    return teams

def encontrar_time_por_abrev(abbr: str):
    teams = obter_times()
    for tid, t in teams.items():
        if t.get("abbreviation", "").upper() == abbr.upper() or t.get("full_name", "").lower() == abbr.lower():
            return t
    return None

# =============================
# GAMES: obter jogos por data (cache)
# =============================
def obter_jogos_data(data_str: str) -> list:
    cache = carregar_cache_games()
    key = f"games_{data_str}"
    if key in cache:
        return cache[key]

    jogos = []
    per_page = 100
    page = 1
    while True:
        params = {"dates[]": data_str, "per_page": per_page, "page": page}
        resp = balldontlie_get("games", params=params)
        if not resp or "data" not in resp:
            break
        jogos.extend(resp["data"])
        meta = resp.get("meta")
        if not meta or page >= meta.get("total_pages", 1):
            break
        page += 1
        time.sleep(0.05)
    cache[key] = jogos
    salvar_cache_games(cache)
    return jogos

# =============================
# ESTATÍSTICAS RECENTES DO TIME
# =============================
def obter_estatisticas_recentes_time(team_id: int, window_games: int = 20) -> dict:
    """
    Usa endpoint /games para agregar últimos jogos do time e retornar:
    pts_for_avg, pts_against_avg, games_count, first_half_avg (home_q1+home_q2 or visitor_q1+visitor_q2)
    """
    cache = carregar_cache_stats()
    key = f"team_{team_id}_{window_games}"
    if key in cache:
        return cache[key]

    # buscar jogos recentes: usar start_date = hoje - 120 dias (suficiente) e filtrar por team_id
    end = date.today()
    start = end - timedelta(days=365)  # pegar até 1 ano para garantir window
    per_page = 100
    page = 1
    games = []
    while len(games) < window_games:
        params = {
            "team_ids[]": team_id,
            "per_page": per_page,
            "page": page,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d")
        }
        resp = balldontlie_get("games", params=params)
        if not resp or "data" not in resp:
            break
        # ordenar por date desc - BallDontLie doesn't guarantee order; we will append then sort
        games.extend(resp["data"])
        meta = resp.get("meta")
        if not meta or page >= meta.get("total_pages", 1):
            break
        page += 1
        time.sleep(0.05)

    # ordenar por data desc
    def _gdate(g):
        d = g.get("datetime") or g.get("date") or g.get("game_date")
        try:
            return datetime.fromisoformat((d or "").replace("Z", "+00:00"))
        except Exception:
            return datetime.min
    games_sorted = sorted(games, key=_gdate, reverse=True)[:window_games]

    if not games_sorted:
        stats = {"pts_for_avg": 0.0, "pts_against_avg": 0.0, "games": 0, "pts_diff_avg": 0.0, "first_half_avg": 0.0}
        cache[key] = stats
        salvar_cache_stats(cache)
        return stats

    pts_for = 0
    pts_against = 0
    first_half_total = 0
    count = 0
    seen_game_ids = set()

    for g in games_sorted:
        gid = g.get("id")
        if not gid or gid in seen_game_ids:
            continue
        seen_game_ids.add(gid)
        # identificar se team_id jogou em casa ou fora
        home_id = g.get("home_team", {}).get("id")
        visitor_id = g.get("visitor_team", {}).get("id")
        home_score = g.get("home_team_score")
        visitor_score = g.get("visitor_team_score")

        # ignorar jogos sem placar
        if home_score is None or visitor_score is None:
            continue

        if home_id == team_id:
            pts_for += home_score
            pts_against += visitor_score
            # first half: home_q1 + home_q2 if present
            q1 = g.get("home_q1") or 0
            q2 = g.get("home_q2") or 0
            fh = q1 + q2
        else:
            pts_for += visitor_score
            pts_against += home_score
            q1 = g.get("visitor_q1") or 0
            q2 = g.get("visitor_q2") or 0
            fh = q1 + q2

        first_half_total += fh
        count += 1

    if count == 0:
        stats = {"pts_for_avg": 0.0, "pts_against_avg": 0.0, "games": 0, "pts_diff_avg": 0.0, "first_half_avg": 0.0}
    else:
        stats = {
            "pts_for_avg": pts_for / count,
            "pts_against_avg": pts_against / count,
            "games": count,
            "pts_diff_avg": (pts_for - pts_against) / count,
            "first_half_avg": first_half_total / count
        }

    cache[key] = stats
    salvar_cache_stats(cache)
    return stats

# =============================
# LÓGICA DE PREVISÃO — 4 MODALIDADES
# =============================
def prever_total_points(home_id: int, away_id: int, window_games: int = 20) -> tuple[float, float, str]:
    """
    Estima o total combinado e decide tendência (Over/Under thresholds).
    Retorna: (estimativa_total, confianca_percent, tendencia_str)
    """
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)

    # fallback neutro
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return 215.0, 55.0, "Mais 215.5"

    estimativa = home_stats["pts_for_avg"] + away_stats["pts_for_avg"]
    # confiança baseada em quantidade de jogos e balanceamento
    jogos = min(home_stats["games"], away_stats["games"])
    conf = 45 + min(30, jogos)  # base 45 + até 30
    # penalizar quando a diferença de ritmo é grande (less confident)
    diff = abs(home_stats["pts_diff_avg"] - away_stats["pts_diff_avg"])
    conf -= max(0, min(10, diff))  # reduce confidence up to 10 points
    conf = max(35.0, min(95.0, conf))

    # thresholds típicos da NBA (ajustáveis)
    if estimativa >= 235:
        tendencia = "Mais 235.5"
    elif estimativa >= 225:
        tendencia = "Mais 225.5"
    elif estimativa >= 215:
        tendencia = "Mais 215.5"
    elif estimativa >= 210:
        tendencia = "Mais 210.5"
    else:
        tendencia = "Menos 210.5"

    return round(estimativa, 1), round(conf, 1), tendencia

def prever_moneyline(home_id: int, away_id: int, window_games: int = 20) -> tuple[str, float]:
    """
    Estima favorito (home advantage considered) e retorna (favorito_str, confianca%).
    """
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)

    if home_stats["games"] == 0 and away_stats["games"] == 0:
        return "Empate", 50.0

    diff = home_stats["pts_diff_avg"] - away_stats["pts_diff_avg"]
    # adicionar pequeno bônus de casa
    home_bonus = 1.5
    diff += home_bonus

    if abs(diff) < 2.5:
        return "Empate", 50.0
    elif diff > 0:
        conf = min(95.0, 55 + diff * 3.5)
        return "Casa vencer", round(max(50.0, conf), 1)
    else:
        conf = min(95.0, 55 + abs(diff) * 3.5)
        return "Fora vencer", round(max(50.0, conf), 1)

def prever_handicap(home_id: int, away_id: int, window_games: int = 20) -> dict:
    """
    Estima margem esperada (home_margin = home_pts_avg - away_pts_avg) e sugere um spread.
    Retorna dict com margem_estimada, spread_sugerido, prob_cover_home approximada.
    """
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)

    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return {"margem": 0.0, "spread": "0.5", "prob_cover_home": 50.0}

    margem = (home_stats["pts_for_avg"] - away_stats["pts_for_avg"]) + 1.5  # incluir vantagem casa
    # sugerir spread arredondado para .5
    spread = round(margem)
    if spread >= 0:
        spread_str = f"-{abs(spread)}.5"  # casa favorita
    else:
        spread_str = f"+{abs(spread)}.5"  # visitante favorito
    # probabilidade aproximada de cobrir (simples sigmoidal approx)
    prob = 50 + (margem * 3)  # escala arbitrária
    prob = max(10.0, min(95.0, prob))
    return {"margem": round(margem, 1), "spread": spread_str, "prob_cover_home": round(prob, 1)}

def prever_first_half(home_id: int, away_id: int, window_games: int = 20) -> tuple[float, float, str]:
    """
    Estima total combinado do 1º tempo e tendência (Over/Under First Half).
    Usa first_half_avg das estatísticas recentes.
    """
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)

    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return 105.0, 50.0, "Mais 105.5"

    estimativa = home_stats["first_half_avg"] + away_stats["first_half_avg"]
    jogos = min(home_stats["games"], away_stats["games"])
    conf = 45 + min(30, jogos)
    conf = max(35.0, min(95.0, conf))
    if estimativa >= 115:
        tendencia = "Mais 115.5 (1º tempo)"
    elif estimativa >= 110:
        tendencia = "Mais 110.5 (1º tempo)"
    else:
        tendencia = "Menos 105.5 (1º tempo)"
    return round(estimativa, 1), round(conf, 1), tendencia

# =============================
# ALERTAS / TELEGRAM
# =============================
def enviar_telegram(msg: str, chat_id: str = TELEGRAM_CHAT_ID) -> bool:
    try:
        resp = requests.get(BASE_URL_TG, params={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
        return resp.status_code == 200
    except requests.RequestException as e:
        st.error(f"Erro enviar Telegram: {e}")
        return False

def formatar_msg_alerta(game: dict, predictions: dict) -> str:
    home = game["home_team"]["full_name"]
    away = game["visitor_team"]["full_name"]
    data_str, hora_str = formatar_data_brt(game.get("datetime") or game.get("date"))
    status = game.get("status", "SCHEDULED")

    msg = f"🏀 <b>Alerta NBA - {data_str} {hora_str} (BRT)</b>\n"
    msg += f"🏟️ {home} vs {away}\n"
    msg += f"📌 Status: {status}\n\n"

    # Total
    t = predictions["total"]
    msg += f"📈 <b>Total</b>: {t['tendencia']} | Estimativa: <b>{t['estimativa']:.1f}</b> | Conf: {t['confianca']:.0f}%\n"
    # Moneyline
    ml = predictions["moneyline"]
    msg += f"🎯 <b>Moneyline</b>: {ml[0]} ({ml[1]:.0f}%)\n"
    # Handicap
    h = predictions["handicap"]
    msg += f"📐 <b>Handicap</b>: Spread sugerido {h['spread']} | Margem estimada: {h['margem']:.1f} | Prob cover casa: {h['prob_cover_home']:.0f}%\n"
    # First Half
    fh = predictions["first_half"]
   # msg += f"⏱️ <b>1º Tempo</b>: {fh[2]} | Estimativa: <b>{fh[0'] if False else fh[0]:.1f}</b> | Conf: {fh[1]:.0f}%\n"
    msg += f"⏱️ <b>1º Tempo</b>: {fh[2]} | Estimativa: <b>{fh[0]:.1f}</b> | Conf: {fh[1]:.0f}%\n"                                                     

    return msg

def verificar_e_enviar_alerta(game: dict, predictions: dict):
    alertas = carregar_alertas()
    fid = str(game.get("id"))
    if fid not in alertas:
        # salvar previsão
        alertas[fid] = {
            "game_id": fid,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "conferido": False
        }
        salvar_alertas(alertas)
        msg = formatar_msg_alerta(game, predictions)
        enviar_telegram(msg)

# =============================
# RESULTADOS & CONFERÊNCIA
# =============================
def processar_resultado_nba(game: dict, alerta_info: dict) -> dict:
    """
    Recebe um jogo (possivelmente final) e a previsão salva e retorna info de conferência.
    """
    home = game["home_team"]["full_name"]
    away = game["visitor_team"]["full_name"]
    status = game.get("status", "").upper()
    home_score = game.get("home_team_score")
    vis_score = game.get("visitor_team_score")
    total = (home_score or 0) + (vis_score or 0)
    resultado = "⏳ Aguardando"

    pred = alerta_info.get("predictions", {})

    # checar finalizado
    if status in ("FINAL", "FINALIZED"):
        # Total
        t = pred.get("total", {})
        tendencia = t.get("tendencia", "")
        # extrair threshold numérico do tendencia (último token)
        try:
            th = float(tendencia.split()[-1])
        except Exception:
            th = 215.5
        # se "Mais" in tendencia
        if "Mais" in tendencia:
            total_res = "🟢 GREEN" if total > th else "🔴 RED"
        elif "Menos" in tendencia:
            total_res = "🟢 GREEN" if total < th else "🔴 RED"
        else:
            total_res = "⚪ INDEFINIDO"

        # Primeiro tempo (se disponível)
        fh_pred = pred.get("first_half", {})
        try:
            th_fh = float(fh_pred.get("tendencia", "105.5").split()[-1])
        except Exception:
            th_fh = 105.5
        # attempt to read first half actual points from game fields
        home_q1 = game.get("home_q1") or 0
        home_q2 = game.get("home_q2") or 0
        vis_q1 = game.get("visitor_q1") or 0
        vis_q2 = game.get("visitor_q2") or 0
        first_half_total = home_q1 + home_q2 + vis_q1 + vis_q2
        if "Mais" in fh_pred.get("tendencia", ""):
            fh_res = "🟢 GREEN" if first_half_total > th_fh else "🔴 RED"
        else:
            fh_res = "🟢 GREEN" if first_half_total < th_fh else "🔴 RED"

        result = {
            "home": home,
            "away": away,
            "status": status,
            "placar": f"{home_score} x {vis_score}",
            "total": total,
            "total_result": total_res,
            "first_half_total": first_half_total,
            "first_half_result": fh_res
        }
        return result
    else:
        return {
            "home": home,
            "away": away,
            "status": status,
            "placar": "-" ,
            "total": total,
            "total_result": "⏳ Aguardando",
            "first_half_total": None,
            "first_half_result": "⏳ Aguardando"
        }

def enviar_resultado_telegram_nba(resultado: dict):
    msg = (
        f"📊 <b>Resultado Conferido (NBA)</b>\n"
        f"🏟️ {resultado['home']} vs {resultado['away']}\n"
        f"📌 Status: {resultado['status']}\n"
        f"📊 Placar Final: <b>{resultado['placar']}</b>\n"
        f"🏀 Total: {resultado['total']} -> {resultado['total_result']}\n"
        f"⏱️ 1º Tempo: {resultado.get('first_half_total')} -> {resultado.get('first_half_result')}"
    )
    enviar_telegram(msg, TELEGRAM_CHAT_ID_ALT2)

# =============================
# PDF (mantendo estilo)
# =============================
def gerar_relatorio_pdf(rows: list) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)
    data = [["Jogo", "Modalidade", "Estimativa", "Confiança", "Placar", "Status", "Resultado", "Hora"]] + rows
    table = Table(data, repeatRows=1, colWidths=[150, 90, 80, 70, 70, 70, 80, 70])
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1f2937")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ])
    for i in range(1, len(data)):
        bg = colors.HexColor("#F3F4F6") if i % 2 == 0 else colors.white
        style.add('BACKGROUND', (0,i), (-1,i), bg)
    table.setStyle(style)
    doc.build([table])
    buffer.seek(0)
    return buffer

# =============================
# STREAMLIT UI & FLUXO
# =============================
def main():
    st.set_page_config(page_title="🏀 Elite Master - NBA Alerts", layout="wide")
    st.title("🏀 Elite Master — Sistema de Alertas Automáticos (NBA)")

    with st.sidebar:
        st.header("Configurações")
        top_n = st.selectbox("📊 Top jogos a enviar", [3,5,10], index=0)
        janela = st.slider("Janela (nº jogos recentes p/ médias)", min_value=5, max_value=40, value=20)
        enviar_auto = st.checkbox("📤 Enviar Top automaticamente ao Telegram", value=False)
        st.markdown("API: BallDontLie (com chave). Ajuste se necessário nos env vars.")

    col1, col2 = st.columns([2,1])
    with col1:
        data_sel = st.date_input("📅 Data para análise:", value=date.today())
    with col2:
        st.write(" ")
        if st.button("🔍 Buscar & Analisar (NBA)"):
            processar_dia_nba(data_sel, top_n, janela, enviar_auto)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Atualizar Cache / Forçar Re-fetch"):
            limpar_caches()
            st.success("Cache limpo.")
    with col2:
        if st.button("📊 Conferir Resultados (salvos)"):
            conferir_resultados_nba()
    with col3:
        if st.button("🧹 Limpar Alertas"):
            try:
                if os.path.exists(ALERTAS_PATH):
                    os.remove(ALERTAS_PATH)
                st.success("Alertas limpos.")
            except Exception as e:
                st.error(f"Erro limpar alertas: {e}")

def processar_dia_nba(data_sel: date, top_n: int, janela: int, enviar_auto: bool):
    data_str = data_sel.strftime("%Y-%m-%d")
    st.info(f"Buscando jogos NBA para {data_sel.strftime('%d/%m/%Y')} ...")
    games = obter_jogos_data(data_str)
    if not games:
        st.warning("Nenhum jogo encontrado para a data selecionada.")
        return

    rows_for_pdf = []
    progress = st.progress(0)
    total = len(games)
    for i, g in enumerate(games):
        # previsões
        home_id = g["home_team"]["id"]
        away_id = g["visitor_team"]["id"]

        total_pred = {}
        total_pred_values = prever_total_points(home_id, away_id, window_games=janela)
        total_pred["estimativa"], total_pred["confianca"], total_pred["tendencia"] = total_pred_values

        moneyline_pred = prever_moneyline(home_id, away_id, window_games=janela)
        handicap_pred = prever_handicap(home_id, away_id, window_games=janela)
        fh_pred_values = prever_first_half(home_id, away_id, window_games=janela)

        predictions = {
            "total": {"estimativa": total_pred["estimativa"], "confianca": total_pred["confianca"], "tendencia": total_pred["tendencia"]},
            "moneyline": moneyline_pred,
            "handicap": handicap_pred,
            "first_half": {"estimativa": fh_pred_values[0], "confianca": fh_pred_values[1], "tendencia": fh_pred_values[2]}
        }

        verificar_e_enviar_alerta(g, predictions)

        hora = datetime.fromisoformat((g.get("datetime") or g.get("date")).replace("Z", "+00:00")) - timedelta(hours=3)
        rows_for_pdf.append([
            f"{abreviar(g['home_team']['full_name'])} vs {abreviar(g['visitor_team']['full_name'])}",
            "Total",
            f"{predictions['total']['estimativa']:.1f}",
            f"{predictions['total']['confianca']:.0f}%",
            "-", g.get("status", "SCHEDULED"), "-", hora.strftime("%d/%m %H:%M")
        ])
        progress.progress((i+1)/total)

    # montar Top N para envio/visualização
    alertas = carregar_alertas()
    jogos_list = []
    for fid, info in alertas.items():
        # recuperar jogo via API para infos atuais
        g = balldontlie_get(f"games/{fid}")
        # se não existir no cache, pular
        if not g:
            continue
        pred = info.get("predictions", {})
        jogos_list.append({
            "id": fid,
            "home": g.get("home_team", {}).get("full_name"),
            "away": g.get("visitor_team", {}).get("full_name"),
            "estimativa": pred.get("total", {}).get("estimativa", 0),
            "confianca": pred.get("total", {}).get("confianca", 0),
            "tendencia": pred.get("total", {}).get("tendencia", "")
        })

    jogos_sorted = sorted(jogos_list, key=lambda x: x["confianca"], reverse=True)[:top_n]
    msg_top = f"📢 TOP {top_n} Jogos NBA - {date.today().strftime('%d/%m/%Y')}\n\n"
    for j in jogos_sorted:
        msg_top += (f"🏟️ {j['home']} vs {j['away']}\n"
                    f"📈 {j['tendencia']} | Estim: {j['estimativa']:.1f} | Conf: {j['confianca']:.0f}%\n\n")

    if enviar_auto:
        enviar_telegram(msg_top, TELEGRAM_CHAT_ID_ALT2)
        st.success("Top jogos enviados ao Telegram.")
    else:
        st.code(msg_top)

    if rows_for_pdf:
        buffer = gerar_relatorio_pdf(rows_for_pdf)
        st.download_button("📄 Baixar Relatório PDF", data=buffer, file_name=f"jogos_nba_{data_str}.pdf", mime="application/pdf")

    st.success("Análise concluída.")

def conferir_resultados_nba():
    alertas = carregar_alertas()
    if not alertas:
        st.info("Nenhum alerta salvo.")
        return
    rows_pdf = []
    mudou = False
    for fid, info in list(alertas.items()):
        # get latest game state
        g = balldontlie_get(f"games/{fid}")
        if not g:
            continue
        res = processar_resultado_nba(g, info)
        exibir_resultado_streamlit(res)
        if res["status"] in ("FINAL", "FINALIZED"):
            enviar_resultado_telegram_nba(res)
            alertas[fid]["conferido"] = True
            mudou = True
        rows_pdf.append([
            f"{abreviar(res['home'])} vs {abreviar(res['away'])}",
            "Total/1H",
            res.get("total", 0),
            "-",
            res.get("placar", "-"),
            res.get("status", "-"),
            res.get("total_result", "-"),
            "-"
        ])
    if mudou:
        salvar_alertas(alertas)
    if rows_pdf:
        buffer = gerar_relatorio_pdf(rows_pdf)
        st.download_button("📄 Baixar Relatório de Conferência", data=buffer, file_name=f"conferencia_nba_{date.today().strftime('%Y-%m-%d')}.pdf", mime="application/pdf")

def exibir_resultado_streamlit(res: dict):
    bg = "#1e4620" if "🟢" in res.get("total_result", "") else ("#5a1e1e" if "🔴" in res.get("total_result", "") else "#2c2c2c")
    st.markdown(f"""
    <div style="border:1px solid #444; border-radius:10px; padding:12px; margin-bottom:10px;
                background-color:{bg}; color:#fff;">
      <b>🏟️ {res.get('home')} vs {res.get('away')}</b><br>
      📌 Status: <b>{res.get('status')}</b><br>
      📊 Placar: <b>{res.get('placar')}</b><br>
      🏀 Total: <b>{res.get('total')}</b> -> {res.get('total_result')}<br>
      ⏱️ 1º Tempo: {res.get('first_half_total')} -> {res.get('first_half_result')}
    </div>
    """, unsafe_allow_html=True)

def limpar_caches():
    for f in [CACHE_GAMES, CACHE_TEAMS, CACHE_STATS]:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

if __name__ == "__main__":
    main()
