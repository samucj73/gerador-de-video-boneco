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
# CONFIGURAÇÕES - VERSÃO TESTE
# =============================
# ✅ CHAVES DIRETAS NO CÓDIGO (APENAS PARA TESTE)
BALLDONTLIE_API_KEY = "7da89f74-317a-45a0-88f9-57cccfef5a00"
TELEGRAM_TOKEN = "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY"
TELEGRAM_CHAT_ID = "-1003073115320"
TELEGRAM_CHAT_ID_ALT2 = "-1002754276285"

BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"
BASE_URL_TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

ALERTAS_PATH = "alertas_nba.json"
CACHE_GAMES = "cache_games_nba.json"
CACHE_TEAMS = "cache_teams_nba.json"
CACHE_STATS = "cache_stats_nba.json"
CACHE_TIMEOUT = 3600  # 1h

HEADERS_BDL = {"Authorization": BALLDONTLIE_API_KEY}

# ✅ Rate Limiting
REQUEST_TIMEOUT = 15
LAST_REQUEST_TIME = 0
MIN_REQUEST_INTERVAL = 1.1  # 1.1 segundo entre requests

# =============================
# UTILITÁRIOS DE CACHE E IO
# =============================
def carregar_json(caminho: str) -> dict:
    try:
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)
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
# REQUISIÇÕES À BALLDONTLIE COM RATE LIMITING
# =============================
def balldontlie_get(path: str, params: dict | None = None, timeout: int = REQUEST_TIMEOUT) -> dict | None:
    global LAST_REQUEST_TIME
    
    # ✅ Rate Limiting
    current_time = time.time()
    time_since_last_request = current_time - LAST_REQUEST_TIME
    if time_since_last_request < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - time_since_last_request
        time.sleep(sleep_time)
    
    try:
        url = BALLDONTLIE_BASE.rstrip("/") + "/" + path.lstrip("/")
        resp = requests.get(url, headers=HEADERS_BDL, params=params, timeout=timeout)
        
        LAST_REQUEST_TIME = time.time()
        
        if resp.status_code == 429:
            st.warning("⚠️ Rate limit atingido. Aguardando 60 segundos...")
            time.sleep(60)
            resp = requests.get(url, headers=HEADERS_BDL, params=params, timeout=timeout)
            LAST_REQUEST_TIME = time.time()
        
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Erro BallDontLie {path}: {e}")
        return None

# =============================
# TIMES (cache)
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
    max_pages = 3
    
    while page <= max_pages:
        params = {"dates[]": data_str, "per_page": per_page, "page": page}
        resp = balldontlie_get("games", params=params)
        if not resp or "data" not in resp:
            break
            
        data_chunk = resp["data"]
        if not data_chunk:
            break
            
        jogos.extend(data_chunk)
        meta = resp.get("meta", {})
        
        current_page = meta.get("current_page", page)
        total_pages = meta.get("total_pages", 1)
        
        if current_page >= total_pages:
            break
            
        page += 1

    cache[key] = jogos
    salvar_cache_games(cache)
    return jogos

# =============================
# ESTATÍSTICAS RECENTES DO TIME
# =============================
def obter_estatisticas_recentes_time(team_id: int, window_games: int = 20) -> dict:
    cache = carregar_cache_stats()
    key = f"team_{team_id}_{window_games}"
    
    if key in cache:
        cached_data = cache[key]
        if cached_data.get("games", 0) > 0:
            return cached_data

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=90)
    
    per_page = 100
    page = 1
    games = []
    max_pages = 2
    
    while len(games) < window_games * 2 and page <= max_pages:
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
        games.extend(resp["data"])
        meta = resp.get("meta")
        if not meta or page >= meta.get("total_pages", 1):
            break
        page += 1

    def _gdate(g):
        d = g.get("datetime") or g.get("date") or g.get("game_date")
        try:
            return datetime.fromisoformat((d or "").replace("Z", "+00:00"))
        except Exception:
            return datetime.min

    games_finalizados = [g for g in games if g.get("status", "").upper() in ("FINAL", "FINAL/OT")]
    games_sorted = sorted(games_finalizados, key=_gdate, reverse=True)[:window_games]

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
        home_id = g.get("home_team", {}).get("id")
        visitor_id = g.get("visitor_team", {}).get("id")
        home_score = g.get("home_team_score")
        visitor_score = g.get("visitor_team_score")
        if home_score is None or visitor_score is None:
            continue

        if home_id == team_id:
            pts_for += home_score
            pts_against += visitor_score
            q1 = g.get("home_periods", [0, 0, 0, 0])[0] if g.get("home_periods") else g.get("home_q1", 0)
            q2 = g.get("home_periods", [0, 0, 0, 0])[1] if g.get("home_periods") else g.get("home_q2", 0)
            fh = q1 + q2
        else:
            pts_for += visitor_score
            pts_against += home_score
            q1 = g.get("visitor_periods", [0, 0, 0, 0])[0] if g.get("visitor_periods") else g.get("visitor_q1", 0)
            q2 = g.get("visitor_periods", [0, 0, 0, 0])[1] if g.get("visitor_periods") else g.get("visitor_q2", 0)
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
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)
    
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return 215.0, 50.0, "Dados Insuficientes"
    
    estimativa = (home_stats["pts_for_avg"] + away_stats["pts_for_avg"]) * 0.98
    
    jogos_minimos = min(home_stats["games"], away_stats["games"])
    conf_base = 40 + min(25, jogos_minimos)
    
    diff_qualidade = abs(home_stats["pts_diff_avg"] - away_stats["pts_diff_avg"])
    conf_ajustada = conf_base - min(15, diff_qualidade * 2)
    
    confianca = max(30.0, min(85.0, conf_ajustada))
    
    if estimativa >= 230:
        tendencia = "Mais 230.5"
    elif estimativa >= 220:
        tendencia = "Mais 220.5"
    elif estimativa >= 215:
        tendencia = "Mais 215.5"
    elif estimativa >= 205:
        tendencia = "Mais 205.5"
    else:
        tendencia = "Menos 205.5"
        
    return round(estimativa, 1), round(confianca, 1), tendencia

def prever_moneyline(home_id: int, away_id: int, window_games: int = 20) -> tuple[str, float]:
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)
    if home_stats["games"] == 0 and away_stats["games"] == 0:
        return "Empate", 50.0
    diff = home_stats["pts_diff_avg"] - away_stats["pts_diff_avg"]
    home_bonus = 1.5
    diff += home_bonus
    if abs(diff) < 2.5:
        return "Empate", 50.0
    elif diff > 0:
        conf = min(90.0, 50 + diff * 3.0)
        return "Casa vencer", round(max(50.0, conf), 1)
    else:
        conf = min(90.0, 50 + abs(diff) * 3.0)
        return "Fora vencer", round(max(50.0, conf), 1)

def prever_handicap(home_id: int, away_id: int, window_games: int = 20) -> dict:
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return {"margem": 0.0, "spread": "0.5", "prob_cover_home": 50.0}
    margem = (home_stats["pts_for_avg"] - away_stats["pts_for_avg"]) + 1.5
    spread = round(margem)
    if spread >= 0:
        spread_str = f"-{abs(spread)}.5"
    else:
        spread_str = f"+{abs(spread)}.5"
    prob = 50 + (margem * 2.5)
    prob = max(15.0, min(90.0, prob))
    return {"margem": round(margem, 1), "spread": spread_str, "prob_cover_home": round(prob, 1)}

def prever_first_half(home_id: int, away_id: int, window_games: int = 20) -> tuple[float, float, str]:
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return 105.0, 50.0, "Mais 105.5 (1H)"
    estimativa = home_stats["first_half_avg"] + away_stats["first_half_avg"]
    jogos = min(home_stats["games"], away_stats["games"])
    conf = 40 + min(25, jogos)
    conf = max(30.0, min(85.0, conf))
    if estimativa >= 115:
        tendencia = "Mais 115.5 (1H)"
    elif estimativa >= 110:
        tendencia = "Mais 110.5 (1H)"
    else:
        tendencia = "Menos 105.5 (1H)"
    return round(estimativa, 1), round(conf, 1), tendencia

# =============================
# ALERTAS / FORMATAÇÃO / TELEGRAM
# =============================
def enviar_telegram(msg: str, chat_id: str = TELEGRAM_CHAT_ID) -> bool:
    try:
        resp = requests.get(BASE_URL_TG, params={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
        return resp.status_code == 200
    except requests.RequestException as e:
        st.error(f"Erro enviar Telegram: {e}")
        return False

def formatar_msg_alerta(game: dict, predictions: dict) -> str:
    try:
        home = game.get("home_team", {}).get("full_name", "Casa")
        away = game.get("visitor_team", {}).get("full_name", "Visitante")
        data_str, hora_str = formatar_data_brt(game.get("datetime") or game.get("date") or "")
        status = game.get("status", "SCHEDULED")

        msg = f"🏀 <b>Alerta NBA - {data_str} {hora_str} (BRT)</b>\n"
        msg += f"🏟️ {home} vs {away}\n"
        msg += f"📌 Status: {status}\n\n"

        t = predictions.get("total", {})
        if t:
            estim_t = t.get("estimativa", 0)
            conf_t = t.get("confianca", 0)
            tend_t = t.get("tendencia", "Mais 215.5")
            msg += f"📈 <b>Total</b>: {tend_t} | Estimativa: <b>{estim_t:.1f}</b> | Conf: {conf_t:.0f}%\n"
        
        ml = predictions.get("moneyline", None)
        if ml:
            try:
                msg += f"🎯 <b>Moneyline</b>: {ml[0]} ({ml[1]:.0f}%)\n"
            except Exception:
                msg += f"🎯 <b>Moneyline</b>: Dados insuficientes\n"
        
        h = predictions.get("handicap", None)
        if isinstance(h, dict):
            msg += f"📐 <b>Handicap</b>: Spread sugerido {h.get('spread','-')} | Margem: {h.get('margem',0):.1f} | Prob cover casa: {h.get('prob_cover_home',0):.0f}%\n"
        
        fh = predictions.get("first_half", {})
        if fh:
            try:
                estim_fh = fh.get("estimativa") if isinstance(fh, dict) else fh[0]
                conf_fh = fh.get("confianca") if isinstance(fh, dict) else fh[1]
                tend_fh = fh.get("tendencia") if isinstance(fh, dict) else fh[2]
                msg += f"⏱️ <b>1º Tempo</b>: {tend_fh} | Estimativa: <b>{float(estim_fh):.1f}</b> | Conf: {float(conf_fh):.0f}%\n"
            except Exception:
                msg += "⏱️ <b>1º Tempo</b>: Dados indisponíveis\n"

        msg += "\n🏆 <b>Elite Master</b> - Análise Automática NBA"
        return msg
    except Exception as e:
        return f"⚠️ Erro ao formatar alerta: {e}"

def verificar_e_enviar_alerta(game: dict, predictions: dict, send_to_telegram: bool = False):
    alertas = carregar_alertas()
    fid = str(game.get("id"))
    if fid not in alertas:
        alertas[fid] = {
            "game_id": fid,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "conferido": False
        }
        salvar_alertas(alertas)
        msg = formatar_msg_alerta(game, predictions)
        st.info(msg)
        if send_to_telegram:
            enviar_telegram(msg)

# =============================
# RESULTADOS & CONFERÊNCIA
# =============================
def processar_resultado_nba(game: dict, alerta_info: dict) -> dict:
    home = game.get("home_team", {}).get("full_name", "Casa")
    away = game.get("visitor_team", {}).get("full_name", "Visitante")
    status = (game.get("status") or "").upper()
    home_score = game.get("home_team_score")
    vis_score = game.get("visitor_team_score")
    
    if home_score is None or vis_score is None:
        return {
            "home": home,
            "away": away,
            "status": status,
            "placar": "Indisponível",
            "total": 0,
            "total_result": "⏳ Aguardando Dados",
            "first_half_total": None,
            "first_half_result": "⏳ Aguardando Dados"
        }
    
    total = home_score + vis_score
    pred = alerta_info.get("predictions", {})

    if status in ("FINAL", "FINAL/OT"):
        t = pred.get("total", {})
        tendencia = t.get("tendencia", "")
        
        try:
            th_str = tendencia.split()[-1].replace("(", "").replace(")", "")
            th = float(th_str)
        except (ValueError, IndexError):
            th = 215.5

        if "Mais" in tendencia:
            total_res = "🟢 GREEN" if total > th else "🔴 RED"
        elif "Menos" in tendencia:
            total_res = "🟢 GREEN" if total < th else "🔴 RED"
        else:
            total_res = "⚪ INDEFINIDO"

        home_q1 = game.get("home_periods", [0, 0, 0, 0])[0] if game.get("home_periods") else game.get("home_q1", 0)
        home_q2 = game.get("home_periods", [0, 0, 0, 0])[1] if game.get("home_periods") else game.get("home_q2", 0)
        vis_q1 = game.get("visitor_periods", [0, 0, 0, 0])[0] if game.get("visitor_periods") else game.get("visitor_q1", 0)
        vis_q2 = game.get("visitor_periods", [0, 0, 0, 0])[1] if game.get("visitor_periods") else game.get("visitor_q2", 0)
        
        first_half_total = (home_q1 or 0) + (home_q2 or 0) + (vis_q1 or 0) + (vis_q2 or 0)

        fh_pred = pred.get("first_half", {})
        try:
            if isinstance(fh_pred, dict):
                fh_tendencia = fh_pred.get("tendencia", "105.5")
                th_fh_str = fh_tendencia.split()[-1].replace("(", "").replace(")", "")
                th_fh = float(th_fh_str)
            else:
                fh_tendencia = fh_pred[2] if len(fh_pred) > 2 else "105.5"
                th_fh_str = fh_tendencia.split()[-1].replace("(", "").replace(")", "")
                th_fh = float(th_fh_str)
        except (ValueError, IndexError):
            th_fh = 105.5

        if "Mais" in (fh_pred.get("tendencia", "") if isinstance(fh_pred, dict) else str(fh_pred[2]) if fh_pred else ""):
            fh_res = "🟢 GREEN" if first_half_total > th_fh else "🔴 RED"
        else:
            fh_res = "🟢 GREEN" if first_half_total < th_fh else "🔴 RED"

        return {
            "home": home,
            "away": away,
            "status": status,
            "placar": f"{home_score} x {vis_score}",
            "total": total,
            "total_result": total_res,
            "first_half_total": first_half_total,
            "first_half_result": fh_res
        }
    else:
        return {
            "home": home,
            "away": away,
            "status": status,
            "placar": "-",
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
# PDF
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
    
    # Status das configurações
    st.sidebar.header("🔧 Configurações")
    st.sidebar.success("✅ BallDontLie API: Configurada")
    st.sidebar.success("✅ Telegram: Configurado")
    
    with st.sidebar:
        st.header("Configurações de Análise")
        top_n = st.selectbox("📊 Top jogos a enviar", [3,5,10], index=0)
        janela = st.slider("Janela (nº jogos recentes p/ médias)", min_value=5, max_value=40, value=20)
        enviar_auto = st.checkbox("📤 Enviar alertas ao Telegram", value=False)
        
        st.markdown("---")
        st.markdown("**API:** BallDontLie (60 req/min)")
        st.markdown("**Rate Limiting:** Ativo")
        st.markdown("**Modo:** Teste")

    col1, col2 = st.columns([2,1])
    with col1:
        data_sel = st.date_input("📅 Data para análise:", value=date.today())
    with col2:
        st.write(" ")
        if st.button("🔍 Buscar & Analisar (NBA)", type="primary"):
            processar_dia_nba(data_sel, top_n, janela, enviar_auto)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Atualizar Cache"):
            limpar_caches()
            st.success("Cache limpo.")
    with col2:
        if st.button("📊 Conferir Resultados"):
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
    st.warning("⚠️ Rate Limiting ativo: ~1 requisição/segundo")
    
    games = obter_jogos_data(data_str)
    if not games:
        st.warning("Nenhum jogo encontrado para a data selecionada.")
        return

    rows_for_pdf = []
    progress = st.progress(0)
    total = len(games)
    
    for i, g in enumerate(games):
        home_id = g["home_team"]["id"]
        away_id = g["visitor_team"]["id"]

        estim_total, conf_total, tend_total = prever_total_points(home_id, away_id, window_games=janela)
        ml_pred = prever_moneyline(home_id, away_id, window_games=janela)
        handicap_pred = prever_handicap(home_id, away_id, window_games=janela)
        fh_estim, fh_conf, fh_tend = prever_first_half(home_id, away_id, window_games=janela)

        predictions = {
            "total": {"estimativa": estim_total, "confianca": conf_total, "tendencia": tend_total},
            "moneyline": ml_pred,
            "handicap": handicap_pred,
            "first_half": {"estimativa": fh_estim, "confianca": fh_conf, "tendencia": fh_tend}
        }

        verificar_e_enviar_alerta(g, predictions, send_to_telegram=enviar_auto)

        hora_str = "-"
        try:
            hora = datetime.fromisoformat((g.get("datetime") or g.get("date")).replace("Z", "+00:00")) - timedelta(hours=3)
            hora_str = hora.strftime("%d/%m %H:%M")
        except Exception:
            pass

        rows_for_pdf.append([
            f"{abreviar(g['home_team']['full_name'])} vs {abreviar(g['visitor_team']['full_name'])}",
            "Total",
            f"{predictions['total']['estimativa']:.1f}",
            f"{predictions['total']['confianca']:.0f}%",
            "-", g.get("status", "SCHEDULED"), "-", hora_str
        ])
        progress.progress((i+1)/total)

    alertas = carregar_alertas()
    jogos_list = []
    for fid, info in alertas.items():
        cache_games = carregar_cache_games()
        g_cached = None
        for data_key, games_in_cache in cache_games.items():
            for game_in_cache in games_in_cache:
                if str(game_in_cache.get("id")) == fid:
                    g_cached = game_in_cache
                    break
            if g_cached:
                break
        
        if g_cached:
            g = g_cached
        else:
            g = balldontlie_get(f"games/{fid}")
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

    st.code(msg_top)
    if rows_for_pdf:
        buffer = gerar_relatorio_pdf(rows_for_pdf)
        st.download_button("📄 Baixar Relatório PDF", data=buffer, file_name=f"jogos_nba_{data_str}.pdf", mime="application/pdf")

    st.success("Análise concluída!")

def conferir_resultados_nba():
    alertas = carregar_alertas()
    if not alertas:
        st.info("Nenhum alerta salvo.")
        return
    rows_pdf = []
    mudou = False
    
    st.warning("⚠️ Conferindo resultados com rate limiting ativo...")
    
    for fid, info in list(alertas.items()):
        cache_games = carregar_cache_games()
        g_cached = None
        for data_key, games_in_cache in cache_games.items():
            for game_in_cache in games_in_cache:
                if str(game_in_cache.get("id")) == fid:
                    g_cached = game_in_cache
                    break
            if g_cached:
                break
        
        if g_cached:
            g = g_cached
        else:
            g = balldontlie_get(f"games/{fid}")
            if not g:
                continue
                
        res = processar_resultado_nba(g, info)
        exibir_resultado_streamlit(res)
        if res["status"] in ("FINAL", "FINAL/OT") and not info.get("conferido", False):
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
