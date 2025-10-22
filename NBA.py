# app_nba_elite_master.py
import streamlit as st
from datetime import datetime, timedelta, date
import requests
import json
import os
import io
import pandas as pd
import math
import time
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

# =============================
# CONFIGURAÇÕES E SEGURANÇA
# =============================
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", "7da89f74-317a-45a0-88f9-57cccfef5a00")
BALLDONTLIE_BASE = os.getenv("BALLDONTLIE_BASE", "https://api.balldontlie.io/v1")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003073115320")
TELEGRAM_CHAT_ID_ALT2 = os.getenv("TELEGRAM_CHAT_ID_ALT2", "-1002754276285")
BASE_URL_TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# Arquivos de cache/estado
ALERTAS_PATH = "alertas_nba.json"            # alertas salvos (predictions)
SENT_ALERTS_PATH = "alerts_sent_nba.json"    # dedupe: mensagens enviadas (hashes)
CACHE_GAMES = "cache_games_nba.json"
CACHE_TEAMS = "cache_teams_nba.json"
CACHE_STATS = "cache_stats_nba.json"
CACHE_PLAYERS = "cache_players_nba.json"
CACHE_TIMEOUT = 3600  # 1h

HEADERS_BDL = {"Authorization": BALLDONTLIE_API_KEY}

# =============================
# UTILITÁRIOS DE CACHE E IO
# =============================
def carregar_json(caminho: str) -> dict:
    try:
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                return json.load(f)
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

def carregar_sent_alerts():
    return carregar_json(SENT_ALERTS_PATH) or {}

def salvar_sent_alerts(dados):
    salvar_json(SENT_ALERTS_PATH, dados)

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

def carregar_cache_players():
    return carregar_json(CACHE_PLAYERS) or {}

def salvar_cache_players(dados):
    salvar_json(CACHE_PLAYERS, dados)

# =============================
# HELPERS e FORMATAÇÃO
# =============================
def formatar_data_brt(date_iso: str) -> tuple[str, str]:
    if not date_iso:
        return "Data inválida", "Hora inválida"
    try:
        dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00")) - timedelta(hours=3)
        return dt.strftime("%d/%m/%Y"), dt.strftime("%H:%M")
    except Exception:
        return "Data inválida", "Hora inválida"

def abreviar(nome: str, l=20):
    if not nome:
        return ""
    return nome if len(nome) <= l else nome[:l-3] + "..."

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

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
# TEAMS & PLAYERS (cache)
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

def obter_players_por_time(team_id: int, per_page=100):
    cache = carregar_cache_players()
    key = f"team_{team_id}"
    if key in cache:
        return cache[key]

    players = []
    page = 1
    while True:
        params = {"team_ids[]": team_id, "per_page": per_page, "page": page}
        data = balldontlie_get("players", params=params)
        if not data or "data" not in data:
            break
        players.extend(data.get("data", []))
        meta = data.get("meta", {})
        total_pages = meta.get("total_pages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.05)
    cache[key] = players
    salvar_cache_players(cache)
    return players

def obter_season_averages_for_players(player_ids: list[int], season: int = None):
    """
    Chama endpoint season_averages para uma lista de player_ids — retorna dict player_id -> averages dict
    """
    if not player_ids:
        return {}
    season = season or datetime.today().year - 1  # fallback para temporada atual/última
    averages = {}
    # ballDontLie supports multiple player_ids by passing player_ids[]=id...
    # mas limite por request — faremos batches de 50
    batch_size = 50
    for i in range(0, len(player_ids), batch_size):
        batch = player_ids[i:i+batch_size]
        params = {"season": season}
        for pid in batch:
            params.setdefault("player_ids[]", []).append(pid)
        resp = balldontlie_get("season_averages", params=params)
        if not resp:
            continue
        for item in resp.get("data", []):
            pid = item.get("player", {}).get("id") or item.get("player")
            averages[pid] = item
    return averages

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
        meta = resp.get("meta", {})
        total_pages = meta.get("total_pages", 1)
        if page >= total_pages:
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
    cache = carregar_cache_stats()
    key = f"team_{team_id}_{window_games}"
    if key in cache:
        return cache[key]

    end = date.today()
    start = end - timedelta(days=365)
    per_page = 100
    page = 1
    games = []
    while len(games) < max(window_games, 10):
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
        meta = resp.get("meta", {})
        total_pages = meta.get("total_pages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.05)

    def _gdate(g):
        d = g.get("datetime") or g.get("date") or g.get("game_date")
        try:
            return datetime.fromisoformat((d or "").replace("Z", "+00:00"))
        except Exception:
            return datetime.min

    games_sorted = sorted(games, key=_gdate, reverse=True)[:window_games]

    if not games_sorted:
        stats = {"pts_for_avg": 0.0, "pts_against_avg": 0.0, "games": 0, "pts_diff_avg": 0.0, "first_half_avg": 0.0, "totals": []}
        cache[key] = stats
        salvar_cache_stats(cache)
        return stats

    pts_for = 0
    pts_against = 0
    first_half_total = 0
    count = 0
    totals = []
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

        total = (home_score or 0) + (visitor_score or 0)
        totals.append(total)

        if home_id == team_id:
            pts_for += home_score
            pts_against += visitor_score
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
        stats = {"pts_for_avg": 0.0, "pts_against_avg": 0.0, "games": 0, "pts_diff_avg": 0.0, "first_half_avg": 0.0, "totals": totals}
    else:
        stats = {
            "pts_for_avg": pts_for / count,
            "pts_against_avg": pts_against / count,
            "games": count,
            "pts_diff_avg": (pts_for - pts_against) / count,
            "first_half_avg": first_half_total / count,
            "totals": totals
        }

    cache[key] = stats
    salvar_cache_stats(cache)
    return stats

# =============================
# CALIBRAÇÃO DE THRESHOLDS (AUTOMÁTICA)
# =============================
def calibrar_thresholds(home_id: int, away_id: int, window_games: int = 30) -> dict:
    """
    Gera thresholds calibrados para Over/Under a partir do histórico de jogos recentes
    de ambos os times: calcula média dos totais e desvio padrão.
    Retorna dict com thresholds para lines (mean, mean+std, mean+1.5*std).
    """
    # pegar estatísticas recentes de ambos
    h_stats = obter_estatisticas_recentes_time(home_id, window_games)
    a_stats = obter_estatisticas_recentes_time(away_id, window_games)

    # juntar listas de totais (cada time traz os totais das suas partidas)
    totals = []
    totals.extend(h_stats.get("totals", []) or [])
    totals.extend(a_stats.get("totals", []) or [])

    # se não houver dados suficientes, usar heurística padrão (215 +/-)
    if not totals:
        mean = 215.0
        std = 8.0
    else:
        mean = sum(totals) / len(totals)
        # cálculo de std populacional
        var = sum((x - mean) ** 2 for x in totals) / len(totals)
        std = math.sqrt(var)

    thresholds = {
        "mean": round(mean, 1),
        "std": round(std, 2),
        "line_mean": f"{round(mean,1)}",
        "line_plus_std": f"{round(mean + std, 1)}",
        "line_plus_1_5std": f"{round(mean + 1.5 * std, 1)}"
    }
    # recomendações (mapear para linhas de apostas comuns .5)
    def to_05(x):
        # converte para nearest .5 e adiciona .5 to represent typical sportsbook format
        return f"{round(x - 0.5 if (x - math.floor(x)) < 0.5 else math.floor(x)+0.5, 1)}"
    thresholds["recommended_lines"] = {
        "conservative": to_05(mean + std),
        "balanced": to_05(mean),
        "aggressive": to_05(mean + 1.5*std)
    }
    return thresholds

# =============================
# LÓGICA DE PREVISÃO — 4 MODALIDADES (usando calibração)
# =============================
def prever_total_points(home_id: int, away_id: int, window_games: int = 20, calibrate: bool = True) -> tuple[float, float, str]:
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        # fallback neutro
        return 215.0, 55.0, "Mais 215.5"

    # estimativa direta: soma das médias de pontos por time
    estimativa = home_stats["pts_for_avg"] + away_stats["pts_for_avg"]

    # confiança baseada em número de jogos e proximidade das médias
    jogos = min(home_stats["games"], away_stats["games"])
    conf = 45 + min(30, jogos)
    diff = abs(home_stats["pts_diff_avg"] - away_stats["pts_diff_avg"])
    conf -= max(0, min(10, diff))
    conf = max(35.0, min(95.0, conf))

    # usar calibração de thresholds se desejado
    if calibrate:
        cal = calibrar_thresholds(home_id, away_id, window_games)
        mean = safe_float(cal.get("mean", 215.0))
    else:
        mean = 215.0

    # selecionar tendência linha mais próxima
    # linhas predefinidas comuns
    lines = [205.5, 210.5, 215.5, 220.5, 225.5, 230.5, 235.5]
    # escolher a linha cuja diferença absoluta para estimativa é mínima OR prefer mean
    candidate = min(lines, key=lambda x: abs(x - estimativa))
    tendencia = f"Mais {candidate:.1f}" if estimativa >= candidate else f"Menos {candidate:.1f}"

    return round(estimativa, 1), round(conf, 1), tendencia

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
        conf = min(95.0, 55 + diff * 3.5)
        return "Casa vencer", round(max(50.0, conf), 1)
    else:
        conf = min(95.0, 55 + abs(diff) * 3.5)
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
    prob = 50 + (margem * 3)
    prob = max(10.0, min(95.0, prob))
    return {"margem": round(margem, 1), "spread": spread_str, "prob_cover_home": round(prob, 1)}

def prever_first_half(home_id: int, away_id: int, window_games: int = 20) -> tuple[float, float, str]:
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return 105.0, 50.0, "Mais 105.5 (1H)"
    estimativa = home_stats["first_half_avg"] + away_stats["first_half_avg"]
    jogos = min(home_stats["games"], away_stats["games"])
    conf = 45 + min(30, jogos)
    conf = max(35.0, min(95.0, conf))
    if estimativa >= 115:
        tendencia = "Mais 115.5 (1H)"
    elif estimativa >= 110:
        tendencia = "Mais 110.5 (1H)"
    else:
        tendencia = "Menos 105.5 (1H)"
    return round(estimativa, 1), round(conf, 1), tendencia

# =============================
# TELEGRAM: dedupe + backoff seguro
# =============================
def send_telegram_with_backoff(msg: str, chat_id: str = TELEGRAM_CHAT_ID_ALT2, max_retries: int = 4) -> bool:
    """
    Envia mensagem para telegram usando retries exponenciais e dedupe (via hash).
    Salva mensagem enviada em SENT_ALERTS_PATH para dedupe.
    """
    # dedupe: hash simples (home+away+msg trimmed)
    key = str(abs(hash(msg)))  # not cryptographic but good enough for dedupe
    sent = carregar_sent_alerts()

    # se já enviado recentemente (24h), evitar reenviar
    if key in sent:
        ts = sent[key].get("ts")
        try:
            sent_time = datetime.fromisoformat(ts)
            if (datetime.now() - sent_time).total_seconds() < 60 * 60 * 24:
                return True  # já enviado recentemente
        except Exception:
            pass

    url = BASE_URL_TG
    params = {"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                # salvar dedupe info
                sent[key] = {"ts": datetime.now().isoformat(), "status_code": 200}
                salvar_sent_alerts(sent)
                return True
            else:
                # registrar tentativa
                sent[key] = {"last_status": resp.status_code, "last_text": resp.text, "ts": datetime.now().isoformat()}
                salvar_sent_alerts(sent)
                # retry with backoff
                time.sleep(backoff)
                backoff *= 2
        except requests.RequestException as e:
            # salvar erro e tentar novamente com backoff
            sent[key] = {"error": str(e), "ts": datetime.now().isoformat()}
            salvar_sent_alerts(sent)
            time.sleep(backoff)
            backoff *= 2
    return False

# =============================
# ALERTAS / FORMATAÇÃO / STREAMLIT
# =============================
def formatar_msg_alerta(game: dict, predictions: dict) -> str:
    try:
        home = game.get("home_team", {}).get("full_name", "Casa")
        away = game.get("visitor_team", {}).get("full_name", "Visitante")
        data_str, hora_str = formatar_data_brt(game.get("datetime") or game.get("date") or "")
        status = game.get("status", "SCHEDULED")

        msg = f"🏀 <b>Alerta NBA - {data_str} {hora_str} (BRT)</b>\n"
        msg += f"🏟️ {home} vs {away}\n"
        msg += f"📌 Status: {status}\n\n"

        # Total
        t = predictions.get("total", {})
        if t:
            estim_t = safe_float(t.get("estimativa", 0))
            conf_t = safe_float(t.get("confianca", 0))
            tend_t = t.get("tendencia", "Mais 215.5")
            msg += f"📈 <b>Total</b>: {tend_t} | Estimativa: <b>{estim_t:.1f}</b> | Conf: {conf_t:.0f}%\n"

        # Moneyline
        ml = predictions.get("moneyline", None)
        if ml and isinstance(ml, (list, tuple)):
            try:
                msg += f"🎯 <b>Moneyline</b>: {ml[0]} ({ml[1]:.0f}%)\n"
            except Exception:
                msg += "🎯 <b>Moneyline</b>: Dados insuficientes\n"

        # Handicap
        h = predictions.get("handicap", None)
        if isinstance(h, dict):
            msg += f"📐 <b>Handicap</b>: Spread {h.get('spread','-')} | Margem: {h.get('margem',0):.1f} | Prob cover casa: {h.get('prob_cover_home',0):.0f}%\n"

        # First Half
        fh = predictions.get("first_half", {})
        if fh:
            try:
                if isinstance(fh, dict):
                    estim_fh = safe_float(fh.get("estimativa", 0))
                    conf_fh = safe_float(fh.get("confianca", 0))
                    tend_fh = fh.get("tendencia", "")
                else:
                    estim_fh = safe_float(fh[0])
                    conf_fh = safe_float(fh[1])
                    tend_fh = fh[2] if len(fh) > 2 else ""
                msg += f"⏱️ <b>1º Tempo</b>: {tend_fh} | Estimativa: <b>{estim_fh:.1f}</b> | Conf: {conf_fh:.0f}%\n"
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
        # Mostrar no Streamlit
        st.info(msg)
        # enviar de forma segura (dedupe+backoff) se habilitado
        if send_to_telegram:
            ok = send_telegram_with_backoff(msg, TELEGRAM_CHAT_ID_ALT2)
            if ok:
                st.success("Alerta enviado ao Telegram.")
            else:
                st.warning("Falha ao enviar alerta ao Telegram (ver logs).")

# =============================
# RESULTADOS & CONFERÊNCIA
# =============================
def processar_resultado_nba(game: dict, alerta_info: dict) -> dict:
    home = game.get("home_team", {}).get("full_name", "Casa")
    away = game.get("visitor_team", {}).get("full_name", "Visitante")
    status = (game.get("status") or "").upper()
    home_score = game.get("home_team_score")
    vis_score = game.get("visitor_team_score")
    total = (home_score or 0) + (vis_score or 0)
    pred = alerta_info.get("predictions", {})

    if status in ("FINAL", "FINALIZED"):
        t = pred.get("total", {})
        tendencia = t.get("tendencia", "")
        try:
            th = float(tendencia.split()[-1])
        except Exception:
            th = 215.5
        if "Mais" in tendencia:
            total_res = "🟢 GREEN" if total > th else "🔴 RED"
        elif "Menos" in tendencia:
            total_res = "🟢 GREEN" if total < th else "🔴 RED"
        else:
            total_res = "⚪ INDEFINIDO"

        fh_pred = pred.get("first_half", {})
        try:
            if isinstance(fh_pred, dict):
                th_fh = float(str(fh_pred.get("tendencia","105.5")).split()[-1])
            else:
                th_fh = float(str(fh_pred[2]).split()[-1]) if fh_pred and len(fh_pred) > 2 else 105.5
        except Exception:
            th_fh = 105.5

        home_q1 = game.get("home_q1") or 0
        home_q2 = game.get("home_q2") or 0
        vis_q1 = game.get("visitor_q1") or 0
        vis_q2 = game.get("visitor_q2") or 0
        first_half_total = home_q1 + home_q2 + vis_q1 + vis_q2
        if "Mais" in (fh_pred.get("tendencia", "") if isinstance(fh_pred, dict) else (str(fh_pred[2]) if fh_pred and len(fh_pred)>2 else "")):
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
    send_telegram_with_backoff(msg, TELEGRAM_CHAT_ID_ALT2)

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
# UI STREAMLIT - Fluxo principal
# =============================
def main():
    st.set_page_config(page_title="🏀 Elite Master - NBA Alerts", layout="wide")
    st.title("🏀 Elite Master — Sistema de Alertas Automáticos (NBA)")

    # Sidebar
    with st.sidebar:
        st.header("Configurações")
        top_n = st.selectbox("📊 Top jogos a enviar / exibir", [3,5,10], index=0)
        janela = st.slider("Janela (nº jogos recentes p/ médias)", min_value=5, max_value=60, value=20)
        enviar_auto = st.checkbox("📤 Enviar alertas ao Telegram (opcional)", value=False)
        calibrar_check = st.checkbox("🔧 Calibrar thresholds automaticamente", value=True)
        st.markdown("Guarde sua chave em BALLDONTLIE_API_KEY (recomendado).")
        st.markdown("Obs: envio Telegram usa dedupe e backoff.")

    col1, col2 = st.columns([2,1])
    with col1:
        data_sel = st.date_input("📅 Data para análise:", value=date.today())
    with col2:
        st.write(" ")
        if st.button("🔍 Buscar & Analisar (NBA)"):
            processar_dia_nba(data_sel, top_n, janela, enviar_auto, calibrar_check)

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

    # Player metrics quick tool
    st.markdown("---")
    st.subheader("🔎 Métricas de Jogadores (Season Averages)")
    team_input = st.text_input("Time (abreviação ou nome) — ex: GSW, Lakers (deixe vazio para pular)")
    season_input = st.number_input("Season (ano inicial) — ex: 2024", min_value=2000, max_value=2100, value=datetime.today().year - 1)
    if st.button("📈 Buscar métricas do time"):
        if team_input:
            show_player_metrics(team_input, season_input)

def processar_dia_nba(data_sel: date, top_n: int, janela: int, enviar_auto: bool, calibrar_check: bool):
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
        # ids
        home_id = g.get("home_team", {}).get("id")
        away_id = g.get("visitor_team", {}).get("id")

        # previsões — se calibrar_check, passe calibrate=True
        estim_total, conf_total, tend_total = prever_total_points(home_id, away_id, window_games=janela, calibrate=calibrar_check)
        ml_pred = prever_moneyline(home_id, away_id, window_games=janela)
        handicap_pred = prever_handicap(home_id, away_id, window_games=janela)
        fh_estim, fh_conf, fh_tend = prever_first_half(home_id, away_id, window_games=janela)

        predictions = {
            "total": {"estimativa": estim_total, "confianca": conf_total, "tendencia": tend_total},
            "moneyline": ml_pred,
            "handicap": handicap_pred,
            "first_half": {"estimativa": fh_estim, "confianca": fh_conf, "tendencia": fh_tend}
        }

        # salvar alerta e (opcional) enviar telegram
        verificar_e_enviar_alerta(g, predictions, send_to_telegram=enviar_auto)

        # linha para PDF
        try:
            hora = datetime.fromisoformat((g.get("datetime") or g.get("date")).replace("Z", "+00:00")) - timedelta(hours=3)
            hora_str = hora.strftime("%d/%m %H:%M")
        except Exception:
            hora_str = "-"

        rows_for_pdf.append([
            f"{abreviar(g.get('home_team', {}).get('full_name','Casa'))} vs {abreviar(g.get('visitor_team', {}).get('full_name','Visitante'))}",
            "Total",
            f"{predictions['total']['estimativa']:.1f}",
            f"{predictions['total']['confianca']:.0f}%",
            "-", g.get("status", "SCHEDULED"), "-", hora_str
        ])
        progress.progress((i+1)/total)

    # top N exibição
    alertas = carregar_alertas()
    jogos_list = []
    for fid, info in alertas.items():
        g = balldontlie_get(f"games/{fid}")
        if not g or "data" in g and isinstance(g["data"], dict) and "id" not in g["data"]:
            # some endpoints return direct object or wrapped; try both shapes
            pass
        # g may be a dict with 'data' wrapper when fetching a single game
        if isinstance(g, dict) and "data" in g:
            g_data = g["data"]
        else:
            g_data = g

        if not g_data:
            continue
        pred = info.get("predictions", {})
        jogos_list.append({
            "id": fid,
            "home": g_data.get("home_team", {}).get("full_name"),
            "away": g_data.get("visitor_team", {}).get("full_name"),
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

    st.success("Análise concluída.")

def conferir_resultados_nba():
    alertas = carregar_alertas()
    if not alertas:
        st.info("Nenhum alerta salvo.")
        return
    rows_pdf = []
    mudou = False
    for fid, info in list(alertas.items()):
        g = balldontlie_get(f"games/{fid}")
        if isinstance(g, dict) and "data" in g:
            g_data = g["data"]
        else:
            g_data = g
        if not g_data:
            continue
        res = processar_resultado_nba(g_data, info)
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
    for f in [CACHE_GAMES, CACHE_TEAMS, CACHE_STATS, CACHE_PLAYERS]:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

# =============================
# PLAYER METRICS DISPLAY
# =============================
def show_player_metrics(team_input: str, season: int):
    """
    Busca jogadores do time e exibe season averages (quando disponível).
    Aceita abreviação ou parte do nome.
    """
    st.info(f"Buscando jogadores e season averages para time '{team_input}' (season {season})...")
    teams = obter_times()
    # tentar encontrar team pela abreviação ou pelo nome
    matched = None
    for tid, t in teams.items():
        if team_input.lower() in (t.get("abbreviation","").lower(), t.get("full_name","").lower(), t.get("name","").lower()):
            matched = t
            break
    if not matched:
        st.warning("Time não encontrado localmente — buscando lista completa...")
        # fallback listar times
        df = pd.DataFrame([{"id": k, **v} for k,v in teams.items()])
        st.dataframe(df[["id","full_name","abbreviation","city"]])
        return

    st.success(f"Time encontrado: {matched.get('full_name')} ({matched.get('abbreviation')})")
    players = obter_players_por_time(matched.get("id"))
    if not players:
        st.warning("Nenhum jogador encontrado para o time.")
        return

    # coletar ids e buscar season averages
    player_ids = [p.get("id") for p in players if p.get("id")]
    avgs = obter_season_averages_for_players(player_ids, season=season)

    rows = []
    for p in players:
        pid = p.get("id")
        avg = avgs.get(pid, {})
        stats = avg.get("stats") if isinstance(avg, dict) else None
        pts = stats.get("pts") if stats else None
        reb = stats.get("reb") if stats else None
        ast = stats.get("ast") if stats else None
        rows.append({
            "player_id": pid,
            "name": f"{p.get('first_name','')} {p.get('last_name','')}",
            "position": p.get("position"),
            "pts_avg": round(pts,1) if pts is not None else "-",
            "reb_avg": round(reb,1) if reb is not None else "-",
            "ast_avg": round(ast,1) if ast is not None else "-"
        })

    df = pd.DataFrame(rows)
    st.dataframe(df.sort_values(by="pts_avg", ascending=False).reset_index(drop=True))

# =============================
# Entrypoint
# =============================
if __name__ == "__main__":
    main()
