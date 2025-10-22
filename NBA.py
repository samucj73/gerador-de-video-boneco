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
# CONFIGURAÇÕES
# =============================
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
CACHE_TIMEOUT = 86400  # 24h

HEADERS_BDL = {"Authorization": BALLDONTLIE_API_KEY}

# Rate limiting balanceado
REQUEST_TIMEOUT = 10
LAST_REQUEST_TIME = 0
MIN_REQUEST_INTERVAL = 0.8

# =============================
# CACHE E IO
# =============================
def carregar_json(caminho: str) -> dict:
    try:
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)
            if datetime.now().timestamp() - os.path.getmtime(caminho) > CACHE_TIMEOUT:
                return {}
            return dados
    except Exception:
        return {}
    return {}

def salvar_json(caminho: str, dados: dict):
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

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
# REQUISIÇÕES À API
# =============================
def balldontlie_get(path: str, params: dict | None = None, timeout: int = REQUEST_TIMEOUT) -> dict | None:
    global LAST_REQUEST_TIME
    
    current_time = time.time()
    time_since_last_request = current_time - LAST_REQUEST_TIME
    if time_since_last_request < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - time_since_last_request)
    
    try:
        url = BALLDONTLIE_BASE.rstrip("/") + "/" + path.lstrip("/")
        resp = requests.get(url, headers=HEADERS_BDL, params=params, timeout=timeout)
        LAST_REQUEST_TIME = time.time()
        
        if resp.status_code == 429:
            st.warning("⏳ Rate limit atingido. Aguardando 45 segundos...")
            time.sleep(45)
            resp = requests.get(url, headers=HEADERS_BDL, params=params, timeout=timeout)
            LAST_REQUEST_TIME = time.time()
        
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Erro na API: {e}")
        return None

# =============================
# DADOS DOS TIMES
# =============================
def obter_times():
    cache = carregar_cache_teams()
    if "teams" in cache and cache["teams"]:
        return cache["teams"]
    
    st.info("📥 Buscando dados dos times...")
    data = balldontlie_get("teams")
    if not data or "data" not in data:
        return {}
    
    teams = {t["id"]: t for t in data.get("data", [])}
    cache["teams"] = teams
    salvar_cache_teams(cache)
    return teams

# =============================
# BUSCA DE JOGOS - OTIMIZADA
# =============================
def obter_jogos_data(data_str: str) -> list:
    cache = carregar_cache_games()
    key = f"games_{data_str}"
    
    if key in cache and cache[key]:
        return cache[key]

    st.info(f"📥 Buscando jogos para {data_str}...")
    jogos = []
    page = 1
    max_pages = 2
    
    while page <= max_pages:
        params = {
            "dates[]": data_str, 
            "per_page": 50,
            "page": page
        }
        
        resp = balldontlie_get("games", params=params)
        if not resp or "data" not in resp:
            break
            
        data_chunk = resp["data"]
        if not data_chunk:
            break
            
        jogos.extend(data_chunk)
        
        meta = resp.get("meta", {})
        total_pages = meta.get("total_pages", 1)
        if page >= total_pages:
            break
            
        page += 1

    cache[key] = jogos
    salvar_cache_games(cache)
    return jogos

# =============================
# ESTATÍSTICAS REAIS - CORRIGIDA
# =============================
def obter_estatisticas_recentes_time(team_id: int, window_games: int = 8) -> dict:
    """Busca estatísticas reais com cache inteligente - VERSÃO CORRIGIDA"""
    cache = carregar_cache_stats()
    key = f"team_{team_id}_{window_games}"
    
    if key in cache:
        cached_data = cache[key]
        if cached_data.get("games", 0) > 0:
            return cached_data

    # Busca dados reais da API
    end_date = date.today()
    start_date = end_date - timedelta(days=60)
    
    games = []
    page = 1
    max_pages = 2
    
    while len(games) < window_games * 2 and page <= max_pages:
        params = {
            "team_ids[]": team_id,
            "per_page": 25,
            "page": page,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        
        resp = balldontlie_get("games", params=params)
        if not resp or "data" not in resp:
            break
            
        games.extend(resp["data"])
        page += 1

    # CORREÇÃO: Processamento mais robusto dos jogos
    games_finalizados = []
    for game in games:
        try:
            status = game.get("status", "").upper()
            home_score = game.get("home_team_score")
            visitor_score = game.get("visitor_team_score")
            game_date = game.get("date")
            
            # Verifica se é um jogo finalizado com placar válido
            if (status in ("FINAL", "FINAL/OT") and 
                home_score is not None and 
                visitor_score is not None and
                game_date is not None):
                games_finalizados.append(game)
        except Exception:
            continue

    # CORREÇÃO: Ordenação mais segura
    try:
        games_finalizados.sort(key=lambda x: x.get("date", ""), reverse=True)
        games_finalizados = games_finalizados[:window_games]
    except Exception as e:
        st.warning(f"⚠️ Erro ao ordenar jogos: {e}")
        games_finalizados = games_finalizados[:window_games]

    if not games_finalizados:
        stats = {
            "pts_for_avg": 0.0, 
            "pts_against_avg": 0.0, 
            "games": 0, 
            "pts_diff_avg": 0.0, 
            "first_half_avg": 0.0
        }
    else:
        pts_for = 0
        pts_against = 0
        first_half_total = 0
        count = 0

        for game in games_finalizados:
            try:
                home_id = game.get("home_team", {}).get("id")
                home_score = game.get("home_team_score", 0)
                visitor_score = game.get("visitor_team_score", 0)
                
                if home_id == team_id:
                    pts_for += home_score
                    pts_against += visitor_score
                    # Primeiro tempo - time da casa
                    period_scores = game.get("home_periods")
                    if period_scores and len(period_scores) >= 2:
                        first_half_total += sum(period_scores[:2])
                    else:
                        q1 = game.get("home_q1", 0) or 0
                        q2 = game.get("home_q2", 0) or 0
                        first_half_total += q1 + q2
                else:
                    pts_for += visitor_score
                    pts_against += home_score
                    # Primeiro tempo - time visitante
                    period_scores = game.get("visitor_periods")
                    if period_scores and len(period_scores) >= 2:
                        first_half_total += sum(period_scores[:2])
                    else:
                        q1 = game.get("visitor_q1", 0) or 0
                        q2 = game.get("visitor_q2", 0) or 0
                        first_half_total += q1 + q2
                
                count += 1
            except Exception as e:
                st.warning(f"⚠️ Erro ao processar jogo: {e}")
                continue

        if count > 0:
            stats = {
                "pts_for_avg": pts_for / count,
                "pts_against_avg": pts_against / count,
                "games": count,
                "pts_diff_avg": (pts_for - pts_against) / count,
                "first_half_avg": first_half_total / count
            }
        else:
            stats = {
                "pts_for_avg": 0.0, 
                "pts_against_avg": 0.0, 
                "games": 0, 
                "pts_diff_avg": 0.0, 
                "first_half_avg": 0.0
            }

    cache[key] = stats
    salvar_cache_stats(cache)
    return stats

# =============================
# LÓGICA DE PREVISÃO COM DADOS REAIS
# =============================
def prever_total_points(home_id: int, away_id: int, window_games: int = 8) -> tuple[float, float, str]:
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)
    
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return 215.0, 50.0, "Dados Insuficientes"
    
    estimativa = (home_stats["pts_for_avg"] + away_stats["pts_for_avg"])
    
    jogos_minimos = min(home_stats["games"], away_stats["games"])
    conf_base = 40 + min(30, jogos_minimos * 3)
    
    diff_qualidade = abs(home_stats["pts_diff_avg"] - away_stats["pts_diff_avg"])
    conf_ajustada = conf_base - min(20, diff_qualidade * 2)
    
    confianca = max(35.0, min(90.0, conf_ajustada))
    
    if estimativa >= 235:
        tendencia = "Mais 235.5"
    elif estimativa >= 230:
        tendencia = "Mais 230.5"
    elif estimativa >= 225:
        tendencia = "Mais 225.5"
    elif estimativa >= 220:
        tendencia = "Mais 220.5"
    elif estimativa >= 215:
        tendencia = "Mais 215.5"
    elif estimativa >= 210:
        tendencia = "Mais 210.5"
    else:
        tendencia = "Menos 210.5"
        
    return round(estimativa, 1), round(confianca, 1), tendencia

def prever_moneyline(home_id: int, away_id: int, window_games: int = 8) -> tuple[str, float]:
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)
    
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return "Dados Insuficientes", 50.0
    
    diff = home_stats["pts_diff_avg"] - away_stats["pts_diff_avg"]
    home_bonus = 3.0
    diff += home_bonus
    
    if abs(diff) < 2.0:
        return "Empate", 50.0
    elif diff > 0:
        conf = min(85.0, 55 + diff * 2.5)
        return "Casa vencer", round(max(50.0, conf), 1)
    else:
        conf = min(85.0, 55 + abs(diff) * 2.5)
        return "Fora vencer", round(max(50.0, conf), 1)

def prever_handicap(home_id: int, away_id: int, window_games: int = 8) -> dict:
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)
    
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return {"margem": 0.0, "spread": "0.5", "prob_cover_home": 50.0}
    
    margem = (home_stats["pts_for_avg"] - away_stats["pts_for_avg"]) + 2.5
    spread = round(margem * 2) / 2
    
    if spread >= 0:
        spread_str = f"-{abs(spread)}"
    else:
        spread_str = f"+{abs(spread)}"
    
    prob = 50 + (margem * 2)
    prob = max(25.0, min(85.0, prob))
    
    return {
        "margem": round(margem, 1), 
        "spread": spread_str, 
        "prob_cover_home": round(prob, 1)
    }

def prever_first_half(home_id: int, away_id: int, window_games: int = 8) -> tuple[float, float, str]:
    home_stats = obter_estatisticas_recentes_time(home_id, window_games)
    away_stats = obter_estatisticas_recentes_time(away_id, window_games)
    
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return 105.0, 50.0, "Mais 105.5 (1H)"
    
    estimativa = home_stats["first_half_avg"] + away_stats["first_half_avg"]
    
    jogos_minimos = min(home_stats["games"], away_stats["games"])
    confianca = max(40.0, min(85.0, 50 + jogos_minimos * 4))
    
    if estimativa >= 115:
        tendencia = "Mais 115.5 (1H)"
    elif estimativa >= 112:
        tendencia = "Mais 112.5 (1H)"
    elif estimativa >= 110:
        tendencia = "Mais 110.5 (1H)"
    elif estimativa >= 108:
        tendencia = "Mais 108.5 (1H)"
    else:
        tendencia = "Menos 108.5 (1H)"
        
    return round(estimativa, 1), round(confianca, 1), tendencia

# =============================
# ALERTAS E TELEGRAM
# =============================
def enviar_telegram(msg: str, chat_id: str = TELEGRAM_CHAT_ID) -> bool:
    try:
        resp = requests.get(BASE_URL_TG, params={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False

def formatar_msg_alerta(game: dict, predictions: dict) -> str:
    try:
        home = game.get("home_team", {}).get("full_name", "Casa")
        away = game.get("visitor_team", {}).get("full_name", "Visitante")
        
        data_hora = game.get("datetime") or game.get("date") or ""
        if data_hora:
            try:
                dt = datetime.fromisoformat(data_hora.replace("Z", "+00:00")) - timedelta(hours=3)
                data_str = dt.strftime("%d/%m/%Y")
                hora_str = dt.strftime("%H:%M")
            except:
                data_str, hora_str = "-", "-"
        else:
            data_str, hora_str = "-", "-"

        msg = f"🏀 <b>Alerta NBA - {data_str} {hora_str} (BRT)</b>\n"
        msg += f"🏟️ {home} vs {away}\n"
        msg += f"📌 Status: {game.get('status', 'SCHEDULED')}\n\n"

        t = predictions.get("total", {})
        if t:
            msg += f"📈 <b>Total</b>: {t.get('tendencia', 'N/A')} | Estimativa: <b>{t.get('estimativa', 0):.1f}</b> | Conf: {t.get('confianca', 0):.0f}%\n"
        
        ml = predictions.get("moneyline", ())
        if ml and len(ml) == 2:
            msg += f"🎯 <b>Moneyline</b>: {ml[0]} ({ml[1]:.0f}%)\n"
        
        h = predictions.get("handicap", {})
        if h:
            msg += f"📐 <b>Handicap</b>: {h.get('spread', '-')} | Prob: {h.get('prob_cover_home', 0):.0f}%\n"
        
        fh = predictions.get("first_half", {})
        if fh and len(fh) == 3:
            msg += f"⏱️ <b>1º Tempo</b>: {fh[2]} | Estimativa: <b>{fh[0]:.1f}</b> | Conf: {fh[1]:.0f}%\n"

        msg += "\n🏆 <b>Elite Master</b> - Análise com Dados Reais"
        return msg
    except Exception as e:
        return f"⚠️ Erro ao formatar: {e}"

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
        
        with st.expander(f"🎯 {game['home_team']['full_name']} vs {game['visitor_team']['full_name']}", expanded=True):
            st.write(f"**Total:** {predictions['total']['tendencia']} (Estimativa: {predictions['total']['estimativa']:.1f}, Conf: {predictions['total']['confianca']}%)")
            if predictions.get('moneyline'):
                st.write(f"**Moneyline:** {predictions['moneyline'][0]} ({predictions['moneyline'][1]}%)")
            if predictions.get('handicap'):
                st.write(f"**Handicap:** {predictions['handicap']['spread']} (Prob: {predictions['handicap']['prob_cover_home']}%)")
            if predictions.get('first_half'):
                st.write(f"**1º Tempo:** {predictions['first_half'][2]} (Estimativa: {predictions['first_half'][0]:.1f}, Conf: {predictions['first_half'][1]}%)")
        
        if send_to_telegram:
            if enviar_telegram(msg):
                st.success("📤 Enviado para Telegram")
            else:
                st.error("❌ Erro ao enviar para Telegram")

# =============================
# INTERFACE STREAMLIT
# =============================
def main():
    st.set_page_config(page_title="🏀 Elite Master - NBA Alerts", layout="wide")
    st.title("🏀 Elite Master — Análise com Dados Reais")
    
    st.sidebar.header("⚙️ Configurações")
    st.sidebar.info("📊 **Fonte:** API BallDontLie (Dados Reais)")
    st.sidebar.warning("⏱️ **Rate Limit:** 0.8s entre requisições")
    
    with st.sidebar:
        top_n = st.slider("🎯 Número de jogos para analisar", 1, 15, 8)
        janela = st.slider("📈 Jogos recentes para análise", 5, 15, 8)
        enviar_auto = st.checkbox("📤 Enviar alertas para Telegram", value=False)
        
        st.markdown("---")
        if st.button("🔄 Limpar Cache", type="secondary"):
            for f in [CACHE_GAMES, CACHE_STATS]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                        st.success(f"🗑️ {f} removido")
                except:
                    pass
            st.rerun()

    col1, col2 = st.columns([2, 1])
    with col1:
        data_sel = st.date_input("📅 Data para análise:", value=date.today())
    with col2:
        st.write("")
        if st.button("🚀 ANALISAR JOGOS", type="primary", use_container_width=True):
            analisar_jogos_reais(data_sel, top_n, janela, enviar_auto)

def analisar_jogos_reais(data_sel: date, top_n: int, janela: int, enviar_auto: bool):
    data_str = data_sel.strftime("%Y-%m-%d")
    
    progress_placeholder = st.empty()
    results_placeholder = st.empty()
    
    with progress_placeholder:
        st.info(f"🔍 Buscando jogos reais para {data_sel.strftime('%d/%m/%Y')}...")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    jogos = obter_jogos_data(data_str)
    
    if not jogos:
        st.error("❌ Nenhum jogo encontrado para esta data")
        return
    
    jogos = jogos[:top_n]
    
    status_text.text(f"📊 Analisando {len(jogos)} jogos...")
    
    results = []
    with results_placeholder:
        st.subheader(f"🎯 Análise em Tempo Real ({len(jogos)} jogos)")
        
        for i, jogo in enumerate(jogos):
            progress = (i + 1) / len(jogos)
            progress_bar.progress(progress)
            
            status_text.text(f"🔍 Analisando: {jogo['home_team']['full_name']} vs {jogo['visitor_team']['full_name']} ({i+1}/{len(jogos)})")
            
            home_id = jogo["home_team"]["id"]
            away_id = jogo["visitor_team"]["id"]
            
            try:
                total_estim, total_conf, total_tend = prever_total_points(home_id, away_id, janela)
                ml_pred = prever_moneyline(home_id, away_id, janela)
                handicap_pred = prever_handicap(home_id, away_id, janela)
                fh_pred = prever_first_half(home_id, away_id, janela)
                
                predictions = {
                    "total": {"estimativa": total_estim, "confianca": total_conf, "tendencia": total_tend},
                    "moneyline": ml_pred,
                    "handicap": handicap_pred,
                    "first_half": fh_pred
                }
                
                results.append({
                    "jogo": jogo,
                    "predictions": predictions
                })
                
                verificar_e_enviar_alerta(jogo, predictions, enviar_auto)
                
            except Exception as e:
                st.error(f"❌ Erro ao analisar jogo {jogo['home_team']['full_name']} vs {jogo['visitor_team']['full_name']}: {e}")
                continue
    
    progress_placeholder.empty()
    status_text.empty()
    
    st.success(f"✅ Análise concluída! {len(results)} jogos processados com dados reais.")

if __name__ == "__main__":
    main()
