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

# Rate limiting
REQUEST_TIMEOUT = 10
LAST_REQUEST_TIME = 0
MIN_REQUEST_INTERVAL = 1.2

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
            st.error("🚨 RATE LIMIT ATINGIDO! Aguardando 60 segundos...")
            time.sleep(60)
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
# BUSCA DE JOGOS REAIS
# =============================
def obter_jogos_data(data_str: str) -> list:
    cache = carregar_cache_games()
    key = f"games_{data_str}"
    
    if key in cache and cache[key]:
        return cache[key]

    st.info(f"📥 Buscando jogos para {data_str}...")
    jogos = []
    page = 1
    max_pages = 2  # Aumentado para buscar mais jogos
    
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
# ESTATÍSTICAS REAIS COM DADOS PASSADOS
# =============================
def obter_estatisticas_time_completo(team_id: int, season: int = 2024, window_games: int = 15) -> dict:
    """Busca estatísticas reais de temporadas passadas"""
    cache = carregar_cache_stats()
    key = f"team_{team_id}_season_{season}"
    
    if key in cache:
        cached_data = cache[key]
        if cached_data.get("games", 0) > 0:
            return cached_data

    # Busca jogos da temporada atual e anterior
    start_date = f"{season-1}-10-01"  # Início da temporada anterior (outubro)
    end_date = f"{season}-06-30"      # Fim da temporada atual (junho)
    
    games = []
    page = 1
    max_pages = 3  # Busca mais páginas para ter dados suficientes
    
    st.info(f"📊 Buscando estatísticas do time {team_id}...")
    
    while page <= max_pages:
        params = {
            "team_ids[]": team_id,
            "per_page": 25,
            "page": page,
            "start_date": start_date,
            "end_date": end_date,
            "seasons[]": season
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

    # Filtra apenas jogos finalizados com placar válido
    games_validos = []
    for game in games:
        try:
            status = game.get("status", "").upper()
            home_score = game.get("home_team_score")
            visitor_score = game.get("visitor_team_score")
            
            if (status in ("FINAL", "FINAL/OT") and 
                home_score is not None and 
                visitor_score is not None and
                home_score > 0 and visitor_score > 0):  # Garante placares válidos
                games_validos.append(game)
        except Exception:
            continue

    # Ordena por data (mais recentes primeiro) e limita pela janela
    try:
        games_validos.sort(key=lambda x: x.get("date", ""), reverse=True)
        games_validos = games_validos[:window_games]
    except Exception:
        games_validos = games_validos[:window_games]

    # Se não encontrou jogos válidos, tenta buscar da temporada anterior
    if not games_validos and season > 2020:
        return obter_estatisticas_time_completo(team_id, season-1, window_games)

    # Calcula estatísticas
    if not games_validos:
        stats = {
            "pts_for_avg": 110.0,  # Fallback para média da NBA
            "pts_against_avg": 110.0,
            "games": 0,
            "pts_diff_avg": 0.0,
            "win_rate": 0.5
        }
    else:
        pts_for = 0
        pts_against = 0
        wins = 0
        count = len(games_validos)

        for game in games_validos:
            try:
                home_id = game.get("home_team", {}).get("id")
                home_score = game.get("home_team_score", 0)
                visitor_score = game.get("visitor_team_score", 0)
                
                if home_id == team_id:
                    pts_for += home_score
                    pts_against += visitor_score
                    if home_score > visitor_score:
                        wins += 1
                else:
                    pts_for += visitor_score
                    pts_against += home_score
                    if visitor_score > home_score:
                        wins += 1
                        
            except Exception:
                continue

        if count > 0:
            stats = {
                "pts_for_avg": pts_for / count,
                "pts_against_avg": pts_against / count,
                "games": count,
                "pts_diff_avg": (pts_for - pts_against) / count,
                "win_rate": wins / count
            }
        else:
            stats = {
                "pts_for_avg": 110.0,
                "pts_against_avg": 110.0,
                "games": 0,
                "pts_diff_avg": 0.0,
                "win_rate": 0.5
            }

    cache[key] = stats
    salvar_cache_stats(cache)
    return stats

# =============================
# PREVISÕES COM DADOS REAIS
# =============================
def prever_total_points(home_id: int, away_id: int, window_games: int = 15) -> tuple[float, float, str]:
    """Previsão baseada em dados reais de temporadas passadas"""
    home_stats = obter_estatisticas_time_completo(home_id, 2024, window_games)
    away_stats = obter_estatisticas_time_completo(away_id, 2024, window_games)
    
    # Usa dados reais ou fallback se não houver dados suficientes
    home_avg = home_stats["pts_for_avg"]
    away_avg = away_stats["pts_for_avg"]
    
    # Ajuste para vantagem de casa
    home_advantage = 2.5
    estimativa = home_avg + away_avg + home_advantage
    
    # Calcula confiança baseada na quantidade de dados
    home_games = home_stats["games"]
    away_games = away_stats["games"]
    min_games = min(home_games, away_games)
    
    if min_games >= 10:
        confianca = 75.0
    elif min_games >= 5:
        confianca = 65.0
    elif min_games > 0:
        confianca = 55.0
    else:
        confianca = 45.0  # Dados insuficientes
    
    # Ajusta confiança baseado na consistência dos times
    home_consistency = min(10, home_stats.get("pts_diff_avg", 0) * 0.5)
    away_consistency = min(10, away_stats.get("pts_diff_avg", 0) * 0.5)
    confianca += (home_consistency + away_consistency)
    confianca = min(85.0, max(40.0, confianca))
    
    # Determina tendência baseada em dados reais
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

def prever_vencedor(home_id: int, away_id: int, window_games: int = 15) -> tuple[str, float, str]:
    """Previsão de vencedor baseada em dados reais"""
    home_stats = obter_estatisticas_time_completo(home_id, 2024, window_games)
    away_stats = obter_estatisticas_time_completo(away_id, 2024, window_games)
    
    # Calcula vantagem baseada em performance histórica
    home_win_rate = home_stats["win_rate"]
    away_win_rate = away_stats["win_rate"]
    home_pts_diff = home_stats["pts_diff_avg"]
    away_pts_diff = away_stats["pts_diff_avg"]
    
    # Vantagem de jogar em casa (NBA: ~3-4 pontos)
    home_advantage = 0.1  # ~10% de aumento na win rate
    
    # Calcula probabilidade
    home_strength = home_win_rate + home_pts_diff * 0.01
    away_strength = away_win_rate + away_pts_diff * 0.01
    
    home_prob = home_strength / (home_strength + away_strength) + home_advantage
    away_prob = 1 - home_prob
    
    # Determina vencedor e confiança
    if home_prob > 0.6:
        vencedor = "Casa"
        confianca = min(85.0, home_prob * 100)
        detalhe = f"Forte vantagem da casa ({home_win_rate:.1%} win rate)"
    elif away_prob > 0.6:
        vencedor = "Visitante"
        confianca = min(85.0, away_prob * 100)
        detalhe = f"Visitante favorito ({away_win_rate:.1%} win rate)"
    elif home_prob > away_prob:
        vencedor = "Casa"
        confianca = home_prob * 100
        detalhe = f"Ligeira vantagem da casa"
    elif away_prob > home_prob:
        vencedor = "Visitante"
        confianca = away_prob * 100
        detalhe = f"Ligeira vantagem do visitante"
    else:
        vencedor = "Empate"
        confianca = 50.0
        detalhe = "Jogo muito equilibrado"
    
    # Ajusta confiança baseada na quantidade de dados
    min_games = min(home_stats["games"], away_stats["games"])
    if min_games < 5:
        confianca = max(40.0, confianca * 0.8)
    
    return vencedor, round(confianca, 1), detalhe

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

        total_pred = predictions.get("total", {})
        if total_pred:
            msg += f"📈 <b>Total Pontos</b>: {total_pred.get('tendencia', 'N/A')}\n"
            msg += f"   📊 Estimativa: <b>{total_pred.get('estimativa', 0):.1f}</b> | Confiança: {total_pred.get('confianca', 0):.0f}%\n\n"

        vencedor_pred = predictions.get("vencedor", {})
        if vencedor_pred:
            msg += f"🎯 <b>Vencedor</b>: {vencedor_pred.get('vencedor', 'N/A')}\n"
            msg += f"   💪 Confiança: {vencedor_pred.get('confianca', 0):.0f}% | {vencedor_pred.get('detalhe', '')}\n"

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
            "game_data": game,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "enviado_telegram": send_to_telegram,
            "conferido": False
        }
        salvar_alertas(alertas)
        
        msg = formatar_msg_alerta(game, predictions)
        
        # Se marcado para enviar ao Telegram, envia
        if send_to_telegram:
            if enviar_telegram(msg):
                alertas[fid]["enviado_telegram"] = True
                salvar_alertas(alertas)
                return True
            else:
                return False
        return True
    return False

# =============================
# EXIBIÇÃO DOS JOGOS ANALISADOS
# =============================
def exibir_jogos_analisados():
    st.header("📈 Jogos Analisados")
    
    alertas = carregar_alertas()
    if not alertas:
        st.info("Nenhum jogo analisado ainda.")
        return
    
    alertas_ordenados = sorted(
        alertas.items(), 
        key=lambda x: x[1].get("timestamp", ""), 
        reverse=True
    )
    
    st.subheader(f"🎯 {len(alertas_ordenados)} Jogos Analisados")
    
    for alerta_id, alerta in alertas_ordenados:
        game_data = alerta.get("game_data", {})
        predictions = alerta.get("predictions", {})
        
        home_team = game_data.get("home_team", {}).get("full_name", "Casa")
        away_team = game_data.get("visitor_team", {}).get("full_name", "Visitante")
        status = game_data.get("status", "SCHEDULED")
        
        total_pred = predictions.get("total", {})
        vencedor_pred = predictions.get("vencedor", {})
        
        # Card do jogo
        with st.expander(f"🏀 {home_team} vs {away_team} - {status}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Total de Pontos**")
                st.write(f"Tendência: {total_pred.get('tendencia', 'N/A')}")
                st.write(f"Estimativa: {total_pred.get('estimativa', 0):.1f}")
                st.write(f"Confiança: {total_pred.get('confianca', 0):.0f}%")
            
            with col2:
                st.write("**🎯 Vencedor**")
                st.write(f"Previsão: {vencedor_pred.get('vencedor', 'N/A')}")
                st.write(f"Confiança: {vencedor_pred.get('confianca', 0):.0f}%")
                st.write(f"Detalhe: {vencedor_pred.get('detalhe', '')}")
            
            if alerta.get("enviado_telegram", False):
                st.success("📤 Enviado para Telegram")
            else:
                st.info("📝 Salvo localmente")
            
            timestamp = alerta.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    st.caption(f"Analisado em: {dt.strftime('%d/%m/%Y %H:%M')}")
                except:
                    pass

# =============================
# CONFERÊNCIA DE RESULTADOS
# =============================
def conferir_resultados():
    st.header("📊 Conferência de Resultados")
    
    alertas = carregar_alertas()
    if not alertas:
        st.info("Nenhum alerta salvo para conferência.")
        return
    
    jogos_para_conferir = []
    for alerta_id, alerta in alertas.items():
        game_data = alerta.get("game_data", {})
        status = game_data.get("status", "").upper()
        
        if status in ["FINAL", "FINAL/OT"]:
            jogos_para_conferir.append((alerta_id, alerta))
    
    if not jogos_para_conferir:
        st.info("Nenhum jogo finalizado para conferência.")
        return
    
    st.subheader(f"🎯 {len(jogos_para_conferir)} Jogos Finalizados")
    
    for alerta_id, alerta in jogos_para_conferir:
        game_data = alerta.get("game_data", {})
        predictions = alerta.get("predictions", {})
        
        home_team = game_data.get("home_team", {}).get("full_name", "Casa")
        away_team = game_data.get("visitor_team", {}).get("full_name", "Visitante")
        home_score = game_data.get("home_team_score", 0)
        away_score = game_data.get("visitor_team_score", 0)
        status = game_data.get("status", "")
        
        total_pontos = home_score + away_score
        
        # Determina resultado do Total
        total_pred = predictions.get("total", {})
        tendencia_total = total_pred.get("tendencia", "")
        resultado_total = "⏳ Aguardando"
        
        if "Mais" in tendencia_total:
            try:
                limite = float(tendencia_total.split()[-1])
                resultado_total = "🟢 GREEN" if total_pontos > limite else "🔴 RED"
            except:
                resultado_total = "⚪ INDEFINIDO"
        elif "Menos" in tendencia_total:
            try:
                limite = float(tendencia_total.split()[-1])
                resultado_total = "🟢 GREEN" if total_pontos < limite else "🔴 RED"
            except:
                resultado_total = "⚪ INDEFINIDO"
        
        # Determina resultado do Vencedor
        vencedor_pred = predictions.get("vencedor", {})
        vencedor_previsto = vencedor_pred.get("vencedor", "")
        resultado_vencedor = "⏳ Aguardando"
        
        if vencedor_previsto == "Casa" and home_score > away_score:
            resultado_vencedor = "🟢 GREEN"
        elif vencedor_previsto == "Visitante" and away_score > home_score:
            resultado_vencedor = "🟢 GREEN"
        elif vencedor_previsto == "Empate" and home_score == away_score:
            resultado_vencedor = "🟢 GREEN"
        elif vencedor_previsto in ["Casa", "Visitante", "Empate"]:
            resultado_vencedor = "🔴 RED"
        else:
            resultado_vencedor = "⚪ INDEFINIDO"
        
        # Exibe card do jogo
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.write(f"**{home_team}** vs **{away_team}**")
            st.write(f"📊 **Placar:** {home_score} x {away_score}")
            st.write(f"🏀 **Total:** {total_pontos} pontos")
        
        with col2:
            st.write(f"**Total:** {tendencia_total}")
            st.write(f"**Resultado:** {resultado_total}")
            st.write(f"**Vencedor:** {resultado_vencedor}")
        
        with col3:
            if not alerta.get("conferido", False):
                if st.button("✅ Confirmar", key=f"conf_{alerta_id}"):
                    alertas[alerta_id]["conferido"] = True
                    salvar_alertas(alertas)
                    st.rerun()
            else:
                st.success("✅ Conferido")
        
        st.markdown("---")

# =============================
# INTERFACE STREAMLIT
# =============================
def main():
    st.set_page_config(page_title="🏀 Elite Master - NBA Alerts", layout="wide")
    st.title("🏀 Elite Master — Análise com Dados Reais")
    
    st.sidebar.header("⚙️ Configurações")
    st.sidebar.info("🎯 **Fonte:** Dados Reais da API")
    st.sidebar.warning("📊 **Período:** Temporada 2023-2024")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Análise", "📊 Jogos Analisados", "✅ Conferência"])
    
    with tab1:
        exibir_aba_analise()
    
    with tab2:
        exibir_jogos_analisados()
    
    with tab3:
        conferir_resultados()

def exibir_aba_analise():
    st.header("🎯 Análise com Dados Reais")
    
    with st.sidebar:
        st.subheader("Controles de Análise")
        top_n = st.slider("Número de jogos para analisar", 1, 15, 5)
        janela = st.slider("Jogos recentes para análise", 2, 20, 15)
        enviar_auto = st.checkbox("Enviar alertas automaticamente para Telegram", value=True)
        
        st.markdown("---")
        st.subheader("Gerenciamento")
        if st.button("🧹 Limpar Cache", type="secondary"):
            for f in [CACHE_GAMES, CACHE_STATS, ALERTAS_PATH]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                        st.success(f"🗑️ {f} removido")
                except:
                    pass
            st.rerun()

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        data_sel = st.date_input("Selecione a data:", value=date.today())
    with col2:
        st.write("")
        st.write("")
        if st.button("🚀 ANALISAR COM DADOS REAIS", type="primary", use_container_width=True):
            analisar_jogos_com_dados_reais(data_sel, top_n, janela, enviar_auto)
    with col3:
        st.write("")
        st.write("")
        if st.button("🔄 Atualizar Dados", type="secondary"):
            st.rerun()

def analisar_jogos_com_dados_reais(data_sel: date, top_n: int, janela: int, enviar_auto: bool):
    data_str = data_sel.strftime("%Y-%m-%d")
    
    progress_placeholder = st.empty()
    results_placeholder = st.empty()
    
    with progress_placeholder:
        st.info(f"🔍 Buscando dados reais para {data_sel.strftime('%d/%m/%Y')}...")
        st.success("📊 Analisando com dados das temporadas 2023-2024")
        if enviar_auto:
            st.warning("📤 Alertas serão enviados para Telegram")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Busca jogos
    jogos = obter_jogos_data(data_str)
    
    if not jogos:
        st.error("❌ Nenhum jogo encontrado para esta data")
        return
    
    jogos = jogos[:top_n]
    
    status_text.text(f"📊 Analisando {len(jogos)} jogos com dados históricos...")
    
    resultados = []
    alertas_enviados = 0
    
    with results_placeholder:
        st.subheader(f"🎯 Análise com Dados Reais")
        
        for i, jogo in enumerate(jogos):
            progress = (i + 1) / len(jogos)
            progress_bar.progress(progress)
            
            home_team = jogo['home_team']['full_name']
            away_team = jogo['visitor_team']['full_name']
            status_text.text(f"🔍 Analisando: {home_team} vs {away_team} ({i+1}/{len(jogos)})")
            
            home_id = jogo["home_team"]["id"]
            away_id = jogo["visitor_team"]["id"]
            
            try:
                # Previsões com dados reais
                total_estim, total_conf, total_tend = prever_total_points(home_id, away_id, janela)
                vencedor, vencedor_conf, vencedor_detalhe = prever_vencedor(home_id, away_id, janela)
                
                predictions = {
                    "total": {
                        "estimativa": total_estim, 
                        "confianca": total_conf, 
                        "tendencia": total_tend
                    },
                    "vencedor": {
                        "vencedor": vencedor,
                        "confianca": vencedor_conf,
                        "detalhe": vencedor_detalhe
                    }
                }
                
                # Envia alerta
                enviado = verificar_e_enviar_alerta(jogo, predictions, enviar_auto)
                if enviado and enviar_auto:
                    alertas_enviados += 1
                
                # Exibe resultado
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{home_team}** vs **{away_team}**")
                    st.write(f"📍 **Status:** {jogo.get('status', 'SCHEDULED')}")
                
                with col2:
                    st.write(f"📊 **Total:** {total_tend}")
                    st.write(f"🎯 **Vencedor:** {vencedor}")
                    st.write(f"💪 **Confiança:** {vencedor_conf}%")
                
                with col3:
                    st.write(f"📈 **Estimativa:** {total_estim:.1f}")
                    st.write(f"🔒 **Confiança:** {total_conf}%")
                    if enviado and enviar_auto:
                        st.success("✅ Telegram")
                    else:
                        st.info("💾 Salvo")
                
                st.markdown("---")
                
                resultados.append({
                    "jogo": jogo,
                    "predictions": predictions
                })
                
            except Exception as e:
                st.error(f"❌ Erro ao analisar {home_team} vs {away_team}: {e}")
                continue
    
    progress_placeholder.empty()
    
    # Resumo final
    st.success(f"✅ Análise com dados reais concluída!")
    st.info(f"""
    **📊 Resumo da Análise:**
    - 🏀 {len(resultados)} jogos analisados com dados históricos
    - 📤 {alertas_enviados} alertas enviados para Telegram
    - 📈 Estatísticas baseadas na temporada 2023-2024
    - 💾 Dados salvos para conferência futura
    """)

# =============================
# EXECUÇÃO PRINCIPAL
# =============================
if __name__ == "__main__":
    main()
