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

# Rate limiting mais conservador
REQUEST_TIMEOUT = 10
LAST_REQUEST_TIME = 0
MIN_REQUEST_INTERVAL = 1.2  # Aumentado para evitar rate limit

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
    max_pages = 1  # Apenas 1 página para reduzir requisições
    
    while page <= max_pages:
        params = {
            "dates[]": data_str, 
            "per_page": 25,
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
# ESTATÍSTICAS SIMPLIFICADAS - APENAS DADOS ESSENCIAIS
# =============================
def obter_estatisticas_simples_time(team_id: int, window_games: int = 6) -> dict:
    """Versão simplificada que busca apenas dados essenciais"""
    cache = carregar_cache_stats()
    key = f"team_{team_id}_simple"
    
    if key in cache:
        cached_data = cache[key]
        if cached_data.get("games", 0) > 0:
            return cached_data

    # Busca apenas dados básicos - período mais curto
    end_date = date.today()
    start_date = end_date - timedelta(days=30)  # Apenas 30 dias
    
    games = []
    page = 1
    max_pages = 1  # Apenas 1 página
    
    while page <= max_pages:
        params = {
            "team_ids[]": team_id,
            "per_page": 15,
            "page": page,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        
        resp = balldontlie_get("games", params=params)
        if not resp or "data" not in resp:
            break
            
        games.extend(resp["data"])
        page += 1

    # Processa apenas jogos finalizados com placar
    games_validos = []
    for game in games:
        try:
            status = game.get("status", "").upper()
            home_score = game.get("home_team_score")
            visitor_score = game.get("visitor_team_score")
            
            if (status in ("FINAL", "FINAL/OT") and 
                home_score is not None and 
                visitor_score is not None):
                games_validos.append(game)
        except Exception:
            continue

    # Ordena por data e pega os mais recentes
    try:
        games_validos.sort(key=lambda x: x.get("date", ""), reverse=True)
        games_validos = games_validos[:window_games]
    except Exception:
        games_validos = games_validos[:window_games]

    if not games_validos:
        stats = {
            "pts_for_avg": 110.0,  # Valor padrão
            "pts_against_avg": 110.0,
            "games": 0,
            "pts_diff_avg": 0.0
        }
    else:
        pts_for = 0
        pts_against = 0
        count = 0

        for game in games_validos:
            try:
                home_id = game.get("home_team", {}).get("id")
                home_score = game.get("home_team_score", 0)
                visitor_score = game.get("visitor_team_score", 0)
                
                if home_id == team_id:
                    pts_for += home_score
                    pts_against += visitor_score
                else:
                    pts_for += visitor_score
                    pts_against += home_score
                
                count += 1
            except Exception:
                continue

        if count > 0:
            stats = {
                "pts_for_avg": pts_for / count,
                "pts_against_avg": pts_against / count,
                "games": count,
                "pts_diff_avg": (pts_for - pts_against) / count
            }
        else:
            stats = {
                "pts_for_avg": 110.0,
                "pts_against_avg": 110.0,
                "games": 0,
                "pts_diff_avg": 0.0
            }

    cache[key] = stats
    salvar_cache_stats(cache)
    return stats

# =============================
# PREVISÕES SIMPLIFICADAS - APENAS 2 MODALIDADES
# =============================
def prever_total_points(home_id: int, away_id: int, window_games: int = 6) -> tuple[float, float, str]:
    """Previsão simplificada para total de pontos"""
    home_stats = obter_estatisticas_simples_time(home_id, window_games)
    away_stats = obter_estatisticas_simples_time(away_id, window_games)
    
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return 215.0, 50.0, "Dados Insuficientes"
    
    # Cálculo simples da estimativa
    estimativa = (home_stats["pts_for_avg"] + away_stats["pts_for_avg"])
    
    # Confiança baseada na quantidade de dados
    jogos_minimos = min(home_stats["games"], away_stats["games"])
    confianca = max(40.0, min(85.0, 50 + jogos_minimos * 5))
    
    # Tendência simplificada
    if estimativa >= 230:
        tendencia = "Mais 230.5"
    elif estimativa >= 225:
        tendencia = "Mais 225.5"
    elif estimativa >= 220:
        tendencia = "Mais 220.5"
    elif estimativa >= 215:
        tendencia = "Mais 215.5"
    else:
        tendencia = "Menos 215.5"
        
    return round(estimativa, 1), round(confianca, 1), tendencia

def prever_vencedor(home_id: int, away_id: int, window_games: int = 6) -> tuple[str, float, str]:
    """Previsão simplificada para vencedor"""
    home_stats = obter_estatisticas_simples_time(home_id, window_games)
    away_stats = obter_estatisticas_simples_time(away_id, window_games)
    
    if home_stats["games"] == 0 or away_stats["games"] == 0:
        return "Dados Insuficientes", 50.0, "Empate"
    
    # Diferença de performance + vantagem de casa
    diff = home_stats["pts_diff_avg"] - away_stats["pts_diff_avg"] + 3.0
    
    # Determina vencedor e confiança
    if abs(diff) < 2.0:
        vencedor = "Empate"
        confianca = 50.0
        detalhe = "Jogo muito equilibrado"
    elif diff > 0:
        vencedor = "Casa"
        confianca = min(85.0, 55 + diff * 3)
        detalhe = f"Vantagem de {abs(diff):.1f} pontos"
    else:
        vencedor = "Visitante"
        confianca = min(85.0, 55 + abs(diff) * 3)
        detalhe = f"Vantagem de {abs(diff):.1f} pontos"
    
    return vencedor, round(confianca, 1), detalhe

# =============================
# ALERTAS E TELEGRAM SIMPLIFICADOS
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

        # Total de pontos
        total_pred = predictions.get("total", {})
        if total_pred:
            msg += f"📈 <b>Total Pontos</b>: {total_pred.get('tendencia', 'N/A')}\n"
            msg += f"   📊 Estimativa: <b>{total_pred.get('estimativa', 0):.1f}</b> | Confiança: {total_pred.get('confianca', 0):.0f}%\n\n"

        # Vencedor
        vencedor_pred = predictions.get("vencedor", {})
        if vencedor_pred:
            msg += f"🎯 <b>Vencedor</b>: {vencedor_pred.get('vencedor', 'N/A')}\n"
            msg += f"   💪 Confiança: {vencedor_pred.get('confianca', 0):.0f}% | {vencedor_pred.get('detalhe', '')}\n"

        msg += "\n🏆 <b>Elite Master</b> - Análise Simplificada"
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
        
        # Display simplificado no Streamlit
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader(f"🏀 {game['home_team']['full_name']} vs {game['visitor_team']['full_name']}")
        with col2:
            status = game.get('status', 'SCHEDULED')
            st.write(f"**Status:** {status}")
        
        # Total de pontos
        total_pred = predictions.get("total", {})
        if total_pred:
            st.write(f"**📊 Total de Pontos:** {total_pred.get('tendencia', 'N/A')}")
            st.write(f"   Estimativa: {total_pred.get('estimativa', 0):.1f} | Confiança: {total_pred.get('confianca', 0):.0f}%")
        
        # Vencedor
        vencedor_pred = predictions.get("vencedor", {})
        if vencedor_pred:
            st.write(f"**🎯 Vencedor:** {vencedor_pred.get('vencedor', 'N/A')}")
            st.write(f"   Confiança: {vencedor_pred.get('confianca', 0):.0f}% | {vencedor_pred.get('detalhe', '')}")
        
        st.markdown("---")
        
        if send_to_telegram:
            if enviar_telegram(msg):
                st.success("📤 Enviado para Telegram")
            else:
                st.error("❌ Erro ao enviar para Telegram")

# =============================
# INTERFACE STREAMLIT SIMPLIFICADA
# =============================
def main():
    st.set_page_config(page_title="🏀 Elite Master - NBA Alerts", layout="wide")
    st.title("🏀 Elite Master — Análise Simplificada")
    
    st.sidebar.header("⚙️ Configurações")
    st.sidebar.info("🎯 **Modalidades:** Vencedor + Total Pontos")
    st.sidebar.warning("⏱️ **Rate Limit:** 1.2s entre requisições")
    
    with st.sidebar:
        st.subheader("Controles de Análise")
        top_n = st.slider("Número de jogos para analisar", 1, 10, 5)
        janela = st.slider("Jogos recentes para análise", 3, 8, 6)
        enviar_auto = st.checkbox("Enviar alertas para Telegram", value=False)
        
        st.markdown("---")
        st.subheader("Gerenciamento")
        if st.button("🧹 Limpar Cache", type="secondary"):
            for f in [CACHE_GAMES, CACHE_STATS]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                        st.success(f"Cache {f} limpo")
                except:
                    pass
            st.rerun()

    # Interface principal
    st.header("📅 Análise de Jogos")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        data_sel = st.date_input("Selecione a data:", value=date.today())
    with col2:
        st.write("")
        st.write("")
        if st.button("🚀 ANALISAR JOGOS", type="primary", use_container_width=True):
            analisar_jogos_otimizado(data_sel, top_n, janela, enviar_auto)
    with col3:
        st.write("")
        st.write("")
        if st.button("📊 Ver Cache Stats", type="secondary"):
            mostrar_status_cache()

def analisar_jogos_otimizado(data_sel: date, top_n: int, janela: int, enviar_auto: bool):
    """Análise otimizada com apenas 2 modalidades"""
    
    data_str = data_sel.strftime("%Y-%m-%d")
    
    # Container para progresso
    progress_placeholder = st.empty()
    
    with progress_placeholder:
        st.info(f"🔍 Iniciando análise para {data_sel.strftime('%d/%m/%Y')}...")
        st.warning("⚠️ Modo econômico: Apenas 2 modalidades por jogo")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Busca jogos
    jogos = obter_jogos_data(data_str)
    
    if not jogos:
        st.error("❌ Nenhum jogo encontrado para esta data")
        return
    
    # Limita pelo número de jogos
    jogos = jogos[:top_n]
    
    status_text.text(f"📊 Analisando {len(jogos)} jogos...")
    
    results = []
    for i, jogo in enumerate(jogos):
        progress = (i + 1) / len(jogos)
        progress_bar.progress(progress)
        
        status_text.text(f"🔍 Processando: {jogo['home_team']['full_name']} vs {jogo['visitor_team']['full_name']} ({i+1}/{len(jogos)})")
        
        home_id = jogo["home_team"]["id"]
        away_id = jogo["visitor_team"]["id"]
        
        try:
            # Apenas 2 previsões por jogo
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
            
            results.append({
                "jogo": jogo,
                "predictions": predictions
            })
            
            # Gera alerta
            verificar_e_enviar_alerta(jogo, predictions, enviar_auto)
            
        except Exception as e:
            st.error(f"❌ Erro ao analisar jogo: {e}")
            continue
    
    # Limpa placeholders
    progress_placeholder.empty()
    
    st.success(f"✅ Análise concluída! {len(results)} jogos processados.")
    st.info(f"📊 Estatísticas: {len(jogos)} jogos analisados | 2 modalidades por jogo")

def mostrar_status_cache():
    """Mostra status do cache"""
    st.subheader("📊 Status do Cache")
    
    cache_files = {
        "Jogos": CACHE_GAMES,
        "Times": CACHE_TEAMS,
        "Estatísticas": CACHE_STATS,
        "Alertas": ALERTAS_PATH
    }
    
    for nome, arquivo in cache_files.items():
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{nome}:**")
        with col2:
            if os.path.exists(arquivo):
                tamanho = os.path.getsize(arquivo)
                mod_time = datetime.fromtimestamp(os.path.getmtime(arquivo))
                st.write(f"{tamanho} bytes | {mod_time.strftime('%d/%m %H:%M')}")
            else:
                st.write("Não existe")
        with col3:
            if os.path.exists(arquivo):
                if st.button(f"🗑️", key=f"del_{arquivo}"):
                    try:
                        os.remove(arquivo)
                        st.rerun()
                    except:
                        st.error("Erro ao deletar")

# =============================
# EXECUÇÃO PRINCIPAL
# =============================
if __name__ == "__main__":
    main()
