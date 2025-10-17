# Futebol_Live_AllSports.py
import streamlit as st
import pandas as pd
import asyncio
import websockets
import json
from datetime import datetime
import requests
from collections import defaultdict

# =============================
# CONFIGURAÇÕES
# =============================
API_KEY_AS = "SUA_API_KEY_AQUI"  # AllSportsAPI
TIMEZONE = "America/Sao_Paulo"

TELEGRAM_TOKEN = "SEU_TELEGRAM_TOKEN"
TELEGRAM_CHAT_ID = "SEU_CHAT_ID"

# Cache local (para histórico e resultados)
CACHE_RESULTADOS = "cache_resultados.json"

# =============================
# Funções auxiliares
# =============================
def carregar_cache():
    try:
        with open(CACHE_RESULTADOS, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def salvar_cache(dados):
    with open(CACHE_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

def enviar_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.get(url, params={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode":"Markdown"})
        return True
    except:
        return False

# =============================
# Tendência com base no histórico
# =============================
def calcular_tendencia(event):
    # Pegamos gols do histórico se existir
    home = event["event_home_team"]
    away = event["event_away_team"]
    
    # Média de gols simples baseada nos últimos resultados (ou estimativa 1.8)
    cache = carregar_cache()
    jogos_home = cache.get(home, [])
    jogos_away = cache.get(away, [])
    
    media_home = sum([int(r.split("-")[0]) for r in jogos_home[-5:]] or [1.8])/max(1,len(jogos_home[-5:]) or 1)
    media_away = sum([int(r.split("-")[1]) for r in jogos_away[-5:]] or [1.8])/max(1,len(jogos_away[-5:]) or 1)
    
    estimativa = (media_home + media_away)
    
    if estimativa >= 3.0:
        tendencia = "Mais 2.5"
        confianca = min(95, 70 + (estimativa - 3.0)*10)
    elif estimativa >= 2.0:
        tendencia = "Mais 1.5"
        confianca = min(90, 60 + (estimativa - 2.0)*10)
    else:
        tendencia = "Menos 2.5"
        confianca = min(85, 55 + (2.0 - estimativa)*10)
        
    return round(estimativa,2), round(confianca,0), tendencia

# =============================
# Processar evento ao vivo e enviar alerta
# =============================
def processar_evento(event):
    estimativa, confianca, tendencia = calcular_tendencia(event)
    
    data_event = event["event_date"]
    hora_event = event["event_time"]
    liga = event["league_name"]
    home = event["event_home_team"]
    away = event["event_away_team"]
    
    msg = (
        f"⚽ *Alerta de Gol / Evento AO VIVO!*\n"
        f"🏆 {liga}\n"
        f"🏟️ {home} x {away}\n"
        f"⏰ {hora_event} {data_event}\n"
        f"🎯 Tendência: {tendencia}\n"
        f"📊 Estimativa: {estimativa:.2f} gols | ✅ Confiança: {confianca}%"
    )
    
    enviar_telegram(msg)

# =============================
# WebSocket cliente
# =============================
async def websocket_client():
    url = f"wss://wss.allsportsapi.com/live_events?APIkey={API_KEY_AS}&timezone={TIMEZONE}"
    async with websockets.connect(url) as ws:
        st.info("Conectado à AllSportsAPI AO VIVO...")
        async for message in ws:
            try:
                events = json.loads(message)
                for e in events:
                    processar_evento(e)
                    # Atualiza cache histórico
                    cache = carregar_cache()
                    home = e["event_home_team"]
                    away = e["event_away_team"]
                    score = e.get("event_final_result","0 - 0")
                    cache.setdefault(home, []).append(score)
                    cache.setdefault(away, []).append(score)
                    salvar_cache(cache)
                    # Atualiza Streamlit
                    st.session_state.jogos_encontrados.append({
                        "home": home, "away": away,
                        "liga": e["league_name"],
                        "hora": e["event_time"],
                        "data": e["event_date"],
                        "status": e["event_status"]
                    })
            except Exception as ex:
                st.error(f"Erro ao processar evento: {ex}")

# =============================
# Streamlit UI
# =============================
def main():
    st.set_page_config(page_title="⚽ Futebol AO VIVO", layout="wide")
    if "jogos_encontrados" not in st.session_state:
        st.session_state.jogos_encontrados = []

    st.title("⚽ Sistema de Futebol AO VIVO (AllSportsAPI)")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Iniciar AO VIVO WebSocket"):
            st.session_state.jogos_encontrados = []
            asyncio.run(websocket_client())
            
    with col2:
        if st.button("📤 Enviar TOP 5 Alertas Manual"):
            top5 = sorted(st.session_state.jogos_encontrados, key=lambda x: x["hora"], reverse=True)[:5]
            for t in top5:
                enviar_telegram(f"🏆 {t['home']} x {t['away']} ({t['liga']}) - {t['hora']} {t['data']}")
            st.success("Top 5 alertas enviados!")

    st.subheader("📋 Jogos AO VIVO")
    if st.session_state.jogos_encontrados:
        df = pd.DataFrame(st.session_state.jogos_encontrados)
        st.dataframe(df)

if __name__ == "__main__":
    main()
