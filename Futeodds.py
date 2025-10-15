# Futebol_Alertas_Oddstop.py - Versão otimizada
import streamlit as st
from datetime import datetime, timedelta, date
import requests
import os
import json
import logging
from functools import lru_cache
from dotenv import load_dotenv

# =============================
# Configuração Inicial
# =============================
load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HISTORICO_FILE = "historico_alertas.json"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================
# Funções Auxiliares
# =============================

@st.cache_data(ttl=60)  # Cache por 60s
def puxar_eventos_odds():
    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={API_KEY}&regions=eu&markets=h2h,totals"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Erro ao puxar Odds API: {e}")
        return []

def calcular_probabilidade(odds_total):
    # Converte odds em probabilidade % simples
    if odds_total <= 1:
        return 0
    return round(100 / odds_total, 2)

def enviar_telegram(mensagem):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": mensagem, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Erro ao enviar mensagem Telegram: {e}")

def carregar_historico():
    if os.path.exists(HISTORICO_FILE):
        with open(HISTORICO_FILE, "r") as f:
            return json.load(f)
    return {}

def salvar_historico(historico):
    with open(HISTORICO_FILE, "w") as f:
        json.dump(historico, f, indent=2)

def limpar_historico():
    if os.path.exists(HISTORICO_FILE):
        os.remove(HISTORICO_FILE)
        st.success("Histórico apagado!")

# =============================
# Função de Processamento
# =============================
def processar_eventos(eventos, historico):
    alertas_enviados = historico.get("alertas", {})
    novas_alertas = []

    for ev in eventos:
        try:
            match_id = ev.get("id")
            home = ev.get("home_team")
            away = ev.get("away_team")
            start_time = ev.get("commence_time")

            # Odds Totais
            totals = ev.get("bookmakers", [])
            over1_5 = []
            over2_5 = []
            over3_5 = []

            for b in totals:
                for m in b.get("markets", []):
                    if m["key"] == "totals":
                        for o in m["outcomes"]:
                            if o["name"] == "Over 1.5 Goals":
                                over1_5.append(o["price"])
                            elif o["name"] == "Over 2.5 Goals":
                                over2_5.append(o["price"])
                            elif o["name"] == "Over 3.5 Goals":
                                over3_5.append(o["price"])

            # Média das odds
            prob1_5 = calcular_probabilidade(sum(over1_5)/len(over1_5)) if over1_5 else 0
            prob2_5 = calcular_probabilidade(sum(over2_5)/len(over2_5)) if over2_5 else 0
            prob3_5 = calcular_probabilidade(sum(over3_5)/len(over3_5)) if over3_5 else 0

            # Seleciona a tendência
            tendencia = None
            if prob1_5 > 65:
                tendencia = "+1.5"
            if prob2_5 > 60:
                tendencia = "+2.5"
            if prob3_5 > 55:
                tendencia = "+3.5"

            if not tendencia:
                continue

            # Evita alerta duplicado
            if str(match_id) in alertas_enviados and alertas_enviados[str(match_id)] == tendencia:
                continue

            # Mensagem
            mensagem = (
                f"🏟️ *{home} vs {away}*\n"
                f"🕒 {datetime.fromisoformat(start_time).strftime('%d/%m %H:%M')}\n"
                f"⚽ Tendência: {tendencia}\n"
                f"📊 Probabilidades: 1.5={prob1_5}%, 2.5={prob2_5}%, 3.5={prob3_5}%"
            )
            enviar_telegram(mensagem)
            novas_alertas.append({match_id: tendencia})
            alertas_enviados[str(match_id)] = tendencia

        except Exception as e:
            logging.error(f"Erro ao processar evento {ev.get('id')}: {e}")

    historico["alertas"] = alertas_enviados
    salvar_historico(historico)
    return novas_alertas

# =============================
# Interface Streamlit
# =============================
st.set_page_config(page_title="Futebol Alertas Oddstop", layout="wide")
st.title("📊 Futebol Alertas Oddstop")

# Botão limpar histórico
if st.sidebar.button("🗑️ Apagar Histórico"):
    limpar_historico()

# Data do dia
hoje = date.today()
st.sidebar.date_input("Data", value=hoje)

# Puxar eventos
st.info("🔄 Buscando eventos...")
eventos = puxar_eventos_odds()
st.success(f"✅ {len(eventos)} eventos encontrados.")

# Carregar histórico
historico = carregar_historico()

# Processar alertas
novos_alertas = processar_eventos(eventos, historico)

# Exibir alertas
if novos_alertas:
    st.subheader("📢 Novos Alertas Enviados:")
    for a in novos_alertas:
        for match_id, tend in a.items():
            st.write(f"ID {match_id} -> Tendência: {tend}")
else:
    st.info("Nenhum novo alerta hoje.")

# Lista completa dos eventos
st.subheader("📋 Eventos do Dia")
for ev in eventos:
    home = ev.get("home_team")
    away = ev.get("away_team")
    start_time = datetime.fromisoformat(ev.get("commence_time")).strftime("%d/%m %H:%M")
    st.write(f"{start_time} - {home} vs {away}")
