import streamlit as st
import json
import os
import requests
import logging
import numpy as np
from collections import Counter, deque
from alertas import enviar_previsao, enviar_resultado
from streamlit_autorefresh import st_autorefresh
import base64

# =============================
# Configurações
# =============================
HISTORICO_PATH = "historico_dominantes.json"
API_URL = "https://api.casinodata.com/roulette"  # coloque a URL correta da API

# =============================
# Funções auxiliares
# =============================
def carregar_historico():
    if os.path.exists(HISTORICO_PATH):
        with open(HISTORICO_PATH, "r") as f:
            return deque(json.load(f), maxlen=500)
    return deque(maxlen=500)

def salvar_historico(historico):
    with open(HISTORICO_PATH, "w") as f:
        json.dump(list(historico), f)

def get_terminal(numero):
    return numero % 10

# =============================
# Critério A: Mais repetidos nos últimos 12
# =============================
def criterio_a(ultimos):
    terminais = [get_terminal(n) for n in ultimos]
    contagem = Counter(terminais)
    dominantes = [t for t, _ in contagem.most_common(2)]
    return dominantes

# =============================
# Critério B: O 13º está nos últimos 12
# =============================
def criterio_b(ultimos):
    if len(ultimos) < 13:
        return None
    ult_12 = ultimos[-13:-1]  # pega 12 anteriores
    alvo = ultimos[-13]       # 13º número
    if alvo in ult_12:
        terminais = [get_terminal(n) for n in ult_12]
        contagem = Counter(terminais)
        dominantes = [t for t, _ in contagem.most_common(2)]
        return dominantes
    return None

# =============================
# Avaliação do resultado
# =============================
def avaliar_resultado(ultimo_numero, terminais_previstos):
    if get_terminal(ultimo_numero) in terminais_previstos:
        return "🟢 GREEN"
    else:
        return "🔴 RED"

# =============================
# Streamlit App
# =============================
st.set_page_config("IA Dominantes", page_icon="🎰", layout="centered")
st.title("🎰 IA de Terminais Dominantes")

# Autorefresh a cada 10s
st_autorefresh(interval=10000, key="refresh")

# Histórico
historico = carregar_historico()

# Busca novo número da API
try:
    response = requests.get(API_URL, timeout=10)
    response.raise_for_status()
    data = response.json()
    ultimo_numero = data.get("ultimo_numero")
except Exception as e:
    st.error(f"Erro API: {e}")
    ultimo_numero = None

if ultimo_numero is not None:
    if not historico or historico[-1] != ultimo_numero:
        historico.append(ultimo_numero)
        salvar_historico(historico)

        st.write(f"🎲 Último número: **{ultimo_numero}**")

        if len(historico) >= 13:
            ultimos = list(historico)[-13:]

            # Testa Critério A
            dominantes_a = criterio_a(ultimos[-12:])
            # Testa Critério B
            dominantes_b = criterio_b(ultimos)

            if dominantes_b:
                terminais_previstos = dominantes_b
                criterio = "Critério B"
            else:
                terminais_previstos = dominantes_a
                criterio = "Critério A"

            # Envia previsão
            msg_prev = f"🎯 Previsão: terminais {terminais_previstos} ({criterio})"
            enviar_previsao(msg_prev)
            st.success(msg_prev)

            # Avalia resultado
            resultado = avaliar_resultado(ultimo_numero, terminais_previstos)
            msg_res = f"Resultado: {ultimo_numero} | Terminais: {terminais_previstos} | {resultado}"
            enviar_resultado(msg_res)
            st.info(msg_res)
else:
    st.warning("⏳ Aguardando próximo número para calcular dominantes.")
