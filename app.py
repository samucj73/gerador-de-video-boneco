import streamlit as st
import json
import os
import requests
import logging
import numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from streamlit_autorefresh import st_autorefresh
import joblib
import matplotlib.pyplot as plt

# --- Configurações ---
HISTORICO_PATH = "historico_coluna_duzia.json"
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {"User-Agent": "Mozilla/5.0"}
MODELO_DIR = "modelos"
os.makedirs(MODELO_DIR, exist_ok=True)

# --- Funções auxiliares ---
def fetch_latest_result():
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        game_data = data.get("data", {})
        result = game_data.get("result", {})
        outcome = result.get("outcome", {})
        number = outcome.get("number")
        timestamp = game_data.get("startedAt")
        return {"number": number, "timestamp": timestamp}
    except Exception as e:
        logging.error(f"Erro ao buscar resultado: {e}")
        return None

def get_coluna(n):
    return (n - 1) % 3 + 1 if n != 0 else 0

def get_duzia(n):
    if n == 0:
        return 0
    elif 1 <= n <= 12:
        return 1
    elif 13 <= n <= 24:
        return 2
    elif 25 <= n <= 36:
        return 3
    return None

def salvar_resultado_em_arquivo(historico):
    with open(HISTORICO_PATH, "w") as f:
        json.dump(historico, f, indent=2)

def grupo_mais_frequente(numeros, func, n=30):
    grupos = [func(x) for x in numeros[-n:] if x >= 0]
    return Counter(grupos).most_common(1)[0][0] if grupos else None

# --- Classe para IA ---
class ModeloIA:
    def __init__(self, tipo, janela=20):
        self.tipo = tipo   # "coluna", "duzia" ou "numero"
        self.janela = janela
        self.modelo = None
        self.encoder = LabelEncoder()
        self.treinado = False
        self.path = os.path.join(MODELO_DIR, f"modelo_{tipo}.joblib")
    
    def construir_features(self, numeros):
        ultimos = numeros[-self.janela:]
        atual = ultimos[-1]
        anteriores = ultimos[:-1]
        
        features = [
            atual % 2,
            int(str(atual)[-1]),
            atual % 3,
            abs(atual - (anteriores[-1] if anteriores else atual)),
            int(anteriores[-1] == atual) if anteriores else 0,
            1 if (anteriores[-1] if anteriores else atual) < atual else -1 if (anteriores[-1] if anteriores else atual) > atual else 0,
            sum(1 for x in anteriores[-3:] if x == atual),
            Counter(numeros[-30:]).get(atual, 0),
            int(atual in [n for n, _ in Counter(numeros[-30:]).most_common(5)]),
            int(np.mean(anteriores) < atual) if anteriores else 0,
            int(atual == 0),
        ]
        if self.tipo == "coluna":
            grupo = get_coluna(atual)
            freq = Counter(get_coluna(n) for n in numeros[-20:])
            features.append(grupo)
            features.append(freq.get(grupo, 0))
        elif self.tipo == "duzia":
            grupo = get_duzia(atual)
            freq = Counter(get_duzia(n) for n in numeros[-20:])
            features.append(grupo)
            features.append(freq.get(grupo, 0))
        # Para "numero" não adicionamos features de agrupamento
        return features

    def treinar(self, historico):
        # Filtra apenas números válidos
        numeros = [h["number"] for h in historico if isinstance(h.get("number"), int) and 0 <= h["number"] <= 36]
        X, y = [], []
        for i in range(self.janela, len(numeros)-1):
            janela = numeros[i - self.janela: i+1]
            target = None
            if self.tipo == "coluna":
                target = get_coluna(numeros[i])
            elif self.tipo == "duzia":
                target = get_duzia(numeros[i])
            elif self.tipo == "numero":
                target = numeros[i]
            if target is not None:
                X.append(self.construir_features(janela))
                y.append(target)
        if X:
            X = np.array(X, dtype=np.float32)
            y = self.encoder.fit_transform(y)
            if self.tipo == "numero":
                self.modelo = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            else:
                self.modelo = HistGradientBoostingClassifier(max_iter=300, max_depth=6, random_state=42)
            self.modelo.fit(X, y)
            self.treinado = True
            joblib.dump((self.modelo, self.encoder), self.path)

    def carregar(self):
        if os.path.exists(self.path):
            self.modelo, self.encoder = joblib.load(self.path)
            self.treinado = True

    def prever(self, historico, top_n=1):
        if not self.treinado:
            return None
        numeros = [h["number"] for h in historico if isinstance(h.get("number"), int) and 0 <= h["number"] <= 36]
        if len(numeros) < self.janela + 1:
            return None
        janela = numeros[-(self.janela+1):]
        entrada = np.array([self.construir_features(janela)], dtype=np.float32)
        proba = self.modelo.predict_proba(entrada)[0]
        indices = np.argsort(proba)[::-1][:top_n]
        return list(self.encoder.inverse_transform(indices))

# --- App Streamlit ---
st.set_page_config(page_title="IA Roleta XXXtreme", layout="centered")
st.title("🎯 IA Roleta — Coluna, Dúzia e Top 3 Números")

# Inicialização do histórico
if "historico" not in st.session_state:
    if os.path.exists(HISTORICO_PATH):
        with open(HISTORICO_PATH, "r") as f:
            st.session_state.historico = json.load(f)
    else:
        st.session_state.historico = []

# Inicialização dos modelos
if "modelo_coluna" not in st.session_state:
    st.session_state.modelo_coluna = ModeloIA("coluna")
    st.session_state.modelo_coluna.carregar()
if "modelo_duzia" not in st.session_state:
    st.session_state.modelo_duzia = ModeloIA("duzia")
    st.session_state.modelo_duzia.carregar()
if "modelo_numero" not in st.session_state:
    st.session_state.modelo_numero = ModeloIA("numero")
    st.session_state.modelo_numero.carregar()

# Inicialização dos acertos
if "acertos" not in st.session_state:
    st.session_state.acertos = {"coluna": 0, "duzia": 0, "numero": 0}
if "historico_taxas" not in st.session_state:
    st.session_state.historico_taxas = []

# Entrada manual
st.subheader("✍️ Inserir Números Manualmente")
entrada = st.text_area("Digite números (0-36) separados por espaço:", key="entrada")
if st.button("Adicionar Números"):
    try:
        nums = [int(n) for n in entrada.split() if n.isdigit() and 0 <= int(n) <= 36]
        for n in nums:
            st.session_state.historico.append({"number": n, "timestamp": f"manual_{len(st.session_state.historico)}"})
        salvar_resultado_em_arquivo(st.session_state.historico)
        st.success(f"{len(nums)} números adicionados.")
    except Exception as e:
        st.error("Erro ao adicionar números.")

# Atualização automática
st_autorefresh(interval=10000, key="refresh")

# Captura automática do último resultado
res = fetch_latest_result()
ultimo_ts = st.session_state.historico[-1]["timestamp"] if st.session_state.historico else None
if res and res["timestamp"] != ultimo_ts:
    st.session_state.historico.append(res)
    salvar_resultado_em_arquivo(st.session_state.historico)
    st.toast(f"🎲 Novo número: {res['number']}")
    # Verificação de acertos
    previsao_coluna = st.session_state.modelo_coluna.prever(st.session_state.historico)
    previsao_duzia = st.session_state.modelo_duzia.prever(st.session_state.historico)
    previsao_numero = st.session_state.modelo_numero.prever(st.session_state.historico, top_n=3)
    if previsao_coluna and get_coluna(res["number"]) in previsao_coluna:
        st.session_state.acertos["coluna"] += 1
        st.toast("✅ Acertou a coluna!")
    if previsao_duzia and get_duzia(res["number"]) in previsao_duzia:
        st.session_state.acertos["duzia"] += 1
        st.toast("✅ Acertou a dúzia!")
    if previsao_numero and res["number"] in previsao_numero:
        st.session_state.acertos["numero"] += 1
        st.toast("🎯 Acertou o número!")

# Treinamento dos modelos
st.session_state.modelo_coluna.treinar(st.session_state.historico)
st.session_state.modelo_duzia.treinar(st.session_state.historico)
st.session_state.modelo_numero.treinar(st.session_state.historico)

# Obtenção das previsões
coluna = st.session_state.modelo_coluna.prever(st.session_state.historico)
duzia = st.session_state.modelo_duzia.prever(st.session_state.historico)
numeros = st.session_state.modelo_numero.prever(st.session_state.historico, top_n=3)

# Exibição dos últimos 10 números
st.subheader("🔁 Últimos 10 Números")
ultimos = st.session_state.historico[-10:]
st.write(" ".join(str(item["number"]) for item in ultimos))

# Bloco seguro de previsões
st.subheader("🔮 Previsões")
if coluna and isinstance(coluna, list) and len(coluna) > 0:
    st.info(f"🧱 Coluna provável: {coluna[0]}")
else:
    st.warning("📉 Col
