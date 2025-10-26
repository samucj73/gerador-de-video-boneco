import streamlit as st
import json
import os
import requests
import logging
from collections import Counter, deque
from alertas import enviar_previsao, enviar_resultado
from streamlit_autorefresh import st_autorefresh

# =============================
# Configurações
# =============================
HISTORICO_PATH = "historico_coluna_duzia.json"
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =============================
# ESTRATÉGIAS MIDAS
# =============================
class EstrategiasMidas:
    def __init__(self):
        self.terminais = {
            '0': [0, 10, 20, 30],
            '1': [1, 11, 21, 31],
            '2': [2, 12, 22, 32],
            '3': [3, 13, 23, 33],
            '4': [4, 14, 24, 34],
            '5': [5, 15, 25, 35],
            '6': [6, 16, 26, 36],
            '7': [7, 17, 27],
            '8': [8, 18, 28],
            '9': [9, 19, 29]
        }
        
        self.vizinhos_race = {
            '33': [1, 20, 36, 11, 30, 4, 21, 2, 14, 31, 9],
            '21': [2, 25, 28, 12, 35, 9, 22, 18, 0, 32, 15],
            '35': [3, 26, 27, 13, 36, 8, 23, 10, 16, 33, 1],
            '19': [4, 21, 20, 14, 31, 5, 24, 16, 17, 34, 6],
            '10': [5, 24, 32, 15, 19, 2, 25, 17, 12, 35, 3],
            '34': [6, 27, 24, 16, 33, 3, 26, 0, 13, 36, 11],
            '29': [7, 28, 25, 17, 34, 6, 27, 13, 26, 0, 32],
            '30': [8, 23, 22, 18, 29, 7, 28, 12],
            '31': [9, 22, 15, 19, 4, 18, 29, 7],
            '23': [10, 5, 1, 20, 14, 11, 30, 8]
        }

    def get_vizinhos_race(self, numero):
        return self.vizinhos_race.get(str(numero), [])

    def analisar_estrategias_midas(self, ultimos_numeros):
        if len(ultimos_numeros) < 5:
            return []

        estrategias_ativas = []

        # Padrão do Zero
        if any(num in self.terminais['0'] for num in ultimos_numeros[-5:]):
            numeros_apostar = self.terminais['0'].copy()
            for num in self.terminais['0']:
                numeros_apostar.extend(self.get_vizinhos_race(num))
            estrategias_ativas.append({
                'nome': 'Padrão do Zero',
                'numeros_apostar': list(set(numeros_apostar)),
                'gatilho': 'Terminal 0 (0,10,20,30)'
            })

        # Padrão do Sete
        if any(num in self.terminais['7'] for num in ultimos_numeros[-5:]):
            numeros_apostar = self.terminais['7'].copy()
            for num in self.terminais['7']:
                numeros_apostar.extend(self.get_vizinhos_race(num))
            estrategias_ativas.append({
                'nome': 'Padrão do Sete',
                'numeros_apostar': list(set(numeros_apostar)),
                'gatilho': 'Terminal 7 (7,17,27)'
            })

        # Padrão do Cinco
        if any(num in self.terminais['5'] for num in ultimos_numeros[-5:]):
            numeros_apostar = self.terminais['5'].copy()
            for num in self.terminais['5']:
                numeros_apostar.extend(self.get_vizinhos_race(num))
            estrategias_ativas.append({
                'nome': 'Padrão do Cinco',
                'numeros_apostar': list(set(numeros_apostar)),
                'gatilho': 'Terminal 5 (5,15,25,35)'
            })

        # Padrão Gêmeos
        if any(num in [11, 22, 33] for num in ultimos_numeros[-5:]):
            numeros_apostar = [11, 22, 33].copy()
            for num in [11, 22, 33]:
                numeros_apostar.extend(self.get_vizinhos_race(num))
            estrategias_ativas.append({
                'nome': 'Padrão Gêmeos',
                'numeros_apostar': list(set(numeros_apostar)),
                'gatilho': 'Gêmeos (11,22,33)'
            })

        return estrategias_ativas

# =============================
# ESTRATÉGIA TERMINAIS DOMINANTES
# =============================
class EstrategiaTerminaisDominantes:
    def __init__(self, janela=12):
        self.janela = janela
        self.historico = deque(maxlen=janela+1)
        self.roleta = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]

    def extrair_terminal(self, numero):
        return numero % 10

    def adicionar_numero(self, numero):
        self.historico.append(numero)

    def calcular_dominantes(self):
        if len(self.historico) < self.janela:
            return []
        ultimos_13 = list(self.historico)
        ultimos_12 = ultimos_13[:-1] if len(ultimos_13) >= 13 else ultimos_13
        terminais = [self.extrair_terminal(n) for n in ultimos_12]
        contagem = Counter(terminais)
        return [t for t, _ in contagem.most_common(2)]

    def adicionar_vizinhos_fisicos(self, numeros):
        conjunto = set()
        for n in numeros:
            if n not in self.roleta:
                continue
            idx = self.roleta.index(n)
            for offset in range(-2, 3):
                vizinho = self.roleta[(idx + offset) % len(self.roleta)]
                conjunto.add(vizinho)
        return conjunto

    def verificar_entrada(self):
        if len(self.historico) < self.janela + 1:
            return None

        ultimos = list(self.historico)
        ultimos_12 = ultimos[:-1]
        numero_13 = ultimos[-1]
        dominantes = self.calcular_dominantes()
        terminal_13 = self.extrair_terminal(numero_13)

        condicao_a = numero_13 in ultimos_12
        condicao_b = terminal_13 in [self.extrair_terminal(n) for n in ultimos_12]

        if condicao_a or condicao_b:
            jogar_nos_terminais = {}
            for t in dominantes:
                base = [n for n in range(37) if self.extrair_terminal(n) == t]
                jogar_nos_terminais[t] = sorted(self.adicionar_vizinhos_fisicos(base))

            return {
                "entrada": True,
                "estrategia": "Terminais Dominantes",
                "criterio": "A" if condicao_a else "B",
                "numero_13": numero_13,
                "dominantes": dominantes,
                "numeros_apostar": list(set().union(*jogar_nos_terminais.values()))
            }
        return None

# =============================
# GESTOR PRINCIPAL
# =============================
class GestorEstrategias:
    def __init__(self):
        self.terminais_dominantes = EstrategiaTerminaisDominantes()
        self.midas = EstrategiasMidas()
        
    def adicionar_numero(self, numero):
        self.terminais_dominantes.adicionar_numero(numero)
        
    def analisar_todas_estrategias(self):
        todas_estrategias = []
        
        # 1. Terminais Dominantes
        entrada_terminais = self.terminais_dominantes.verificar_entrada()
        if entrada_terminais and entrada_terminais.get("entrada"):
            todas_estrategias.append(entrada_terminais)
        
        # 2. Midas
        historico_numeros = list(self.terminais_dominantes.historico)
        estrategias_midas = self.midas.analisar_estrategias_midas(historico_numeros)
        for estrategia in estrategias_midas:
            todas_estrategias.append({
                "estrategia": estrategia['nome'],
                "numeros_apostar": estrategia['numeros_apostar'],
                "logica": estrategia['gatilho']
            })
        
        return todas_estrategias

# =============================
# FUNÇÕES AUXILIARES
# =============================
def tocar_som_moeda():
    st.markdown("""<audio autoplay><source src="" type="audio/mp3"></audio>""", unsafe_allow_html=True)

def salvar_resultado_em_arquivo(historico, caminho=HISTORICO_PATH):
    try:
        with open(caminho, "w") as f:
            json.dump(historico, f, indent=2)
    except Exception as e:
        logging.error(f"Erro ao salvar histórico: {e}")

def fetch_latest_result():
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=5)
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

# =============================
# APP STREAMLIT
# =============================
st.set_page_config(page_title="IA Roleta — Todas as Estratégias", layout="centered")
st.title("🎯 IA Roleta — Todas as Estratégias Ativas")

# --- Estado ---
if "historico" not in st.session_state:
    st.session_state.historico = json.load(open(HISTORICO_PATH)) if os.path.exists(HISTORICO_PATH) else []

if "gestor" not in st.session_state:
    st.session_state.gestor = GestorEstrategias()

if "estrategias_ativas" not in st.session_state:
    st.session_state.estrategias_ativas = []

if "acertos" not in st.session_state:
    st.session_state.acertos = 0

if "erros" not in st.session_state:
    st.session_state.erros = 0

# --- Inicialização ---
if "inicializado" not in st.session_state:
    for h in st.session_state.historico[-13:]:
        try:
            st.session_state.gestor.adicionar_numero(int(h["number"]))
        except Exception:
            pass
    st.session_state.inicializado = True

# --- Entrada manual ---
st.subheader("✍️ Inserir Sorteios")
entrada = st.text_input("Digite números (0-36) separados por espaço:")
if st.button("Adicionar") and entrada:
    try:
        nums = [int(n) for n in entrada.split() if n.isdigit() and 0 <= int(n) <= 36]
        for n in nums:
            item = {"number": n, "timestamp": f"manual_{len(st.session_state.historico)}"}
            st.session_state.historico.append(item)
            st.session_state.gestor.adicionar_numero(n)
        salvar_resultado_em_arquivo(st.session_state.historico)
        st.success(f"{len(nums)} números adicionados!")
    except Exception as e:
        st.error(f"Erro: {e}")

# --- Atualização automática ---
st_autorefresh(interval=3000, key="refresh")

# --- Buscar resultado da API ---
resultado = fetch_latest_result()
ultimo_ts = st.session_state.historico[-1]["timestamp"] if st.session_state.historico else None

if resultado and resultado.get("timestamp") and resultado["timestamp"] != ultimo_ts:
    numero_atual = resultado.get("number")
    if numero_atual is not None:
        st.session_state.historico.append(resultado)
        st.session_state.gestor.adicionar_numero(int(numero_atual))
        salvar_resultado_em_arquivo(st.session_state.historico)

# --- Analisar estratégias ---
st.session_state.estrategias_ativas = st.session_state.gestor.analisar_todas_estrategias()

# --- Interface ---
st.subheader("🔁 Últimos Números")
st.write(" ".join(str(h["number"]) for h in st.session_state.historico[-10:]))

st.subheader("🎯 Estratégias Ativas")

if st.session_state.estrategias_ativas:
    for i, estrategia in enumerate(st.session_state.estrategias_ativas):
        with st.expander(f"✅ {estrategia['estrategia']}", expanded=True):
            st.write(f"**Lógica:** {estrategia.get('logica', estrategia.get('criterio', 'N/A'))}")
            st.write(f"**Números para apostar ({len(estrategia['numeros_apostar'])}):**")
            st.write(", ".join(map(str, sorted(estrategia['numeros_apostar']))))
else:
    st.info("🔎 Aguardando gatilhos para as estratégias...")

st.subheader("📊 Desempenho")
total = st.session_state.acertos + st.session_state.erros
taxa = (st.session_state.acertos / total * 100) if total > 0 else 0.0
col1, col2, col3 = st.columns(3)
col1.metric("🟢 GREEN", st.session_state.acertos)
col2.metric("🔴 RED", st.session_state.erros)
col3.metric("✅ Taxa", f"{taxa:.1f}%")

# --- Download histórico ---
if os.path.exists(HISTORICO_PATH):
    with open(HISTORICO_PATH, "r") as f:
        conteudo = f.read()
    st.download_button("📥 Baixar histórico", data=conteudo, file_name="historico_roleta.json")
