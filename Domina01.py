# Domina01.py
import streamlit as st
import json
import os
import requests
import logging
from collections import Counter, deque
from alertas import enviar_previsao, enviar_resultado
from streamlit_autorefresh import st_autorefresh
import base64

# =============================
# Configurações
# =============================
HISTORICO_PATH = "historico_coluna_duzia.json"
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =============================
# Funções auxiliares
# =============================
def tocar_som_moeda():
    som_base64 = (
        "SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjI2LjEwNAAAAAAAAAAAAAAA//tQxAADBQAB"
        "VAAAAnEAAACcQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAA//sQxAADAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
        "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
        "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
        "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
    )
    st.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{som_base64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )

def salvar_resultado_em_arquivo(historico, caminho=HISTORICO_PATH):
    try:
        with open(caminho, "w") as f:
            json.dump(historico, f, indent=2)
    except Exception as e:
        logging.error(f"Erro ao salvar histórico: {e}")

# ===== API CORRETA (como você enviou) =====
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

# # =============================
# Estratégia baseada em terminais dominantes
# =============================
from collections import deque, Counter

class EstrategiaRoleta:
    def __init__(self, janela=12):
        self.janela = janela
        # mantém até 13 números (12 + o 13º para validação A/B)
        self.historico = deque(maxlen=janela+1)

    def extrair_terminal(self, numero):
        return numero % 10

    def adicionar_numero(self, numero):
        self.historico.append(numero)

    def calcular_dominantes(self):
        # usa os 12 anteriores ao 13º
        if len(self.historico) < self.janela:
            return []
        ultimos_13 = list(self.historico)
        ultimos_12 = ultimos_13[:-1] if len(ultimos_13) >= 13 else ultimos_13
        terminais = [self.extrair_terminal(n) for n in ultimos_12]
        contagem = Counter(terminais)
        return [t for t, _ in contagem.most_common(2)]

    def verificar_entrada(self):
        # precisa de 13 números (12 base + 13º gatilho)
        if len(self.historico) < self.janela + 1:
            return None

        ultimos = list(self.historico)
        ultimos_12 = ultimos[:-1]
        numero_13 = ultimos[-1]
        dominantes = self.calcular_dominantes()
        terminal_13 = self.extrair_terminal(numero_13)

        # Critério A: número inteiro do 13º já saiu nos 12 anteriores
        condicao_a = numero_13 in ultimos_12

        # Critério B (atualizado): terminal do 13º está entre os terminais dos últimos 12
        condicao_b = terminal_13 in [self.extrair_terminal(n) for n in ultimos_12]

        if condicao_a or condicao_b:
            return {
                "entrada": True,
                "criterio": "A" if condicao_a else "B",
                "numero_13": numero_13,
                "dominantes": dominantes,
                "jogar_nos_terminais": {
                    t: [n for n in range(37) if self.extrair_terminal(n) == t]
                    for t in dominantes
                }
            }
        else:
            return {
                "entrada": False,
                "numero_13": numero_13,
                "dominantes": dominantes
            }

# =============================
# App Streamlit
# =============================
st.set_page_config(page_title="IA Roleta — Terminais Dominantes (A/B)", layout="centered")
st.title("🎯 IA Roleta XXXtreme — Estratégia dos Terminais Dominantes (Critérios A e B)")

# --- Estado ---
if "historico" not in st.session_state:
    st.session_state.historico = json.load(open(HISTORICO_PATH)) if os.path.exists(HISTORICO_PATH) else []

if "estrategia" not in st.session_state:
    st.session_state.estrategia = EstrategiaRoleta(janela=12)

# Pré-carrega a estratégia com até 13 últimos números já salvos
if "estrategia_inicializada" not in st.session_state:
    for h in st.session_state.historico[-13:]:
        try:
            st.session_state.estrategia.adicionar_numero(int(h["number"]))
        except Exception:
            pass
    st.session_state.estrategia_inicializada = True

# Previsão/resultado & métricas
for k, v in {
    "terminais_previstos": None,
    "criterio": None,
    "previsao_enviada": False,
    "resultado_enviado": False,
    "previsao_base_timestamp": None,  # timestamp do giro que gerou a entrada (aposta vale para o PRÓXIMO giro)
    "acertos": 0,
    "erros": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Entrada manual ---
st.subheader("✍️ Inserir Sorteios Manualmente")
entrada = st.text_area("Digite números (0–36), separados por espaço — até 100:", height=100, key="entrada_manual")
if st.button("Adicionar Sorteios"):
    try:
        nums = [int(n) for n in entrada.split() if n.isdigit() and 0 <= int(n) <= 36]
        if len(nums) > 100:
            st.warning("Limite de 100 números.")
        else:
            for n in nums:
                item = {"number": n, "timestamp": f"manual_{len(st.session_state.historico)}"}
                st.session_state.historico.append(item)
                st.session_state.estrategia.adicionar_numero(n)

                # Se houver previsão pendente, o número manual conta como próximo giro → avaliar resultado
                if st.session_state.previsao_enviada and not st.session_state.resultado_enviado:
                    terminais = st.session_state.terminais_previstos or []
                    terminal_ultimo = n % 10
                    green = terminal_ultimo in terminais
                    msg = f"Resultado: {n} | Terminais: {terminais} | {'🟢 GREEN' if green else '🔴 RED'}"
                    enviar_resultado(msg)
                    st.session_state.resultado_enviado = True
                    st.session_state.previsao_enviada = False
                    if green:
                        st.session_state.acertos += 1
                        tocar_som_moeda()
                    else:
                        st.session_state.erros += 1

            salvar_resultado_em_arquivo(st.session_state.historico)
            st.success(f"{len(nums)} números adicionados.")
    except Exception as e:
        st.error(f"Erro ao adicionar números: {e}")

# --- Atualização automática ---
st_autorefresh(interval=3000, key="refresh_dominantes")

# Busca resultado mais recente da API correta
resultado = fetch_latest_result()
ultimo_ts = st.session_state.historico[-1]["timestamp"] if st.session_state.historico else None

if resultado and resultado.get("timestamp") and resultado["timestamp"] != ultimo_ts:
    numero_atual = resultado["number"]
    ts_atual = resultado["timestamp"]

    # Atualiza histórico e estratégia
    st.session_state.historico.append(resultado)
    try:
        st.session_state.estrategia.adicionar_numero(int(numero_atual))
    except Exception:
        pass
    salvar_resultado_em_arquivo(st.session_state.historico)

    # 1) Se havia previsão pendente, este é o PRÓXIMO giro → avaliar GREEN/RED agora
    if st.session_state.previsao_enviada and not st.session_state.resultado_enviado:
        terminais = st.session_state.terminais_previstos or []
        terminal_ultimo = int(numero_atual) % 10
        green = terminal_ultimo in terminais
        msg = f"Resultado: {numero_atual} | Terminais: {terminais} | {'🟢 GREEN' if green else '🔴 RED'}"
        enviar_resultado(msg)
        st.session_state.resultado_enviado = True
        st.session_state.previsao_enviada = False
        if green:
            st.session_state.acertos += 1
            tocar_som_moeda()
        else:
            st.session_state.erros += 1

    # 2) Com o número atual adicionado, verifica se gera NOVA entrada para o PRÓXIMO giro
    entrada_info = st.session_state.estrategia.verificar_entrada()
    if entrada_info:
        dominantes = entrada_info["dominantes"]
        if entrada_info.get("entrada") and not st.session_state.previsao_enviada:
            st.session_state.terminais_previstos = dominantes
            st.session_state.criterio = entrada_info.get("criterio")
            st.session_state.previsao_base_timestamp = ts_atual  # aposta vale para o próximo giro
            st.session_state.resultado_enviado = False
            st.session_state.previsao_enviada = True
            enviar_previsao(f"🎯 Previsão: terminais {dominantes} (Critério {st.session_state.criterio})")

# --- Interface ---
st.subheader("🔁 Últimos 13 Números")
st.write(" ".join(str(h["number"]) for h in st.session_state.historico[-13:]))

st.subheader("🔮 Previsão de Entrada")
if st.session_state.terminais_previstos:
    if st.session_state.criterio:
        st.write(f"🎯 Terminais dominantes: {st.session_state.terminais_previstos} (Critério {st.session_state.criterio})")
    else:
        st.write(f"🎯 Terminais dominantes: {st.session_state.terminais_previstos}")
else:
    st.info("🔎 Aguardando próximo número para calcular dominantes.")

st.subheader("📊 Desempenho")
total = st.session_state.acertos + st.session_state.erros
taxa = (st.session_state.acertos / total * 100) if total > 0 else 0.0
col1, col2, col3 = st.columns(3)
col1.metric("🟢 GREEN", st.session_state.acertos)
col2.metric("🔴 RED", st.session_state.erros)
col3.metric("✅ Taxa de acerto", f"{taxa:.1f}%")

# --- Download histórico ---
if os.path.exists(HISTORICO_PATH):
    with open(HISTORICO_PATH, "r") as f:
        conteudo = f.read()
    st.download_button("📥 Baixar histórico", data=conteudo, file_name="historico_coluna_duzia.json")
