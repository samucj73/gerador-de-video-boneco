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
# ESTRATÉGIAS MIDAS - ADICIONADAS
# =============================
class MidasStrategies:
    def __init__(self):
        # Terminais da roleta (último dígito)
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

        # Race da roleta (ordem física)
        self.race = [
            5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3,
            10, 23, 8, 30, 11, 36, 13, 27, 6, 34, 17, 25, 2, 21, 4, 19, 15, 32, 0
        ]

        # Vizinhos no race (baseado no PDF)
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

        self.historico_gatilhos = deque(maxlen=10)

    def get_vizinhos_race(self, numero):
        """Retorna os vizinhos no race para um número específico"""
        numero_str = str(numero)
        return self.vizinhos_race.get(numero_str, [])

    def estrategia_padrao_zero(self, ultimos_numeros):
        """Padrão do Zero - Gatilho: terminal 0"""
        trigger = self.terminais['0']
        
        # Verifica se algum número do terminal 0 saiu nos últimos 5 números
        gatilho_ativado = any(num in trigger for num in ultimos_numeros[-5:])
        
        if gatilho_ativado:
            numeros_apostar = self.terminais['0'].copy()
            # Adiciona vizinhos do race para cada número do terminal
            for num in self.terminais['0']:
                numeros_apostar.extend(self.get_vizinhos_race(num))
            
            return {
                'nome': 'Padrão do Zero',
                'numeros_apostar': list(set(numeros_apostar)),  # Remove duplicatas
                'gatilho': 'Terminal 0 (0,10,20,30)',
                'repeticoes': 2
            }
        return None

    def estrategia_padrao_sete(self, ultimos_numeros):
        """Padrão do Sete - Gatilho: terminal 7"""
        trigger = self.terminais['7']
        
        gatilho_ativado = any(num in trigger for num in ultimos_numeros[-5:])
        
        if gatilho_ativado:
            numeros_apostar = self.terminais['7'].copy()
            for num in self.terminais['7']:
                numeros_apostar.extend(self.get_vizinhos_race(num))
            
            return {
                'nome': 'Padrão do Sete',
                'numeros_apostar': list(set(numeros_apostar)),
                'gatilho': 'Terminal 7 (7,17,27)',
                'repeticoes': 2
            }
        return None

    def estrategia_padrao_cinco(self, ultimos_numeros):
        """Padrão do Cinco - Gatilho: terminal 5"""
        trigger = self.terminais['5']
        
        gatilho_ativado = any(num in trigger for num in ultimos_numeros[-5:])
        
        if gatilho_ativado:
            numeros_apostar = self.terminais['5'].copy()
            for num in self.terminais['5']:
                numeros_apostar.extend(self.get_vizinhos_race(num))
            
            return {
                'nome': 'Padrão do Cinco',
                'numeros_apostar': list(set(numeros_apostar)),
                'gatilho': 'Terminal 5 (5,15,25,35)',
                'repeticoes': 2
            }
        return None

    def estrategia_padrao_gemeos(self, ultimos_numeros):
        """Padrão Gêmeos - Gatilho: números 11, 22, 33"""
        trigger = [11, 22, 33]
        
        gatilho_ativado = any(num in trigger for num in ultimos_numeros[-5:])
        
        if gatilho_ativado:
            numeros_apostar = trigger.copy()
            for num in trigger:
                numeros_apostar.extend(self.get_vizinhos_race(num))
            
            return {
                'nome': 'Padrão Gêmeos',
                'numeros_apostar': list(set(numeros_apostar)),
                'gatilho': 'Gêmeos (11,22,33)',
                'repeticoes': 2
            }
        return None

    def analisar_estrategias(self, ultimos_numeros):
        """Analisa todas as estratégias e retorna as ativas"""
        if len(ultimos_numeros) < 5:
            return []
            
        estrategias_ativas = []
        
        # Verifica cada estratégia
        estrategia_zero = self.estrategia_padrao_zero(ultimos_numeros)
        if estrategia_zero:
            estrategias_ativas.append(estrategia_zero)
            
        estrategia_sete = self.estrategia_padrao_sete(ultimos_numeros)
        if estrategia_sete:
            estrategias_ativas.append(estrategia_sete)
            
        estrategia_cinco = self.estrategia_padrao_cinco(ultimos_numeros)
        if estrategia_cinco:
            estrategias_ativas.append(estrategia_cinco)
            
        estrategia_gemeos = self.estrategia_padrao_gemeos(ultimos_numeros)
        if estrategia_gemeos:
            estrategias_ativas.append(estrategia_gemeos)
            
        return estrategias_ativas

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

# ===== API CORRETA =====
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
# Estratégia baseada em terminais dominantes + vizinhos físicos Race
# =============================
class EstrategiaRoleta:
    def __init__(self, janela=12):
        self.janela = janela
        self.historico = deque(maxlen=janela+1)

        # ordem física da roleta Race (europeia, 37 casas)
        self.roleta = [
            0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27,
            13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33,
            1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12,
            35, 3, 26
        ]

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
        """Expande cada número com 2 vizinhos físicos antes e 2 depois (ordem Race)."""
        conjunto = set()
        for n in numeros:
            if n not in self.roleta:
                continue
            idx = self.roleta.index(n)
            for offset in range(-2, 3):  # 2 antes até 2 depois
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

        # Critério A → número repetido
        condicao_a = numero_13 in ultimos_12

        # Critério B → terminal repetido
        condicao_b = terminal_13 in [self.extrair_terminal(n) for n in ultimos_12]

        # Critério C → não repetiu nem número nem terminal
        condicao_c = not condicao_a and not condicao_b

        if condicao_a or condicao_b:
            jogar_nos_terminais = {}
            for t in dominantes:
                base = [n for n in range(37) if self.extrair_terminal(n) == t]
                jogar_nos_terminais[t] = sorted(self.adicionar_vizinhos_fisicos(base))

            return {
                "entrada": True,
                "criterio": "A" if condicao_a else "B",
                "numero_13": numero_13,
                "dominantes": dominantes,
                "jogar_nos_terminais": jogar_nos_terminais
            }

        elif condicao_c:
            return {
                "entrada": False,
                "criterio": "C",
                "numero_13": numero_13,
                "dominantes": dominantes
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
st.set_page_config(page_title="IA Roleta — Terminais Dominantes + Midas", layout="centered")
st.title("🎯 IA Roleta XXXtreme — Estratégias Midas + Terminais")

# --- Estado ---
if "rodadas_bloqueadas" not in st.session_state:
    st.session_state.rodadas_bloqueadas = 0
    
if "historico" not in st.session_state:
    st.session_state.historico = json.load(open(HISTORICO_PATH)) if os.path.exists(HISTORICO_PATH) else []

if "estrategia" not in st.session_state:
    st.session_state.estrategia = EstrategiaRoleta(janela=12)

# INICIALIZAÇÃO DAS ESTRATÉGIAS MIDAS - ADICIONADO
if "midas_strategies" not in st.session_state:
    st.session_state.midas_strategies = MidasStrategies()

# Seleção de estratégia
estrategia_selecionada = st.sidebar.selectbox(
    "🎯 Selecione a Estratégia:",
    ["Terminais Dominantes", "Estratégias Midas"]
)

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
    "previsao_base_timestamp": None,
    "acertos": 0,
    "erros": 0,
    "midas_estrategias_ativas": [],  # ADICIONADO
    "midas_numeros_apostar": []      # ADICIONADO
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

                # GREEN/RED atualizado com vizinhos
                if st.session_state.previsao_enviada and not st.session_state.resultado_enviado:
                    if estrategia_selecionada == "Terminais Dominantes":
                        terminais = st.session_state.terminais_previstos or []
                        numeros_validos = set()
                        for t in terminais:
                            base = [num for num in range(37) if num % 10 == t]
                            numeros_validos.update(st.session_state.estrategia.adicionar_vizinhos_fisicos(base))
                        green = n in numeros_validos
                    else:
                        # Estratégias Midas
                        green = n in st.session_state.midas_numeros_apostar

                    msg = f"Resultado: {n} | {'🟢 GREEN' if green else '🔴 RED'}"
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

# FIX: Add proper validation for numero_atual
numero_atual = None
ts_atual = None

if resultado and resultado.get("timestamp") and resultado["timestamp"] != ultimo_ts:
    numero_atual = resultado.get("number")
    ts_atual = resultado.get("timestamp")

    # Validate that we have a valid number before proceeding
    if numero_atual is not None:
        # Atualiza histórico e estratégia
        st.session_state.historico.append(resultado)
        try:
            st.session_state.estrategia.adicionar_numero(int(numero_atual))
        except Exception:
            pass
        salvar_resultado_em_arquivo(st.session_state.historico)

        # GREEN/RED atualizado com vizinhos - FIX: Add proper validation
        if st.session_state.previsao_enviada and not st.session_state.resultado_enviado:
            if estrategia_selecionada == "Terminais Dominantes":
                terminais = st.session_state.terminais_previstos or []
                numeros_validos = set()
                for t in terminais:
                    base = [num for num in range(37) if num % 10 == t]
                    numeros_validos.update(st.session_state.estrategia.adicionar_vizinhos_fisicos(base))
                
                # FIX: Safe conversion with error handling
                try:
                    green = int(numero_atual) in numeros_validos
                except (ValueError, TypeError):
                    green = False
            else:
                # Estratégias Midas
                try:
                    green = int(numero_atual) in st.session_state.midas_numeros_apostar
                except (ValueError, TypeError):
                    green = False

            msg = f"Resultado: {numero_atual} | {'🟢 GREEN' if green else '🔴 RED'}"
            enviar_resultado(msg)
            st.session_state.resultado_enviado = True
            st.session_state.previsao_enviada = False
            if green:
                st.session_state.acertos += 1
                tocar_som_moeda()
            else:
                st.session_state.erros += 1

# =============================
# Verifica nova entrada (AMBAS AS ESTRATÉGIAS)
# =============================
if estrategia_selecionada == "Terminais Dominantes":
    # ESTRATÉGIA ORIGINAL
    entrada_info = None
    if "estrategia" in st.session_state:
        entrada_info = st.session_state.estrategia.verificar_entrada()

    if entrada_info:
        dominantes = entrada_info.get("dominantes", [])

        # --- Critérios A/B: entrar na aposta ---
        if entrada_info.get("entrada") and not st.session_state.previsao_enviada:
            st.session_state.terminais_previstos = dominantes
            st.session_state.criterio = entrada_info.get("criterio")
            st.session_state.previsao_base_timestamp = ts_atual
            st.session_state.resultado_enviado = False
            st.session_state.previsao_enviada = True

            # Monta números que compõem cada terminal dominante
            linhas_numeros = []
            for t in dominantes:
                numeros_terminal = [n for n in range(37) if n % 10 == t]
                linhas_numeros.append(" ".join(str(n) for n in numeros_terminal))

            msg_alerta = "\n".join(linhas_numeros)
            enviar_previsao(msg_alerta)

        # --- Critério C: 13º número não bate com os 12 anteriores ---
        elif (entrada_info or {}).get("criterio") == "C":
            st.session_state.previsao_enviada = False
            st.session_state.terminais_previstos = None
            st.session_state.criterio = None
            msg_alerta = "⏳ Nenhum terminal\nAguardando próximo giro..."
            enviar_previsao(msg_alerta)

else:
    # ESTRATÉGIAS MIDAS - NOVAS
    if st.session_state.historico:
        ultimos_numeros = [int(h["number"]) for h in st.session_state.historico[-10:]]
        
        # Analisa estratégias Midas
        estrategias_ativas = st.session_state.midas_strategies.analisar_estrategias(ultimos_numeros)
        st.session_state.midas_estrategias_ativas = estrategias_ativas
        
        # Se há estratégias ativas e ainda não enviou previsão
        if estrategias_ativas and not st.session_state.previsao_enviada:
            # Pega a primeira estratégia ativa (poderia escolher por prioridade)
            estrategia = estrategias_ativas[0]
            st.session_state.midas_numeros_apostar = estrategia['numeros_apostar']
            st.session_state.previsao_enviada = True
            st.session_state.resultado_enviado = False
            
            # Envia alerta
            msg_alerta = f"{estrategia['nome']} - Gatilho: {estrategia['gatilho']}\n"
            msg_alerta += f"Números: {', '.join(map(str, estrategia['numeros_apostar']))}"
            enviar_previsao(msg_alerta)

# =============================
# Interface - ATUALIZADA
# =============================
st.subheader("🔁 Últimos 10 Números")
st.write(" ".join(str(h["number"]) for h in st.session_state.historico[-10:]))

st.subheader("🔮 Previsão de Entrada")

if estrategia_selecionada == "Terminais Dominantes":
    # Interface original
    if st.session_state.terminais_previstos:
        if st.session_state.criterio:
            st.write(f"🎯 Terminais dominantes: {st.session_state.terminais_previstos} (Critério {st.session_state.criterio})")
        else:
            st.write(f"🎯 Terminais dominantes: {st.session_state.terminais_previstos}")
    else:
        st.info("🔎 Aguardando próximo número para calcular dominantes.")
else:
    # Interface Midas
    if st.session_state.midas_estrategias_ativas:
        for estrategia in st.session_state.midas_estrategias_ativas:
            with st.expander(f"🎯 {estrategia['nome']} - ATIVA", expanded=True):
                st.write(f"**Gatilho:** {estrategia['gatilho']}")
                st.write(f"**Números para apostar ({len(estrategia['numeros_apostar'])}):**")
                st.write(", ".join(map(str, sorted(estrategia['numeros_apostar']))))
                st.write(f"**Repetições:** {estrategia['repeticoes']}")
    else:
        st.info("🔎 Aguardando gatilhos para as estratégias Midas...")

# =============================
# Informações das Estratégias Midas - NOVA SEÇÃO
# =============================
with st.expander("📚 Estratégias Midas - Como Funcionam"):
    st.write("""
    **Padrão do Zero**
    - Gatilho: Aparição de números do Terminal 0 (0,10,20,30)
    - Entrada: Aposta no Terminal 0 + 2 vizinhos no Race
    
    **Padrão do Sete**
    - Gatilho: Aparição de números do Terminal 7 (7,17,27)
    - Entrada: Aposta no Terminal 7 + 2 vizinhos no Race
    
    **Padrão do Cinco**
    - Gatilho: Aparição de números do Terminal 5 (5,15,25,35)
    - Entrada: Aposta no Terminal 5 + 2 vizinhos no Race
    
    **Padrão Gêmeos**
    - Gatilho: Aparição dos números 11, 22, 33
    - Entrada: Aposta nos Gêmeos + 2 vizinhos no Race
    
    *Repetir a entrada 2 vezes caso não acerte na primeira*
    """)

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
