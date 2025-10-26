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
# ESTRATÉGIAS MIDAS MELHORADAS
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

    def verificar_gatilho_midas(self, ultimo_numero, historico_recente):
        """Verifica qual estratégia Midas foi ativada pelo último número"""
        
        # Padrão do Zero - Mais restritivo
        if ultimo_numero in self.terminais['0']:
            # Só ativa se apareceu pelo menos 2 vezes nos últimos 8 sorteios
            count_terminal_0 = sum(1 for n in historico_recente[-8:] if n in self.terminais['0'])
            if count_terminal_0 >= 2:
                numeros_apostar = self.terminais['0'].copy()
                for num in self.terminais['0']:
                    numeros_apostar.extend(self.get_vizinhos_race(num))
                return {
                    'nome': 'Padrão do Zero',
                    'numeros_apostar': list(set(numeros_apostar))[:8],
                    'gatilho': f'Terminal 0 ativado ({count_terminal_2}x)'
                }

        # Padrão do Sete
        elif ultimo_numero in self.terminais['7']:
            count_terminal_7 = sum(1 for n in historico_recente[-8:] if n in self.terminais['7'])
            if count_terminal_7 >= 2:
                numeros_apostar = self.terminais['7'].copy()
                for num in self.terminais['7']:
                    numeros_apostar.extend(self.get_vizinhos_race(num))
                return {
                    'nome': 'Padrão do Sete',
                    'numeros_apostar': list(set(numeros_apostar))[:8],
                    'gatilho': f'Terminal 7 ativado ({count_terminal_7}x)'
                }

        # Padrão do Cinco
        elif ultimo_numero in self.terminais['5']:
            count_terminal_5 = sum(1 for n in historico_recente[-8:] if n in self.terminais['5'])
            if count_terminal_5 >= 2:
                numeros_apostar = self.terminais['5'].copy()
                for num in self.terminais['5']:
                    numeros_apostar.extend(self.get_vizinhos_race(num))
                return {
                    'nome': 'Padrão do Cinco',
                    'numeros_apostar': list(set(numeros_apostar))[:8],
                    'gatilho': f'Terminal 5 ativado ({count_terminal_5}x)'
                }

        # Padrão Gêmeos - Mais restritivo
        elif ultimo_numero in [11, 22, 33]:
            count_gemeos = sum(1 for n in historico_recente[-10:] if n in [11, 22, 33])
            if count_gemeos >= 2:
                numeros_apostar = [11, 22, 33].copy()
                for num in [11, 22, 33]:
                    numeros_apostar.extend(self.get_vizinhos_race(num))
                return {
                    'nome': 'Padrão Gêmeos',
                    'numeros_apostar': list(set(numeros_apostar))[:8],
                    'gatilho': f'Gêmeos ativado ({count_gemeos}x)'
                }

        return None

# =============================
# ESTRATÉGIA TERMINAIS DOMINANTES MELHORADA
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

    def selecionar_numeros_estrategicos(self, terminais_dominantes, numero_13):
        """Seleciona no máximo 8 números mais estratégicos"""
        numeros_candidatos = set()
        
        # 1. Adicionar números dos terminais dominantes (máximo 3 por terminal)
        for t in terminais_dominantes[:2]:  # Apenas os 2 mais dominantes
            base = [n for n in range(37) if self.extrair_terminal(n) == t]
            # Pegar números que não saíram recentemente
            numeros_recentes = set(list(self.historico)[-5:])
            numeros_frios = [n for n in base if n not in numeros_recentes]
            
            if numeros_frios:
                numeros_candidatos.update(numeros_frios[:3])
            else:
                numeros_candidatos.update(base[:3])
        
        # 2. Adicionar vizinhos próximos do número 13 no Race (apenas 2)
        if numero_13 in self.roleta:
            idx = self.roleta.index(numero_13)
            vizinhos = [
                self.roleta[(idx - 1) % len(self.roleta)],
                self.roleta[(idx + 1) % len(self.roleta)]
            ]
            numeros_candidatos.update(vizinhos)
        
        # 3. Adicionar números opostos no Race (2 números)
        if numero_13 in self.roleta:
            idx = self.roleta.index(numero_13)
            opostos = [
                self.roleta[(idx + 18) % len(self.roleta)],
                self.roleta[(idx - 18) % len(self.roleta)]
            ]
            numeros_candidatos.update(opostos)
        
        return list(numeros_candidatos)[:8]  # Máximo 8 números

    def verificar_gatilho_terminais(self):
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
            numeros_apostar = self.selecionar_numeros_estrategicos(dominantes, numero_13)
            
            return {
                "nome": "Terminais Dominantes",
                "numeros_apostar": numeros_apostar,
                "gatilho": f"Critério {'A' if condicao_a else 'B'} - Número {numero_13}",
                "terminais_dominantes": dominantes
            }
        return None

# =============================
# GESTOR PRINCIPAL
# =============================
class GestorEstrategias:
    def __init__(self):
        self.terminais_dominantes = EstrategiaTerminaisDominantes()
        self.midas = EstrategiasMidas()
        self.previsao_ativa = None
        
    def adicionar_numero(self, numero):
        self.terminais_dominantes.adicionar_numero(numero)
        
    def verificar_estrategia_ativa(self):
        """Verifica qual estratégia foi ativada pelo último número"""
        historico = list(self.terminais_dominantes.historico)
        
        if not historico:
            return None
            
        ultimo_numero = historico[-1]
        
        # Primeiro verifica Terminais Dominantes
        estrategia_terminais = self.terminais_dominantes.verificar_gatilho_terminais()
        if estrategia_terminais:
            return estrategia_terminais
        
        # Depois verifica estratégias Midas
        estrategia_midas = self.midas.verificar_gatilho_midas(ultimo_numero, historico[-10:])
        if estrategia_midas:
            return estrategia_midas
        
        return None

    def definir_previsao(self, estrategia):
        """Define uma previsão ativa"""
        self.previsao_ativa = {
            'estrategia': estrategia['nome'],
            'numeros_apostar': estrategia['numeros_apostar'],
            'gatilho': estrategia['gatilho'],
            'timestamp': len(self.terminais_dominantes.historico)
        }

    def conferir_previsao(self, numero_sorteado):
        """Conferir se a previsão atual acertou"""
        if not self.previsao_ativa:
            return None
            
        acerto = numero_sorteado in self.previsao_ativa['numeros_apostar']
        resultado = {
            'acerto': acerto,
            'numero_sorteado': numero_sorteado,
            'estrategia': self.previsao_ativa['estrategia'],
            'previsao': self.previsao_ativa['numeros_apostar']
        }
        
        # Limpa a previsão após conferir
        self.previsao_ativa = None
        
        return resultado

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
# APP STREAMLIT - INTERFACE ORIGINAL
# =============================
st.set_page_config(page_title="IA Roleta — Sistema Inteligente", layout="centered")
st.title("🎯 IA Roleta — Estratégias Melhoradas")

# --- Estado ---
if "historico" not in st.session_state:
    st.session_state.historico = json.load(open(HISTORICO_PATH)) if os.path.exists(HISTORICO_PATH) else []

if "gestor" not in st.session_state:
    st.session_state.gestor = GestorEstrategias()

if "ultima_conferencia" not in st.session_state:
    st.session_state.ultima_conferencia = None

if "acertos" not in st.session_state:
    st.session_state.acertos = 0

if "erros" not in st.session_state:
    st.session_state.erros = 0

if "historico_desempenho" not in st.session_state:
    st.session_state.historico_desempenho = []

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
            
            # Primeiro conferir previsão anterior se existir
            if st.session_state.gestor.previsao_ativa:
                resultado_conferencia = st.session_state.gestor.conferir_previsao(n)
                if resultado_conferencia:
                    st.session_state.ultima_conferencia = resultado_conferencia
                    if resultado_conferencia['acerto']:
                        st.session_state.acertos += 1
                        tocar_som_moeda()
                    else:
                        st.session_state.erros += 1
                    
                    # Salva no histórico de desempenho
                    st.session_state.historico_desempenho.append(resultado_conferencia)
            
            # Adiciona número ao histórico
            st.session_state.historico.append(item)
            st.session_state.gestor.adicionar_numero(n)
            
            # Verifica se ativou nova estratégia
            estrategia = st.session_state.gestor.verificar_estrategia_ativa()
            if estrategia:
                st.session_state.gestor.definir_previsao(estrategia)
                # Envia alerta
                msg = f"🎯 {estrategia['nome']}\n"
                msg += f"Gatilho: {estrategia['gatilho']}\n"
                msg += f"Números: {', '.join(map(str, sorted(estrategia['numeros_apostar'])))}"
                enviar_previsao(msg)
            
        salvar_resultado_em_arquivo(st.session_state.historico)
        st.success(f"{len(nums)} números adicionados!")
        st.rerun()
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
        # Primeiro conferir previsão anterior se existir
        if st.session_state.gestor.previsao_ativa:
            resultado_conferencia = st.session_state.gestor.conferir_previsao(numero_atual)
            if resultado_conferencia:
                st.session_state.ultima_conferencia = resultado_conferencia
                if resultado_conferencia['acerto']:
                    st.session_state.acertos += 1
                    tocar_som_moeda()
                else:
                    st.session_state.erros += 1
                
                # Salva no histórico de desempenho
                st.session_state.historico_desempenho.append(resultado_conferencia)
        
        # Adiciona número ao histórico
        st.session_state.historico.append(resultado)
        st.session_state.gestor.adicionar_numero(int(numero_atual))
        
        # Verifica se ativou nova estratégia
        estrategia = st.session_state.gestor.verificar_estrategia_ativa()
        if estrategia:
            st.session_state.gestor.definir_previsao(estrategia)
            # Envia alerta
            msg = f"🎯 {estrategia['nome']}\n"
            msg += f"Gatilho: {estrategia['gatilho']}\n"
            msg += f"Números: {', '.join(map(str, sorted(estrategia['numeros_apostar'])))}"
            enviar_previsao(msg)
        
        salvar_resultado_em_arquivo(st.session_state.historico)

# --- Interface ---
st.subheader("🔁 Últimos Números")
if st.session_state.historico:
    st.write(" ".join(str(h["number"]) for h in st.session_state.historico[-10:]))
else:
    st.write("Nenhum número registrado")

st.subheader("🎯 Previsão Ativa")

if st.session_state.gestor.previsao_ativa:
    previsao = st.session_state.gestor.previsao_ativa
    st.success(f"**{previsao['estrategia']}**")
    st.write(f"**Gatilho:** {previsao['gatilho']}")
    st.write(f"**Números para apostar ({len(previsao['numeros_apostar'])}):**")
    st.write(", ".join(map(str, sorted(previsao['numeros_apostar']))))
    st.info("⏳ Aguardando próximo sorteio para conferência...")
else:
    st.info("⏳ Aguardando gatilho para nova previsão...")

# --- Última Conferência ---
if st.session_state.ultima_conferencia:
    st.subheader("📊 Última Conferência")
    conferencia = st.session_state.ultima_conferencia
    if conferencia['acerto']:
        st.success(f"🎉 **ACERTOU!** Número {conferencia['numero_sorteado']} estava na previsão!")
    else:
        st.error(f"❌ **ERROU!** Número {conferencia['numero_sorteado']} não estava na previsão.")
    st.write(f"Estratégia: {conferencia['estrategia']}")

# --- Desempenho Detalhado ---
st.subheader("📈 Desempenho Detalhado")

total = st.session_state.acertos + st.session_state.erros
taxa = (st.session_state.acertos / total * 100) if total > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Acertos", st.session_state.acertos)
col2.metric("🔴 Erros", st.session_state.erros)
col3.metric("📊 Total", total)
col4.metric("✅ Taxa", f"{taxa:.1f}%")

# Histórico recente de conferências
if st.session_state.historico_desempenho:
    st.write("**Últimas 5 conferências:**")
    for i, conf in enumerate(st.session_state.historico_desempenho[-5:]):
        resultado_str = "🟢" if conf['acerto'] else "🔴"
        st.write(f"{resultado_str} {conf['estrategia']}: Número {conf['numero_sorteado']}")

# --- Informações das Estratégias ---
with st.expander("📚 Estratégias Disponíveis"):
    st.write("""
    **🎯 Terminais Dominantes (Melhorado)**
    - Gatilho: Critério A (número repetido) ou B (terminal repetido)
    - **Máximo 8 números**: Terminais dominantes + vizinhos + opostos
    - Foco em números frios (não sorteados recentemente)
    
    **🎯 Padrão do Zero**
    - Gatilho: Terminal 0 aparecer ≥2x nos últimos 8 sorteios
    - **Máximo 8 números**: Terminal 0 + vizinhos
    
    **🎯 Padrão do Sete**
    - Gatilho: Terminal 7 aparecer ≥2x nos últimos 8 sorteios
    - **Máximo 8 números**: Terminal 7 + vizinhos
    
    **🎯 Padrão do Cinco**
    - Gatilho: Terminal 5 aparecer ≥2x nos últimos 8 sorteios
    - **Máximo 8 números**: Terminal 5 + vizinhos
    
    **🎯 Padrão Gêmeos**
    - Gatilho: Gêmeos (11,22,33) aparecer ≥2x nos últimos 10 sorteios
    - **Máximo 8 números**: Gêmeos + vizinhos
    """)

# --- Download histórico ---
if os.path.exists(HISTORICO_PATH):
    with open(HISTORICO_PATH, "r") as f:
        conteudo = f.read()
    st.download_button("📥 Baixar histórico", data=conteudo, file_name="historico_roleta.json")
