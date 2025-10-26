import streamlit as st
import json
import os
import requests
import logging
from collections import Counter, deque
from alertas import enviar_previsao, enviar_resultado
from streamlit_autorefresh import st_autorefresh
import random

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
        
        # Apenas ativa se o número não apareceu nos últimos 3 sorteios (evita repetição)
        if historico_recente and ultimo_numero in historico_recente[-3:]:
            return None
        
        # Padrão do Zero - Só ativa se houver pelo menos 2 números do terminal nos últimos 5
        if ultimo_numero in self.terminais['0']:
            count_terminal_0 = sum(1 for n in historico_recente[-5:] if n in self.terminais['0'])
            if count_terminal_0 >= 2:
                numeros_apostar = self.terminais['0'].copy()
                for num in self.terminais['0']:
                    numeros_apostar.extend(self.get_vizinhos_race(num))
                return {
                    'nome': 'Padrão do Zero',
                    'numeros_apostar': list(set(numeros_apostar))[:8],  # Reduz para 8 números
                    'gatilho': f'Terminal 0 apareceu {count_terminal_0}x nos últimos 5 sorteios'
                }

        # Padrão do Sete
        elif ultimo_numero in self.terminais['7']:
            count_terminal_7 = sum(1 for n in historico_recente[-5:] if n in self.terminais['7'])
            if count_terminal_7 >= 2:
                numeros_apostar = self.terminais['7'].copy()
                for num in self.terminais['7']:
                    numeros_apostar.extend(self.get_vizinhos_race(num))
                return {
                    'nome': 'Padrão do Sete',
                    'numeros_apostar': list(set(numeros_apostar))[:8],
                    'gatilho': f'Terminal 7 apareceu {count_terminal_7}x nos últimos 5 sorteios'
                }

        # Padrão do Cinco
        elif ultimo_numero in self.terminais['5']:
            count_terminal_5 = sum(1 for n in historico_recente[-5:] if n in self.terminais['5'])
            if count_terminal_5 >= 2:
                numeros_apostar = self.terminais['5'].copy()
                for num in self.terminais['5']:
                    numeros_apostar.extend(self.get_vizinhos_race(num))
                return {
                    'nome': 'Padrão do Cinco',
                    'numeros_apostar': list(set(numeros_apostar))[:8],
                    'gatilho': f'Terminal 5 apareceu {count_terminal_5}x nos últimos 5 sorteios'
                }

        # Padrão Gêmeos - Só ativa se houver pelo menos 2 gêmeos nos últimos 8 sorteios
        elif ultimo_numero in [11, 22, 33]:
            count_gemeos = sum(1 for n in historico_recente[-8:] if n in [11, 22, 33])
            if count_gemeos >= 2:
                numeros_apostar = [11, 22, 33].copy()
                for num in [11, 22, 33]:
                    numeros_apostar.extend(self.get_vizinhos_race(num))
                return {
                    'nome': 'Padrão Gêmeos',
                    'numeros_apostar': list(set(numeros_apostar))[:8],
                    'gatilho': f'Gêmeos apareceram {count_gemeos}x nos últimos 8 sorteios'
                }

        return None

# =============================
# ESTRATÉGIA TERMINAIS DOMINANTES INTELIGENTE MELHORADA
# =============================
class EstrategiaTerminaisDominantes:
    def __init__(self, janela=12):
        self.janela = janela
        self.historico = deque(maxlen=janela+1)
        self.roleta = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
        
        # Histórico de desempenho por terminal
        self.desempenho_terminais = {str(i): {'acertos': 0, 'tentativas': 0} for i in range(10)}

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

    def atualizar_desempenho_terminal(self, terminal, acerto):
        """Atualiza estatísticas de desempenho por terminal"""
        terminal_str = str(terminal)
        self.desempenho_terminais[terminal_str]['tentativas'] += 1
        if acerto:
            self.desempenho_terminais[terminal_str]['acertos'] += 1

    def get_taxa_acerto_terminal(self, terminal):
        """Retorna taxa de acerto de um terminal"""
        terminal_str = str(terminal)
        dados = self.desempenho_terminais[terminal_str]
        if dados['tentativas'] == 0:
            return 0.0
        return (dados['acertos'] / dados['tentativas']) * 100

    def selecionar_numeros_inteligentes(self, terminais_dominantes, numero_13):
        """Seleciona no máximo 8 números mais prováveis baseado em desempenho histórico"""
        numeros_candidatos = set()
        
        # 1. Ordenar terminais por desempenho histórico (melhores primeiro)
        terminais_ordenados = sorted(terminais_dominantes, 
                                   key=lambda t: self.get_taxa_acerto_terminal(t), 
                                   reverse=True)
        
        # 2. Pegar números dos 2 melhores terminais baseado em desempenho
        for terminal in terminais_ordenados[:2]:
            base_terminal = [n for n in range(37) if self.extrair_terminal(n) == terminal]
            
            # Priorizar números que não saíram recentemente
            numeros_recentes = set(list(self.historico)[-5:])
            numeros_frios = [n for n in base_terminal if n not in numeros_recentes]
            
            if numeros_frios:
                # Pegar até 3 números frios do terminal
                numeros_candidatos.update(numeros_frios[:3])
            else:
                # Se não há números frios, pegar 2 números aleatórios do terminal
                numeros_candidatos.update(random.sample(base_terminal, min(2, len(base_terminal))))
        
        # 3. Adicionar vizinhos estratégicos do número 13 (apenas 2 melhores vizinhos)
        if numero_13 in self.roleta:
            idx = self.roleta.index(numero_13)
            # Escolher apenas 2 vizinhos (1 antes e 1 depois)
            vizinhos_estrategicos = [
                self.roleta[(idx - 1) % len(self.roleta)],
                self.roleta[(idx + 1) % len(self.roleta)]
            ]
            numeros_candidatos.update(vizinhos_estrategicos)
        
        # 4. Se ainda temos poucos números, adicionar números opostos no Race
        if len(numeros_candidatos) < 6:
            idx_13 = self.roleta.index(numero_13)
            oposto_1 = self.roleta[(idx_13 + 18) % len(self.roleta)]  # Metade da roleta
            oposto_2 = self.roleta[(idx_13 - 18) % len(self.roleta)]
            numeros_candidatos.add(oposto_1)
            numeros_candidatos.add(oposto_2)
        
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

        # Só ativa se o terminal tiver pelo menos 30% de taxa de acerto OU se for nova tentativa
        terminal_aceitavel = any(self.get_taxa_acerto_terminal(t) >= 30.0 for t in dominantes)
        
        if (condicao_a or condicao_b) and (terminal_aceitavel or len(ultimos) < 20):
            numeros_apostar = self.selecionar_numeros_inteligentes(dominantes, numero_13)
            
            return {
                "nome": "Terminais Dominantes",
                "numeros_apostar": numeros_apostar,
                "gatilho": f"Critério {'A' if condicao_a else 'B'} - Número {numero_13}",
                "terminais_dominantes": dominantes
            }
        return None

# =============================
# SISTEMA DE ANALISE DE PADRÕES
# =============================
class AnalisadorPadroes:
    def __init__(self):
        self.historico_long = deque(maxlen=50)
        
    def adicionar_numero(self, numero):
        self.historico_long.append(numero)
        
    def analisar_tendencias(self):
        """Analisa tendências para melhorar as estratégias"""
        if len(self.historico_long) < 10:
            return {}
            
        ultimos_10 = list(self.historico_long)[-10:]
        
        # Analisar terminais quentes e frios
        terminais_10 = [n % 10 for n in ultimos_10]
        contagem_terminais = Counter(terminais_10)
        
        # Terminais quentes (mais frequentes)
        terminais_quentes = [t for t, count in contagem_terminais.most_common(3) if count >= 3]
        
        # Terminais frios (menos frequentes)
        todos_terminais = list(range(10))
        terminais_frios = [t for t in todos_terminais if t not in contagem_terminais or contagem_terminais[t] == 0]
        
        return {
            'terminais_quentes': terminais_quentes,
            'terminais_frios': terminais_frios[:3],  # Top 3 mais frios
            'ultimos_numeros': ultimos_10
        }

# =============================
# GESTOR PRINCIPAL MELHORADO
# =============================
class GestorEstrategias:
    def __init__(self):
        self.terminais_dominantes = EstrategiaTerminaisDominantes()
        self.midas = EstrategiasMidas()
        self.analisador = AnalisadorPadroes()
        self.previsao_ativa = None
        self.contador_estrategias = {
            'Terminais Dominantes': {'acertos': 0, 'tentativas': 0},
            'Padrão do Zero': {'acertos': 0, 'tentativas': 0},
            'Padrão do Sete': {'acertos': 0, 'tentativas': 0},
            'Padrão do Cinco': {'acertos': 0, 'tentativas': 0},
            'Padrão Gêmeos': {'acertos': 0, 'tentativas': 0}
        }
        
    def adicionar_numero(self, numero):
        self.terminais_dominantes.adicionar_numero(numero)
        self.analisador.adicionar_numero(numero)
        
    def atualizar_desempenho_estrategia(self, estrategia_nome, acerto):
        if estrategia_nome in self.contador_estrategias:
            self.contador_estrategias[estrategia_nome]['tentativas'] += 1
            if acerto:
                self.contador_estrategias[estrategia_nome]['acertos'] += 1

    def get_taxa_acerto_estrategia(self, estrategia_nome):
        dados = self.contador_estrategias.get(estrategia_nome, {'acertos': 0, 'tentativas': 0})
        if dados['tentativas'] == 0:
            return 0.0
        return (dados['acertos'] / dados['tentativas']) * 100

    def verificar_estrategia_ativa(self):
        """Verifica qual estratégia foi ativada pelo último número com base em desempenho"""
        historico = list(self.terminais_dominantes.historico)
        
        if not historico:
            return None
            
        ultimo_numero = historico[-1]
        tendencias = self.analisador.analisar_tendencias()
        
        # Primeiro verifica Terminais Dominantes (só se tiver boa taxa)
        taxa_terminais = self.get_taxa_acerto_estrategia('Terminais Dominantes')
        if taxa_terminais >= 25.0 or len(historico) < 15:  # Dá chance inicial
            estrategia_terminais = self.terminais_dominantes.verificar_gatilho_terminais()
            if estrategia_terminais:
                return estrategia_terminais
        
        # Depois verifica estratégias Midas (com filtro de desempenho)
        estrategia_midas = self.midas.verificar_gatilho_midas(ultimo_numero, historico[-8:])
        if estrategia_midas:
            taxa_midas = self.get_taxa_acerto_estrategia(estrategia_midas['nome'])
            if taxa_midas >= 20.0 or len(historico) < 20:  # Filtro por desempenho
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
        
        # Atualizar desempenho da estratégia
        self.atualizar_desempenho_estrategia(self.previsao_ativa['estrategia'], acerto)
        
        # Atualizar desempenho dos terminais (se for estratégia de terminais)
        if self.previsao_ativa['estrategia'] == 'Terminais Dominantes':
            for numero in self.previsao_ativa['numeros_apostar']:
                terminal = self.terminais_dominantes.extrair_terminal(numero)
                self.terminais_dominantes.atualizar_desempenho_terminal(terminal, acerto)
        
        # Limpa a previsão após conferir
        self.previsao_ativa = None
        
        return resultado

# =============================
# FUNÇÕES AUXILIARES (manter as mesmas)
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
# APP STREAMLIT - INTERFACE MELHORADA
# =============================
st.set_page_config(page_title="IA Roleta — Sistema Inteligente", layout="centered")
st.title("🎯 IA Roleta — Sistema com Análise de Desempenho")

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

# --- Resto do código mantido igual até a seção de desempenho ---

# [MANTER TODO O CÓDIGO ANTERIOR ATÉ A SEÇÃO DE DESEMPENHO]

# --- DESEMPENHO DETALHADO MELHORADO ---
st.subheader("📈 Análise de Desempenho Avançada")

total = st.session_state.acertos + st.session_state.erros
taxa = (st.session_state.acertos / total * 100) if total > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Acertos", st.session_state.acertos)
col2.metric("🔴 Erros", st.session_state.erros)
col3.metric("📊 Total", total)
col4.metric("✅ Taxa Geral", f"{taxa:.1f}%")

# Estatísticas por estratégia
st.write("**📊 Desempenho por Estratégia:**")
for estrategia, dados in st.session_state.gestor.contador_estrategias.items():
    if dados['tentativas'] > 0:
        taxa_estrategia = (dados['acertos'] / dados['tentativas']) * 100
        cor = "🟢" if taxa_estrategia >= 40 else "🟡" if taxa_estrategia >= 25 else "🔴"
        st.write(f"{cor} {estrategia}: {dados['acertos']}/{dados['tentativas']} ({taxa_estrategia:.1f}%)")

# Estatísticas por terminal
st.write("**🎯 Desempenho por Terminal:**")
for terminal in range(10):
    taxa_terminal = st.session_state.gestor.terminais_dominantes.get_taxa_acerto_terminal(terminal)
    tentativas = st.session_state.gestor.terminais_dominantes.desempenho_terminais[str(terminal)]['tentativas']
    if tentativas > 0:
        cor = "🟢" if taxa_terminal >= 40 else "🟡" if taxa_terminal >= 25 else "🔴"
        st.write(f"{cor} Terminal {terminal}: {taxa_terminal:.1f}% ({tentativas} tentativas)")

# Tendências atuais
tendencias = st.session_state.gestor.analisador.analisar_tendencias()
if tendencias:
    st.write("**🔍 Tendências Atuais:**")
    st.write(f"Terminais Quentes: {tendencias.get('terminais_quentes', [])}")
    st.write(f"Terminais Frios: {tendencias.get('terminais_frios', [])}")

# Histórico recente de conferências
if st.session_state.historico_desempenho:
    st.write("**📋 Últimas 5 conferências:**")
    for i, conf in enumerate(st.session_state.historico_desempenho[-5:]):
        resultado_str = "🟢" if conf['acerto'] else "🔴"
        st.write(f"{resultado_str} {conf['estrategia']}: Número {conf['numero_sorteado']}")

# --- Resto do código mantido igual ---
