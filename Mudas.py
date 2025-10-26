import streamlit as st
import json
import os
import requests
import logging
from collections import Counter, deque
from alertas import enviar_previsao, enviar_resultado
from streamlit_autorefresh import st_autorefresh

# =============================
# CONFIGURAÇÕES
# =============================
HISTORICO_PATH = "historico_coluna_duzia.json"
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =============================
# CLASSE PRINCIPAL DA ROLETA
# =============================
class RoletaInteligente:
    def __init__(self):
        self.race = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
        self.terminais = {
            '0': [0, 10, 20, 30], '1': [1, 11, 21, 31], '2': [2, 12, 22, 32],
            '3': [3, 13, 23, 33], '4': [4, 14, 24, 34], '5': [5, 15, 25, 35],
            '6': [6, 16, 26, 36], '7': [7, 17, 27], '8': [8, 18, 28], '9': [9, 19, 29]
        }
        self.vizinhos_race = {
            '0': [32, 15, 19, 4, 21, 2, 25, 17], '32': [0, 15, 19, 4, 21, 2, 25, 17],
            '15': [0, 32, 19, 4, 21, 2, 25, 17], '19': [0, 32, 15, 4, 21, 2, 25, 17],
            '4': [0, 32, 15, 19, 21, 2, 25, 17], '21': [0, 32, 15, 19, 4, 2, 25, 17],
            '2': [0, 32, 15, 19, 4, 21, 25, 17], '25': [17, 34, 6, 27, 13, 36, 11, 30],
            '17': [25, 34, 6, 27, 13, 36, 11, 30], '34': [25, 17, 6, 27, 13, 36, 11, 30],
            '6': [25, 17, 34, 27, 13, 36, 11, 30], '27': [25, 17, 34, 6, 13, 36, 11, 30],
            '13': [25, 17, 34, 6, 27, 36, 11, 30], '36': [25, 17, 34, 6, 27, 13, 11, 30],
            '11': [30, 8, 23, 10, 5, 24, 16, 33], '30': [11, 8, 23, 10, 5, 24, 16, 33],
            '8': [11, 30, 23, 10, 5, 24, 16, 33], '23': [11, 30, 8, 10, 5, 24, 16, 33],
            '10': [11, 30, 8, 23, 5, 24, 16, 33], '5': [11, 30, 8, 23, 10, 24, 16, 33],
            '24': [11, 30, 8, 23, 10, 5, 16, 33], '16': [33, 1, 20, 14, 31, 9, 22, 18],
            '33': [16, 1, 20, 14, 31, 9, 22, 18], '1': [16, 33, 20, 14, 31, 9, 22, 18],
            '20': [16, 33, 1, 14, 31, 9, 22, 18], '14': [16, 33, 1, 20, 31, 9, 22, 18],
            '31': [16, 33, 1, 20, 14, 9, 22, 18], '9': [16, 33, 1, 20, 14, 31, 22, 18],
            '22': [18, 29, 7, 28, 12, 35, 3, 26], '18': [22, 29, 7, 28, 12, 35, 3, 26],
            '29': [22, 18, 7, 28, 12, 35, 3, 26], '7': [22, 18, 29, 28, 12, 35, 3, 26],
            '28': [22, 18, 29, 7, 12, 35, 3, 26], '12': [22, 18, 29, 7, 28, 35, 3, 26],
            '35': [22, 18, 29, 7, 28, 12, 3, 26], '3': [26, 0, 32, 15, 19, 4, 21, 2],
            '26': [3, 0, 32, 15, 19, 4, 21, 2]
        }

    def get_vizinhos_proximos(self, numero, quantidade=6):
        """Retorna os vizinhos mais próximos no race"""
        if str(numero) not in self.vizinhos_race:
            return []
        return self.vizinhos_race[str(numero)][:quantidade]

    def get_terminal(self, numero):
        return numero % 10

    def get_numeros_por_terminal(self, terminal):
        return self.terminais.get(str(terminal), [])

# =============================
# ESTRATÉGIA INTELIGENTE COM 10 NÚMEROS
# =============================
class EstrategiaRoletaRevisada:
    def __init__(self):
        self.roleta = RoletaInteligente()
        self.historico = deque(maxlen=15)
        self.estatisticas = {
            'total_estrategias': 0,
            'estrategias_vencedoras': 0,
            'historico_apostas': []
        }

    def adicionar_numero(self, numero):
        self.historico.append(numero)

    def analisar_padrao_inteligente(self):
        """Estratégia mais conservadora e focada - ATÉ 10 NÚMEROS"""
        if len(self.historico) < 8:
            return None

        ultimos = list(self.historico)
        padrao = self._identificar_padrao_assertivo(ultimos)
        
        if padrao:
            # Limitar a 10 números no máximo
            numeros_apostar = padrao['numeros_apostar'][:10]
            return {
                'nome': padrao['nome'],
                'numeros_apostar': numeros_apostar,
                'gatilho': padrao['gatilho'],
                'confianca': padrao.get('confianca', 'Média')
            }
        return None

    def _identificar_padrao_assertivo(self, ultimos):
        """Identifica padrões com maior probabilidade real"""
        
        # PADRÃO 1: Repetição de Terminal com Confirmação (ATÉ 10 NÚMEROS)
        terminal_padrao = self._verificar_repeticao_terminal_confirmada(ultimos)
        if terminal_padrao:
            return terminal_padrao
        
        # PADRÃO 2: Sequência no Race (ATÉ 10 NÚMEROS)
        sequencia_padrao = self._verificar_sequencia_race(ultimos)
        if sequencia_padrao:
            return sequencia_padrao
        
        # PADRÃO 3: Zona Quente (ATÉ 10 NÚMEROS)
        zona_padrao = self._verificar_zona_quente(ultimos)
        if zona_padrao:
            return zona_padrao
        
        # PADRÃO 4: Padrões Midas (ATÉ 10 NÚMEROS)
        midas_padrao = self._verificar_padroes_midas(ultimos)
        if midas_padrao:
            return midas_padrao
        
        return None

    def _verificar_repeticao_terminal_confirmada(self, ultimos):
        """Padrão mais conservador - terminal aparece 3+ vezes nos últimos 8 números"""
        if len(ultimos) < 8:
            return None
            
        ultimos_8 = ultimos[-8:]
        terminais = [self.roleta.get_terminal(n) for n in ultimos_8]
        contador = Counter(terminais)
        
        # Buscar terminal que apareceu pelo menos 3 vezes
        for terminal, count in contador.most_common(2):
            if count >= 3:
                numeros_terminal = self.roleta.get_numeros_por_terminal(terminal)
                
                # Adicionar mais vizinhos (6 em vez de 2) para chegar a ~10 números
                ultimo_numero = ultimos[-1]
                if self.roleta.get_terminal(ultimo_numero) == terminal:
                    vizinhos = self.roleta.get_vizinhos_proximos(ultimo_numero, 6)
                    numeros_apostar = list(set(numeros_terminal + vizinhos))
                    
                    return {
                        'nome': 'Terminal Dominante Confirmado',
                        'numeros_apostar': numeros_apostar,
                        'gatilho': f'Terminal {terminal} repetido {count}x',
                        'confianca': 'Alta'
                    }
        return None

    def _verificar_sequencia_race(self, ultimos):
        """Verifica sequências próximas no race - ATÉ 10 NÚMEROS"""
        if len(ultimos) < 4:
            return None
            
        # Verificar se últimos 2-3 números estão próximos no race
        ultimos_3 = ultimos[-3:]
        posicoes = [self.roleta.race.index(n) for n in ultimos_3 if n in self.roleta.race]
        
        if len(posicoes) < 2:
            return None
            
        # Verificar proximidade (máximo 5 posições de diferença)
        diferencas = [abs(posicoes[i] - posicoes[i-1]) for i in range(1, len(posicoes))]
        if all(diff <= 5 for diff in diferencas):
            # Apostar nos vizinhos do último número (mais vizinhos - 8 em vez de 4)
            ultimo_numero = ultimos[-1]
            vizinhos = self.roleta.get_vizinhos_proximos(ultimo_numero, 8)
            
            # Adicionar também os números da sequência
            numeros_apostar = list(set(vizinhos + ultimos_3))
            
            return {
                'nome': 'Sequência Race',
                'numeros_apostar': numeros_apostar,
                'gatilho': f'Sequência próxima no Race: {ultimos_3}',
                'confianca': 'Média'
            }
        return None

    def _verificar_zona_quente(self, ultimos):
        """Identifica zonas quentes baseadas em frequência - ATÉ 10 NÚMEROS"""
        if len(ultimos) < 10:
            return None
            
        ultimos_10 = ultimos[-10:]
        
        # Agrupar por quadrantes no race (agora com mais números)
        quadrantes = {
            'Q1': self.roleta.race[0:10],   # Primeiros 10 números
            'Q2': self.roleta.race[10:20],  # Próximos 10 números  
            'Q3': self.roleta.race[20:30],  # Próximos 10 números
            'Q4': self.roleta.race[30:]     # Últimos números
        }
        
        # Contar frequência por quadrante
        freq_quadrantes = {}
        for nome, numeros in quadrantes.items():
            count = sum(1 for n in ultimos_10 if n in numeros)
            freq_quadrantes[nome] = count
        
        # Encontrar quadrante mais quente
        quadrante_quente = max(freq_quadrantes, key=freq_quadrantes.get)
        if freq_quadrantes[quadrante_quente] >= 4:  # Pelo menos 40% dos números
            numeros_quadrante = quadrantes[quadrante_quente]
            
            # Pegar mais números do quadrante (até 8) + 2 vizinhos do último número
            ultimo_numero = ultimos[-1]
            vizinhos = self.roleta.get_vizinhos_proximos(ultimo_numero, 2)
            numeros_apostar = list(set(numeros_quadrante[:8] + vizinhos))
            
            return {
                'nome': 'Zona Quente',
                'numeros_apostar': numeros_apostar,
                'gatilho': f'Quadrante {quadrante_quente} ({freq_quadrantes[quadrante_quente]}/10 números)',
                'confianca': 'Média'
            }
        return None

    def _verificar_padroes_midas(self, ultimos):
        """Implementa os padrões Midas do PDF - ATÉ 10 NÚMEROS"""
        if len(ultimos) < 5:
            return None
            
        ultimo_numero = ultimos[-1]
        historico_recente = ultimos[-5:]
        
        # Padrão do Zero
        if ultimo_numero in [0, 10, 20, 30]:
            count_zero = sum(1 for n in historico_recente if n in [0, 10, 20, 30])
            if count_zero >= 2:
                numeros_zero = [0, 10, 20, 30]
                vizinhos = []
                for num in numeros_zero:
                    vizinhos.extend(self.roleta.get_vizinhos_proximos(num, 3))
                numeros_apostar = list(set(numeros_zero + vizinhos))
                return {
                    'nome': 'Padrão do Zero',
                    'numeros_apostar': numeros_apostar,
                    'gatilho': f'Terminal 0 ativado ({count_zero}x)',
                    'confianca': 'Média'
                }

        # Padrão do Sete
        if ultimo_numero in [7, 17, 27]:
            count_sete = sum(1 for n in historico_recente if n in [7, 17, 27])
            if count_sete >= 2:
                numeros_sete = [7, 17, 27]
                vizinhos = []
                for num in numeros_sete:
                    vizinhos.extend(self.roleta.get_vizinhos_proximos(num, 4))
                numeros_apostar = list(set(numeros_sete + vizinhos))
                return {
                    'nome': 'Padrão do Sete',
                    'numeros_apostar': numeros_apostar,
                    'gatilho': f'Terminal 7 ativado ({count_sete}x)',
                    'confianca': 'Média'
                }

        # Padrão do Cinco
        if ultimo_numero in [5, 15, 25, 35]:
            count_cinco = sum(1 for n in historico_recente if n in [5, 15, 25, 35])
            if count_cinco >= 2:
                numeros_cinco = [5, 15, 25, 35]
                vizinhos = []
                for num in numeros_cinco:
                    vizinhos.extend(self.roleta.get_vizinhos_proximos(num, 3))
                numeros_apostar = list(set(numeros_cinco + vizinhos))
                return {
                    'nome': 'Padrão do Cinco',
                    'numeros_apostar': numeros_apostar,
                    'gatilho': f'Terminal 5 ativado ({count_cinco}x)',
                    'confianca': 'Média'
                }

        # Padrão Gêmeos
        if ultimo_numero in [11, 22, 33]:
            count_gemeos = sum(1 for n in historico_recente if n in [11, 22, 33])
            if count_gemeos >= 2:
                numeros_gemeos = [11, 22, 33]
                vizinhos = []
                for num in numeros_gemeos:
                    vizinhos.extend(self.roleta.get_vizinhos_proximos(num, 4))
                numeros_apostar = list(set(numeros_gemeos + vizinhos))
                return {
                    'nome': 'Padrão Gêmeos',
                    'numeros_apostar': numeros_apostar,
                    'gatilho': f'Gêmeos ativado ({count_gemeos}x)',
                    'confianca': 'Média'
                }

        return None

# =============================
# SISTEMA DE GESTÃO
# =============================
class SistemaRoleta:
    def __init__(self):
        self.estrategia = EstrategiaRoletaRevisada()
        self.previsao_ativa = None
        self.historico_desempenho = []
        self.acertos = 0
        self.erros = 0
        
    def processar_novo_numero(self, numero):
        # Conferir previsão anterior se existir
        if self.previsao_ativa:
            acerto = numero in self.previsao_ativa['numeros_apostar']
            self.historico_desempenho.append({
                'numero': numero,
                'acerto': acerto,
                'estrategia': self.previsao_ativa['nome'],
                'previsao': self.previsao_ativa['numeros_apostar']
            })
            
            if acerto:
                self.acertos += 1
                tocar_som_moeda()
            else:
                self.erros += 1
            
            self.previsao_ativa = None
        
        # Adicionar número e verificar nova estratégia
        self.estrategia.adicionar_numero(numero)
        nova_estrategia = self.estrategia.analisar_padrao_inteligente()
        
        if nova_estrategia:
            self.previsao_ativa = nova_estrategia
            # Enviar alerta
            msg = f"🎯 {nova_estrategia['nome']} ({nova_estrategia['confianca']})\n"
            msg += f"Gatilho: {nova_estrategia['gatilho']}\n"
            msg += f"Números: {', '.join(map(str, sorted(nova_estrategia['numeros_apostar'])))}"
            enviar_previsao(msg)

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
# APLICAÇÃO STREAMLIT
# =============================
st.set_page_config(page_title="IA Roleta — Estratégia 10 Números", layout="centered")
st.title("🎯 IA Roleta — Estratégia com 10 Números")

# Inicialização
if "sistema" not in st.session_state:
    st.session_state.sistema = SistemaRoleta()

if "historico" not in st.session_state:
    st.session_state.historico = json.load(open(HISTORICO_PATH)) if os.path.exists(HISTORICO_PATH) else []

# Entrada manual
st.subheader("✍️ Inserir Sorteios")
entrada = st.text_input("Digite números (0-36) separados por espaço:")
if st.button("Adicionar") and entrada:
    try:
        nums = [int(n) for n in entrada.split() if n.isdigit() and 0 <= int(n) <= 36]
        for n in nums:
            item = {"number": n, "timestamp": f"manual_{len(st.session_state.historico)}"}
            st.session_state.historico.append(item)
            st.session_state.sistema.processar_novo_numero(n)
        salvar_resultado_em_arquivo(st.session_state.historico)
        st.success(f"{len(nums)} números adicionados!")
        st.rerun()
    except Exception as e:
        st.error(f"Erro: {e}")

# Atualização automática
st_autorefresh(interval=3000, key="refresh")

# Buscar resultado da API
resultado = fetch_latest_result()
ultimo_ts = st.session_state.historico[-1]["timestamp"] if st.session_state.historico else None

if resultado and resultado.get("timestamp") and resultado["timestamp"] != ultimo_ts:
    numero_atual = resultado.get("number")
    if numero_atual is not None:
        st.session_state.historico.append(resultado)
        st.session_state.sistema.processar_novo_numero(numero_atual)
        salvar_resultado_em_arquivo(st.session_state.historico)

# Interface
st.subheader("🔁 Últimos Números")
if st.session_state.historico:
    st.write(" ".join(str(h["number"]) for h in st.session_state.historico[-10:]))

st.subheader("🎯 Previsão Ativa")
sistema = st.session_state.sistema

if sistema.previsao_ativa:
    previsao = sistema.previsao_ativa
    st.success(f"**{previsao['nome']}** - Confiança: {previsao['confianca']}")
    st.write(f"**Gatilho:** {previsao['gatilho']}")
    st.write(f"**Números para apostar ({len(previsao['numeros_apostar'])}):**")
    st.write(", ".join(map(str, sorted(previsao['numeros_apostar']))))
    st.info("⏳ Aguardando próximo sorteio para conferência...")
else:
    st.info("⏳ Analisando padrões para nova previsão...")

# Desempenho
st.subheader("📈 Desempenho com 10 Números")

total = sistema.acertos + sistema.erros
taxa = (sistema.acertos / total * 100) if total > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Acertos", sistema.acertos)
col2.metric("🔴 Erros", sistema.erros)
col3.metric("📊 Total", total)
col4.metric("✅ Taxa", f"{taxa:.1f}%")

# Análise de padrões
if sistema.historico_desempenho:
    st.write("**📊 Análise de Estratégias:**")
    estrategias = {}
    for resultado in sistema.historico_desempenho:
        nome = resultado['estrategia']
        if nome not in estrategias:
            estrategias[nome] = {'acertos': 0, 'total': 0}
        estrategias[nome]['total'] += 1
        if resultado['acerto']:
            estrategias[nome]['acertos'] += 1
    
    for nome, dados in estrategias.items():
        taxa_estrategia = (dados['acertos'] / dados['total'] * 100) if dados['total'] > 0 else 0
        cor = "🟢" if taxa_estrategia >= 40 else "🟡" if taxa_estrategia >= 25 else "🔴"
        st.write(f"{cor} {nome}: {dados['acertos']}/{dados['total']} ({taxa_estrategia:.1f}%)")

# Download histórico
if os.path.exists(HISTORICO_PATH):
    with open(HISTORICO_PATH, "r") as f:
        conteudo = f.read()
    st.download_button("📥 Baixar histórico", data=conteudo, file_name="historico_roleta.json")
