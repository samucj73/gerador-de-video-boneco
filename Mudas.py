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
        
    def get_vizinhos_zona(self, numero_central, quantidade=6):
        """Retorna 6 vizinhos antes e 6 depois do número central no race"""
        if numero_central not in self.race:
            return []
        
        posicao = self.race.index(numero_central)
        vizinhos = []
        
        # Pegar 6 números antes e 6 depois (incluindo o central = 13 números)
        for offset in range(-quantidade, quantidade + 1):
            vizinho = self.race[(posicao + offset) % len(self.race)]
            vizinhos.append(vizinho)
        
        return vizinhos

# =============================
# ESTRATÉGIA DAS ZONAS OTIMIZADA (VERSÃO 2.0)
# =============================
class EstrategiaZonas:
    def __init__(self):
        self.roleta = RoletaInteligente()
        self.historico = deque(maxlen=25)  # Aumentado para 25
        self.nome = "Estratégia das Zonas v2"
        
        # Zonas otimizadas
        self.zonas = {
            'Amarela': 2,   # Número central da zona amarela
            'Vermelha': 7,  # Número central da zona vermelha  
            'Azul': 10      # Número central da zona azul
        }
        
        # Pré-calcular os números de cada zona
        self.numeros_zonas = {}
        for nome, central in self.zonas.items():
            self.numeros_zonas[nome] = self.roleta.get_vizinhos_zona(central, 6)

        # Estatísticas de performance por zona
        self.stats_zonas = {zona: {'acertos': 0, 'tentativas': 0} for zona in self.zonas.keys()}

    def adicionar_numero(self, numero):
        self.historico.append(numero)
        self.atualizar_stats(numero)

    def atualizar_stats(self, ultimo_numero):
        """Atualiza estatísticas de performance das zonas"""
        # Verificar se havia previsão ativa e atualizar stats
        for zona, numeros in self.numeros_zonas.items():
            if ultimo_numero in numeros:
                self.stats_zonas[zona]['acertos'] += 1
            self.stats_zonas[zona]['tentativas'] += 1

    def get_zona_mais_quente(self):
        """Identifica a zona com melhor performance usando múltiplos critérios"""
        if len(self.historico) < 12:  # Aumentado o mínimo
            return None
            
        # Sistema de pontuação multi-critério
        zonas_score = {}
        
        for zona in self.zonas.keys():
            score = 0
            
            # Critério 1: Frequência geral (40% do score)
            freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
            percentual_geral = freq_geral / len(self.historico)
            score += percentual_geral * 40
            
            # Critério 2: Frequência recente (35% do score)
            ultimos_10 = list(self.historico)[-10:] if len(self.historico) >= 10 else list(self.historico)
            freq_recente = sum(1 for n in ultimos_10 if n in self.numeros_zonas[zona])
            percentual_recente = freq_recente / len(ultimos_10) if ultimos_10 else 0
            score += percentual_recente * 35
            
            # Critério 3: Performance histórica (25% do score)
            if self.stats_zonas[zona]['tentativas'] > 0:
                taxa_acerto = (self.stats_zonas[zona]['acertos'] / self.stats_zonas[zona]['tentativas']) * 100
                score += min(taxa_acerto * 0.25, 25)  # Normalizado para 25%
            
            zonas_score[zona] = score
        
        # Retorna a zona com maior score, desde que atinja threshold mínimo
        zona_vencedora = max(zonas_score, key=zonas_score.get) if zonas_score else None
        return zona_vencedora if zona_vencedora and zonas_score[zona_vencedora] >= 25 else None

    def analisar_zonas(self):
        """Versão otimizada com múltiplos critérios"""
        if len(self.historico) < 15:  # Aumentei o mínimo para mais confiabilidade
            return None

        zona_alvo = self.get_zona_mais_quente()
        
        if zona_alvo:
            numeros_apostar = self.numeros_zonas[zona_alvo]
            
            # Calcular confiança baseada em múltiplos fatores
            confianca = self.calcular_confianca(zona_alvo)
            
            return {
                'nome': f'Zona {zona_alvo}',
                'numeros_apostar': numeros_apostar,
                'gatilho': f'Zona {zona_alvo} - Score: {self.get_zona_score(zona_alvo):.1f} - Conf: {confianca}',
                'confianca': confianca,
                'zona': zona_alvo
            }
        
        return None

    def get_zona_score(self, zona):
        """Calcula o score atual de uma zona específica"""
        if len(self.historico) < 12:
            return 0
            
        score = 0
        # Frequência geral
        freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
        percentual_geral = freq_geral / len(self.historico)
        score += percentual_geral * 40
        
        # Frequência recente
        ultimos_10 = list(self.historico)[-10:] if len(self.historico) >= 10 else list(self.historico)
        freq_recente = sum(1 for n in ultimos_10 if n in self.numeros_zonas[zona])
        percentual_recente = freq_recente / len(ultimos_10) if ultimos_10 else 0
        score += percentual_recente * 35
        
        # Performance histórica
        if self.stats_zonas[zona]['tentativas'] > 0:
            taxa_acerto = (self.stats_zonas[zona]['acertos'] / self.stats_zonas[zona]['tentativas']) * 100
            score += min(taxa_acerto * 0.25, 25)
            
        return score

    def calcular_confianca(self, zona):
        """Calcula nível de confiança baseado em múltiplos indicadores"""
        indicadores = []
        
        # Indicador 1: Frequência geral
        freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
        perc_geral = (freq_geral / len(self.historico)) * 100
        if perc_geral > 40: 
            indicadores.append(3)
        elif perc_geral > 30: 
            indicadores.append(2)
        else: 
            indicadores.append(1)
        
        # Indicador 2: Frequência recente
        ultimos_8 = list(self.historico)[-8:]
        freq_recente = sum(1 for n in ultimos_8 if n in self.numeros_zonas[zona])
        perc_recente = (freq_recente / len(ultimos_8)) * 100 if ultimos_8 else 0
        if perc_recente > 50: 
            indicadores.append(3)
        elif perc_recente > 35: 
            indicadores.append(2)
        else: 
            indicadores.append(1)
        
        # Indicador 3: Performance histórica
        if self.stats_zonas[zona]['tentativas'] > 10:
            taxa = (self.stats_zonas[zona]['acertos'] / self.stats_zonas[zona]['tentativas']) * 100
            if taxa > 30: 
                indicadores.append(3)
            elif taxa > 20: 
                indicadores.append(2)
            else: 
                indicadores.append(1)
        else:
            indicadores.append(1)  # Default se não há dados suficientes
        
        score_confianca = sum(indicadores) / len(indicadores)
        
        if score_confianca >= 2.5: 
            return 'Muito Alta'
        elif score_confianca >= 2.0: 
            return 'Alta'
        elif score_confianca >= 1.5: 
            return 'Média'
        else: 
            return 'Baixa'

    def get_info_zonas(self):
        """Retorna informações sobre as zonas para display"""
        info = {}
        for zona, numeros in self.numeros_zonas.items():
            info[zona] = {
                'numeros': sorted(numeros),
                'quantidade': len(numeros),
                'central': self.zonas[zona]
            }
        return info

    def get_analise_detalhada(self):
        """Análise completa com insights acionáveis"""
        if len(self.historico) == 0:
            return "Aguardando dados..."
        
        analise = "🎯 ANÁLISE DETALHADA DAS ZONAS v2\n"
        analise += "=" * 50 + "\n"
        
        # Performance por zona
        analise += "📊 PERFORMANCE POR ZONA:\n"
        for zona in self.zonas.keys():
            tentativas = self.stats_zonas[zona]['tentativas']
            acertos = self.stats_zonas[zona]['acertos']
            taxa = (acertos / tentativas * 100) if tentativas > 0 else 0
            
            analise += f"📍 {zona}: {acertos}/{tentativas} → {taxa:.1f}%\n"
        
        analise += "\n📈 FREQUÊNCIA ATUAL:\n"
        for zona in self.zonas.keys():
            freq = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
            perc = (freq / len(self.historico)) * 100
            score = self.get_zona_score(zona)
            analise += f"📍 {zona}: {freq}/{len(self.historico)} → {perc:.1f}% | Score: {score:.1f}\n"
        
        # Recomendações
        zona_recomendada = self.get_zona_mais_quente()
        if zona_recomendada:
            analise += f"\n💡 RECOMENDAÇÃO: Zona {zona_recomendada}\n"
            analise += f"🎯 Números: {sorted(self.numeros_zonas[zona_recomendada])}\n"
            analise += f"📈 Confiança: {self.calcular_confianca(zona_recomendada)}\n"
            analise += f"🔥 Score: {self.get_zona_score(zona_recomendada):.1f}\n"
        else:
            analise += "\n⚠️  AGUARDAR: Nenhuma zona com confiança suficiente\n"
            analise += f"📋 Mínimo necessário: Score 25+ | Histórico: {len(self.historico)}/15 números\n"
        
        return analise

    def get_analise_atual(self):
        """Mantido para compatibilidade - usa a nova análise detalhada"""
        return self.get_analise_detalhada()

# =============================
# ESTRATÉGIA MIDAS (EXISTENTE - MANTIDA)
# =============================
class EstrategiaMidas:
    def __init__(self):
        self.roleta = RoletaInteligente()
        self.historico = deque(maxlen=15)
        self.terminais = {
            '0': [0, 10, 20, 30], '1': [1, 11, 21, 31], '2': [2, 12, 22, 32],
            '3': [3, 13, 23, 33], '4': [4, 14, 24, 34], '5': [5, 15, 25, 35],
            '6': [6, 16, 26, 36], '7': [7, 17, 27], '8': [8, 18, 28], '9': [9, 19, 29]
        }

    def adicionar_numero(self, numero):
        self.historico.append(numero)

    def analisar_midas(self):
        if len(self.historico) < 5:
            return None
            
        ultimo_numero = self.historico[-1]
        historico_recente = self.historico[-5:]

        # Padrão do Zero
        if ultimo_numero in [0, 10, 20, 30]:
            count_zero = sum(1 for n in historico_recente if n in [0, 10, 20, 30])
            if count_zero >= 1:
                return {
                    'nome': 'Padrão do Zero',
                    'numeros_apostar': [0, 10, 20, 30],
                    'gatilho': f'Terminal 0 ativado ({count_zero}x)',
                    'confianca': 'Média'
                }

        # Padrão do Sete
        if ultimo_numero in [7, 17, 27]:
            count_sete = sum(1 for n in historico_recente if n in [7, 17, 27])
            if count_sete >= 1:
                return {
                    'nome': 'Padrão do Sete',
                    'numeros_apostar': [7, 17, 27],
                    'gatilho': f'Terminal 7 ativado ({count_sete}x)',
                    'confianca': 'Média'
                }

        # Padrão do Cinco
        if ultimo_numero in [5, 15, 25, 35]:
            count_cinco = sum(1 for n in historico_recente if n in [5, 15, 25, 35])
            if count_cinco >= 1:
                return {
                    'nome': 'Padrão do Cinco',
                    'numeros_apostar': [5, 15, 25, 35],
                    'gatilho': f'Terminal 5 ativado ({count_cinco}x)',
                    'confianca': 'Média'
                }

        return None

# =============================
# SISTEMA DE GESTÃO COM MÚLTIPLAS ESTRATÉGIAS
# =============================
class SistemaRoletaCompleto:
    def __init__(self):
        self.estrategia_zonas = EstrategiaZonas()
        self.estrategia_midas = EstrategiaMidas()
        self.previsao_ativa = None
        self.historico_desempenho = []
        self.acertos = 0
        self.erros = 0
        self.estrategias_contador = {}
        self.modo_estrategia = "Todas"  # "Todas", "Apenas Zonas", "Apenas Midas"

    def set_modo_estrategia(self, modo):
        self.modo_estrategia = modo

    def processar_novo_numero(self, numero):
        # Conferir previsão anterior se existir
        if self.previsao_ativa:
            acerto = numero in self.previsao_ativa['numeros_apostar']
            
            # Atualizar contador de estratégias
            nome_estrategia = self.previsao_ativa['nome']
            if nome_estrategia not in self.estrategias_contador:
                self.estrategias_contador[nome_estrategia] = {'acertos': 0, 'total': 0}
            
            self.estrategias_contador[nome_estrategia]['total'] += 1
            if acerto:
                self.estrategias_contador[nome_estrategia]['acertos'] += 1
                self.acertos += 1
                tocar_som_moeda()
            else:
                self.erros += 1
            
            self.historico_desempenho.append({
                'numero': numero,
                'acerto': acerto,
                'estrategia': nome_estrategia,
                'previsao': self.previsao_ativa['numeros_apostar']
            })
            
            self.previsao_ativa = None
        
        # Adicionar número a todas as estratégias
        self.estrategia_zonas.adicionar_numero(numero)
        self.estrategia_midas.adicionar_numero(numero)
        
        # Verificar nova estratégia baseada no modo selecionado
        nova_estrategia = None
        
        if self.modo_estrategia == "Apenas Zonas":
            nova_estrategia = self.estrategia_zonas.analisar_zonas()
        elif self.modo_estrategia == "Apenas Midas":
            nova_estrategia = self.estrategia_midas.analisar_midas()
        else:  # Todas as estratégias
            # Prioridade para Zonas (mais conservadora)
            nova_estrategia = self.estrategia_zonas.analisar_zonas()
            if not nova_estrategia:
                nova_estrategia = self.estrategia_midas.analisar_midas()
        
        if nova_estrategia:
            self.previsao_ativa = nova_estrategia
            # Enviar alerta
            msg = f"🎯 {nova_estrategia['nome']} - {nova_estrategia['confianca']}\n"
            msg += f"🎲 Gatilho: {nova_estrategia['gatilho']}\n"
            msg += f"🔢 Números: {', '.join(map(str, sorted(nova_estrategia['numeros_apostar'])))}"
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
st.set_page_config(page_title="IA Roleta — Estratégia das Zonas v2", layout="centered")
st.title("🎯 IA Roleta — Estratégia das Zonas v2 + Midas")

# Inicialização
if "sistema" not in st.session_state:
    st.session_state.sistema = SistemaRoletaCompleto()

if "historico" not in st.session_state:
    st.session_state.historico = json.load(open(HISTORICO_PATH)) if os.path.exists(HISTORICO_PATH) else []

# Sidebar - Seleção de Estratégia
st.sidebar.title("⚙️ Configurações")
modo_estrategia = st.sidebar.selectbox(
    "🎯 Estratégia:",
    ["Todas as Estratégias", "Apenas Zonas", "Apenas Midas"],
    key="modo_estrategia"
)

# Aplicar modo selecionado
st.session_state.sistema.set_modo_estrategia(modo_estrategia)

# Informações sobre as Zonas
with st.sidebar.expander("📊 Informações das Zonas"):
    info_zonas = st.session_state.sistema.estrategia_zonas.get_info_zonas()
    for zona, dados in info_zonas.items():
        st.write(f"**Zona {zona}** (Núcleo: {dados['central']})")
        st.write(f"Números: {', '.join(map(str, dados['numeros']))}")
        st.write(f"Total: {dados['quantidade']} números")
        st.write("---")

# DEBUG: Análise detalhada das zonas
with st.sidebar.expander("🔍 Análise Detalhada - Zonas v2"):
    analise = st.session_state.sistema.estrategia_zonas.get_analise_detalhada()
    st.text_area("Análise detalhada:", analise, height=400)

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
else:
    st.write("Nenhum número registrado")

st.subheader("🎯 Previsão Ativa")
sistema = st.session_state.sistema

if sistema.previsao_ativa:
    previsao = sistema.previsao_ativa
    st.success(f"**{previsao['nome']}**")
    st.write(f"**Confiança:** {previsao['confianca']}")
    st.write(f"**Gatilho:** {previsao['gatilho']}")
    st.write(f"**Números para apostar ({len(previsao['numeros_apostar'])}):**")
    st.write(", ".join(map(str, sorted(previsao['numeros_apostar']))))
    st.info("⏳ Aguardando próximo sorteio para conferência...")
else:
    st.info(f"🎲 Analisando padrões ({modo_estrategia})...")

# Desempenho
st.subheader("📈 Desempenho")

total = sistema.acertos + sistema.erros
taxa = (sistema.acertos / total * 100) if total > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Acertos", sistema.acertos)
col2.metric("🔴 Erros", sistema.erros)
col3.metric("📊 Total", total)
col4.metric("✅ Taxa", f"{taxa:.1f}%")

# Análise detalhada por estratégia
if sistema.estrategias_contador:
    st.write("**📊 Performance por Estratégia:**")
    for nome, dados in sistema.estrategias_contador.items():
        if dados['total'] > 0:
            taxa_estrategia = (dados['acertos'] / dados['total'] * 100)
            cor = "🟢" if taxa_estrategia >= 50 else "🟡" if taxa_estrategia >= 30 else "🔴"
            st.write(f"{cor} {nome}: {dados['acertos']}/{dados['total']} ({taxa_estrategia:.1f}%)")

# Últimas conferências
if sistema.historico_desempenho:
    st.write("**🔍 Últimas 5 Conferências:**")
    for i, resultado in enumerate(sistema.historico_desempenho[-5:]):
        emoji = "🎉" if resultado['acerto'] else "❌"
        st.write(f"{emoji} {resultado['estrategia']}: Número {resultado['numero']}")

# Download histórico
if os.path.exists(HISTORICO_PATH):
    with open(HISTORICO_PATH, "r") as f:
        conteudo = f.read()
    st.download_button("📥 Baixar histórico", data=conteudo, file_name="historico_roleta.json")
