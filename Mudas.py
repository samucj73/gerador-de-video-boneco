import streamlit as st
import json
import os
import requests
import logging
import numpy as np
import pandas as pd
from collections import Counter, deque
import joblib
from streamlit_autorefresh import st_autorefresh

# =============================
# CONFIGURAÇÕES
# =============================
HISTORICO_PATH = "historico_coluna_duzia.json"
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =============================
# SISTEMA ULTRA CONSERVADOR - PADRÕES ESPECÍFICOS
# =============================
class SistemaUltraAssertivo:
    def __init__(self):
        self.historico = deque(maxlen=50)
        self.previsao_ativa = None
        self.acertos = 0
        self.erros = 0
        self.historico_desempenho = []
        
        # Estatísticas de performance
        self.stats_estrategias = {}
        
    def adicionar_numero(self, numero):
        if isinstance(numero, dict) and 'number' in numero:
            numero_real = numero['number']
        else:
            numero_real = numero
            
        self.historico.append(numero_real)
        
        # Conferir previsão anterior
        self._conferir_previsao_anterior(numero_real)
        
        # Gerar nova previsão
        self._gerar_nova_previsao()
        
    def _conferir_previsao_anterior(self, numero_sorteado):
        if self.previsao_ativa:
            acerto = numero_sorteado in self.previsao_ativa['numeros_apostar']
            
            estrategia_nome = self.previsao_ativa['nome']
            if estrategia_nome not in self.stats_estrategias:
                self.stats_estrategias[estrategia_nome] = {'acertos': 0, 'total': 0}
                
            self.stats_estrategias[estrategia_nome]['total'] += 1
            
            if acerto:
                self.acertos += 1
                self.stats_estrategias[estrategia_nome]['acertos'] += 1
                enviar_resultado(f"🎉 ACERTO! Número {numero_sorteado} - {estrategia_nome}")
            else:
                self.erros += 1
                enviar_resultado(f"❌ ERRO! Número {numero_sorteado} - {estrategia_nome}")
                
            self.historico_desempenho.append({
                'numero': numero_sorteado,
                'acerto': acerto,
                'estrategia': estrategia_nome,
                'previsao': self.previsao_ativa['numeros_apostar']
            })
            
            self.previsao_ativa = None
    
    def _gerar_nova_previsao(self):
        """Gera previsões usando apenas estratégias ULTRA conservadoras"""
        if len(self.historico) < 5:
            return
            
        estrategias = [
            self._estrategia_repeticao_imediata(),
            self._estrategia_duplas_alternadas(),
            self._estrategia_zeramento_sequencial(),
            self._estrategia_padrao_fibonacci(),
            self._estrategia_mirror_numbers()
        ]
        
        # Escolher a estratégia com maior confiança
        estrategia_escolhida = None
        for estrategia in estrategias:
            if estrategia and (not estrategia_escolhida or estrategia['confianca_score'] > estrategia_escolhida['confianca_score']):
                estrategia_escolhida = estrategia
                
        if estrategia_escolhida and estrategia_escolhida['confianca_score'] >= 8:
            self.previsao_ativa = estrategia_escolhida
            
            # Enviar alerta
            msg = f"🎯 {estrategia_escolhida['nome']} - CONFIANÇA: {estrategia_escolhida['confianca']}\n"
            msg += f"📊 {estrategia_escolhida['gatilho']}\n"
            msg += f"🔢 NÚMEROS: {', '.join(map(str, estrategia_escolhida['numeros_apostar']))}"
            enviar_previsao(msg)
            
            # Alerta Telegram
            self._enviar_alerta_telegram_ultra(estrategia_escolhida)
    
    def _estrategia_repeticao_imediata(self):
        """Estratégia: Números que se repetem em sequências curtas"""
        if len(self.historico) < 6:
            return None
            
        ultimos_6 = list(self.historico)[-6:]
        
        # Verificar repetição nos últimos 2-3 sorteios
        for i in range(len(ultimos_6) - 2):
            if ultimos_6[i] == ultimos_6[i+1]:
                return {
                    'nome': 'REPETIÇÃO IMEDIATA',
                    'numeros_apostar': [ultimos_6[i]],
                    'gatilho': f'Número {ultimos_6[i]} repetido consecutivamente',
                    'confianca': 'MUITO ALTA',
                    'confianca_score': 9
                }
                
        # Verificar padrão A-B-A
        if len(ultimos_6) >= 3 and ultimos_6[-3] == ultimos_6[-1]:
            return {
                'nome': 'PADRÃO A-B-A',
                'numeros_apostar': [ultimos_6[-3]],
                'gatilho': f'Padrão {ultimos_6[-3]}-{ultimos_6[-2]}-{ultimos_6[-3]} identificado',
                'confianca': 'ALTA',
                'confianca_score': 8
            }
            
        return None
    
    def _estrategia_duplas_alternadas(self):
        """Estratégia: Duplas que se alternam"""
        if len(self.historico) < 8:
            return None
            
        ultimos_8 = list(self.historico)[-8:]
        
        # Verificar padrão de duplas (ex: 5,12,5,12)
        for i in range(len(ultimos_8) - 3):
            if (ultimos_8[i] == ultimos_8[i+2] and 
                ultimos_8[i+1] == ultimos_8[i+3] and
                ultimos_8[i] != ultimos_8[i+1]):
                return {
                    'nome': 'DUPLAS ALTERNADAS',
                    'numeros_apostar': [ultimos_8[i], ultimos_8[i+1]],
                    'gatilho': f'Dupla {ultimos_8[i]}-{ultimos_8[i+1]} alternando',
                    'confianca': 'ALTA',
                    'confianca_score': 8
                }
                
        return None
    
    def _estrategia_zeramento_sequencial(self):
        """Estratégia: Sequências que "zeram" ou padrões matemáticos"""
        if len(self.historico) < 5:
            return None
            
        ultimos_5 = list(self.historico)[-5:]
        
        # Verificar sequência de mesma dezena (ex: 5,15,25,35)
        ultima_dezena = ultimos_5[-1] % 10
        count_mesma_dezena = sum(1 for n in ultimos_5 if n % 10 == ultima_dezena)
        
        if count_mesma_dezena >= 3:
            proximo_numero = (ultimos_5[-1] + 10) % 37
            if proximo_numero <= 36:
                return {
                    'nome': 'SEQUÊNCIA DE DEZENA',
                    'numeros_apostar': [proximo_numero],
                    'gatilho': f'Sequência de dezena {ultima_dezena} ({count_mesma_dezena}x)',
                    'confianca': 'ALTA',
                    'confianca_score': 8
                }
        
        # Verificar sequência aritmética simples
        if len(ultimos_5) >= 3:
            diferencas = [ultimos_5[i+1] - ultimos_5[i] for i in range(len(ultimos_5)-1)]
            if len(set(diferencas[-2:])) == 1:  # Últimas 2 diferenças iguais
                proximo = ultimos_5[-1] + diferencas[-1]
                if 0 <= proximo <= 36:
                    return {
                        'nome': 'SEQUÊNCIA ARITMÉTICA',
                        'numeros_apostar': [proximo],
                        'gatilho': f'Progressão +{diferencas[-1]} identificada',
                        'confianca': 'MÉDIA-ALTA',
                        'confianca_score': 7
                    }
                    
        return None
    
    def _estrategia_padrao_fibonacci(self):
        """Estratégia: Padrões baseados em sequências Fibonacci-like"""
        if len(self.historico) < 6:
            return None
            
        ultimos_6 = list(self.historico)[-6:]
        
        # Verificar padrão de soma (ex: 1,3,4,7,11...)
        for i in range(len(ultimos_6) - 2):
            if ultimos_6[i+2] == (ultimos_6[i] + ultimos_6[i+1]) % 37:
                proximo = (ultimos_6[i+1] + ultimos_6[i+2]) % 37
                if proximo <= 36:
                    return {
                        'nome': 'PADRÃO FIBONACCI',
                        'numeros_apostar': [proximo],
                        'gatilho': f'Sequência Fibonacci: {ultimos_6[i]}+{ultimos_6[i+1]}→{ultimos_6[i+2]}',
                        'confianca': 'ALTA',
                        'confianca_score': 8
                    }
                    
        return None
    
    def _estrategia_mirror_numbers(self):
        """Estratégia: Números espelho (ex: 12 e 21, 13 e 31)"""
        if len(self.historico) < 4:
            return None
            
        ultimo = self.historico[-1]
        
        # Calcular número espelho (inverter dígitos)
        if ultimo < 10:
            mirror = int(str(ultimo) + '0') if ultimo != 0 else 0
        else:
            digits = str(ultimo)
            mirror = int(digits[1] + digits[0])
            
        if mirror <= 36 and mirror != ultimo:
            # Verificar se o espelho apareceu recentemente
            ultimos_10 = list(self.historico)[-10:-1]  # Excluir o último
            if mirror in ultimos_10:
                return {
                    'nome': 'NÚMEROS ESPELHO',
                    'numeros_apostar': [mirror],
                    'gatilho': f'Espelho de {ultimo} → {mirror}',
                    'confianca': 'MÉDIA',
                    'confianca_score': 6
                }
                
        return None
    
    def _enviar_alerta_telegram_ultra(self, estrategia):
        """Envia alerta ULTRA conservador para Telegram"""
        try:
            if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
                token = st.session_state.telegram_token
                chat_id = st.session_state.telegram_chat_id
                
                if token and chat_id:
                    mensagem = f"""
🎯 <b>ALERTA ULTRA CONSERVADOR</b>

🏆 <b>Estratégia:</b> {estrategia['nome']}
💎 <b>Confiança:</b> {estrategia['confianca']}
📊 <b>Gatilho:</b> {estrategia['gatilho']}

🎲 <b>NÚMERO PARA APOSTAR:</b>
{estrategia['numeros_apostar'][0]}

📈 <b>Performance do Sistema:</b>
{self.acertos}/{self.acertos + self.erros} acertos ({self._calcular_taxa_acertos():.1f}%)

⚡ <b>ENTRADA ALTAMENTE SELETIVA - MÁXIMA ASSERTIVIDADE</b>
"""
                    
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {
                        "chat_id": chat_id,
                        "text": mensagem,
                        "parse_mode": "HTML"
                    }
                    
                    requests.post(url, json=payload, timeout=10)
        except Exception as e:
            logging.error(f"Erro no alerta ultra: {e}")
    
    def _calcular_taxa_acertos(self):
        total = self.acertos + self.erros
        return (self.acertos / total * 100) if total > 0 else 0.0

# =============================
# ESTRATÉGIA DE GESTÃO DE BANCA
# =============================
class GestaoBanca:
    def __init__(self):
        self.saldo = 1000  # Saldo inicial
        self.aposta_base = 5
        self.sequencia_perdas = 0
        self.max_sequencia_perdas = 5
        
    def calcular_aposta(self, confianca):
        """Calcula valor da aposta baseado na confiança e sequência"""
        multiplicador = 1
        
        if confianca in ['MUITO ALTA', 'ALTA']:
            multiplicador = 2
        elif confianca == 'MÉDIA-ALTA':
            multiplicador = 1.5
            
        # Martingale moderado
        if self.sequencia_perdas > 0:
            multiplicador *= min(2 ** self.sequencia_perdas, 8)  # Limite de 8x
            
        return int(self.aposta_base * multiplicador)
    
    def registrar_resultado(self, acerto):
        """Registra resultado e atualiza sequência"""
        if acerto:
            self.sequencia_perdas = 0
            self.saldo += self.calcular_aposta('ALTA') * 35  # Ganho da roleta
        else:
            self.sequencia_perdas += 1
            self.saldo -= self.calcular_aposta('ALTA')
            
    def get_status(self):
        return {
            'saldo': self.saldo,
            'sequencia_perdas': self.sequencia_perdas,
            'proxima_aposta': self.calcular_aposta('ALTA')
        }

# =============================
# FUNÇÕES DE NOTIFICAÇÃO
# =============================
def enviar_previsao(mensagem):
    try:
        st.toast(f"🎯 {mensagem}", icon="🔥")
        st.warning(f"🔔 PREVISÃO: {mensagem}")
        
        if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
            if st.session_state.telegram_token and st.session_state.telegram_chat_id:
                enviar_telegram(mensagem)
    except Exception as e:
        logging.error(f"Erro ao enviar previsão: {e}")

def enviar_resultado(mensagem):
    try:
        st.toast(f"🎲 {mensagem}", icon="✅")
        st.success(f"📢 RESULTADO: {mensagem}")
        
        if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
            if st.session_state.telegram_token and st.session_state.telegram_chat_id:
                enviar_telegram(f"RESULTADO: {mensagem}")
    except Exception as e:
        logging.error(f"Erro ao enviar resultado: {e}")

def enviar_telegram(mensagem):
    try:
        token = st.session_state.telegram_token
        chat_id = st.session_state.telegram_chat_id
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": mensagem,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logging.info("Mensagem enviada para Telegram")
    except Exception as e:
        logging.error(f"Erro no Telegram: {e}")

# =============================
# FUNÇÕES AUXILIARES
# =============================
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
st.set_page_config(page_title="Roleta - Sistema Ultra Assertivo", layout="centered")
st.title("🎯 SISTEMA ULTRA ASSERTIVO - ROULETTE")

# Inicialização
if "sistema" not in st.session_state:
    st.session_state.sistema = SistemaUltraAssertivo()

if "banca" not in st.session_state:
    st.session_state.banca = GestaoBanca()

if "historico" not in st.session_state:
    if os.path.exists(HISTORICO_PATH):
        try:
            with open(HISTORICO_PATH, "r") as f:
                st.session_state.historico = json.load(f)
        except:
            st.session_state.historico = []
    else:
        st.session_state.historico = []

# Configurações do Telegram
if "telegram_token" not in st.session_state:
    st.session_state.telegram_token = ""
if "telegram_chat_id" not in st.session_state:
    st.session_state.telegram_chat_id = ""

# Sidebar
st.sidebar.title("⚙️ CONFIGURAÇÕES ULTRA")

with st.sidebar.expander("🔔 Telegram", expanded=False):
    telegram_token = st.text_input(
        "Bot Token:",
        value=st.session_state.telegram_token,
        type="password"
    )
    
    telegram_chat_id = st.text_input(
        "Chat ID:",
        value=st.session_state.telegram_chat_id
    )
    
    if st.button("Salvar Configurações"):
        st.session_state.telegram_token = telegram_token
        st.session_state.telegram_chat_id = telegram_chat_id
        st.success("✅ Configurações salvas!")

# Gestão de Banca
with st.sidebar.expander("💰 GESTÃO DE BANCA", expanded=True):
    banca = st.session_state.banca.get_status()
    
    st.metric("💵 Saldo Atual", f"R$ {banca['saldo']}")
    st.metric("📉 Sequência de Perdas", banca['sequencia_perdas'])
    st.metric("🎯 Próxima Aposta", f"R$ {banca['proxima_aposta']}")
    
    if st.button("🔄 Reiniciar Banca"):
        st.session_state.banca = GestaoBanca()
        st.success("Banca reiniciada!")

# Entrada manual
st.subheader("✍️ INSERIR NÚMEROS MANUALMENTE")
col1, col2 = st.columns([3, 1])

with col1:
    entrada = st.text_input("Números (0-36) separados por espaço:", key="entrada_manual")

with col2:
    if st.button("🎯 Adicionar", use_container_width=True) and entrada:
        try:
            nums = [int(n) for n in entrada.split() if n.isdigit() and 0 <= int(n) <= 36]
            for n in nums:
                item = {"number": n, "timestamp": f"manual_{len(st.session_state.historico)}"}
                st.session_state.historico.append(item)
                st.session_state.sistema.adicionar_numero(n)
            salvar_resultado_em_arquivo(st.session_state.historico)
            st.success(f"✅ {len(nums)} números adicionados!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Erro: {e}")

# Atualização automática
st_autorefresh(interval=3000, key="refresh")

# Buscar resultado da API
resultado = fetch_latest_result()
if st.session_state.historico:
    ultimo_ts = st.session_state.historico[-1].get("timestamp") if st.session_state.historico else None
else:
    ultimo_ts = None

if resultado and resultado.get("timestamp") and resultado["timestamp"] != ultimo_ts:
    numero_atual = resultado.get("number")
    if numero_atual is not None:
        st.session_state.historico.append(resultado)
        st.session_state.sistema.adicionar_numero(resultado)
        salvar_resultado_em_arquivo(st.session_state.historico)

# Interface principal
st.subheader("🔁 ÚLTIMOS NÚMEROS")
if st.session_state.historico:
    ultimos_12 = st.session_state.historico[-12:]
    numeros_display = []
    for item in ultimos_12:
        num = item['number'] if isinstance(item, dict) else item
        numeros_display.append(f"{num:02d}")
    
    # Mostrar em formato de grid
    cols = st.columns(6)
    for i, num_str in enumerate(numeros_display[-12:]):
        with cols[i % 6]:
            st.metric(f"{i+1}º", num_str)
else:
    st.info("⏳ Aguardando dados...")

st.subheader("🎯 PREVISÃO ATIVA")
sistema = st.session_state.sistema

if sistema.previsao_ativa:
    previsao = sistema.previsao_ativa
    banca_info = st.session_state.banca.get_status()
    
    st.success(f"**{previsao['nome']}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Confiança", previsao['confianca'])
    with col2:
        st.metric("🔢 Números", len(previsao['numeros_apostar']))
    with col3:
        st.metric("💰 Aposta", f"R$ {banca_info['proxima_aposta']}")
    
    st.write(f"**📊 Gatilho:** {previsao['gatilho']}")
    st.write(f"**🎲 Números para apostar:**")
    
    # Destacar números
    for num in previsao['numeros_apostar']:
        st.write(f"**🎯 {num}**")
    
    st.info("⏳ Aguardando próximo sorteio...")
else:
    st.info("🎯 Analisando padrões ultra conservadores...")
    st.write("📊 **Estratégias ativas:**")
    st.write("• 🔁 Repetição Imediata")
    st.write("• 🔀 Duplas Alternadas") 
    st.write("• 📈 Sequências Aritméticas")
    st.write("• 🔢 Padrão Fibonacci")
    st.write("• 👥 Números Espelho")

# Desempenho
st.subheader("📈 DESEMPENHO DETALHADO")

total = sistema.acertos + sistema.erros
taxa = sistema._calcular_taxa_acertos()

col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Acertos", sistema.acertos)
col2.metric("❌ Erros", sistema.erros)
col3.metric("📊 Total", total)
col4.metric("✅ Taxa", f"{taxa:.1f}%")

# Performance por estratégia
if sistema.stats_estrategias:
    st.write("**📊 PERFORMANCE POR ESTRATÉGIA:**")
    for nome, dados in sistema.stats_estrategias.items():
        if dados['total'] > 0:
            taxa_estrategia = (dados['acertos'] / dados['total'] * 100)
            # Cor baseada na performance
            if taxa_estrategia >= 40:
                emoji = "🔥"
            elif taxa_estrategia >= 20:
                emoji = "⚡"
            else:
                emoji = "⚠️"
                
            st.write(f"{emoji} **{nome}**: {dados['acertos']}/{dados['total']} ({taxa_estrategia:.1f}%)")

# Últimas conferências
if sistema.historico_desempenho:
    st.write("**🔍 ÚLTIMAS 8 CONFERÊNCIAS:**")
    cols = st.columns(4)
    for i, resultado in enumerate(sistema.historico_desempenho[-8:]):
        with cols[i % 4]:
            if resultado['acerto']:
                st.success(f"🎉 {resultado['estrategia']}\nNº {resultado['numero']}")
            else:
                st.error(f"❌ {resultado['estrategia']}\nNº {resultado['numero']}")

# Estatísticas avançadas
with st.expander("📊 ESTATÍSTICAS AVANÇADAS"):
    if st.session_state.historico:
        todos_numeros = []
        for item in st.session_state.historico:
            if isinstance(item, dict) and 'number' in item:
                todos_numeros.append(item['number'])
            elif isinstance(item, (int, float)):
                todos_numeros.append(int(item))
        
        if todos_numeros:
            contador = Counter(todos_numeros)
            mais_frequentes = contador.most_common(5)
            
            st.write("**🔢 NÚMEROS MAIS FREQUENTES:**")
            for num, count in mais_frequentes:
                st.write(f"• **{num}**: {count} vezes ({count/len(todos_numeros)*100:.1f}%)")

# Download histórico
if os.path.exists(HISTORICO_PATH):
    with open(HISTORICO_PATH, "r") as f:
        conteudo = f.read()
    st.download_button("📥 BAIXAR HISTÓRICO COMPLETO", 
                      data=conteudo, 
                      file_name="historico_roleta_ultra.json",
                      use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🎯 **SISTEMA ULTRA ASSERTIVO** - Estratégias conservadoras para máxima eficiência")
