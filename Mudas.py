import streamlit as st
import json
import os
import requests
import logging
import numpy as np
import pandas as pd
from collections import Counter, deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import joblib
from streamlit_autorefresh import st_autorefresh
import pickle
from datetime import datetime

# =============================
# CONFIGURAÇÕES DE PERSISTÊNCIA
# =============================
SESSION_DATA_PATH = "session_data.pkl"
HISTORICO_PATH = "historico_coluna_duzia.json"
ML_MODEL_PATH = "ml_roleta_model.pkl"
SCALER_PATH = "ml_scaler.pkl"
META_PATH = "ml_meta.pkl"

def salvar_sessao():
    """Salva todos os dados da sessão em arquivo"""
    try:
        session_data = {
            'historico': st.session_state.historico,
            'telegram_token': st.session_state.telegram_token,
            'telegram_chat_id': st.session_state.telegram_chat_id,
            'sistema_acertos': st.session_state.sistema.acertos,
            'sistema_erros': st.session_state.sistema.erros,
            'sistema_estrategias_contador': st.session_state.sistema.estrategias_contador,
            'sistema_historico_desempenho': st.session_state.sistema.historico_desempenho,
            'sistema_contador_sorteios_global': st.session_state.sistema.contador_sorteios_global,
            'sistema_sequencia_erros': st.session_state.sistema.sequencia_erros,
            'sistema_ultima_estrategia_erro': st.session_state.sistema.ultima_estrategia_erro,
            # Dados da estratégia Zonas
            'zonas_historico': list(st.session_state.sistema.estrategia_zonas.historico),
            'zonas_stats': st.session_state.sistema.estrategia_zonas.stats_zonas,
            # Dados da estratégia Midas
            'midas_historico': list(st.session_state.sistema.estrategia_midas.historico),
            # Dados da estratégia ML
            'ml_historico': list(st.session_state.sistema.estrategia_ml.historico),
            'ml_contador_sorteios': st.session_state.sistema.estrategia_ml.contador_sorteios,
            'ml_sequencias_padroes': st.session_state.sistema.estrategia_ml.sequencias_padroes,
            'ml_metricas_padroes': st.session_state.sistema.estrategia_ml.metricas_padroes,
            'estrategia_selecionada': st.session_state.sistema.estrategia_selecionada,
            # Novos dados do sistema de rotação inteligente
            'rotacao_performance': st.session_state.sistema.rotacao_inteligente.performance_historica,
            'aprendizado_horario': st.session_state.sistema.aprendizado_continuo.performance_por_horario
        }
        
        with open(SESSION_DATA_PATH, 'wb') as f:
            pickle.dump(session_data, f)
        
        logging.info("✅ Sessão salva com sucesso")
    except Exception as e:
        logging.error(f"❌ Erro ao salvar sessão: {e}")

def carregar_sessao():
    """Carrega todos os dados da sessão do arquivo"""
    try:
        if os.path.exists(SESSION_DATA_PATH):
            with open(SESSION_DATA_PATH, 'rb') as f:
                session_data = pickle.load(f)
            
            # Restaurar dados básicos
            st.session_state.historico = session_data.get('historico', [])
            st.session_state.telegram_token = session_data.get('telegram_token', '')
            st.session_state.telegram_chat_id = session_data.get('telegram_chat_id', '')
            
            # Restaurar sistema
            if 'sistema' in st.session_state:
                st.session_state.sistema.acertos = session_data.get('sistema_acertos', 0)
                st.session_state.sistema.erros = session_data.get('sistema_erros', 0)
                st.session_state.sistema.estrategias_contador = session_data.get('sistema_estrategias_contador', {})
                st.session_state.sistema.historico_desempenho = session_data.get('sistema_historico_desempenho', [])
                st.session_state.sistema.contador_sorteios_global = session_data.get('sistema_contador_sorteios_global', 0)
                st.session_state.sistema.sequencia_erros = session_data.get('sistema_sequencia_erros', 0)
                st.session_state.sistema.ultima_estrategia_erro = session_data.get('sistema_ultima_estrategia_erro', '')
                st.session_state.sistema.estrategia_selecionada = session_data.get('estrategia_selecionada', 'Zonas')
                
                # Restaurar estratégia Zonas
                zonas_historico = session_data.get('zonas_historico', [])
                st.session_state.sistema.estrategia_zonas.historico = deque(zonas_historico, maxlen=70)
                st.session_state.sistema.estrategia_zonas.stats_zonas = session_data.get('zonas_stats', {
                    'Vermelha': {'acertos': 0, 'tentativas': 0, 'sequencia_atual': 0, 'sequencia_maxima': 0, 'performance_media': 0},
                    'Azul': {'acertos': 0, 'tentativas': 0, 'sequencia_atual': 0, 'sequencia_maxima': 0, 'performance_media': 0},
                    'Amarela': {'acertos': 0, 'tentativas': 0, 'sequencia_atual': 0, 'sequencia_maxima': 0, 'performance_media': 0}
                })
                
                # Restaurar estratégia Midas
                midas_historico = session_data.get('midas_historico', [])
                st.session_state.sistema.estrategia_midas.historico = deque(midas_historico, maxlen=15)
                
                # Restaurar estratégia ML
                ml_historico = session_data.get('ml_historico', [])
                st.session_state.sistema.estrategia_ml.historico = deque(ml_historico, maxlen=30)
                st.session_state.sistema.estrategia_ml.contador_sorteios = session_data.get('ml_contador_sorteios', 0)
                st.session_state.sistema.estrategia_ml.sequencias_padroes = session_data.get('ml_sequencias_padroes', {
                    'sequencias_ativas': {},
                    'historico_sequencias': [],
                    'padroes_detectados': []
                })
                st.session_state.sistema.estrategia_ml.metricas_padroes = session_data.get('ml_metricas_padroes', {
                    'padroes_detectados_total': 0,
                    'padroes_acertados': 0,
                    'padroes_errados': 0,
                    'eficiencia_por_tipo': {},
                    'historico_validacao': []
                })
                
                # Restaurar sistema de rotação inteligente
                rotacao_performance = session_data.get('rotacao_performance', {})
                st.session_state.sistema.rotacao_inteligente.performance_historica = rotacao_performance
                
                # Restaurar aprendizado contínuo
                aprendizado_horario = session_data.get('aprendizado_horario', {})
                st.session_state.sistema.aprendizado_continuo.performance_por_horario = aprendizado_horario
            
            logging.info("✅ Sessão carregada com sucesso")
            return True
    except Exception as e:
        logging.error(f"❌ Erro ao carregar sessão: {e}")
    return False

def limpar_sessao():
    """Limpa todos os dados da sessão"""
    try:
        if os.path.exists(SESSION_DATA_PATH):
            os.remove(SESSION_DATA_PATH)
        if os.path.exists(HISTORICO_PATH):
            os.remove(HISTORICO_PATH)
        # Limpar session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
        logging.info("🗑️ Sessão limpa com sucesso")
    except Exception as e:
        logging.error(f"❌ Erro ao limpar sessão: {e}")

# =============================
# CONFIGURAÇÕES DE NOTIFICAÇÃO - SUPER SIMPLIFICADAS
# =============================
def enviar_previsao_super_simplificada(previsao):
    """Envia notificação de previsão super simplificada"""
    try:
        nome_estrategia = previsao['nome']
        
        if 'Zonas' in nome_estrategia:
            # Mensagem super simplificada para Zonas - apenas o número da zona
            zona = previsao.get('zona', '')
            # Mostrar número do núcleo
            if zona == 'Vermelha':
                mensagem = "📍 Núcleo 7"
            elif zona == 'Azul':
                mensagem = "📍 Núcleo 10"
            elif zona == 'Amarela':
                mensagem = "📍 Núcleo 2"
            else:
                mensagem = f"📍 Núcleo {zona}"
            
        elif 'Machine Learning' in nome_estrategia or 'ML' in nome_estrategia or 'CatBoost' in nome_estrategia:
            # CORREÇÃO: Verificar múltiplas possibilidades do nome ML
            zona_ml = previsao.get('zona_ml', '')
            
            # NOVA LÓGICA: Verificar se há números específicos na previsão
            numeros_apostar = previsao.get('numeros_apostar', [])
            
            # Verificar se o número 2 está nos números para apostar
            if 2 in numeros_apostar:
                mensagem = "🤖 Zona 2"
            # Verificar se o número 7 está nos números para apostar
            elif 7 in numeros_apostar:
                mensagem = "🤖 Zona 7"
            # Verificar se o número 10 está nos números para apostar
            elif 10 in numeros_apostar:
                mensagem = "🤖 Zona 10"
            else:
                # Fallback para a lógica original
                if zona_ml == 'Vermelha':
                    mensagem = "🤖 Zona 7"
                elif zona_ml == 'Azul':
                    mensagem = "🤖 Zona 10"  
                elif zona_ml == 'Amarela':
                    mensagem = "🤖 Zona 2"
                else:
                    mensagem = f"🤖 Zona {zona_ml}"
            
        else:
            # Mensagem para Midas
            mensagem = f"💰 {previsao['nome']}"
        
        st.toast(f"🎯 Nova Previsão", icon="🔥")
        st.warning(f"🔔 {mensagem}")
        
        if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
            if st.session_state.telegram_token and st.session_state.telegram_chat_id:
                enviar_telegram(f"🔔 PREVISÃO\n{mensagem}")
                
        # Salvar sessão após nova previsão
        salvar_sessao()
    except Exception as e:
        logging.error(f"Erro ao enviar previsão: {e}")

def enviar_resultado_super_simplificado(numero_real, acerto, nome_estrategia, zona_acertada=None):
    """Envia notificação de resultado super simplificado"""
    try:
        if acerto:
            if 'Zonas' in nome_estrategia and zona_acertada:
                # CORREÇÃO: Mostrar número do núcleo em vez do nome da zona
                if zona_acertada == 'Vermelha':
                    nucleo = "7"
                elif zona_acertada == 'Azul':
                    nucleo = "10"
                elif zona_acertada == 'Amarela':
                    nucleo = "2"
                else:
                    nucleo = zona_acertada
                mensagem = f"✅ Acerto Núcleo {nucleo}\n🎲 Número: {numero_real}"
            elif 'ML' in nome_estrategia and zona_acertada:
                # CORREÇÃO: Mostrar número do núcleo em vez do nome da zona
                if zona_acertada == 'Vermelha':
                    nucleo = "7"
                elif zona_acertada == 'Azul':
                    nucleo = "10"
                elif zona_acertada == 'Amarela':
                    nucleo = "2"
                else:
                    nucleo = zona_acertada
                mensagem = f"✅ Acerto Núcleo {nucleo}\n🎲 Número: {numero_real}"
            else:
                mensagem = f"✅ Acerto\n🎲 Número: {numero_real}"
        else:
            mensagem = f"❌ Erro\n🎲 Número: {numero_real}"
        
        st.toast(f"🎲 Resultado", icon="✅" if acerto else "❌")
        st.success(f"📢 {mensagem}") if acerto else st.error(f"📢 {mensagem}")
        
        if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
            if st.session_state.telegram_token and st.session_state.telegram_chat_id:
                enviar_telegram(f"📢 RESULTADO\n{mensagem}")
                
        # Salvar sessão após resultado
        salvar_sessao()
    except Exception as e:
        logging.error(f"Erro ao enviar resultado: {e}")

def enviar_rotacao_automatica(estrategia_anterior, estrategia_nova):
    """Envia notificação de rotação automática"""
    try:
        mensagem = f"🔄 ROTAÇÃO AUTOMÁTICA\n{estrategia_anterior} → {estrategia_nova}"
        
        st.toast("🔄 Rotação Automática", icon="🔄")
        st.warning(f"🔄 {mensagem}")
        
        if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
            if st.session_state.telegram_token and st.session_state.telegram_chat_id:
                enviar_telegram(f"🔄 ROTAÇÃO\n{mensagem}")
                
    except Exception as e:
        logging.error(f"Erro ao enviar rotação: {e}")

def enviar_telegram(mensagem):
    """Envia mensagem para o Telegram"""
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
            logging.info("Mensagem enviada para Telegram com sucesso")
        else:
            logging.error(f"Erro ao enviar para Telegram: {response.status_code}")
    except Exception as e:
        logging.error(f"Erro na conexão com Telegram: {e}")

# =============================
# CONFIGURAÇÕES
# =============================
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =============================
# CLASSE PRINCIPAL DA ROLETA ATUALIZADA
# =============================
class RoletaInteligente:
    def __init__(self):
        # ORDEM FÍSICA DA ROLETA EUROPEIA (sentido horário)
        self.race = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
        
    def get_vizinhos_zona(self, numero_central, quantidade=6):
        """Retorna 6 vizinhos antes e 6 depois do número central no race (ordem física)"""
        if numero_central not in self.race:
            return []
        
        posicao = self.race.index(numero_central)
        vizinhos = []
        
        # 6 números ANTES (sentido anti-horário)
        for offset in range(-quantidade, 0):
            vizinho = self.race[(posicao + offset) % len(self.race)]
            vizinhos.append(vizinho)
        
        # Número central
        vizinhos.append(numero_central)
        
        # 6 números DEPOIS (sentido horário)  
        for offset in range(1, quantidade + 1):
            vizinho = self.race[(posicao + offset) % len(self.race)]
            vizinhos.append(vizinho)
        
        return vizinhos

    def get_posicao_race(self, numero):
        """Retorna a posição física do número na roda"""
        return self.race.index(numero) if numero in self.race else -1

    def get_vizinhos_fisicos(self, numero, raio=3):
        """Retorna vizinhos físicos na roda"""
        if numero not in self.race:
            return []
        
        posicao = self.race.index(numero)
        vizinhos = []
        
        for offset in range(-raio, raio + 1):
            if offset != 0:  # Exclui o próprio número
                vizinho = self.race[(posicao + offset) % len(self.race)]
                vizinhos.append(vizinho)
        
        return vizinhos

# =============================
# SISTEMA DE ROTAÇÃO INTELIGENTE
# =============================
class SistemaRotacaoInteligente:
    def __init__(self):
        self.performance_historica = {
            'Zonas': {'acertos': 0, 'total': 0, 'performance_media': 0},
            'ML': {'acertos': 0, 'total': 0, 'performance_media': 0},
            'Midas': {'acertos': 0, 'total': 0, 'performance_media': 0}
        }
        self.estrategia_atual = 'Zonas'
        self.performance_minima = 0.35  # 35% de performance mínima
        self.janela_analise = 20  # Analisar últimos 20 resultados
        
    def calcular_performance_estrategia(self, estrategia):
        """Calcula a performance atual de uma estratégia"""
        dados = self.performance_historica[estrategia]
        if dados['total'] > 0:
            performance = dados['acertos'] / dados['total']
            dados['performance_media'] = performance
            return performance
        return 0.0
    
    def decidir_rotacao(self, resultado_ultimo):
        """Decide se deve rotacionar a estratégia baseado na performance"""
        estrategia_ultima = resultado_ultimo['estrategia']
        
        # Atualizar performance da estratégia usada
        self.performance_historica[estrategia_ultima]['total'] += 1
        if resultado_ultimo['acerto']:
            self.performance_historica[estrategia_ultima]['acertos'] += 1
        
        # Calcular performance atual
        perf_atual = self.calcular_performance_estrategia(self.estrategia_atual)
        
        # Se performance abaixo do mínimo, considerar rotação
        if perf_atual < self.performance_minima:
            # Escolher melhor estratégia baseada em performance histórica
            estrategias_disponiveis = ['Zonas', 'ML', 'Midas']
            performances = {e: self.calcular_performance_estrategia(e) 
                          for e in estrategias_disponiveis if e != self.estrategia_atual}
            
            if performances:
                melhor_estrategia = max(performances, key=performances.get)
                if performances[melhor_estrategia] > perf_atual:
                    estrategia_anterior = self.estrategia_atual
                    self.estrategia_atual = melhor_estrategia
                    logging.info(f"🔄 Rotação inteligente: {estrategia_anterior} ({perf_atual:.1%}) → {melhor_estrategia} ({performances[melhor_estrategia]:.1%})")
                    return True, estrategia_anterior, self.estrategia_atual
        
        return False, self.estrategia_atual, self.estrategia_atual

    def get_status_rotacao(self):
        """Retorna o status atual do sistema de rotação"""
        performances = {e: self.calcular_performance_estrategia(e) for e in ['Zonas', 'ML', 'Midas']}
        return {
            'estrategia_atual': self.estrategia_atual,
            'performances': performances,
            'performance_minima': self.performance_minima,
            'proxima_avaliacao_em': self.janela_analise
        }

# =============================
# SISTEMA DE APRENDIZADO CONTÍNUO
# =============================
class AprendizadoContinuo:
    def __init__(self):
        self.performance_por_horario = {}
        self.padroes_sazonais = {}
        self.adaptacao_rapida = True
        
    def analisar_performance_temporal(self):
        """Analisa performance por período do dia"""
        hora_atual = datetime.now().hour
        periodo = self._classificar_periodo(hora_atual)
        
        # Inicializar período se não existir
        if periodo not in self.performance_por_horario:
            self.performance_por_horario[periodo] = {'acertos': 0, 'total': 0, 'performance': 0.0}
        
        return self.ajustar_estrategia_por_periodo(periodo)
    
    def _classificar_periodo(self, hora):
        """Classifica o período do dia"""
        if 6 <= hora < 12:
            return "manha"
        elif 12 <= hora < 18:
            return "tarde" 
        elif 18 <= hora < 24:
            return "noite"
        else:
            return "madrugada"
    
    def ajustar_estrategia_por_periodo(self, periodo):
        """Ajusta parâmetros baseado no período do dia"""
        ajustes = {
            'manha': {'threshold': -2, 'sensibilidade': 1.1, 'agressividade': 1.2},
            'tarde': {'threshold': 0, 'sensibilidade': 1.0, 'agressividade': 1.0},
            'noite': {'threshold': +3, 'sensibilidade': 0.9, 'agressividade': 0.8},
            'madrugada': {'threshold': +5, 'sensibilidade': 0.8, 'agressividade': 0.7}
        }
        return ajustes.get(periodo, ajustes['tarde'])
    
    def atualizar_performance_periodo(self, periodo, acerto):
        """Atualiza a performance do período"""
        dados = self.performance_por_horario[periodo]
        dados['total'] += 1
        if acerto:
            dados['acertos'] += 1
        dados['performance'] = dados['acertos'] / dados['total'] if dados['total'] > 0 else 0.0

    def get_analise_temporal(self):
        """Retorna análise de performance por período"""
        if not self.performance_por_horario:
            return "📊 Análise temporal: Dados insuficientes"
        
        analise = "🕒 PERFORMANCE POR PERÍODO:\n"
        for periodo, dados in self.performance_por_horario.items():
            if dados['total'] > 0:
                perf = dados['performance']
                analise += f"   {periodo.upper()}: {dados['acertos']}/{dados['total']} ({perf:.1%})\n"
        
        periodo_atual = self._classificar_periodo(datetime.now().hour)
        ajustes = self.ajustar_estrategia_por_periodo(periodo_atual)
        analise += f"\n🎯 AJUSTES ATUAIS ({periodo_atual.upper()}):\n"
        analise += f"   Threshold: {ajustes['threshold']:+d}\n"
        analise += f"   Sensibilidade: {ajustes['sensibilidade']}x\n"
        analise += f"   Agressividade: {ajustes['agressividade']}x"
        
        return analise

# =============================
# MÓDULO DE MACHINE LEARNING HIPER OTIMIZADO
# =============================
class MLRoletaHiperOtimizada:
    def __init__(
        self,
        roleta_obj,
        min_training_samples: int = 150,  # OTIMIZADO: 200 → 150 para treinar mais cedo
        max_history: int = 1000,
        retrain_every_n: int = 10,        # OTIMIZADO: 15 → 10 (mais frequente)
        seed: int = 42
    ):
        self.roleta = roleta_obj
        self.min_training_samples = min_training_samples
        self.max_history = max_history
        self.retrain_every_n = retrain_every_n
        self.seed = seed

        self.models = []
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.contador_treinamento = 0
        self.meta = {}

        # Configurações otimizadas
        self.window_for_features = [3, 8, 15, 30, 60, 120]
        self.k_vizinhos = 2
        self.numeros = list(range(37))
        
        # Ensemble maior
        self.ensemble_size = 4  # OTIMIZADO: 3 → 4 modelos
        
        # Melhores parâmetros para CatBoost
        self.catboost_params = {
            'iterations': 2000,
            'learning_rate': 0.03,
            'depth': 12,
            'l2_leaf_reg': 3,
            'random_strength': 0.8,
            'bagging_temperature': 1.0,
        }

    def get_neighbors(self, numero, k=None):
        if k is None:
            k = self.k_vizinhos
        try:
            race = list(self.roleta.race)
            n = len(race)
            idx = race.index(numero)
            neighbors = []
            for offset in range(-k, k+1):
                neighbors.append(race[(idx + offset) % n])
            return neighbors
        except Exception:
            return [numero]

    def extrair_features(self, historico, numero_alvo=None):
        try:
            historico = list(historico)
            N = len(historico)
            
            if N < 10:
                return None, None

            features = []
            names = []

            # --- 1) Últimos K diretos (até 10)
            K_seq = 10
            ultimos = historico[-K_seq:]
            for i in range(K_seq):
                val = ultimos[i] if i < len(ultimos) else -1
                features.append(val)
                names.append(f"ultimo_{i+1}")

            # --- 2) Estatísticas da janela
            for w in self.window_for_features:
                janela = historico[-w:] if N >= w else historico[:]
                arr = np.array(janela, dtype=float)
                features.append(arr.mean() if len(arr) > 0 else 0.0); names.append(f"media_{w}")
                features.append(arr.std() if len(arr) > 1 else 0.0); names.append(f"std_{w}")
                features.append(np.median(arr) if len(arr) > 0 else 0.0); names.append(f"mediana_{w}")

            # --- 3) Frequência por janela
            counter_full = Counter(historico)
            for w in self.window_for_features:
                janela = historico[-w:] if N >= w else historico[:]
                c = Counter(janela)
                features.append(len(c) / (w if w>0 else 1)); names.append(f"diversidade_{w}")
                top1_count = c.most_common(1)[0][1] if len(c)>0 else 0
                features.append(top1_count / (w if w>0 else 1)); names.append(f"top1_prop_{w}")

            # --- 4) Tempo desde último para cada número
            for num in self.numeros:
                try:
                    rev_idx = historico[::-1].index(num)
                    tempo = rev_idx
                except ValueError:
                    tempo = N + 1
                features.append(tempo)
                names.append(f"tempo_desde_{num}")

            # --- 5) Contagens por cor e dúzia
            janela50 = historico[-50:] if N >= 50 else historico[:]
            vermelhos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
            pretos = set(self.numeros[1:]) - vermelhos
            count_verm = sum(1 for x in janela50 if x in vermelhos)
            count_pret = sum(1 for x in janela50 if x in pretos)
            count_zero = sum(1 for x in janela50 if x == 0)
            features.extend([count_verm/len(janela50), count_pret/len(janela50), count_zero/len(janela50)])
            names.extend(["prop_vermelhos_50", "prop_pretos_50", "prop_zero_50"])

            # dúzias
            def duzia_of(x):
                if x == 0: return 0
                if 1 <= x <= 12: return 1
                if 13 <= x <= 24: return 2
                return 3
            for d in [1,2,3]:
                features.append(sum(1 for x in janela50 if duzia_of(x)==d)/len(janela50))
                names.append(f"prop_duzia_{d}_50")

            # --- 6) Vizinhos físicos
            ultimo_num = historico[-1]
            vizinhos_k = self.get_neighbors(ultimo_num, k=6)
            count_in_vizinhos = sum(1 for x in ultimos if x in vizinhos_k) / len(ultimos)
            features.append(count_in_vizinhos); names.append("prop_ultimos_em_vizinhos_6")

            # --- 7) Repetições e padrões binários
            features.append(1 if N>=2 and historico[-1] == historico[-2] else 0); names.append("repetiu_ultimo")
            features.append(1 if N>=2 and (historico[-1] % 2) == (historico[-2] % 2) else 0); names.append("repetiu_paridade")
            features.append(1 if N>=2 and duzia_of(historico[-1]) == duzia_of(historico[-2]) else 0); names.append("repetiu_duzia")

            # --- 8) Diferenças entre janelas
            if N >= max(self.window_for_features):
                small = np.mean(historico[-self.window_for_features[0]:])
                large = np.mean(historico[-self.window_for_features[-1]:])
                features.append(small - large); names.append("delta_media_small_large")
            else:
                features.append(0.0); names.append("delta_media_small_large")

            # --- 9) Estatísticas de transição
            diffs = [abs(historico[i] - historico[i-1]) for i in range(1, len(historico))]
            features.append(np.mean(diffs) if len(diffs)>0 else 0.0); names.append("media_transicoes")
            features.append(np.std(diffs) if len(diffs)>1 else 0.0); names.append("std_transicoes")

            self.feature_names = names
            return features, names

        except Exception as e:
            logging.error(f"[extrair_features] Erro: {e}")
            return None, None

    def preparar_dados_treinamento(self, historico_completo):
        historico_completo = list(historico_completo)
        if len(historico_completo) > self.max_history:
            historico_completo = historico_completo[-self.max_history:]

        X = []
        y = []
        
        start_index = max(50, len(historico_completo) // 10)
        
        for i in range(start_index, len(historico_completo)):
            janela = historico_completo[:i]
            feats, _ = self.extrair_features(janela)
            if feats is None:
                continue
            X.append(feats)
            y.append(historico_completo[i])
        
        if len(X) == 0:
            return np.array([]), np.array([])
        
        class_counts = Counter(y)
        if len(class_counts) < 10:
            logging.warning(f"Pouca variedade de classes: apenas {len(class_counts)} números únicos")
            return np.array([]), np.array([])
        
        return np.array(X), np.array(y)

    def _build_and_train_model(self, X_train, y_train, X_val=None, y_val=None, seed=0):
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                **self.catboost_params,
                loss_function='MultiClass',
                eval_metric='MultiClass',
                random_seed=seed,
                use_best_model=True,
                early_stopping_rounds=150,
                verbose=False
            )
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            else:
                model.fit(X_train, y_train, verbose=False)
            return model, "CatBoost-Hiper"
        except Exception as e:
            logging.warning(f"CatBoost não disponível ou falha ({e}). Usando RandomForest como fallback.")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=-1,
                bootstrap=True
            )
            model.fit(X_train, y_train)
            return model, "RandomForest-Hiper"

    def treinar_modelo(self, historico_completo, force_retrain: bool = False, balance: bool = True):
        try:
            if len(historico_completo) < self.min_training_samples and not force_retrain:
                return False, f"Necessário mínimo de {self.min_training_samples} amostras. Atual: {len(historico_completo)}"

            X, y = self.preparar_dados_treinamento(historico_completo)
            if X.size == 0 or len(X) < 50:
                return False, f"Dados insuficientes para treino: {len(X)} amostras"

            X_scaled = self.scaler.fit_transform(X)

            try:
                class_counts = Counter(y)
                min_samples_per_class = min(class_counts.values())
                
                can_stratify = min_samples_per_class >= 2 and len(class_counts) > 1
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, 
                    test_size=0.2, 
                    random_state=self.seed, 
                    stratify=y if can_stratify else None
                )
                
                logging.info(f"Split realizado: estratificação = {can_stratify}, classes = {len(class_counts)}, min_amostras = {min_samples_per_class}")
                
            except Exception as e:
                logging.warning(f"Erro no split estratificado: {e}. Usando split sem estratificação.")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=self.seed
                )

            if balance and len(X_train) > 0:
                try:
                    df_train = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
                    df_train['y'] = y_train
                    
                    value_counts = df_train['y'].value_counts()
                    if len(value_counts) == 0:
                        raise ValueError("Nenhuma classe encontrada")
                    
                    max_count = value_counts.max()
                    
                    if len(value_counts) < 2:
                        logging.warning("Apenas uma classe disponível, pulando balanceamento")
                        balance = False
                    else:
                        frames = []
                        for cls, grp in df_train.groupby('y'):
                            if len(grp) < max_count:
                                if len(grp) >= 1:
                                    min_samples = max(5, max_count // 3)
                                    n_samples = min(max_count, min_samples)
                                    grp_up = resample(grp, replace=True, n_samples=n_samples, random_state=self.seed)
                                    frames.append(grp_up)
                                else:
                                    frames.append(grp)
                            else:
                                frames.append(grp)
                        
                        if frames:
                            df_bal = pd.concat(frames)
                            y_train = df_bal['y'].values
                            X_train = df_bal.drop(columns=['y']).values
                        else:
                            balance = False
                            
                except Exception as e:
                    logging.warning(f"Erro no balanceamento: {e}. Continuando sem balanceamento.")
                    balance = False

            models = []
            model_names = []
            
            # Ensemble maior (4 modelos)
            for s in [self.seed, self.seed + 7, self.seed + 13, self.seed + 19]:
                try:
                    model, name = self._build_and_train_model(X_train, y_train, X_val, y_val, seed=s)
                    models.append(model)
                    model_names.append(name)
                except Exception as e:
                    logging.error(f"Erro ao treinar modelo {s}: {e}")

            if not models:
                return False, "Todos os modelos falharam no treinamento"

            try:
                probs = []
                for m in models:
                    if hasattr(m, 'predict_proba'):
                        probs.append(m.predict_proba(X_val))
                    else:
                        preds = m.predict(X_val)
                        prob = np.zeros((len(preds), len(self.numeros)))
                        for i, p in enumerate(preds):
                            prob[i, p] = 1.0
                        probs.append(prob)
                
                if probs:
                    avg_prob = np.mean(probs, axis=0)
                    y_pred = np.argmax(avg_prob, axis=1)
                    acc = accuracy_score(y_val, y_pred)
                else:
                    acc = 0.0
                    
            except Exception as e:
                logging.warning(f"Erro na avaliação: {e}")
                acc = 0.0

            self.models = models
            self.is_trained = True
            self.contador_treinamento += 1
            self.meta['last_accuracy'] = acc
            self.meta['trained_on'] = len(historico_completo)
            self.meta['last_training_size'] = len(X)

            try:
                joblib.dump({'models': self.models}, ML_MODEL_PATH)
                joblib.dump(self.scaler, SCALER_PATH)
                joblib.dump(self.meta, META_PATH)
                logging.info(f"Modelos salvos em disco: {ML_MODEL_PATH}")
            except Exception as e:
                logging.warning(f"Falha ao salvar modelos: {e}")

            return True, f"Ensemble treinado ({', '.join(model_names)}) com {len(X)} amostras. Acurácia validação: {acc:.2%}"

        except Exception as e:
            logging.error(f"[treinar_modelo] Erro: {e}", exc_info=True)
            return False, f"Erro no treinamento: {str(e)}"

    def carregar_modelo(self):
        try:
            if os.path.exists(ML_MODEL_PATH) and os.path.exists(SCALER_PATH):
                data = joblib.load(ML_MODEL_PATH)
                self.models = data.get('models', [])
                self.scaler = joblib.load(SCALER_PATH)
                
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ > 0:
                    self.is_trained = len(self.models) > 0
                    logging.info(f"✅ Modelo ML carregado: {len(self.models)} modelos, {self.scaler.n_features_in_} features")
                    return True
                else:
                    logging.warning("❌ Scaler carregado mas não treinado corretamente")
                    self.is_trained = False
                    return False
            return False
        except Exception as e:
            logging.error(f"[carregar_modelo] Erro: {e}")
            self.is_trained = False
            return False

    def _ensemble_predict_proba(self, X_scaled):
        if not self.models:
            return np.ones((len(X_scaled), len(self.numeros))) / len(self.numeros)

        probs = []
        for m in self.models:
            if hasattr(m, 'predict_proba'):
                probs.append(m.predict_proba(X_scaled))
            else:
                preds = m.predict(X_scaled)
                prob = np.zeros((len(preds), len(self.numeros)))
                for i, p in enumerate(preds):
                    prob[i, p] = 1.0
                probs.append(prob)
        return np.mean(probs, axis=0)

    def prever_proximo_numero(self, historico, top_k: int = 25):
        if not self.is_trained:
            return None, "Modelo não treinado"

        feats, _ = self.extrair_features(historico)
        if feats is None:
            return None, "Features insuficientes"

        Xs = np.array([feats])
        
        if not hasattr(self.scaler, 'n_features_in_') or self.scaler.n_features_in_ == 0:
            return None, "Scaler não treinado - necessário treinar modelo primeiro"
        
        if len(feats) != self.scaler.n_features_in_:
            return None, f"Dimensões incompatíveis: features {len(feats)} vs scaler {self.scaler.n_features_in_}"
        
        try:
            Xs_scaled = self.scaler.transform(Xs)
            probs = self._ensemble_predict_proba(Xs_scaled)[0]
            top_idx = np.argsort(probs)[-top_k:][::-1]
            top = [(int(idx), float(probs[idx])) for idx in top_idx]
            return top, "Previsão ML realizada"
        except Exception as e:
            return None, f"Erro na previsão: {str(e)}"

    def prever_blocos_vizinhos(self, historico, k_neighbors: int = 2, top_blocks: int = 5):
        pred, msg = self.prever_proximo_numero(historico, top_k=37)
        if pred is None:
            return None, msg
        prob = {num: p for num, p in pred}
        blocks = []
        for num in range(37):
            neigh = self.get_neighbors(num, k=k_neighbors)
            agg_prob = sum(prob.get(n, 0.0) for n in neigh)
            blocks.append((num, tuple(neigh), agg_prob))
        blocks_sorted = sorted(blocks, key=lambda x: x[2], reverse=True)[:top_blocks]
        formatted = [{"central": b[0], "vizinhos": list(b[1]), "prob": float(b[2])} for b in blocks_sorted]
        return formatted, "Previsão de blocos realizada"

    def registrar_resultado(self, historico, previsao_top, resultado_real):
        try:
            hit = resultado_real in [p for p,_ in previsao_top] if isinstance(previsao_top[0], tuple) else resultado_real in previsao_top
            log_entry = {
                'prev_top': previsao_top,
                'resultado': resultado_real,
                'hit': bool(hit)
            }
            self.meta.setdefault('history_feedback', []).append(log_entry)
            recent = self.meta['history_feedback'][-10:]
            hits = sum(1 for r in recent if r['hit'])
            if len(recent) >= 5 and hits / len(recent) < 0.25:
                logging.info("[feedback] Baixa performance detectada — forçando retreinamento incremental")
                self.treinar_modelo(historico, force_retrain=True, balance=True)
            return True
        except Exception as e:
            logging.error(f"[registrar_resultado] Erro: {e}")
            return False

    def verificar_treinamento_automatico(self, historico_completo):
        try:
            n = len(historico_completo)
            if n >= self.min_training_samples:
                if n % self.retrain_every_n == 0:
                    return self.treinar_modelo(historico_completo)
            return False, "Aguardando próximo ciclo de treinamento"
        except Exception as e:
            return False, f"Erro ao verificar retrain: {e}"

    def resumo_meta(self):
        return {
            "is_trained": self.is_trained,
            "contador_treinamento": self.contador_treinamento,
            "meta": self.meta
        }

# =============================
# SISTEMA DE CONFIRMAÇÃO DE PADRÕES
# =============================
class SistemaConfirmacaoPadroes:
    def __init__(self):
        self.padroes_confirmados = {}
        self.confirmacoes_necessarias = 2
        self.janela_confirmacao = 10
        
    def verificar_confirmacao_padrao(self, padrao_detectado, resultado_real, numeros_zonas):
        """Verifica se um padrão detectado foi confirmado pelo resultado real"""
        chave = f"{padrao_detectado['tipo']}_{padrao_detectado['zona']}"
        
        if chave not in self.padroes_confirmados:
            self.padroes_confirmados[chave] = {
                'detectado_em': padrao_detectado['detectado_em'],
                'confirmacoes': 0,
                'total_ocorrencias': 0,
                'acertos': 0
            }
        
        dados = self.padroes_confirmados[chave]
        dados['total_ocorrencias'] += 1
        
        # Verificar se o padrão acertou
        zona_real = None
        for zona, numeros in numeros_zonas.items():
            if resultado_real in numeros:
                zona_real = zona
                break
        
        if zona_real == padrao_detectado['zona']:
            dados['acertos'] += 1
            dados['confirmacoes'] += 1
        
        # Calcular confiabilidade do padrão
        if dados['total_ocorrencias'] >= 3:
            confiabilidade = dados['acertos'] / dados['total_ocorrencias']
            return confiabilidade > 0.6  # 60% de acerto
        
        return False

    def get_padroes_confiaveis(self):
        """Retorna padrões com alta confiabilidade"""
        padroes_confiaveis = {}
        for chave, dados in self.padroes_confirmados.items():
            if dados['total_ocorrencias'] >= 3:
                confiabilidade = dados['acertos'] / dados['total_ocorrencias']
                if confiabilidade >= 0.6:
                    padroes_confiaveis[chave] = {
                        'confiabilidade': confiabilidade,
                        'total_ocorrencias': dados['total_ocorrencias'],
                        'acertos': dados['acertos']
                    }
        return padroes_confiaveis

# =============================
# ESTRATÉGIA DAS ZONAS HIPER OTIMIZADA
# =============================
class EstrategiaZonasHiperOtimizada:
    def __init__(self):
        self.roleta = RoletaInteligente()
        self.historico = deque(maxlen=70)
        self.nome = "Zonas Hiper Otimizada v7"
        
        self.zonas = {
            'Vermelha': 7,
            'Azul': 10,  
            'Amarela': 2
        }
        
        self.quantidade_zonas = {
            'Vermelha': 6,
            'Azul': 6,
            'Amarela': 6
        }
        
        # Janelas otimizadas
        self.janelas_analise = {
            'curto_prazo': 8,    # Mais sensível
            'medio_prazo': 20,   
            'longo_prazo': 40,
            'performance': 80
        }
        
        self.numeros_zonas = {}
        for nome, central in self.zonas.items():
            qtd = self.quantidade_zonas.get(nome, 6)
            self.numeros_zonas[nome] = self.roleta.get_vizinhos_zona(central, qtd)

        self.stats_zonas = {zona: {
            'acertos': 0, 
            'tentativas': 0, 
            'sequencia_atual': 0,
            'sequencia_maxima': 0,
            'performance_media': 0
        } for zona in self.zonas.keys()}
        
        # Threshold mais agressivo
        self.threshold_base = 25

    def adicionar_numero(self, numero):
        self.historico.append(numero)
        resultado = self.atualizar_stats(numero)
        salvar_sessao()
        return resultado

    def atualizar_stats(self, ultimo_numero):
        acertou_zona = None
        for zona, numeros in self.numeros_zonas.items():
            if ultimo_numero in numeros:
                self.stats_zonas[zona]['acertos'] += 1
                self.stats_zonas[zona]['sequencia_atual'] += 1
                if self.stats_zonas[zona]['sequencia_atual'] > self.stats_zonas[zona]['sequencia_maxima']:
                    self.stats_zonas[zona]['sequencia_maxima'] = self.stats_zonas[zona]['sequencia_atual']
                acertou_zona = zona
            else:
                self.stats_zonas[zona]['sequencia_atual'] = 0
            self.stats_zonas[zona]['tentativas'] += 1
            
            if self.stats_zonas[zona]['tentativas'] > 0:
                self.stats_zonas[zona]['performance_media'] = (
                    self.stats_zonas[zona]['acertos'] / self.stats_zonas[zona]['tentativas'] * 100
                )
        
        return acertou_zona

    def get_threshold_dinamico_otimizado(self, zona):
        """Threshold mais agressivo para zonas quentes"""
        perf = self.stats_zonas[zona]['performance_media']
        sequencia = self.stats_zonas[zona]['sequencia_atual']
        
        # Reduzir threshold para zonas quentes
        if perf > 45:    # Zona muito quente
            return self.threshold_base - 8   # 17
        elif perf > 35:  # Zona quente  
            return self.threshold_base - 5   # 20
        elif perf < 20:  # Zona fria  
            return self.threshold_base + 3   # 28
        
        # Bônus por sequência
        if sequencia >= 2:
            return self.threshold_base - 3   # 22
        
        return self.threshold_base

    def get_zona_mais_quente(self):
        if len(self.historico) < 12:  # Reduzido de 15
            return None
            
        zonas_score = {}
        total_numeros = len(self.historico)
        
        for zona in self.zonas.keys():
            score = 0
            
            # Análise de múltiplas janelas com pesos otimizados
            freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
            percentual_geral = freq_geral / total_numeros
            score += percentual_geral * 20  # Reduzido peso do geral
            
            # Janela de curto prazo (mais peso)
            ultimos_curto = list(self.historico)[-self.janelas_analise['curto_prazo']:] if total_numeros >= self.janelas_analise['curto_prazo'] else list(self.historico)
            freq_curto = sum(1 for n in ultimos_curto if n in self.numeros_zonas[zona])
            percentual_curto = freq_curto / len(ultimos_curto)
            score += percentual_curto * 40  # Aumentado peso do curto prazo
            
            # Performance histórica com peso adaptativo
            if self.stats_zonas[zona]['tentativas'] > 8:  # Reduzido de 10
                taxa_acerto = self.stats_zonas[zona]['performance_media']
                if taxa_acerto > 40: 
                    score += 35  # Mais peso para zonas quentes
                elif taxa_acerto > 35:
                    score += 30
                elif taxa_acerto > 30:
                    score += 25
                elif taxa_acerto > 25:
                    score += 20
                else:
                    score += 15
            else:
                score += 10
            
            # Sequência atual com bônus progressivo mais agressivo
            sequencia = self.stats_zonas[zona]['sequencia_atual']
            if sequencia >= 2:
                score += min(sequencia * 4, 15)  # Mais agressivo
            
            zonas_score[zona] = score
        
        zona_vencedora = max(zonas_score, key=zonas_score.get) if zonas_score else None
        
        if zona_vencedora:
            threshold = self.get_threshold_dinamico_otimizado(zona_vencedora)
            
            # Ajuste adicional por sequência mais agressivo
            if self.stats_zonas[zona_vencedora]['sequencia_atual'] >= 2:
                threshold -= 3  # Mais agressivo
            
            return zona_vencedora if zonas_score[zona_vencedora] >= threshold else None
        
        return None

    def analisar_zonas(self):
        if len(self.historico) < 12:  # Reduzido de 15
            return None
            
        zona_alvo = self.get_zona_mais_quente()
        
        if zona_alvo:
            numeros_apostar = self.numeros_zonas[zona_alvo]
            
            confianca = self.calcular_confianca_hiper(zona_alvo)
            score = self.get_zona_score_otimizado(zona_alvo)
            
            gatilho = f'Zona {zona_alvo} - Score: {score:.1f} | Perf: {self.stats_zonas[zona_alvo]["performance_media"]:.1f}% | Thr: {self.get_threshold_dinamico_otimizado(zona_alvo)}'
            
            return {
                'nome': f'Zona {zona_alvo}',
                'numeros_apostar': numeros_apostar,
                'gatilho': gatilho,
                'confianca': confianca,
                'zona': zona_alvo
            }
        
        return None

    def calcular_confianca_hiper(self, zona):
        if len(self.historico) < 8:  # Reduzido de 10
            return 'Baixa'
            
        fatores = []
        pesos = []
        
        perf_historica = self.stats_zonas[zona]['performance_media']
        if perf_historica > 45: 
            fatores.append(4)  # Mais agressivo
            pesos.append(5)
        elif perf_historica > 35: 
            fatores.append(3)
            pesos.append(5)
        else: 
            fatores.append(2)
            pesos.append(5)
        
        # Análise de múltiplas janelas com foco no curto prazo
        for janela_nome, tamanho in self.janelas_analise.items():
            if janela_nome != 'performance':
                historico_janela = list(self.historico)[-tamanho:] if len(self.historico) >= tamanho else list(self.historico)
                freq_janela = sum(1 for n in historico_janela if n in self.numeros_zonas[zona])
                perc_janela = (freq_janela / len(historico_janela)) * 100
                
                peso = 3 if janela_nome == 'curto_prazo' else 1  # Mais peso no curto prazo
                
                if perc_janela > 50: 
                    fatores.append(4)
                elif perc_janela > 35: 
                    fatores.append(3)
                else: 
                    fatores.append(2)
                pesos.append(peso)
        
        sequencia = self.stats_zonas[zona]['sequencia_atual']
        if sequencia >= 3: 
            fatores.append(4)  # Mais agressivo
            pesos.append(3)
        elif sequencia >= 2: 
            fatores.append(3)
            pesos.append(3)
        else: 
            fatores.append(2)
            pesos.append(3)
        
        if len(self.historico) >= 8:  # Reduzido de 10
            ultimos_4 = list(self.historico)[-4:]  # Janela menor
            anteriores_4 = list(self.historico)[-8:-4] if len(self.historico) >= 8 else []
            
            freq_ultimos = sum(1 for n in ultimos_4 if n in self.numeros_zonas[zona])
            freq_anteriores = sum(1 for n in anteriores_4 if n in self.numeros_zonas[zona]) if anteriores_4 else 0
            
            if freq_ultimos > freq_anteriores: 
                fatores.append(4)  # Mais agressivo
                pesos.append(3)
            elif freq_ultimos == freq_anteriores: 
                fatores.append(3)
                pesos.append(3)
            else: 
                fatores.append(2)
                pesos.append(3)
        
        total_pontos = sum(f * p for f, p in zip(fatores, pesos))
        total_pesos = sum(pesos)
        score_confianca = total_pontos / total_pesos
        
        if score_confianca >= 3.2: 
            return 'Excelente'
        elif score_confianca >= 2.8: 
            return 'Muito Alta'
        elif score_confianca >= 2.4: 
            return 'Alta'
        elif score_confianca >= 2.0: 
            return 'Média'
        else: 
            return 'Baixa'

    def get_zona_score_otimizado(self, zona):
        if len(self.historico) < 8:  # Reduzido de 10
            return 0
            
        score = 0
        total_numeros = len(self.historico)
        
        freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
        percentual_geral = freq_geral / total_numeros
        score += percentual_geral * 20  # Reduzido
        
        # Múltiplas janelas com foco no curto prazo
        for janela_nome, tamanho in self.janelas_analise.items():
            if janela_nome != 'performance':
                historico_janela = list(self.historico)[-tamanho:] if total_numeros >= tamanho else list(self.historico)
                freq_janela = sum(1 for n in historico_janela if n in self.numeros_zonas[zona])
                percentual_janela = freq_janela / len(historico_janela)
                peso = 40 if janela_nome == 'curto_prazo' else 10  # Foco no curto prazo
                score += percentual_janela * peso
        
        if self.stats_zonas[zona]['tentativas'] > 8:  # Reduzido
            taxa_acerto = self.stats_zonas[zona]['performance_media']
            if taxa_acerto > 40: score += 35
            elif taxa_acerto > 35: score += 30
            elif taxa_acerto > 30: score += 25
            elif taxa_acerto > 25: score += 20
            else: score += 15
        else:
            score += 10
        
        sequencia = self.stats_zonas[zona]['sequencia_atual']
        if sequencia >= 2:
            score += min(sequencia * 4, 16)  # Mais agressivo
            
        return score

    def get_info_zonas(self):
        info = {}
        for zona, numeros in self.numeros_zonas.items():
            info[zona] = {
                'numeros': sorted(numeros),
                'quantidade': len(numeros),
                'central': self.zonas[zona],
                'descricao': f"6 antes + 6 depois do {self.zonas[zona]}"
            }
        return info

    def get_analise_detalhada(self):
        if len(self.historico) == 0:
            return "Aguardando dados..."
        
        analise = "🎯 ANÁLISE HIPER OTIMIZADA - ZONAS v7\n"
        analise += "=" * 55 + "\n"
        analise += "🔧 CONFIGURAÇÃO: 6 antes + 6 depois (13 números/zona)\n"
        analise += f"📊 JANELAS: Curto({self.janelas_analise['curto_prazo']}) Médio({self.janelas_analise['medio_prazo']}) Longo({self.janelas_analise['longo_prazo']})\n"
        analise += "🎯 THRESHOLD: Base 25 + ajustes dinâmicos agressivos\n"
        analise += "=" * 55 + "\n"
        
        analise += "📊 PERFORMANCE AVANÇADA:\n"
        for zona in self.zonas.keys():
            tentativas = self.stats_zonas[zona]['tentativas']
            acertos = self.stats_zonas[zona]['acertos']
            taxa = self.stats_zonas[zona]['performance_media']
            sequencia = self.stats_zonas[zona]['sequencia_atual']
            seq_maxima = self.stats_zonas[zona]['sequencia_maxima']
            threshold = self.get_threshold_dinamico_otimizado(zona)
            
            analise += f"📍 {zona}: {acertos}/{tentativas} → {taxa:.1f}% | Seq: {sequencia} | Máx: {seq_maxima} | Thr: {threshold}\n"
        
        analise += "\n📈 FREQUÊNCIA MULTI-JANELAS:\n"
        for zona in self.zonas.keys():
            freq_total = sum(1 for n in self.historico if isinstance(n, (int, float)) and n in self.numeros_zonas[zona])
            perc_total = (freq_total / len(self.historico)) * 100
            
            # Múltiplas janelas
            freq_curto = sum(1 for n in list(self.historico)[-self.janelas_analise['curto_prazo']:] if n in self.numeros_zonas[zona])
            perc_curto = (freq_curto / min(self.janelas_analise['curto_prazo'], len(self.historico))) * 100
            
            score = self.get_zona_score_otimizado(zona)
            qtd_numeros = len(self.numeros_zonas[zona])
            analise += f"📍 {zona}: Total:{freq_total}/{len(self.historico)}({perc_total:.1f}%) | Curto:{freq_curto}/{self.janelas_analise['curto_prazo']}({perc_curto:.1f}%) | Score: {score:.1f}\n"
        
        analise += "\n📊 TENDÊNCIAS AVANÇADAS:\n"
        if len(self.historico) >= 8:
            for zona in self.zonas.keys():
                ultimos_4 = list(self.historico)[-4:]
                anteriores_4 = list(self.historico)[-8:-4] if len(self.historico) >= 8 else []
                
                freq_ultimos = sum(1 for n in ultimos_4 if n in self.numeros_zonas[zona])
                freq_anteriores = sum(1 for n in anteriores_4 if n in self.numeros_zonas[zona]) if anteriores_4 else 0
                
                tendencia = "↗️" if freq_ultimos > freq_anteriores else "↘️" if freq_ultimos < freq_anteriores else "➡️"
                variacao = freq_ultimos - freq_anteriores
                analise += f"📍 {zona}: {freq_ultimos}/4 vs {freq_anteriores}/4 {tendencia} (Δ: {variacao:+d})\n"
        
        zona_recomendada = self.get_zona_mais_quente()
        if zona_recomendada:
            analise += f"\n💡 RECOMENDAÇÃO HIPER: Zona {zona_recomendada}\n"
            analise += f"🎯 Números: {sorted(self.numeros_zonas[zona_recomendada])}\n"
            analise += f"📈 Confiança: {self.calcular_confianca_hiper(zona_recomendada)}\n"
            analise += f"🔥 Score: {self.get_zona_score_otimizado(zona_recomendada):.1f}\n"
            analise += f"🎯 Threshold: {self.get_threshold_dinamico_otimizado(zona_recomendada)}\n"
            analise += f"🔢 Quantidade: {len(self.numeros_zonas[zona_recomendada])} números\n"
            analise += f"📊 Performance: {self.stats_zonas[zona_recomendada]['performance_media']:.1f}%\n"
            
            perf = self.stats_zonas[zona_recomendada]['performance_media']
            if perf > 40:
                analise += f"💎 ESTRATÉGIA: Zona de ALTÍSSIMA performance - Aposta máxima!\n"
            elif perf > 35:
                analise += f"🎯 ESTRATÉGIA: Zona de ALTA performance - Aposta forte\n"
            elif perf > 25:
                analise += f"⚡ ESTRATÉGIA: Zona de performance sólida - Aposta moderada\n"
            else:
                analise += f"🔍 ESTRATÉGIA: Zona em desenvolvimento - Aposta conservadora\n"
        else:
            analise += "\n⚠️  AGUARDAR: Nenhuma zona com confiança suficiente\n"
            analise += f"📋 Histórico atual: {len(self.historico)} números\n"
            analise += f"🎯 Threshold base: {self.threshold_base}+ | Performance >25%\n"
        
        return analise

    def get_analise_atual(self):
        return self.get_analise_detalhada()

    def zerar_estatisticas(self):
        """Zera todas as estatísticas de desempenho"""
        for zona in self.stats_zonas.keys():
            self.stats_zonas[zona] = {
                'acertos': 0, 
                'tentativas': 0, 
                'sequencia_atual': 0,
                'sequencia_maxima': 0,
                'performance_media': 0
            }
        logging.info("📊 Estatísticas das Zonas zeradas")

# =============================
# ESTRATÉGIA MIDAS (MANTIDA)
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
        salvar_sessao()

    def analisar_midas(self):
        if len(self.historico) < 5:
            return None
            
        ultimo_numero = self.historico[-1]
        historico_recente = self.historico[-5:]

        if ultimo_numero in [0, 10, 20, 30]:
            count_zero = sum(1 for n in historico_recente if n in [0, 10, 20, 30])
            if count_zero >= 1:
                return {
                    'nome': 'Padrão do Zero',
                    'numeros_apostar': [0, 10, 20, 30],
                    'gatilho': f'Terminal 0 ativado ({count_zero}x)',
                    'confianca': 'Média'
                }

        if ultimo_numero in [7, 17, 27]:
            count_sete = sum(1 for n in historico_recente if n in [7, 17, 27])
            if count_sete >= 1:
                return {
                    'nome': 'Padrão do Sete',
                    'numeros_apostar': [7, 17, 27],
                    'gatilho': f'Terminal 7 ativado ({count_sete}x)',
                    'confianca': 'Média'
                }

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
# ESTRATÉGIA ML COM SISTEMA DE CONFIRMAÇÃO
# =============================
class EstrategiaML:
    def __init__(self):
        self.roleta = RoletaInteligente()
        # USANDO ML HIPER OTIMIZADA
        self.ml = MLRoletaHiperOtimizada(self.roleta)
        self.historico = deque(maxlen=30)
        self.nome = "Machine Learning (CatBoost-Hiper)"
        self.ml.carregar_modelo()
        self.contador_sorteios = 0
        
        self.zonas_ml = {
            'Vermelha': 7,
            'Azul': 10,  
            'Amarela': 2
        }
        
        self.quantidade_zonas_ml = {
            'Vermelha': 6,
            'Azul': 6,
            'Amarela': 6
        }
        
        self.numeros_zonas_ml = {}
        for nome, central in self.zonas_ml.items():
            qtd = self.quantidade_zonas_ml.get(nome, 6)
            self.numeros_zonas_ml[nome] = self.roleta.get_vizinhos_zona(central, qtd)

        # Sistema de confirmação de padrões
        self.confirmacao_padroes = SistemaConfirmacaoPadroes()
        
        # Sistema de detecção de padrões sequenciais
        self.sequencias_padroes = {
            'sequencias_ativas': {},
            'historico_sequencias': [],
            'padroes_detectados': []
        }
        
        self.adicionar_metricas_padroes()

    def adicionar_metricas_padroes(self):
        """Adiciona métricas de performance dos padrões detectados"""
        self.metricas_padroes = {
            'padroes_detectados_total': 0,
            'padroes_acertados': 0,
            'padroes_errados': 0,
            'eficiencia_por_tipo': {},
            'historico_validacao': []
        }

    def adicionar_numero(self, numero):
        self.historico.append(numero)
        self.contador_sorteios += 1
        
        # Validar padrões do sorteio anterior
        if len(self.historico) > 1:
            numero_anterior = list(self.historico)[-2]
            self.validar_padrao_acerto(numero, self.get_previsao_atual())
        
        # Analisar padrões sequenciais
        self.analisar_padroes_sequenciais(numero)
        
        # Treinamento mais frequente
        if self.contador_sorteios >= 10:  # Reduzido de 15
            self.contador_sorteios = 0
            self.treinar_automatico()
            
        salvar_sessao()

    def get_previsao_atual(self):
        """Obtém a previsão atual para validação"""
        try:
            resultado = self.analisar_ml()
            return resultado
        except:
            return None

    def validar_padrao_acerto(self, numero_sorteado, previsao_ml):
        """Valida se os padrões detectados acertaram"""
        zona_sorteada = None
        for zona, numeros in self.numeros_zonas_ml.items():
            if numero_sorteado in numeros:
                zona_sorteada = zona
                break
        
        if not zona_sorteada:
            return
        
        # Validar padrões com sistema de confirmação
        padroes_recentes = [p for p in self.sequencias_padroes['padroes_detectados'] 
                           if len(self.historico) - p['detectado_em'] <= 3]
        
        for padrao in padroes_recentes:
            # Usar sistema de confirmação
            confirmado = self.confirmacao_padroes.verificar_confirmacao_padrao(
                padrao, numero_sorteado, self.numeros_zonas_ml
            )
            
            self.metricas_padroes['padroes_detectados_total'] += 1
            
            if zona_sorteada == padrao['zona']:
                self.metricas_padroes['padroes_acertados'] += 1
                tipo = padrao['tipo']
                if tipo not in self.metricas_padroes['eficiencia_por_tipo']:
                    self.metricas_padroes['eficiencia_por_tipo'][tipo] = {'acertos': 0, 'total': 0}
                self.metricas_padroes['eficiencia_por_tipo'][tipo]['acertos'] += 1
                self.metricas_padroes['eficiencia_por_tipo'][tipo]['total'] += 1
            else:
                self.metricas_padroes['padroes_errados'] += 1
                tipo = padrao['tipo']
                if tipo in self.metricas_padroes['eficiencia_por_tipo']:
                    self.metricas_padroes['eficiencia_por_tipo'][tipo]['total'] += 1

    def analisar_padroes_sequenciais(self, numero):
        """Versão otimizada da análise de padrões"""
        if len(self.historico) < 6:
            return
            
        historico_recente = list(self.historico)[-8:]
        
        # Identificar zona atual
        zona_atual = None
        for zona, numeros in self.numeros_zonas_ml.items():
            if numero in numeros:
                zona_atual = zona
                break
        
        if not zona_atual:
            return
        
        # Atualizar sequências ativas
        self.atualizar_sequencias_ativas(zona_atual, historico_recente)
        
        # Detecção otimizada de padrões
        self.otimizar_deteccao_padroes(historico_recente)
        
        # Limpar padrões antigos
        self.limpar_padroes_antigos()

    def otimizar_deteccao_padroes(self, historico_recente):
        """Versão otimizada da detecção de padrões com mais sensibilidade"""
        if len(historico_recente) < 6:
            return
        
        # Converter histórico para zonas
        zonas_recentes = []
        for num in historico_recente:
            zona_num = None
            for zona, numeros in self.numeros_zonas_ml.items():
                if num in numeros:
                    zona_num = zona
                    break
            zonas_recentes.append(zona_num)
        
        # Padrão 1: Sequência forte interrompida brevemente (A A A B A A)
        for i in range(len(zonas_recentes) - 5):
            janela = zonas_recentes[i:i+6]
            if (janela[0] and janela[1] and janela[2] and janela[4] and janela[5] and
                janela[0] == janela[1] == janela[2] == janela[4] == janela[5] and
                janela[3] != janela[0]):
                
                self.registrar_padrao_sequencia_interrompida(janela[0], i)

        # Padrão 2: Sequência média com retorno rápido (A A B A A)
        for i in range(len(zonas_recentes) - 4):
            janela = zonas_recentes[i:i+5]
            if (janela[0] and janela[1] and janela[3] and janela[4] and
                janela[0] == janela[1] == janela[3] == janela[4] and
                janela[2] != janela[0]):
                
                self.registrar_padrao_retorno_rapido(janela[0], i)

    def registrar_padrao_sequencia_interrompida(self, zona, posicao):
        """Registra padrão de sequência interrompida com scoring"""
        padrao = {
            'tipo': 'sequencia_interrompida_forte',
            'zona': zona,
            'padrao': 'AAA_B_AA',
            'forca': 0.85,
            'duracao': 6,
            'detectado_em': len(self.historico) - 1,
            'posicao_historico': posicao
        }
        
        if not self.padrao_recente_similar(padrao):
            self.sequencias_padroes['padroes_detectados'].append(padrao)
            logging.info(f"🎯 PADRÃO FORTE: {zona} - {padrao['padrao']}")

    def registrar_padrao_retorno_rapido(self, zona, posicao):
        """Registra padrão de retorno rápido após quebra"""
        padrao = {
            'tipo': 'retorno_rapido',
            'zona': zona,
            'padrao': 'AA_B_AA',
            'forca': 0.75,
            'duracao': 5,
            'detectado_em': len(self.historico) - 1,
            'posicao_historico': posicao
        }
        
        if not self.padrao_recente_similar(padrao):
            self.sequencias_padroes['padroes_detectados'].append(padrao)
            logging.info(f"🎯 PADRÃO RÁPIDO: {zona} - {padrao['padrao']}")

    def padrao_recente_similar(self, novo_padrao, janela=12):
        """Verifica se padrão similar foi detectado recentemente"""
        for padrao in self.sequencias_padroes['padroes_detectados'][-10:]:
            if (padrao['zona'] == novo_padrao['zona'] and 
                padrao['tipo'] == novo_padrao['tipo'] and
                len(self.historico) - padrao['detectado_em'] < janela):
                return True
        return False

    def limpar_padroes_antigos(self, limite=20):
        """Remove padrões muito antigos do histórico"""
        padroes_validos = []
        for padrao in self.sequencias_padroes['padroes_detectados']:
            if len(self.historico) - padrao['detectado_em'] <= limite:
                padroes_validos.append(padrao)
        self.sequencias_padroes['padroes_detectados'] = padroes_validos

    def atualizar_sequencias_ativas(self, zona_atual, historico_recente):
        """Atualiza as sequências ativas por zona"""
        if zona_atual in self.sequencias_padroes['sequencias_ativas']:
            sequencia = self.sequencias_padroes['sequencias_ativas'][zona_atual]
            sequencia['contagem'] += 1
            sequencia['ultimo_numero'] = historico_recente[-1]
        else:
            self.sequencias_padroes['sequencias_ativas'][zona_atual] = {
                'contagem': 1,
                'inicio': len(self.historico) - 1,
                'ultimo_numero': historico_recente[-1],
                'quebras': 0
            }
        
        zonas_ativas = list(self.sequencias_padroes['sequencias_ativas'].keys())
        for zona in zonas_ativas:
            if zona != zona_atual:
                self.sequencias_padroes['sequencias_ativas'][zona]['quebras'] += 1
                
                if self.sequencias_padroes['sequencias_ativas'][zona]['quebras'] >= 3:
                    sequencia_final = self.sequencias_padroes['sequencias_ativas'][zona]
                    if sequencia_final['contagem'] >= 3:
                        self.sequencias_padroes['historico_sequencias'].append({
                            'zona': zona,
                            'tamanho': sequencia_final['contagem'],
                            'finalizado_em': len(self.historico) - 1
                        })
                    del self.sequencias_padroes['sequencias_ativas'][zona]

    def aplicar_padroes_na_previsao(self, distribuicao_zonas):
        """Aplica os padrões detectados para ajustar a previsão"""
        if not self.sequencias_padroes['padroes_detectados']:
            return distribuicao_zonas
        
        distribuicao_ajustada = distribuicao_zonas.copy()
        
        padroes_recentes = [p for p in self.sequencias_padroes['padroes_detectados'] 
                           if len(self.historico) - p['detectado_em'] <= 15]
        
        # Aplicar padrões confirmados com mais força
        padroes_confiaveis = self.confirmacao_padroes.get_padroes_confiaveis()
        
        for padrao in padroes_recentes:
            zona = padrao['zona']
            forca = padrao['forca']
            
            # Aumentar força se o padrão for confirmado
            chave_padrao = f"{padrao['tipo']}_{zona}"
            if chave_padrao in padroes_confiaveis:
                confiabilidade = padroes_confiaveis[chave_padrao]['confiabilidade']
                forca *= (1.0 + confiabilidade)  # Bônus baseado na confiabilidade
            
            if zona in distribuicao_ajustada:
                aumento = max(1, int(distribuicao_ajustada[zona] * forca * 0.4))  # Aumentado de 0.3 para 0.4
                distribuicao_ajustada[zona] += aumento
                logging.info(f"🎯 Aplicando padrão {padrao['tipo']} à zona {zona}: +{aumento}")
        
        return distribuicao_ajustada

    def calcular_confianca_com_padroes(self, distribuicao, zona_alvo):
        """Calcula confiança considerando padrões detectados"""
        confianca_base = self.calcular_confianca_zona_ml({
            'contagem': distribuicao[zona_alvo],
            'total_zonas': 25
        })
        
        padroes_recentes = [p for p in self.sequencias_padroes['padroes_detectados'] 
                           if p['zona'] == zona_alvo and 
                           len(self.historico) - p['detectado_em'] <= 15]
        
        # Bônus maior para padrões confirmados
        bonus_base = len(padroes_recentes) * 0.15
        bonus_confirmacao = 0
        
        padroes_confiaveis = self.confirmacao_padroes.get_padroes_confiaveis()
        for padrao in padroes_recentes:
            chave_padrao = f"{padrao['tipo']}_{zona_alvo}"
            if chave_padrao in padroes_confiaveis:
                bonus_confirmacao += 0.1  # Bônus adicional para padrões confirmados
        
        confianca_final = min(1.0, self.confianca_para_valor(confianca_base) + bonus_base + bonus_confirmacao)
        
        return self.valor_para_confianca(confianca_final)

    def confianca_para_valor(self, confianca_texto):
        """Converte texto de confiança para valor numérico"""
        mapa_confianca = {
            'Muito Baixa': 0.3,
            'Baixa': 0.5,
            'Média': 0.65,
            'Alta': 0.8,
            'Muito Alta': 0.9
        }
        return mapa_confianca.get(confianca_texto, 0.5)

    def valor_para_confianca(self, valor):
        """Converte valor numérico para texto de confiança"""
        if valor >= 0.85: return 'Muito Alta'
        elif valor >= 0.7: return 'Alta'
        elif valor >= 0.6: return 'Média'
        elif valor >= 0.45: return 'Baixa'
        else: return 'Muito Baixa'

    def treinar_automatico(self):
        historico_numeros = self.extrair_numeros_historico()
        
        if len(historico_numeros) >= self.ml.min_training_samples:
            try:
                success, message = self.ml.treinar_modelo(historico_numeros)
                if success:
                    logging.info(f"✅ Treinamento automático ML: {message}")
                else:
                    logging.warning(f"⚠️ Treinamento automático falhou: {message}")
            except Exception as e:
                logging.error(f"❌ Erro no treinamento automático: {e}")

    def extrair_numeros_historico(self):
        historico_numeros = []
        for item in list(self.historico):
            if isinstance(item, dict) and 'number' in item:
                historico_numeros.append(item['number'])
            elif isinstance(item, (int, float)):
                historico_numeros.append(int(item))
        return historico_numeros

    def analisar_ml(self):
        if len(self.historico) < 10:
            return None

        if not self.ml.is_trained:
            return None

        historico_numeros = self.extrair_numeros_historico()

        if len(historico_numeros) < 10:
            return None

        previsao_ml, msg_ml = self.ml.prever_proximo_numero(historico_numeros, top_k=25)
        
        if previsao_ml is None:
            logging.warning(f"❌ Previsão ML falhou: {msg_ml}")
            return None
        
        if previsao_ml:
            top_25_numeros = [num for num, prob in previsao_ml[:25]]
            
            distribuicao_zonas = self.analisar_distribuicao_zonas(top_25_numeros)
            
            if distribuicao_zonas:
                distribuicao_ajustada = self.aplicar_padroes_na_previsao(distribuicao_zonas)
                
                zona_vencedora = max(distribuicao_ajustada, key=distribuicao_ajustada.get)
                numeros_zona = self.numeros_zonas_ml[zona_vencedora]
                contagem_original = distribuicao_zonas[zona_vencedora]
                contagem_ajustada = distribuicao_ajustada[zona_vencedora]
                
                confianca = self.calcular_confianca_com_padroes(distribuicao_ajustada, zona_vencedora)
                
                padroes_aplicados = [p for p in self.sequencias_padroes['padroes_detectados'] 
                                   if p['zona'] == zona_vencedora and 
                                   len(self.historico) - p['detectado_em'] <= 15]
                
                gatilho_extra = ""
                if padroes_aplicados:
                    gatilho_extra = f" | Padrões: {len(padroes_aplicados)}"
                    # Adicionar info de confirmação
                    padroes_confiaveis = self.confirmacao_padroes.get_padroes_confiaveis()
                    padroes_confirmados = [p for p in padroes_aplicados 
                                         if f"{p['tipo']}_{zona_vencedora}" in padroes_confiaveis]
                    if padroes_confirmados:
                        gatilho_extra += f" (Confirmados: {len(padroes_confirmados)})"
                
                return {
                    'nome': 'Machine Learning - CatBoost-Hiper',
                    'numeros_apostar': numeros_zona,
                    'gatilho': f'ML CatBoost-Hiper - Zona {zona_vencedora} ({contagem_original}→{contagem_ajustada}/25){gatilho_extra}',
                    'confianca': confianca,
                    'previsao_ml': previsao_ml,
                    'zona_ml': zona_vencedora,
                    'distribuicao': distribuicao_ajustada,
                    'padroes_aplicados': len(padroes_aplicados)
                }
        
        return None

    def analisar_distribuicao_zonas(self, top_25_numeros):
        contagem_zonas = {}
        
        for zona, numeros in self.numeros_zonas_ml.items():
            count = sum(1 for num in top_25_numeros if num in numeros)
            contagem_zonas[zona] = count
        
        return contagem_zonas if contagem_zonas else None

    def calcular_confianca_zona_ml(self, distribuicao):
        contagem = distribuicao['contagem']
        total = distribuicao['total_zonas']
        percentual = (contagem / total) * 100
        
        if percentual >= 50:
            return 'Muito Alta'
        elif percentual >= 40:
            return 'Alta'
        elif percentual >= 30:
            return 'Média'
        elif percentual >= 25:
            return 'Baixa'
        else:
            return 'Muito Baixa'

    def treinar_modelo_ml(self, historico_completo=None):
        if historico_completo is not None:
            historico_numeros = historico_completo
        else:
            historico_numeros = self.extrair_numeros_historico()
        
        if len(historico_numeros) >= self.ml.min_training_samples:
            success, message = self.ml.treinar_modelo(historico_numeros)
            return success, message
        else:
            return False, f"Histórico insuficiente: {len(historico_numeros)}/{self.ml.min_training_samples} números"

    def get_analise_ml(self):
        if not self.ml.is_trained:
            return "🤖 ML: Modelo não treinado"
        
        if len(self.historico) < 10:
            return "🤖 ML: Aguardando mais dados para análise"
        
        historico_numeros = self.extrair_numeros_historico()
        
        if len(historico_numeros) < 10:
            return "🤖 ML: Histórico insuficiente para análise"
        
        previsao_ml, msg = self.ml.prever_proximo_numero(historico_numeros, top_k=25)
        
        if previsao_ml is None:
            return f"🤖 ML: {msg}"
        
        if previsao_ml:
            if self.ml.models:
                primeiro_modelo = self.ml.models[0]
                modelo_tipo = "CatBoost-Hiper" if hasattr(primeiro_modelo, 'iterations') else "RandomForest-Hiper"
            else:
                modelo_tipo = "Não treinado"
            
            analise = f"🤖 ANÁLISE ML - {modelo_tipo.upper()} (TOP 25):\n"
            analise += f"🔄 Treinamentos realizados: {self.ml.contador_treinamento}\n"
            analise += f"📊 Próximo treinamento: {10 - self.contador_sorteios} sorteios\n"
            analise += f"📈 Ensemble: {len(self.ml.models)} modelos\n"
            
            # Informações do sistema de confirmação
            padroes_confiaveis = self.confirmacao_padroes.get_padroes_confiaveis()
            if padroes_confiaveis:
                analise += f"✅ Padrões confirmados: {len(padroes_confiaveis)}\n"
                for chave, dados in list(padroes_confiaveis.items())[:3]:
                    zona = chave.split('_')[-1]
                    analise += f"   📊 {zona}: {dados['confiabilidade']:.1%} confiabilidade\n"
            
            padroes_recentes = [p for p in self.sequencias_padroes['padroes_detectados'] 
                              if len(self.historico) - p['detectado_em'] <= 20]
            
            if padroes_recentes:
                analise += f"🔍 Padrões ativos: {len(padroes_recentes)}\n"
                for padrao in padroes_recentes[-3:]:
                    idade = len(self.historico) - padrao['detectado_em']
                    analise += f"   📈 {padrao['zona']}: {padrao['tipo']} (há {idade} jogos)\n"
            
            analise += "🎯 Previsões (Top 10):\n"
            for i, (num, prob) in enumerate(previsao_ml[:10]):
                analise += f"  {i+1}. Número {num}: {prob:.2%}\n"
            
            top_25_numeros = [num for num, prob in previsao_ml[:25]]
            distribuicao = self.analisar_distribuicao_zonas(top_25_numeros)
            
            if distribuicao:
                distribuicao_ajustada = self.aplicar_padroes_na_previsao(distribuicao)
                
                analise += f"\n🎯 DISTRIBUIÇÃO POR ZONAS (25 números):\n"
                for zona, count in distribuicao_ajustada.items():
                    count_original = distribuicao[zona]
                    ajuste = count - count_original
                    simbolo_ajuste = f" (+{ajuste})" if ajuste > 0 else ""
                    analise += f"  📍 {zona}: {count_original}→{count}/25{simbolo_ajuste}\n"
                
                zona_vencedora = max(distribuicao_ajustada, key=distribuicao_ajustada.get)
                analise += f"\n💡 ZONA RECOMENDADA: {zona_vencedora}\n"
                analise += f"🎯 Confiança: {self.calcular_confianca_com_padroes(distribuicao_ajustada, zona_vencedora)}\n"
                analise += f"🔢 Números da zona: {sorted(self.numeros_zonas_ml[zona_vencedora])}\n"
                analise += f"📈 Percentual: {(distribuicao_ajustada[zona_vencedora]/25)*100:.1f}%\n"
            else:
                analise += "\n⚠️  Nenhuma zona com predominância suficiente (mínimo 7 números)\n"
            
            return analise
        else:
            return f"🤖 ML: {msg}"

    def get_estatisticas_padroes(self):
        """Retorna estatísticas dos padrões detectados"""
        if not hasattr(self, 'metricas_padroes'):
            return "📊 Métricas de padrões: Não disponível"
        
        total = self.metricas_padroes['padroes_detectados_total']
        if total == 0:
            return "📊 Métricas de padrões: Nenhum padrão validado ainda"
        
        acertos = self.metricas_padroes['padroes_acertados']
        eficiencia = (acertos / total) * 100 if total > 0 else 0
        
        estatisticas = f"📊 EFICIÊNCIA DOS PADRÕES:\n"
        estatisticas += f"✅ Padrões que acertaram: {acertos}/{total} ({eficiencia:.1f}%)\n"
        
        for tipo, dados in self.metricas_padroes['eficiencia_por_tipo'].items():
            if dados['total'] > 0:
                eff_tipo = (dados['acertos'] / dados['total']) * 100
                estatisticas += f"   🎯 {tipo}: {dados['acertos']}/{dados['total']} ({eff_tipo:.1f}%)\n"
        
        # Estatísticas de confirmação
        padroes_confiaveis = self.confirmacao_padroes.get_padroes_confiaveis()
        if padroes_confiaveis:
            estatisticas += f"✅ Padrões confirmados: {len(padroes_confiaveis)}\n"
            for chave, dados in list(padroes_confiaveis.items())[:3]:
                zona = chave.split('_')[-1]
                estatisticas += f"   📈 {zona}: {dados['acertos']}/{dados['total_ocorrencias']} ({dados['confiabilidade']:.1%})\n"
        
        padroes_ativos = [p for p in self.sequencias_padroes['padroes_detectados'] 
                         if len(self.historico) - p['detectado_em'] <= 10]
        
        estatisticas += f"🔍 Padrões ativos: {len(padroes_ativos)}\n"
        for padrao in padroes_ativos[-3:]:
            idade = len(self.historico) - padrao['detectado_em']
            estatisticas += f"   📊 {padrao['zona']}: {padrao['tipo']} (há {idade} jogos)\n"
        
        return estatisticas

    def get_info_zonas_ml(self):
        info = {}
        for zona, numeros in self.numeros_zonas_ml.items():
            info[zona] = {
                'numeros': sorted(numeros),
                'quantidade': len(numeros),
                'central': self.zonas_ml[zona],
                'descricao': f"6 antes + 6 depois do {self.zonas_ml[zona]}"
            }
        return info

    def zerar_padroes(self):
        """Zera todos os padrões detectados"""
        self.sequencias_padroes = {
            'sequencias_ativas': {},
            'historico_sequencias': [],
            'padroes_detectados': []
        }
        self.metricas_padroes = {
            'padroes_detectados_total': 0,
            'padroes_acertados': 0,
            'padroes_errados': 0,
            'eficiencia_por_tipo': {},
            'historico_validacao': []
        }
        self.confirmacao_padroes = SistemaConfirmacaoPadroes()
        logging.info("🔄 Padrões sequenciais e métricas zerados")

# =============================
# SISTEMA DE GESTÃO COMPLETO ATUALIZADO
# =============================
class SistemaRoletaCompleto:
    def __init__(self):
        self.estrategia_zonas = EstrategiaZonasHiperOtimizada()
        self.estrategia_midas = EstrategiaMidas()
        self.estrategia_ml = EstrategiaML()
        self.previsao_ativa = None
        self.historico_desempenho = []
        self.acertos = 0
        self.erros = 0
        self.estrategias_contador = {}
        self.estrategia_selecionada = "Zonas"
        self.contador_sorteios_global = 0
        
        # NOVO: Sistema de rotação inteligente
        self.rotacao_inteligente = SistemaRotacaoInteligente()
        
        # NOVO: Sistema de aprendizado contínuo
        self.aprendizado_continuo = AprendizadoContinuo()
        
        # Sistema de rotação automática (mantido para compatibilidade)
        self.sequencia_erros = 0
        self.ultima_estrategia_erro = ""

    def set_estrategia(self, estrategia):
        self.estrategia_selecionada = estrategia
        self.rotacao_inteligente.estrategia_atual = estrategia
        salvar_sessao()

    def treinar_modelo_ml(self, historico_completo=None):
        return self.estrategia_ml.treinar_modelo_ml(historico_completo)

    def rotacionar_estrategia_inteligente(self, resultado):
        """Usa o sistema inteligente para decidir rotações"""
        rotacionou, estrategia_anterior, estrategia_nova = self.rotacao_inteligente.decidir_rotacao(resultado)
        
        if rotacionou:
            self.estrategia_selecionada = estrategia_nova
            self.sequencia_erros = 0
            
            enviar_rotacao_automatica(estrategia_anterior, estrategia_nova)
            logging.info(f"🔄 ROTAÇÃO INTELIGENTE: {estrategia_anterior} → {estrategia_nova}")
            
            return True
        
        return False

    def processar_novo_numero(self, numero):
        if isinstance(numero, dict) and 'number' in numero:
            numero_real = numero['number']
        else:
            numero_real = numero
            
        self.contador_sorteios_global += 1
            
        if self.previsao_ativa:
            acerto = numero_real in self.previsao_ativa['numeros_apostar']
            nome_estrategia = self.previsao_ativa['nome']
            
            zona_acertada = None
            if acerto:
                if 'Zonas' in nome_estrategia:
                    for zona, numeros in self.estrategia_zonas.numeros_zonas.items():
                        if numero_real in numeros:
                            zona_acertada = zona
                            break
                elif 'ML' in nome_estrategia:
                    for zona, numeros in self.estrategia_ml.numeros_zonas_ml.items():
                        if numero_real in numeros:
                            zona_acertada = zona
                            break
            
            # NOVO: Usar rotação inteligente
            resultado = {
                'estrategia': nome_estrategia,
                'acerto': acerto,
                'numero': numero_real,
                'zona_acertada': zona_acertada
            }
            
            rotacionou = self.rotacionar_estrategia_inteligente(resultado)
            
            # NOVO: Atualizar aprendizado por período
            periodo = self.aprendizado_continuo.analisar_performance_temporal()
            self.aprendizado_continuo.atualizar_performance_periodo(
                periodo['periodo'], acerto
            )
            
            if nome_estrategia not in self.estrategias_contador:
                self.estrategias_contador[nome_estrategia] = {'acertos': 0, 'total': 0}
            
            self.estrategias_contador[nome_estrategia]['total'] += 1
            if acerto:
                self.estrategias_contador[nome_estrategia]['acertos'] += 1
                self.acertos += 1
            else:
                self.erros += 1
            
            enviar_resultado_super_simplificado(numero_real, acerto, nome_estrategia, zona_acertada)
            
            self.historico_desempenho.append({
                'numero': numero_real,
                'acerto': acerto,
                'estrategia': nome_estrategia,
                'previsao': self.previsao_ativa['numeros_apostar'],
                'rotacionou': rotacionou,
                'zona_acertada': zona_acertada,
                'timestamp': datetime.now().isoformat()
            })
            
            self.previsao_ativa = None
        
        self.estrategia_zonas.adicionar_numero(numero_real)
        self.estrategia_midas.adicionar_numero(numero_real)
        self.estrategia_ml.adicionar_numero(numero_real)
        
        nova_estrategia = None
        
        # Aplicar ajustes do aprendizado contínuo
        ajustes_periodo = self.aprendizado_continuo.analisar_performance_temporal()
        
        if self.estrategia_selecionada == "Zonas":
            nova_estrategia = self.estrategia_zonas.analisar_zonas()
        elif self.estrategia_selecionada == "Midas":
            nova_estrategia = self.estrategia_midas.analisar_midas()
        elif self.estrategia_selecionada == "ML":
            nova_estrategia = self.estrategia_ml.analisar_ml()
        
        if nova_estrategia:
            self.previsao_ativa = nova_estrategia
            enviar_previsao_super_simplificada(nova_estrategia)

    def zerar_estatisticas_desempenho(self):
        """Zera todas as estatísticas de desempenho"""
        self.acertos = 0
        self.erros = 0
        self.estrategias_contador = {}
        self.historico_desempenho = []
        self.contador_sorteios_global = 0
        self.sequencia_erros = 0
        self.ultima_estrategia_erro = ""
        
        # Zerar sistemas inteligentes
        self.rotacao_inteligente = SistemaRotacaoInteligente()
        self.aprendizado_continuo = AprendizadoContinuo()
        
        # Zerar estatísticas das estratégias
        self.estrategia_zonas.zerar_estatisticas()
        self.estrategia_ml.zerar_padroes()
        
        logging.info("📊 Todas as estatísticas de desempenho foram zeradas")
        salvar_sessao()

    def reset_recente_estatisticas(self):
        """Faz um reset recente mantendo apenas os últimos 10 resultados"""
        if len(self.historico_desempenho) > 10:
            self.historico_desempenho = self.historico_desempenho[-10:]
            
            self.acertos = sum(1 for resultado in self.historico_desempenho if resultado['acerto'])
            self.erros = len(self.historico_desempenho) - self.acertos
            
            self.estrategias_contador = {}
            for resultado in self.historico_desempenho:
                estrategia = resultado['estrategia']
                if estrategia not in self.estrategias_contador:
                    self.estrategias_contador[estrategia] = {'acertos': 0, 'total': 0}
                
                self.estrategias_contador[estrategia]['total'] += 1
                if resultado['acerto']:
                    self.estrategias_contador[estrategia]['acertos'] += 1
            
            ultimos_resultados = self.historico_desempenho[-5:]
            self.sequencia_erros = 0
            for resultado in reversed(ultimos_resultados):
                if not resultado['acerto']:
                    self.sequencia_erros += 1
                else:
                    break
            
            logging.info("🔄 Estatísticas recentes resetadas (mantidos últimos 10 resultados)")
        else:
            logging.info("ℹ️  Histórico muito pequeno para reset recente")
        
        salvar_sessao()

    def get_status_rotacao(self):
        """Retorna o status atual do sistema de rotação"""
        status_inteligente = self.rotacao_inteligente.get_status_rotacao()
        
        return {
            'estrategia_atual': self.estrategia_selecionada,
            'sequencia_erros': self.sequencia_erros,
            'ultima_estrategia_erro': self.ultima_estrategia_erro,
            'performances': status_inteligente['performances'],
            'performance_minima': status_inteligente['performance_minima'],
            'proxima_avaliacao': status_inteligente['proxima_avaliacao_em']
        }

    def get_analise_temporal(self):
        """Retorna análise de performance temporal"""
        return self.aprendizado_continuo.get_analise_temporal()

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
# APLICAÇÃO STREAMLIT ATUALIZADA
# =============================
st.set_page_config(page_title="IA Roleta — Sistema Multi-Estratégias Hiper", layout="centered")
st.title("🎯 IA Roleta — Sistema Multi-Estratégias Hiper Otimizado")

# Inicialização com persistência
if "sistema" not in st.session_state:
    st.session_state.sistema = SistemaRoletaCompleto()

# Tentar carregar sessão salva
sessao_carregada = carregar_sessao()

if "historico" not in st.session_state:
    if not sessao_carregada and os.path.exists(HISTORICO_PATH):
        try:
            with open(HISTORICO_PATH, "r") as f:
                st.session_state.historico = json.load(f)
        except:
            st.session_state.historico = []
    elif not sessao_carregada:
        st.session_state.historico = []

if "telegram_token" not in st.session_state and not sessao_carregada:
    st.session_state.telegram_token = ""
if "telegram_chat_id" not in st.session_state and not sessao_carregada:
    st.session_state.telegram_chat_id = ""

# Sidebar - Configurações Avançadas
st.sidebar.title("⚙️ Configurações Hiper")

# Gerenciamento de Sessão
with st.sidebar.expander("💾 Gerenciamento de Sessão", expanded=False):
    st.write("**Persistência de Dados**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Salvar Sessão", use_container_width=True):
            salvar_sessao()
            st.success("✅ Sessão salva!")
            
    with col2:
        if st.button("🔄 Carregar Sessão", use_container_width=True):
            if carregar_sessao():
                st.success("✅ Sessão carregada!")
                st.rerun()
            else:
                st.error("❌ Nenhuma sessão salva encontrada")
    
    st.write("---")
    
    st.write("**📊 Gerenciar Estatísticas**")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🔄 Reset Recente", help="Mantém apenas os últimos 10 resultados", use_container_width=True):
            st.session_state.sistema.reset_recente_estatisticas()
            st.success("✅ Estatísticas recentes resetadas!")
            st.rerun()
            
    with col4:
        if st.button("🗑️ Zerar Tudo", type="secondary", help="Zera TODAS as estatísticas", use_container_width=True):
            if st.checkbox("Confirmar zerar TODAS as estatísticas"):
                st.session_state.sistema.zerar_estatisticas_desempenho()
                st.error("🗑️ Todas as estatísticas foram zeradas!")
                st.rerun()
    
    st.write("---")
    
    if st.button("🗑️ Limpar TODOS os Dados", type="secondary", use_container_width=True):
        if st.checkbox("Confirmar limpeza total de todos os dados"):
            limpar_sessao()
            st.error("🗑️ Todos os dados foram limpos!")
            st.stop()

# Configurações do Telegram
with st.sidebar.expander("🔔 Configurações do Telegram", expanded=False):
    st.write("Configure as notificações do Telegram")
    
    telegram_token = st.text_input(
        "Bot Token do Telegram:",
        value=st.session_state.telegram_token,
        type="password",
        help="Obtenha com @BotFather no Telegram"
    )
    
    telegram_chat_id = st.text_input(
        "Chat ID do Telegram:",
        value=st.session_state.telegram_chat_id,
        help="Obtenha com @userinfobot no Telegram"
    )
    
    if st.button("Salvar Configurações Telegram"):
        st.session_state.telegram_token = telegram_token
        st.session_state.telegram_chat_id = telegram_chat_id
        salvar_sessao()
        st.success("✅ Configurações do Telegram salvas!")
        
    if st.button("Testar Conexão Telegram"):
        if telegram_token and telegram_chat_id:
            try:
                enviar_telegram("🔔 Teste de conexão - IA Roleta Hiper funcionando!")
                st.success("✅ Mensagem de teste enviada para Telegram!")
            except Exception as e:
                st.error(f"❌ Erro ao enviar mensagem: {e}")
        else:
            st.error("❌ Preencha token e chat ID primeiro")

# Seleção de Estratégia
estrategia = st.sidebar.selectbox(
    "🎯 Selecione a Estratégia:",
    ["Zonas", "Midas", "ML"],
    key="estrategia_selecionada"
)

# Aplicar estratégia selecionada
if estrategia != st.session_state.sistema.estrategia_selecionada:
    st.session_state.sistema.set_estrategia(estrategia)
    st.toast(f"🔄 Estratégia alterada para: {estrategia}")

# Status da Rotação Inteligente
with st.sidebar.expander("🔄 Rotação Inteligente", expanded=True):
    status_rotacao = st.session_state.sistema.get_status_rotacao()
    
    st.write("**Sistema de Rotação Inteligente:**")
    st.write(f"🎯 **Estratégia Atual:** {status_rotacao['estrategia_atual']}")
    st.write(f"📊 **Performance Mínima:** {status_rotacao['performance_minima']:.1%}")
    
    st.write("**Performances por Estratégia:**")
    for estrategia, performance in status_rotacao['performances'].items():
        cor = "🟢" if performance >= status_rotacao['performance_minima'] else "🔴"
        st.write(f"{cor} **{estrategia}:** {performance:.1%}")
    
    st.write("---")
    st.write("**Regras de Rotação Inteligente:**")
    st.write("• 📊 **Avaliação contínua** de performance")
    st.write("• 🔄 **Rotação automática** quando performance < 35%")
    st.write("• 🎯 **Seleção da melhor** estratégia disponível")
    st.write("• ⏱️ **Janela de análise:** últimos 20 resultados")
    
    # Botão para forçar rotação manual
    if st.button("🔄 Forçar Rotação Inteligente", use_container_width=True):
        estrategia_atual = st.session_state.sistema.estrategia_selecionada
        if estrategia_atual == "Zonas":
            nova_estrategia = "ML"
        else:
            nova_estrategia = "Zonas"
        
        st.session_state.sistema.estrategia_selecionada = nova_estrategia
        st.session_state.sistema.rotacao_inteligente.estrategia_atual = nova_estrategia
        st.success(f"🔄 Rotação forçada: {estrategia_atual} → {nova_estrategia}")
        st.rerun()

# Treinamento ML Hiper
with st.sidebar.expander("🧠 Treinamento ML Hiper", expanded=False):
    numeros_disponiveis = 0
    numeros_lista = []
    
    for item in st.session_state.historico:
        if isinstance(item, dict) and 'number' in item and item['number'] is not None:
            numeros_disponiveis += 1
            numeros_lista.append(item['number'])
        elif isinstance(item, (int, float)) and item is not None:
            numeros_disponiveis += 1
            numeros_lista.append(int(item))
            
    st.write(f"📊 **Números disponíveis:** {numeros_disponiveis}")
    st.write(f"🎯 **Mínimo necessário:** 150 números")  # Reduzido
    st.write(f"🔄 **Treinamento automático:** A cada 10 sorteios")  # Mais frequente
    st.write(f"🤖 **Modelo:** CatBoost-Hiper (mais preciso)")
    st.write(f"🎯 **Ensemble:** 4 modelos")  # Aumentado
    
    if numeros_disponiveis > 0:
        numeros_unicos = len(set(numeros_lista))
        st.write(f"🎲 **Números únicos:** {numeros_unicos}/37")
        
        if numeros_unicos < 10:
            st.warning(f"⚠️ **Pouca variedade:** Necessário pelo menos 10 números diferentes")
        else:
            st.success(f"✅ **Variedade adequada:** {numeros_unicos} números diferentes")
    
    st.write(f"✅ **Status:** {'Dados suficientes' if numeros_disponiveis >= 150 else 'Coletando dados...'}")
    
    if numeros_disponiveis >= 150:
        st.success("✨ **Pronto para treinar!**")
        
        if st.button("🚀 Treinar Modelo ML Hiper", type="primary", use_container_width=True):
            with st.spinner("Treinando modelo ML Hiper com CatBoost... Isso pode levar alguns segundos"):
                try:
                    success, message = st.session_state.sistema.treinar_modelo_ml(numeros_lista)
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"💥 Erro no treinamento: {str(e)}")
    
    else:
        st.warning(f"📥 Colete mais {150 - numeros_disponiveis} números para treinar o ML")
        
    st.write("---")
    st.write("**Status do ML Hiper:**")
    if st.session_state.sistema.estrategia_ml.ml.is_trained:
        if st.session_state.sistema.estrategia_ml.ml.models:
            primeiro_modelo = st.session_state.sistema.estrategia_ml.ml.models[0]
            modelo_tipo = "CatBoost-Hiper" if hasattr(primeiro_modelo, 'iterations') else "RandomForest-Hiper"
        else:
            modelo_tipo = "Não treinado"
            
        st.success(f"✅ Modelo {modelo_tipo} treinado ({st.session_state.sistema.estrategia_ml.ml.contador_treinamento} vezes)")
        if 'last_accuracy' in st.session_state.sistema.estrategia_ml.ml.meta:
            acc = st.session_state.sistema.estrategia_ml.ml.meta['last_accuracy']
            st.info(f"📊 Última acurácia: {acc:.2%}")
        st.info(f"🔄 Próximo treinamento automático em: {10 - st.session_state.sistema.estrategia_ml.contador_sorteios} sorteios")
        st.info(f"🎯 Ensemble: {len(st.session_state.sistema.estrategia_ml.ml.models)} modelos ativos")
    else:
        st.info("🤖 ML Hiper aguardando treinamento")
        
    if st.button("🔄 Forçar Retreinamento ML Hiper", use_container_width=True):
        st.session_state.sistema.estrategia_ml.ml.is_trained = False
        st.session_state.sistema.estrategia_ml.ml.models = []
        st.session_state.sistema.estrategia_ml.ml.scaler = StandardScaler()
        st.info("🔄 Modelo ML Hiper resetado. Treine novamente.")

# Estatísticas de Padrões ML
with st.sidebar.expander("🔍 Estatísticas de Padrões ML", expanded=False):
    if st.session_state.sistema.estrategia_selecionada == "ML":
        estatisticas_padroes = st.session_state.sistema.estrategia_ml.get_estatisticas_padroes()
        st.text(estatisticas_padroes)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if st.button("🔄 Zerar Padrões", use_container_width=True):
                st.session_state.sistema.estrategia_ml.zerar_padroes()
                st.success("✅ Padrões zerados!")
                st.rerun()
                
        with col_p2:
            if st.button("📊 Atualizar Métricas", use_container_width=True):
                st.rerun()
    else:
        st.info("🔍 Ative a estratégia ML para ver estatísticas de padrões")

# Análise Temporal
with st.sidebar.expander("🕒 Análise Temporal", expanded=False):
    analise_temporal = st.session_state.sistema.get_analise_temporal()
    st.text(analise_temporal)

# Informações sobre as Estratégias
with st.sidebar.expander("📊 Informações das Estratégias Hiper"):
    if estrategia == "Zonas":
        info_zonas = st.session_state.sistema.estrategia_zonas.get_info_zonas()
        st.write("**🎯 Estratégia Zonas v7 (Hiper Otimizada):**")
        st.write("**CONFIGURAÇÃO:** 6 antes + 6 depois (13 números/zona)")
        st.write("**OTIMIZAÇÕES HIPER:**")
        st.write("- 📊 Histórico: 70 números")
        st.write("- 🎯 Múltiplas janelas: Curto(8) Médio(20) Longo(40)")
        st.write("- 📈 Threshold dinâmico agressivo: Base 25")
        st.write("- 🔥 Bônus aumentado para sequências")
        st.write("- ⚡ Foco no curto prazo")
        for zona, dados in info_zonas.items():
            st.write(f"**Zona {zona}** (Núcleo: {dados['central']})")
            st.write(f"Descrição: {dados['descricao']}")
            st.write(f"Números: {', '.join(map(str, dados['numeros']))}")
            st.write(f"Total: {dados['quantidade']} números")
            st.write("---")
    
    elif estrategia == "Midas":
        st.write("**🎯 Estratégia Midas:**")
        st.write("Padrões baseados em terminais:")
        st.write("- **Terminal 0**: 0, 10, 20, 30")
        st.write("- **Terminal 7**: 7, 17, 27") 
        st.write("- **Terminal 5**: 5, 15, 25, 35")
        st.write("---")
    
    elif estrategia == "ML":
        st.write("**🤖 Estratégia Machine Learning - CATBOOST HIPER:**")
        st.write("- **Modelo**: CatBoost-Hiper (Gradient Boosting Otimizado)")
        st.write("- **Ensemble**: 4 modelos (3 → 4)")
        st.write("- **Amostras mínimas**: 150 (200 → 150)")
        st.write("- **Histórico máximo**: 1000 números")
        st.write("- **Treinamento**: A cada 10 sorteios (15 → 10)")
        st.write("- **Janelas**: [3, 8, 15, 30, 60, 120]")
        st.write("- **Zonas**: 6 antes + 6 depois (13 números/zona)")
        st.write("- **Sistema de Confirmação**: Padrões validados")
        st.write("- **Saída**: Zona com maior concentração ajustada")
        
        info_zonas_ml = st.session_state.sistema.estrategia_ml.get_info_zonas_ml()
        for zona, dados in info_zonas_ml.items():
            st.write(f"**Zona {zona}** (Núcleo: {dados['central']})")
            st.write(f"Descrição: {dados['descricao']}")
            st.write(f"Números: {', '.join(map(str, dados['numeros']))}")
            st.write(f"Total: {dados['quantidade']} números")
            st.write("---")

# Análise detalhada
with st.sidebar.expander(f"🔍 Análise - {estrategia}", expanded=False):
    if estrategia == "Zonas":
        analise = st.session_state.sistema.estrategia_zonas.get_analise_detalhada()
    elif estrategia == "ML":
        try:
            analise = st.session_state.sistema.estrategia_ml.get_analise_ml()
        except Exception as e:
            analise = f"🤖 ML: Erro na análise - {str(e)}"
    else:
        analise = "🎯 Estratégia Midas ativa\nAnalisando padrões de terminais..."
    
    st.text(analise)

# Interface principal
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
        salvar_sessao()
        st.success(f"{len(nums)} números adicionados!")
        st.rerun()
    except Exception as e:
        st.error(f"Erro: {e}")

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
        st.session_state.sistema.processar_novo_numero(resultado)
        salvar_resultado_em_arquivo(st.session_state.historico)
        salvar_sessao()

# Interface principal
st.subheader("🔁 Últimos Números")
if st.session_state.historico:
    ultimos_10 = st.session_state.historico[-10:]
    numeros_str = " ".join(str(item['number'] if isinstance(item, dict) else item) for item in ultimos_10)
    st.write(numeros_str)
else:
    st.write("Nenhum número registrado")

# Status da Rotação Inteligente na Interface Principal
status_rotacao = st.session_state.sistema.get_status_rotacao()
col_status1, col_status2, col_status3 = st.columns(3)
with col_status1:
    st.metric("🎯 Estratégia Atual", status_rotacao['estrategia_atual'])
with col_status2:
    melhor_estrategia = max(status_rotacao['performances'], key=status_rotacao['performances'].get)
    st.metric("🏆 Melhor Estratégia", f"{melhor_estrategia} ({status_rotacao['performances'][melhor_estrategia]:.1%})")
with col_status3:
    perf_atual = status_rotacao['performances'][status_rotacao['estrategia_atual']]
    st.metric("📊 Performance Atual", f"{perf_atual:.1%}")

st.subheader("🎯 Previsão Ativa")
sistema = st.session_state.sistema

if sistema.previsao_ativa:
    previsao = sistema.previsao_ativa
    st.success(f"**{previsao['nome']}**")
    
    if 'Zonas' in previsao['nome']:
        zona = previsao.get('zona', '')
        if zona == 'Vermelha':
            nucleo = "7"
        elif zona == 'Azul':
            nucleo = "10"
        elif zona == 'Amarela':
            nucleo = "2"
        else:
            nucleo = zona
        st.write(f"**📍 Núcleo:** {nucleo}")
    elif 'ML' in previsao['nome']:
        zona_ml = previsao.get('zona_ml', '')
        if zona_ml == 'Vermelha':
            nucleo = "7"
        elif zona_ml == 'Azul':
            nucleo = "10"
        elif zona_ml == 'Amarela':
            nucleo = "2"
        else:
            nucleo = zona_ml
        st.write(f"**🤖 Núcleo:** {nucleo}")
    
    st.write(f"**🔢 Números para apostar ({len(previsao['numeros_apostar'])}):**")
    st.write(", ".join(map(str, sorted(previsao['numeros_apostar']))))
    
    if 'ML' in previsao['nome'] and previsao.get('padroes_aplicados', 0) > 0:
        st.info(f"🔍 **Padrões aplicados:** {previsao['padroes_aplicados']} padrões sequenciais detectados")
    
    st.info("⏳ Aguardando próximo sorteio para conferência...")
else:
    st.info(f"🎲 Analisando padrões ({estrategia})...")

# Desempenho
st.subheader("📈 Desempenho Hiper")

total = sistema.acertos + sistema.erros
taxa = (sistema.acertos / total * 100) if total > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Acertos", sistema.acertos)
col2.metric("🔴 Erros", sistema.erros)
col3.metric("📊 Total", total)
col4.metric("✅ Taxa", f"{taxa:.1f}%")

# Botões de gerenciamento
st.write("**Gerenciar Estatísticas:**")
col5, col6 = st.columns(2)

with col5:
    if st.button("🔄 Reset Recente", help="Mantém apenas os últimos 10 resultados", use_container_width=True):
        st.session_state.sistema.reset_recente_estatisticas()
        st.success("✅ Estatísticas recentes resetadas!")
        st.rerun()

with col6:
    if st.button("🗑️ Zerar Tudo", type="secondary", help="Zera TODAS as estatísticas", use_container_width=True):
        if st.checkbox("Confirmar zerar TODAS as estatísticas"):
            st.session_state.sistema.zerar_estatisticas_desempenho()
            st.error("🗑️ Todas as estatísticas foram zeradas!")
            st.rerun()

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
        rotacao_emoji = " 🔄" if resultado.get('rotacionou', False) else ""
        zona_info = ""
        if resultado['acerto'] and resultado.get('zona_acertada'):
            if resultado['zona_acertada'] == 'Vermelha':
                nucleo = "7"
            elif resultado['zona_acertada'] == 'Azul':
                nucleo = "10"
            elif resultado['zona_acertada'] == 'Amarela':
                nucleo = "2"
            else:
                nucleo = resultado['zona_acertada']
                
            if 'Zonas' in resultado['estrategia']:
                zona_info = f" (Núcleo {nucleo})"
            elif 'ML' in resultado['estrategia']:
                zona_info = f" (Núcleo {nucleo})"
        st.write(f"{emoji}{rotacao_emoji} {resultado['estrategia']}: Número {resultado['numero']}{zona_info}")

# Download histórico
if os.path.exists(HISTORICO_PATH):
    with open(HISTORICO_PATH, "r") as f:
        conteudo = f.read()
    st.download_button("📥 Baixar histórico", data=conteudo, file_name="historico_roleta.json")

# Salvar sessão automaticamente ao final do script
salvar_sessao()   
