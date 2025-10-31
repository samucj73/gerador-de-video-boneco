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

# =============================
# CONFIGURAÇÕES DE PERSISTÊNCIA - CORRIGIDAS
# =============================
SESSION_DATA_PATH = "session_data.pkl"
HISTORICO_PATH = "historico_coluna_duzia.json"
ML_MODEL_PATH = "ml_roleta_model.pkl"
SCALER_PATH = "ml_scaler.pkl"
META_PATH = "ml_meta.pkl"

def salvar_sessao():
    """Salva todos os dados da sessão em arquivo"""
    try:
        # Verificar se o sistema existe antes de tentar acessar seus atributos
        if 'sistema' not in st.session_state:
            logging.warning("Sistema não encontrado no session_state ao salvar")
            return
            
        session_data = {
            'historico': st.session_state.get('historico', []),
            'telegram_token': st.session_state.get('telegram_token', ''),
            'telegram_chat_id': st.session_state.get('telegram_chat_id', ''),
            'sistema_acertos': st.session_state.sistema.acertos,
            'sistema_erros': st.session_state.sistema.erros,
            'sistema_estrategias_contador': st.session_state.sistema.estrategias_contador,
            'sistema_historico_desempenho': st.session_state.sistema.historico_desempenho,
            'sistema_contador_sorteios_global': st.session_state.sistema.contador_sorteios_global,
            'sistema_sequencia_erros': st.session_state.sistema.sequencia_erros,
            'sistema_ultima_estrategia_erro': st.session_state.sistema.ultima_estrategia_erro,
            'estrategia_selecionada': st.session_state.sistema.estrategia_selecionada
        }
        
        # Adicionar dados das estratégias apenas se existirem
        if hasattr(st.session_state.sistema, 'estrategia_zonas'):
            session_data['zonas_historico'] = list(st.session_state.sistema.estrategia_zonas.historico)
            session_data['zonas_stats'] = st.session_state.sistema.estrategia_zonas.stats_zonas
            
        if hasattr(st.session_state.sistema, 'estrategia_midas'):
            session_data['midas_historico'] = list(st.session_state.sistema.estrategia_midas.historico)
            
        if hasattr(st.session_state.sistema, 'estrategia_ml'):
            session_data['ml_historico'] = list(st.session_state.sistema.estrategia_ml.historico)
            session_data['ml_contador_sorteios'] = st.session_state.sistema.estrategia_ml.contador_sorteios
            session_data['ml_sequencias_padroes'] = getattr(st.session_state.sistema.estrategia_ml, 'sequencias_padroes', {
                'sequencias_ativas': {},
                'historico_sequencias': [],
                'padroes_detectados': []
            })
            session_data['ml_metricas_padroes'] = getattr(st.session_state.sistema.estrategia_ml, 'metricas_padroes', {
                'padroes_detectados_total': 0,
                'padroes_acertados': 0,
                'padroes_errados': 0,
                'eficiencia_por_tipo': {},
                'historico_validacao': []
            })
        
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
            
            # Restaurar sistema apenas se já existir
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
                if hasattr(st.session_state.sistema, 'estrategia_zonas'):
                    zonas_historico = session_data.get('zonas_historico', [])
                    st.session_state.sistema.estrategia_zonas.historico = deque(zonas_historico, maxlen=35)
                    st.session_state.sistema.estrategia_zonas.stats_zonas = session_data.get('zonas_stats', {
                        'Vermelha': {'acertos': 0, 'tentativas': 0, 'sequencia_atual': 0, 'sequencia_maxima': 0, 'performance_media': 0},
                        'Azul': {'acertos': 0, 'tentativas': 0, 'sequencia_atual': 0, 'sequencia_maxima': 0, 'performance_media': 0},
                        'Amarela': {'acertos': 0, 'tentativas': 0, 'sequencia_atual': 0, 'sequencia_maxima': 0, 'performance_media': 0}
                    })
                
                # Restaurar estratégia Midas
                if hasattr(st.session_state.sistema, 'estrategia_midas'):
                    midas_historico = session_data.get('midas_historico', [])
                    st.session_state.sistema.estrategia_midas.historico = deque(midas_historico, maxlen=15)
                
                # Restaurar estratégia ML
                if hasattr(st.session_state.sistema, 'estrategia_ml'):
                    ml_historico = session_data.get('ml_historico', [])
                    st.session_state.sistema.estrategia_ml.historico = deque(ml_historico, maxlen=30)
                    st.session_state.sistema.estrategia_ml.contador_sorteios = session_data.get('ml_contador_sorteios', 0)
                    
                    # Restaurar padrões sequenciais
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
# CONFIGURAÇÕES DE NOTIFICAÇÃO - CORRIGIDAS
# =============================
def enviar_previsao_super_simplificada(previsao):
    """Envia notificação de previsão super simplificada"""
    try:
        nome_estrategia = previsao['nome']
        
        if 'Zonas' in nome_estrategia:
            zona = previsao.get('zona', '')
            # CORREÇÃO: Mostrar sempre "Núcleo X" onde X é o número central
            if zona == 'Vermelha':
                mensagem = "📍 Núcleo 7"
            elif zona == 'Azul':
                mensagem = "📍 Núcleo 10"
            elif zona == 'Amarela':
                mensagem = "📍 Núcleo 2"
            else:
                mensagem = f"📍 Núcleo {zona}"
            
        elif 'ML' in nome_estrategia:
            # CORREÇÃO: Usar campo correto e mostrar núcleo
            zona_ml = previsao.get('zona_ml', previsao.get('zona', ''))
            if zona_ml == 'Vermelha':
                mensagem = "🤖 Núcleo 7"
            elif zona_ml == 'Azul':
                mensagem = "🤖 Núcleo 10"  
            elif zona_ml == 'Amarela':
                mensagem = "🤖 Núcleo 2"
            else:
                mensagem = f"🤖 Núcleo {zona_ml}"
            
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
                # CORREÇÃO: Mostrar sempre número do núcleo
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
                # CORREÇÃO: Mostrar sempre número do núcleo
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
        if acerto:
            st.success(f"📢 {mensagem}")
        else:
            st.error(f"📢 {mensagem}")
        
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
        token = st.session_state.get('telegram_token', '')
        chat_id = st.session_state.get('telegram_chat_id', '')
        
        if not token or not chat_id:
            logging.warning("Token ou Chat ID do Telegram não configurado")
            return
            
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
# CLASSE PRINCIPAL DA ROLETA (MANTIDA)
# =============================
class RoletaInteligente:
    def __init__(self):
        self.race = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
        
    def get_vizinhos_zona(self, numero_central, quantidade=6):
        if numero_central not in self.race:
            return []
        
        posicao = self.race.index(numero_central)
        vizinhos = []
        
        for offset in range(-quantidade, 0):
            vizinho = self.race[(posicao + offset) % len(self.race)]
            vizinhos.append(vizinho)
        
        vizinhos.append(numero_central)
        
        for offset in range(1, quantidade + 1):
            vizinho = self.race[(posicao + offset) % len(self.race)]
            vizinhos.append(vizinho)
        
        return vizinhos

    def get_posicao_race(self, numero):
        return self.race.index(numero) if numero in self.race else -1

    def get_vizinhos_fisicos(self, numero, raio=3):
        if numero not in self.race:
            return []
        
        posicao = self.race.index(numero)
        vizinhos = []
        
        for offset in range(-raio, raio + 1):
            if offset != 0:
                vizinho = self.race[(posicao + offset) % len(self.race)]
                vizinhos.append(vizinho)
        
        return vizinhos

# =============================
# MÓDULO DE MACHINE LEARNING (MANTIDO)
# =============================
class MLRoleta:
    def __init__(
        self,
        roleta_obj,
        min_training_samples: int = 100,
        max_history: int = 500,
        retrain_every_n: int = 10,
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

        self.window_for_features = [5, 10, 20, 50]
        self.k_vizinhos = 2
        self.numeros = list(range(37))

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

            # Features básicas mantidas
            K_seq = 10
            ultimos = historico[-K_seq:]
            for i in range(K_seq):
                val = ultimos[i] if i < len(ultimos) else -1
                features.append(val)
                names.append(f"ultimo_{i+1}")

            for w in self.window_for_features:
                janela = historico[-w:] if N >= w else historico[:]
                arr = np.array(janela, dtype=float)
                features.append(arr.mean() if len(arr) > 0 else 0.0); names.append(f"media_{w}")
                features.append(arr.std() if len(arr) > 1 else 0.0); names.append(f"std_{w}")
                features.append(np.median(arr) if len(arr) > 0 else 0.0); names.append(f"mediana_{w}")

            # ... (restante das features mantido igual)

            self.feature_names = names
            return features, names

        except Exception as e:
            logging.error(f"[extrair_features] Erro: {e}")
            return None, None

    # ... (restante dos métodos mantidos iguais)

# =============================
# ESTRATÉGIA DAS ZONAS (MANTIDA)
# =============================
class EstrategiaZonasOtimizada:
    def __init__(self):
        self.roleta = RoletaInteligente()
        self.historico = deque(maxlen=35)
        self.nome = "Zonas Ultra Otimizada v5"
        
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

    # ... (métodos mantidos iguais)

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

    # ... (métodos mantidos iguais)

# =============================
# ESTRATÉGIA ML - CORREÇÕES APLICADAS
# =============================
class EstrategiaML:
    def __init__(self):
        self.roleta = RoletaInteligente()
        self.ml = MLRoleta(self.roleta)
        self.historico = deque(maxlen=30)
        self.nome = "Machine Learning (CatBoost)"
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

        # CORREÇÃO: Inicialização correta dos padrões sequenciais
        self.sequencias_padroes = {
            'sequencias_ativas': {},
            'historico_sequencias': [],
            'padroes_detectados': []
        }
        
        # CORREÇÃO: Inicialização correta das métricas
        self.metricas_padroes = {
            'padroes_detectados_total': 0,
            'padroes_acertados': 0,
            'padroes_errados': 0,
            'eficiencia_por_tipo': {},
            'historico_validacao': []
        }

    def adicionar_numero(self, numero):
        """Adiciona número ao histórico"""
        self.historico.append(numero)
        self.contador_sorteios += 1
        
        # CORREÇÃO: Verificação segura para validação de padrões
        if len(self.historico) > 1:
            try:
                numero_anterior = list(self.historico)[-2]
                self.validar_padrao_acerto(numero, self.get_previsao_atual())
            except Exception as e:
                logging.error(f"Erro na validação de padrões: {e}")
        
        # Análise de padrões sequenciais
        try:
            self.analisar_padroes_sequenciais(numero)
        except Exception as e:
            logging.error(f"Erro na análise de padrões: {e}")
        
        if self.contador_sorteios >= 10:
            self.contador_sorteios = 0
            self.treinar_automatico()
            
        # Salvar sessão após adicionar número
        if 'sistema' in st.session_state:
            salvar_sessao()

    def get_previsao_atual(self):
        """Obtém a previsão atual para validação"""
        try:
            return self.analisar_ml()
        except:
            return None

    def validar_padrao_acerto(self, numero_sorteado, previsao_ml):
        """Valida se os padrões detectados acertaram"""
        if not previsao_ml:
            return
            
        zona_sorteada = None
        for zona, numeros in self.numeros_zonas_ml.items():
            if numero_sorteado in numeros:
                zona_sorteada = zona
                break
        
        if not zona_sorteada:
            return
        
        # Verificar padrões recentes
        padroes_recentes = [p for p in self.sequencias_padroes['padroes_detectados'] 
                           if len(self.historico) - p['detectado_em'] <= 3]
        
        for padrao in padroes_recentes:
            self.metricas_padroes['padroes_detectados_total'] += 1
            
            if padrao['zona'] == zona_sorteada:
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
        """Análise de padrões sequenciais"""
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
        
        # Detecção de padrões
        self.otimizar_deteccao_padroes(historico_recente)
        
        # Limpar padrões antigos
        self.limpar_padroes_antigos()

    def otimizar_deteccao_padroes(self, historico_recente):
        """Detecção otimizada de padrões"""
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
        
        # Padrões sequenciais
        for i in range(len(zonas_recentes) - 5):
            janela = zonas_recentes[i:i+6]
            if (janela[0] and janela[1] and janela[2] and janela[4] and janela[5] and
                janela[0] == janela[1] == janela[2] == janela[4] == janela[5] and
                janela[3] != janela[0]):
                
                self.registrar_padrao_sequencia_interrompida(janela[0], i)

        for i in range(len(zonas_recentes) - 4):
            janela = zonas_recentes[i:i+5]
            if (janela[0] and janela[1] and janela[3] and janela[4] and
                janela[0] == janela[1] == janela[3] == janela[4] and
                janela[2] != janela[0]):
                
                self.registrar_padrao_retorno_rapido(janela[0], i)

    def registrar_padrao_sequencia_interrompida(self, zona, posicao):
        """Registra padrão de sequência interrompida"""
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
        """Registra padrão de retorno rápido"""
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
        """Remove padrões muito antigos"""
        padroes_validos = []
        for padrao in self.sequencias_padroes['padroes_detectados']:
            if len(self.historico) - padrao['detectado_em'] <= limite:
                padroes_validos.append(padrao)
        self.sequencias_padroes['padroes_detectados'] = padroes_validos

    def atualizar_sequencias_ativas(self, zona_atual, historico_recente):
        """Atualiza sequências ativas por zona"""
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
        """Aplica padrões detectados para ajustar previsão"""
        if not self.sequencias_padroes['padroes_detectados']:
            return distribuicao_zonas
        
        distribuicao_ajustada = distribuicao_zonas.copy()
        
        padroes_recentes = [p for p in self.sequencias_padroes['padroes_detectados'] 
                           if len(self.historico) - p['detectado_em'] <= 15]
        
        for padrao in padroes_recentes:
            zona = padrao['zona']
            forca = padrao['forca']
            
            if zona in distribuicao_ajustada:
                aumento = max(1, int(distribuicao_ajustada[zona] * forca * 0.3))
                distribuicao_ajustada[zona] += aumento
                logging.info(f"🎯 Aplicando padrão {padrao['tipo']} à zona {zona}: +{aumento}")
        
        return distribuicao_ajustada

    def calcular_confianca_com_padroes(self, distribuicao, zona_alvo):
        """Calcula confiança considerando padrões"""
        confianca_base = self.calcular_confianca_zona_ml({
            'contagem': distribuicao[zona_alvo],
            'total_zonas': 25
        })
        
        padroes_recentes = [p for p in self.sequencias_padroes['padroes_detectados'] 
                           if p['zona'] == zona_alvo and 
                           len(self.historico) - p['detectado_em'] <= 15]
        
        bonus_confianca = len(padroes_recentes) * 0.15
        confianca_final = min(1.0, self.confianca_para_valor(confianca_base) + bonus_confianca)
        
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
        """Treinamento automático do ML"""
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
        """Extrai números do histórico"""
        historico_numeros = []
        for item in list(self.historico):
            if isinstance(item, dict) and 'number' in item:
                historico_numeros.append(item['number'])
            elif isinstance(item, (int, float)):
                historico_numeros.append(int(item))
        return historico_numeros

    def analisar_ml(self):
        """Análise principal do ML"""
        if len(self.historico) < 10:
            return None

        if not self.ml.is_trained:
            return None

        historico_numeros = self.extrair_numeros_historico()

        if len(historico_numeros) < 10:
            return None

        previsao_ml, msg_ml = self.ml.prever_proximo_numero(historico_numeros, top_k=25)
        
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
                
                return {
                    'nome': 'Machine Learning - CatBoost',
                    'numeros_apostar': numeros_zona,
                    'gatilho': f'ML CatBoost - Zona {zona_vencedora} ({contagem_original}→{contagem_ajustada}/25){gatilho_extra}',
                    'confianca': confianca,
                    'previsao_ml': previsao_ml,
                    'zona_ml': zona_vencedora,  # CORREÇÃO: campo correto para notificações
                    'distribuicao': distribuicao_ajustada,
                    'padroes_aplicados': len(padroes_aplicados)
                }
        
        return None

    def analisar_distribuicao_zonas(self, top_25_numeros):
        """Analisa distribuição por zonas"""
        contagem_zonas = {}
        
        for zona, numeros in self.numeros_zonas_ml.items():
            count = sum(1 for num in top_25_numeros if num in numeros)
            contagem_zonas[zona] = count
        
        return contagem_zonas if contagem_zonas else None

    def calcular_confianca_zona_ml(self, distribuicao):
        """Calcula confiança baseada na distribuição"""
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
        """Treina o modelo ML"""
        if historico_completo is not None:
            historico_numeros = historico_completo
        else:
            historico_numeros = self.extrair_numeros_historico()
        
        if len(historico_numeros) >= self.ml.min_training_samples:
            success, message = self.ml.treinar_modelo(historico_numeros)
            return success, message
        else:
            return False, f"Histórico insuficiente: {len(historico_numeros)}/{self.ml.min_training_samples} números"

    # ... (restante dos métodos mantidos)

# =============================
# SISTEMA DE GESTÃO - CORREÇÕES APLICADAS
# =============================
class SistemaRoletaCompleto:
    def __init__(self):
        self.estrategia_zonas = EstrategiaZonasOtimizada()
        self.estrategia_midas = EstrategiaMidas()
        self.estrategia_ml = EstrategiaML()
        self.previsao_ativa = None
        self.historico_desempenho = []
        self.acertos = 0
        self.erros = 0
        self.estrategias_contador = {}
        self.estrategia_selecionada = "Zonas"
        self.contador_sorteios_global = 0
        
        # Sistema de rotação automática
        self.sequencia_erros = 0
        self.ultima_estrategia_erro = ""

    def set_estrategia(self, estrategia):
        """Define a estratégia atual"""
        self.estrategia_selecionada = estrategia
        salvar_sessao()

    def treinar_modelo_ml(self, historico_completo=None):
        """Treina o modelo ML"""
        return self.estrategia_ml.treinar_modelo_ml(historico_completo)

    def rotacionar_estrategia_automaticamente(self, acerto, nome_estrategia):
        """Rotaciona automaticamente entre estratégias"""
        if acerto:
            self.sequencia_erros = 0
            self.ultima_estrategia_erro = ""
            return False
        else:
            self.sequencia_erros += 1
            self.ultima_estrategia_erro = nome_estrategia
            
            if self.sequencia_erros >= 2:
                estrategia_atual = self.estrategia_selecionada
                
                if estrategia_atual == "Zonas":
                    nova_estrategia = "ML"
                elif estrategia_atual == "ML":
                    nova_estrategia = "Zonas"
                else:
                    nova_estrategia = "Zonas"
                
                self.estrategia_selecionada = nova_estrategia
                self.sequencia_erros = 0
                
                enviar_rotacao_automatica(estrategia_atual, nova_estrategia)
                logging.info(f"🔄 ROTAÇÃO AUTOMÁTICA: {estrategia_atual} → {nova_estrategia}")
                
                return True
            return False

    def processar_novo_numero(self, numero):
        """Processa novo número sorteado"""
        # CORREÇÃO: Extrair número corretamente
        if isinstance(numero, dict) and 'number' in numero:
            numero_real = numero['number']
        else:
            numero_real = numero
            
        self.contador_sorteios_global += 1
            
        # Processar resultado da previsão anterior
        if self.previsao_ativa:
            acerto = numero_real in self.previsao_ativa['numeros_apostar']
            nome_estrategia = self.previsao_ativa['nome']
            
            # Determinar zona acertada
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
            
            # Aplicar rotação automática
            rotacionou = self.rotacionar_estrategia_automaticamente(acerto, nome_estrategia)
            
            # Atualizar estatísticas
            if nome_estrategia not in self.estrategias_contador:
                self.estrategias_contador[nome_estrategia] = {'acertos': 0, 'total': 0}
            
            self.estrategias_contador[nome_estrategia]['total'] += 1
            if acerto:
                self.estrategias_contador[nome_estrategia]['acertos'] += 1
                self.acertos += 1
            else:
                self.erros += 1
            
            # Enviar resultado
            enviar_resultado_super_simplificado(numero_real, acerto, nome_estrategia, zona_acertada)
            
            self.historico_desempenho.append({
                'numero': numero_real,
                'acerto': acerto,
                'estrategia': nome_estrategia,
                'previsao': self.previsao_ativa['numeros_apostar'],
                'rotacionou': rotacionou,
                'zona_acertada': zona_acertada
            })
            
            self.previsao_ativa = None
        
        # Adicionar número às estratégias
        self.estrategia_zonas.adicionar_numero(numero_real)
        self.estrategia_midas.adicionar_numero(numero_real)
        self.estrategia_ml.adicionar_numero(numero_real)
        
        # Obter nova previsão
        nova_estrategia = None
        
        if self.estrategia_selecionada == "Zonas":
            nova_estrategia = self.estrategia_zonas.analisar_zonas()
        elif self.estrategia_selecionada == "Midas":
            nova_estrategia = self.estrategia_midas.analisar_midas()
        elif self.estrategia_selecionada == "ML":
            nova_estrategia = self.estrategia_ml.analisar_ml()
        
        if nova_estrategia:
            self.previsao_ativa = nova_estrategia
            enviar_previsao_super_simplificada(nova_estrategia)

    # ... (restante dos métodos mantidos)

# =============================
# APLICAÇÃO STREAMLIT - CORREÇÕES APLICADAS
# =============================

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('roleta_system.log', encoding='utf-8')
    ]
)

# Inicialização da aplicação
st.set_page_config(page_title="IA Roleta — Multi-Estratégias", layout="centered")
st.title("🎯 IA Roleta — Sistema Multi-Estratégias")

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

# ... (restante da interface mantida igual)

# Salvar sessão automaticamente ao final
salvar_sessao()
