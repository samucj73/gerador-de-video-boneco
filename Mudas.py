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
import time

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
# CONFIGURAÇÕES DE NOTIFICAÇÃO
# =============================
def enviar_previsao_super_simplificada(previsao):
    """Envia notificação de previsão super simplificada"""
    try:
        nome_estrategia = previsao['nome']
        
        if 'Zonas' in nome_estrategia:
            zona = previsao.get('zona', '')
            if zona == 'Vermelha':
                mensagem = "📍 Núcleo 7"
            elif zona == 'Azul':
                mensagem = "📍 Núcleo 10"
            elif zona == 'Amarela':
                mensagem = "📍 Núcleo 2"
            else:
                mensagem = f"📍 Núcleo {zona}"
            
        elif 'ML' in nome_estrategia:
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
            mensagem = f"💰 {previsao['nome']}"
        
        st.toast(f"🎯 Nova Previsão", icon="🔥")
        st.warning(f"🔔 {mensagem}")
        
        if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
            if st.session_state.telegram_token and st.session_state.telegram_chat_id:
                enviar_telegram(f"🔔 PREVISÃO\n{mensagem}")
                
        salvar_sessao()
    except Exception as e:
        logging.error(f"Erro ao enviar previsão: {e}")

def enviar_resultado_super_simplificado(numero_real, acerto, nome_estrategia, zona_acertada=None):
    """Envia notificação de resultado super simplificado"""
    try:
        if acerto:
            if 'Zonas' in nome_estrategia and zona_acertada:
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
# CONFIGURAÇÕES DA API - CORRIGIDAS
# =============================
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Referer": "https://casinoscores.com/",
    "Origin": "https://casinoscores.com"
}

def fetch_latest_result():
    """CORREÇÃO: Função melhorada para buscar resultados da API"""
    try:
        logging.info("🔍 Buscando último resultado da API...")
        
        response = requests.get(API_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        logging.info(f"📦 Resposta da API: {json.dumps(data, indent=2)}")
        
        # Estrutura mais robusta para extrair o número
        game_data = data.get("data", {})
        result = game_data.get("result", {})
        outcome = result.get("outcome", {})
        
        # Tentar diferentes caminhos para o número
        number = outcome.get("number")
        
        if number is None:
            # Tentar caminho alternativo
            number = result.get("number")
        
        if number is None:
            # Última tentativa - verificar diretamente no outcome
            number = outcome.get("winningNumber")
        
        timestamp = game_data.get("startedAt") or game_data.get("createdAt") or result.get("createdAt")
        
        if number is not None:
            logging.info(f"✅ Número encontrado: {number}, Timestamp: {timestamp}")
            return {
                "number": int(number),
                "timestamp": timestamp,
                "raw_data": data  # Para debug
            }
        else:
            logging.warning("❌ Número não encontrado na resposta da API")
            logging.warning(f"Estrutura da resposta: {json.dumps(data, indent=2)}")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Erro de conexão com a API: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"❌ Erro ao decodificar JSON da API: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ Erro inesperado ao buscar resultado: {e}")
        return None

def salvar_resultado_em_arquivo(historico, caminho=HISTORICO_PATH):
    """Salva histórico em arquivo JSON"""
    try:
        with open(caminho, "w", encoding='utf-8') as f:
            json.dump(historico, f, indent=2, ensure_ascii=False)
        logging.info(f"💾 Histórico salvo com {len(historico)} registros")
    except Exception as e:
        logging.error(f"❌ Erro ao salvar histórico: {e}")

# =============================
# CLASSE PRINCIPAL DA ROLETA
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

# =============================
# MÓDULO DE MACHINE LEARNING
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

            # Features básicas
            K_seq = 10
            ultimos = historico[-K_seq:]
            for i in range(K_seq):
                val = ultimos[i] if i < len(ultimos) else -1
                features.append(val)
                names.append(f"ultimo_{i+1}")

            for w in self.window_for_features:
                janela = historico[-w:] if N >= w else historico[:]
                arr = np.array(janela, dtype=float)
                features.append(arr.mean() if len(arr) > 0 else 0.0)
                names.append(f"media_{w}")
                features.append(arr.std() if len(arr) > 1 else 0.0)
                names.append(f"std_{w}")
                features.append(np.median(arr) if len(arr) > 0 else 0.0)
                names.append(f"mediana_{w}")

            # Frequência por janela
            counter_full = Counter(historico)
            for w in self.window_for_features:
                janela = historico[-w:] if N >= w else historico[:]
                c = Counter(janela)
                features.append(len(c) / (w if w>0 else 1))
                names.append(f"diversidade_{w}")
                top1_count = c.most_common(1)[0][1] if len(c)>0 else 0
                features.append(top1_count / (w if w>0 else 1))
                names.append(f"top1_prop_{w}")

            # Tempo desde último para cada número
            for num in self.numeros:
                try:
                    rev_idx = historico[::-1].index(num)
                    tempo = rev_idx
                except ValueError:
                    tempo = N + 1
                features.append(tempo)
                names.append(f"tempo_desde_{num}")

            # Contagens por cor
            janela50 = historico[-50:] if N >= 50 else historico[:]
            vermelhos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
            pretos = set(self.numeros[1:]) - vermelhos
            count_verm = sum(1 for x in janela50 if x in vermelhos)
            count_pret = sum(1 for x in janela50 if x in pretos)
            count_zero = sum(1 for x in janela50 if x == 0)
            features.extend([count_verm/len(janela50), count_pret/len(janela50), count_zero/len(janela50)])
            names.extend(["prop_vermelhos_50", "prop_pretos_50", "prop_zero_50"])

            # Dúzias
            def duzia_of(x):
                if x == 0: return 0
                if 1 <= x <= 12: return 1
                if 13 <= x <= 24: return 2
                return 3
                
            for d in [1,2,3]:
                features.append(sum(1 for x in janela50 if duzia_of(x)==d)/len(janela50))
                names.append(f"prop_duzia_{d}_50")

            # Vizinhos físicos
            ultimo_num = historico[-1]
            vizinhos_k = self.get_neighbors(ultimo_num, k=6)
            count_in_vizinhos = sum(1 for x in ultimos if x in vizinhos_k) / len(ultimos)
            features.append(count_in_vizinhos)
            names.append("prop_ultimos_em_vizinhos_6")

            # Repetições e padrões binários
            features.append(1 if N>=2 and historico[-1] == historico[-2] else 0)
            names.append("repetiu_ultimo")
            features.append(1 if N>=2 and (historico[-1] % 2) == (historico[-2] % 2) else 0)
            names.append("repetiu_paridade")
            features.append(1 if N>=2 and duzia_of(historico[-1]) == duzia_of(historico[-2]) else 0)
            names.append("repetiu_duzia")

            # Diferenças entre janelas
            if N >= max(self.window_for_features):
                small = np.mean(historico[-self.window_for_features[0]:])
                large = np.mean(historico[-self.window_for_features[-1]:])
                features.append(small - large)
                names.append("delta_media_small_large")
            else:
                features.append(0.0)
                names.append("delta_media_small_large")

            # Estatísticas de transição
            diffs = [abs(historico[i] - historico[i-1]) for i in range(1, len(historico))]
            features.append(np.mean(diffs) if len(diffs)>0 else 0.0)
            names.append("media_transicoes")
            features.append(np.std(diffs) if len(diffs)>1 else 0.0)
            names.append("std_transicoes")

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
        
        start_index = max(30, len(historico_completo) // 10)
        
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
            # Tenta importar CatBoost, se não conseguir usa RandomForest
            try:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.05,
                    depth=8,
                    random_seed=seed,
                    verbose=False
                )
                if X_val is not None and y_val is not None:
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                else:
                    model.fit(X_train, y_train, verbose=False)
                return model, "CatBoost"
            except ImportError:
                logging.warning("CatBoost não disponível. Usando RandomForest.")
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    random_state=seed,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                return model, "RandomForest"
                
        except Exception as e:
            logging.error(f"Erro ao treinar modelo: {e}")
            return None, "Erro"

    def treinar_modelo(self, historico_completo, force_retrain: bool = False, balance: bool = True):
        try:
            if len(historico_completo) < self.min_training_samples and not force_retrain:
                return False, f"Necessário mínimo de {self.min_training_samples} amostras. Atual: {len(historico_completo)}"

            X, y = self.preparar_dados_treinamento(historico_completo)
            if X.size == 0 or len(X) < 50:
                return False, f"Dados insuficientes para treino: {len(X)} amostras"

            X_scaled = self.scaler.fit_transform(X)

            # Split dos dados
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=self.seed
                )
            except Exception as e:
                logging.warning(f"Erro no split: {e}")
                return False, f"Erro no preparo dos dados: {e}"

            models = []
            model_names = []
            
            # Treina apenas um modelo para simplificar
            try:
                model, name = self._build_and_train_model(X_train, y_train, X_val, y_val, seed=self.seed)
                if model is not None:
                    models.append(model)
                    model_names.append(name)
            except Exception as e:
                logging.error(f"Erro ao treinar modelo: {e}")

            if not models:
                return False, "Modelo falhou no treinamento"

            # Avaliação simples
            try:
                acc = 0.0
                if hasattr(models[0], 'score'):
                    acc = models[0].score(X_val, y_val)
            except:
                acc = 0.0

            self.models = models
            self.is_trained = True
            self.contador_treinamento += 1
            self.meta['last_accuracy'] = acc
            self.meta['trained_on'] = len(historico_completo)

            # Tenta salvar os modelos
            try:
                joblib.dump({'models': self.models}, ML_MODEL_PATH)
                joblib.dump(self.scaler, SCALER_PATH)
                joblib.dump(self.meta, META_PATH)
            except Exception as e:
                logging.warning(f"Falha ao salvar modelos: {e}")

            return True, f"Modelo {model_names[0]} treinado com {len(X)} amostras. Acurácia: {acc:.2%}"

        except Exception as e:
            logging.error(f"[treinar_modelo] Erro: {e}")
            return False, f"Erro no treinamento: {str(e)}"

    def carregar_modelo(self):
        """Método carregar_modelo implementado corretamente"""
        try:
            if os.path.exists(ML_MODEL_PATH) and os.path.exists(SCALER_PATH):
                data = joblib.load(ML_MODEL_PATH)
                self.models = data.get('models', [])
                self.scaler = joblib.load(SCALER_PATH)
                if os.path.exists(META_PATH):
                    self.meta = joblib.load(META_PATH)
                self.is_trained = len(self.models) > 0
                logging.info(f"✅ Modelo ML carregado: {len(self.models)} modelos")
                return True
            else:
                logging.info("ℹ️ Nenhum modelo salvo encontrado")
                return False
        except Exception as e:
            logging.error(f"❌ Erro ao carregar modelo: {e}")
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

        try:
            Xs = np.array([feats])
            Xs_scaled = self.scaler.transform(Xs)
            probs = self._ensemble_predict_proba(Xs_scaled)[0]
            top_idx = np.argsort(probs)[-top_k:][::-1]
            top = [(int(idx), float(probs[idx])) for idx in top_idx]
            return top, "Previsão ML realizada"
        except Exception as e:
            return None, f"Erro na previsão: {str(e)}"

    def resumo_meta(self):
        return {
            "is_trained": self.is_trained,
            "contador_treinamento": self.contador_treinamento,
            "meta": self.meta
        }

# =============================
# ESTRATÉGIA DAS ZONAS
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
        
        self.numeros_zonas = {}
        for nome, central in self.zonas.items():
            self.numeros_zonas[nome] = self.roleta.get_vizinhos_zona(central, 6)

        self.stats_zonas = {zona: {
            'acertos': 0, 
            'tentativas': 0, 
            'sequencia_atual': 0,
            'sequencia_maxima': 0,
            'performance_media': 0
        } for zona in self.zonas.keys()}

    def adicionar_numero(self, numero):
        self.historico.append(numero)
        resultado = self.atualizar_stats(numero)
        if 'sistema' in st.session_state:
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

    def get_zona_mais_quente(self):
        if len(self.historico) < 15:
            return None
            
        zonas_score = {}
        total_numeros = len(self.historico)
        
        for zona in self.zonas.keys():
            score = 0
            
            freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
            percentual_geral = freq_geral / total_numeros
            score += percentual_geral * 25
            
            ultimos_15 = list(self.historico)[-15:] if total_numeros >= 15 else list(self.historico)
            freq_recente = sum(1 for n in ultimos_15 if n in self.numeros_zonas[zona])
            percentual_recente = freq_recente / len(ultimos_15)
            score += percentual_recente * 35
            
            if self.stats_zonas[zona]['tentativas'] > 10:
                taxa_acerto = self.stats_zonas[zona]['performance_media']
                if taxa_acerto > 40: 
                    score += 30
                elif taxa_acerto > 35:
                    score += 25
                elif taxa_acerto > 30:
                    score += 20
                elif taxa_acerto > 25:
                    score += 15
                else:
                    score += 10
            else:
                score += 10
            
            sequencia = self.stats_zonas[zona]['sequencia_atual']
            if sequencia >= 2:
                score += min(sequencia * 3, 10)
            
            zonas_score[zona] = score
        
        zona_vencedora = max(zonas_score, key=zonas_score.get) if zonas_score else None
        
        if zona_vencedora:
            threshold = 28
            return zona_vencedora if zonas_score[zona_vencedora] >= threshold else None
        
        return None

    def analisar_zonas(self):
        if len(self.historico) < 15:
            return None
            
        zona_alvo = self.get_zona_mais_quente()
        
        if zona_alvo:
            numeros_apostar = self.numeros_zonas[zona_alvo]
            
            confianca = self.calcular_confianca_ultra(zona_alvo)
            score = self.get_zona_score(zona_alvo)
            
            gatilho = f'Zona {zona_alvo} - Score: {score:.1f} | Perf: {self.stats_zonas[zona_alvo]["performance_media"]:.1f}%'
            
            return {
                'nome': f'Zona {zona_alvo}',
                'numeros_apostar': numeros_apostar,
                'gatilho': gatilho,
                'confianca': confianca,
                'zona': zona_alvo
            }
        
        return None

    def calcular_confianca_ultra(self, zona):
        if len(self.historico) < 10:
            return 'Baixa'
            
        fatores = []
        pesos = []
        
        perf_historica = self.stats_zonas[zona]['performance_media']
        if perf_historica > 40: 
            fatores.append(3)
            pesos.append(4)
        elif perf_historica > 30: 
            fatores.append(2)
            pesos.append(4)
        else: 
            fatores.append(1)
            pesos.append(4)
        
        ultimos_15 = list(self.historico)[-15:] if len(self.historico) >= 15 else list(self.historico)
        freq_recente = sum(1 for n in ultimos_15 if n in self.numeros_zonas[zona])
        perc_recente = (freq_recente / len(ultimos_15)) * 100
        if perc_recente > 50: 
            fatores.append(3)
            pesos.append(3)
        elif perc_recente > 35: 
            fatores.append(2)
            pesos.append(3)
        else: 
            fatores.append(1)
            pesos.append(3)
        
        sequencia = self.stats_zonas[zona]['sequencia_atual']
        if sequencia >= 3: 
            fatores.append(3)
            pesos.append(2)
        elif sequencia >= 2: 
            fatores.append(2)
            pesos.append(2)
        else: 
            fatores.append(1)
            pesos.append(2)
        
        total_pontos = sum(f * p for f, p in zip(fatores, pesos))
        total_pesos = sum(pesos)
        score_confianca = total_pontos / total_pesos
        
        if score_confianca >= 2.5: 
            return 'Excelente'
        elif score_confianca >= 2.2: 
            return 'Muito Alta'
        elif score_confianca >= 1.8: 
            return 'Alta'
        elif score_confianca >= 1.5: 
            return 'Média'
        else: 
            return 'Baixa'

    def get_zona_score(self, zona):
        if len(self.historico) < 10:
            return 0
            
        score = 0
        total_numeros = len(self.historico)
        
        freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
        percentual_geral = freq_geral / total_numeros
        score += percentual_geral * 25
        
        ultimos_15 = list(self.historico)[-15:] if total_numeros >= 15 else list(self.historico)
        freq_recente = sum(1 for n in ultimos_15 if n in self.numeros_zonas[zona])
        percentual_recente = freq_recente / len(ultimos_15)
        score += percentual_recente * 35
        
        if self.stats_zonas[zona]['tentativas'] > 10:
            taxa_acerto = self.stats_zonas[zona]['performance_media']
            if taxa_acerto > 40: score += 30
            elif taxa_acerto > 35: score += 25
            elif taxa_acerto > 30: score += 20
            elif taxa_acerto > 25: score += 15
            else: score += 10
        else:
            score += 10
        
        sequencia = self.stats_zonas[zona]['sequencia_atual']
        if sequencia >= 2:
            score += min(sequencia * 3, 10)
            
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
        
        analise = "🎯 ANÁLISE ULTRA OTIMIZADA - ZONAS v5\n"
        analise += "=" * 55 + "\n"
        
        analise += "📊 PERFORMANCE AVANÇADA:\n"
        for zona in self.zonas.keys():
            tentativas = self.stats_zonas[zona]['tentativas']
            acertos = self.stats_zonas[zona]['acertos']
            taxa = self.stats_zonas[zona]['performance_media']
            sequencia = self.stats_zonas[zona]['sequencia_atual']
            seq_maxima = self.stats_zonas[zona]['sequencia_maxima']
            
            analise += f"📍 {zona}: {acertos}/{tentativas} → {taxa:.1f}% | Seq: {sequencia} | Máx: {seq_maxima}\n"
        
        zona_recomendada = self.get_zona_mais_quente()
        if zona_recomendada:
            analise += f"\n💡 RECOMENDAÇÃO ULTRA: Zona {zona_recomendada}\n"
            analise += f"🎯 Números: {sorted(self.numeros_zonas[zona_recomendada])}\n"
            analise += f"📈 Confiança: {self.calcular_confianca_ultra(zona_recomendada)}\n"
        else:
            analise += "\n⚠️  AGUARDAR: Nenhuma zona com confiança suficiente\n"
        
        return analise

    def zerar_estatisticas(self):
        for zona in self.stats_zonas.keys():
            self.stats_zonas[zona] = {
                'acertos': 0, 
                'tentativas': 0, 
                'sequencia_atual': 0,
                'sequencia_maxima': 0,
                'performance_media': 0
            }

# =============================
# ESTRATÉGIA MIDAS
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
        if 'sistema' in st.session_state:
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

        return None

# =============================
# ESTRATÉGIA ML
# =============================
class EstrategiaML:
    def __init__(self):
        self.roleta = RoletaInteligente()
        self.ml = MLRoleta(self.roleta)
        self.historico = deque(maxlen=30)
        self.nome = "Machine Learning"
        self.contador_sorteios = 0
        
        self.zonas_ml = {
            'Vermelha': 7,
            'Azul': 10,  
            'Amarela': 2
        }
        
        self.numeros_zonas_ml = {}
        for nome, central in self.zonas_ml.items():
            self.numeros_zonas_ml[nome] = self.roleta.get_vizinhos_zona(central, 6)

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
        
        try:
            self.ml.carregar_modelo()
        except Exception as e:
            logging.warning(f"⚠️ Não foi possível carregar modelo ML: {e}")
            self.ml.is_trained = False

    def adicionar_numero(self, numero):
        self.historico.append(numero)
        self.contador_sorteios += 1
        
        if self.contador_sorteios >= 10:
            self.contador_sorteios = 0
            
        if 'sistema' in st.session_state:
            salvar_sessao()

    def analisar_ml(self):
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
                zona_vencedora = max(distribuicao_zonas, key=distribuicao_zonas.get)
                numeros_zona = self.numeros_zonas_ml[zona_vencedora]
                contagem = distribuicao_zonas[zona_vencedora]
                
                return {
                    'nome': 'Machine Learning',
                    'numeros_apostar': numeros_zona,
                    'gatilho': f'ML - Zona {zona_vencedora} ({contagem}/25)',
                    'confianca': 'Média',
                    'zona_ml': zona_vencedora
                }
        
        return None

    def extrair_numeros_historico(self):
        historico_numeros = []
        for item in list(self.historico):
            if isinstance(item, dict) and 'number' in item:
                historico_numeros.append(item['number'])
            elif isinstance(item, (int, float)):
                historico_numeros.append(int(item))
        return historico_numeros

    def analisar_distribuicao_zonas(self, top_25_numeros):
        contagem_zonas = {}
        
        for zona, numeros in self.numeros_zonas_ml.items():
            count = sum(1 for num in top_25_numeros if num in numeros)
            contagem_zonas[zona] = count
        
        return contagem_zonas

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
        
        return "🤖 ML: Modelo treinado e pronto para previsões"

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

# =============================
# SISTEMA DE GESTÃO
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
        
        self.sequencia_erros = 0
        self.ultima_estrategia_erro = ""

    def set_estrategia(self, estrategia):
        self.estrategia_selecionada = estrategia
        salvar_sessao()

    def treinar_modelo_ml(self, historico_completo=None):
        return self.estrategia_ml.treinar_modelo_ml(historico_completo)

    def rotacionar_estrategia_automaticamente(self, acerto, nome_estrategia):
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
            
            rotacionou = self.rotacionar_estrategia_automaticamente(acerto, nome_estrategia)
            
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
                'zona_acertada': zona_acertada
            })
            
            self.previsao_ativa = None
        
        self.estrategia_zonas.adicionar_numero(numero_real)
        self.estrategia_midas.adicionar_numero(numero_real)
        self.estrategia_ml.adicionar_numero(numero_real)
        
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

    def get_status_rotacao(self):
        return {
            'estrategia_atual': self.estrategia_selecionada,
            'sequencia_erros': self.sequencia_erros,
            'ultima_estrategia_erro': self.ultima_estrategia_erro,
            'proxima_rotacao_em': max(0, 2 - self.sequencia_erros)
        }

    def zerar_estatisticas_desempenho(self):
        self.acertos = 0
        self.erros = 0
        self.estrategias_contador = {}
        self.historico_desempenho = []
        self.contador_sorteios_global = 0
        self.sequencia_erros = 0
        self.ultima_estrategia_erro = ""
        
        self.estrategia_zonas.zerar_estatisticas()
        
        logging.info("📊 Estatísticas zeradas")
        salvar_sessao()

    def reset_recente_estatisticas(self):
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
            
            logging.info("🔄 Estatísticas recentes resetadas")
        
        salvar_sessao()

# =============================
# APLICAÇÃO STREAMLIT - CORRIGIDA
# =============================

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

# CORREÇÃO: Controle de atualização da API
if "ultimo_timestamp" not in st.session_state:
    st.session_state.ultimo_timestamp = None

# Sidebar
st.sidebar.title("⚙️ Configurações")

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
    
    # Botões para zerar estatísticas
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
                enviar_telegram("🔔 Teste de conexão - IA Roleta funcionando!")
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

# Status da Rotação Automática
with st.sidebar.expander("🔄 Rotação Automática", expanded=True):
    status_rotacao = st.session_state.sistema.get_status_rotacao()
    
    st.write("**Sistema de Rotação:**")
    st.write(f"🎯 **Estratégia Atual:** {status_rotacao['estrategia_atual']}")
    st.write(f"❌ **Erros Seguidos:** {status_rotacao['sequencia_erros']}/2")
    st.write(f"🔄 **Próxima Rotação em:** {status_rotacao['proxima_rotacao_em']} erro(s)")
    
    if status_rotacao['ultima_estrategia_erro']:
        st.write(f"📊 **Última Estratégia com Erro:** {status_rotacao['ultima_estrategia_erro']}")
    
    st.write("---")
    st.write("**Regras de Rotação:**")
    st.write("• ✅ **Acerto:** Continua na mesma estratégia")
    st.write("• ❌ **1 Erro:** Continua na estratégia") 
    st.write("• ❌❌ **2 Erros Seguidos:** Rotação automática")
    st.write("• 🔄 **Zonas ↔ ML:** Rotação entre as duas principais")
    
    # Botão para forçar rotação manual
    if st.button("🔄 Forçar Rotação", use_container_width=True):
        estrategia_atual = st.session_state.sistema.estrategia_selecionada
        if estrategia_atual == "Zonas":
            nova_estrategia = "ML"
        else:
            nova_estrategia = "Zonas"
        
        st.session_state.sistema.estrategia_selecionada = nova_estrategia
        st.session_state.sistema.sequencia_erros = 0
        st.success(f"🔄 Rotação forçada: {estrategia_atual} → {nova_estrategia}")
        st.rerun()

# Treinamento ML
with st.sidebar.expander("🧠 Treinamento ML", expanded=False):
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
    st.write(f"🎯 **Mínimo necessário:** 100 números")
    st.write(f"🔄 **Treinamento automático:** A cada 10 sorteios")
    st.write(f"🤖 **Modelo:** CatBoost (mais preciso)")
    
    if numeros_disponiveis > 0:
        numeros_unicos = len(set(numeros_lista))
        st.write(f"🎲 **Números únicos:** {numeros_unicos}/37")
        
        if numeros_unicos < 10:
            st.warning(f"⚠️ **Pouca variedade:** Necessário pelo menos 10 números diferentes")
        else:
            st.success(f"✅ **Variedade adequada:** {numeros_unicos} números diferentes")
    
    st.write(f"✅ **Status:** {'Dados suficientes' if numeros_disponiveis >= 100 else 'Coletando dados...'}")
    
    if numeros_disponiveis >= 100:
        st.success("✨ **Pronto para treinar!**")
        
        if st.button("🚀 Treinar Modelo ML", type="primary", use_container_width=True):
            with st.spinner("Treinando modelo ML com CatBoost... Isso pode levar alguns segundos"):
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
        st.warning(f"📥 Colete mais {100 - numeros_disponiveis} números para treinar o ML")
        
    st.write("---")
    st.write("**Status do ML:**")
    if st.session_state.sistema.estrategia_ml.ml.is_trained:
        if st.session_state.sistema.estrategia_ml.ml.models:
            primeiro_modelo = st.session_state.sistema.estrategia_ml.ml.models[0]
            modelo_tipo = "CatBoost" if hasattr(primeiro_modelo, 'iterations') else "RandomForest"
        else:
            modelo_tipo = "Não treinado"
            
        st.success(f"✅ Modelo {modelo_tipo} treinado ({st.session_state.sistema.estrategia_ml.ml.contador_treinamento} vezes)")
        if 'last_accuracy' in st.session_state.sistema.estrategia_ml.ml.meta:
            acc = st.session_state.sistema.estrategia_ml.ml.meta['last_accuracy']
            st.info(f"📊 Última acurácia: {acc:.2%}")
        st.info(f"🔄 Próximo treinamento automático em: {10 - st.session_state.sistema.estrategia_ml.contador_sorteios} sorteios")
    else:
        st.info("🤖 ML aguardando treinamento")

# Informações sobre as Estratégias
with st.sidebar.expander("📊 Informações das Estratégias"):
    if estrategia == "Zonas":
        info_zonas = st.session_state.sistema.estrategia_zonas.get_info_zonas()
        st.write("**🎯 Estratégia Zonas v5:**")
        st.write("**CONFIGURAÇÃO:** 6 antes + 6 depois (13 números/zona)")
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
        st.write("**🤖 Estratégia Machine Learning:**")
        st.write("- **Modelo**: CatBoost (Gradient Boosting)")
        st.write("- **Vantagem**: Mais preciso para dados sequenciais")
        st.write("- **Análise**: Top 25 números previstos")
        st.write("- **Treinamento**: Automático a cada 10 sorteios")
        st.write("- **Zonas**: 6 antes + 6 depois (13 números/zona)")
        st.write("- **Threshold**: Mínimo 7 números na mesma zona")
        st.write("- **Saída**: Zona com maior concentração")
        
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
        analise = st.session_state.sistema.estrategia_ml.get_analise_ml()
    else:
        analise = "🎯 Estratégia Midas ativa\nAnalisando padrões de terminais..."
    
    st.text(analise)

# CORREÇÃO: Seção de captura da API
st.subheader("🌐 Captura Automática da API")
with st.expander("🔧 Configurações da API", expanded=True):
    st.write("**Status da API:**")
    
    # Botão para forçar busca manual
    if st.button("🔄 Buscar Último Resultado da API", use_container_width=True):
        with st.spinner("Buscando último resultado..."):
            resultado = fetch_latest_result()
            if resultado:
                st.success(f"✅ Número encontrado: {resultado['number']}")
                
                # Verificar se é um novo número
                if st.session_state.historico:
                    ultimo_numero = st.session_state.historico[-1].get('number') if isinstance(st.session_state.historico[-1], dict) else st.session_state.historico[-1]
                    if resultado['number'] == ultimo_numero:
                        st.info("ℹ️ Este número já é o último no histórico")
                    else:
                        st.session_state.historico.append(resultado)
                        st.session_state.sistema.processar_novo_numero(resultado)
                        salvar_resultado_em_arquivo(st.session_state.historico)
                        salvar_sessao()
                        st.success(f"🎯 Novo número {resultado['number']} adicionado e processado!")
                        st.rerun()
                else:
                    st.session_state.historico.append(resultado)
                    st.session_state.sistema.processar_novo_numero(resultado)
                    salvar_resultado_em_arquivo(st.session_state.historico)
                    salvar_sessao()
                    st.success(f"🎯 Primeiro número {resultado['number']} adicionado e processado!")
                    st.rerun()
            else:
                st.error("❌ Não foi possível obter resultado da API")
    
    # Mostrar informações de debug da API
    if st.checkbox("Mostrar informações de debug da API"):
        st.write("**Última tentativa de conexão:**")
        try:
            resultado_test = fetch_latest_result()
            if resultado_test:
                st.json(resultado_test.get('raw_data', {}))
            else:
                st.error("Falha na conexão com a API")
        except Exception as e:
            st.error(f"Erro: {e}")

# Entrada manual
st.subheader("✍️ Inserir Sorteios Manualmente")
entrada = st.text_input("Digite números (0-36) separados por espaço:")
if st.button("Adicionar Manualmente") and entrada:
    try:
        nums = [int(n) for n in entrada.split() if n.isdigit() and 0 <= int(n) <= 36]
        for n in nums:
            item = {"number": n, "timestamp": f"manual_{int(time.time())}"}
            st.session_state.historico.append(item)
            st.session_state.sistema.processar_novo_numero(n)
        salvar_resultado_em_arquivo(st.session_state.historico)
        salvar_sessao()
        st.success(f"{len(nums)} números adicionados!")
        st.rerun()
    except Exception as e:
        st.error(f"Erro: {e}")

# Atualização automática
st_autorefresh(interval=5000, key="refresh")  # Aumentado para 5 segundos

# CORREÇÃO: Lógica de captura automática da API
try:
    resultado = fetch_latest_result()
    if resultado and resultado.get("timestamp"):
        # Verificar se é um novo resultado
        if st.session_state.historico:
            ultimo_item = st.session_state.historico[-1]
            ultimo_timestamp = ultimo_item.get('timestamp') if isinstance(ultimo_item, dict) else None
            
            if ultimo_timestamp != resultado["timestamp"]:
                logging.info(f"🆕 Novo número detectado: {resultado['number']}")
                st.session_state.historico.append(resultado)
                st.session_state.sistema.processar_novo_numero(resultado)
                salvar_resultado_em_arquivo(st.session_state.historico)
                salvar_sessao()
                st.rerun()
        else:
            # Primeiro número
            logging.info(f"📝 Primeiro número: {resultado['number']}")
            st.session_state.historico.append(resultado)
            st.session_state.sistema.processar_novo_numero(resultado)
            salvar_resultado_em_arquivo(st.session_state.historico)
            salvar_sessao()
            st.rerun()
except Exception as e:
    logging.error(f"Erro na captura automática: {e}")

# Interface principal
st.subheader("🔁 Últimos Números")
if st.session_state.historico:
    ultimos_10 = st.session_state.historico[-10:]
    numeros_str = " ".join(str(item['number'] if isinstance(item, dict) else item) for item in ultimos_10)
    st.write(f"`{numeros_str}`")
    st.write(f"**Total no histórico:** {len(st.session_state.historico)} números")
else:
    st.write("Nenhum número registrado")
    st.info("💡 Use a seção acima para adicionar números manualmente ou aguarde a captura automática")

# Status da Rotação na Interface Principal
status_rotacao = st.session_state.sistema.get_status_rotacao()
col_status1, col_status2, col_status3 = st.columns(3)
with col_status1:
    st.metric("🎯 Estratégia Atual", status_rotacao['estrategia_atual'])
with col_status2:
    st.metric("❌ Erros Seguidos", f"{status_rotacao['sequencia_erros']}/2")
with col_status3:
    st.metric("🔄 Próxima Rotação", f"Em {status_rotacao['proxima_rotacao_em']} erro(s)")

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
    
    st.info("⏳ Aguardando próximo sorteio para conferência...")
else:
    st.info(f"🎲 Analisando padrões ({estrategia})...")

# Desempenho
st.subheader("📈 Desempenho")

total = sistema.acertos + sistema.erros
taxa = (sistema.acertos / total * 100) if total > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Acertos", sistema.acertos)
col2.metric("🔴 Erros", sistema.erros)
col3.metric("📊 Total", total)
col4.metric("✅ Taxa", f"{taxa:.1f}%")

# Botões de gerenciamento de estatísticas na seção de desempenho
st.write("**Gerenciar Estatísticas:**")
col5, col6 = st.columns(2)

with col5:
    if st.button("🔄 Reset Recente", help="Mantém apenas os últimos 10 resultados", use_container_width=True, key="reset_recente_main"):
        st.session_state.sistema.reset_recente_estatisticas()
        st.success("✅ Estatísticas recentes resetadas!")
        st.rerun()

with col6:
    if st.button("🗑️ Zerar Tudo", type="secondary", help="Zera TODAS as estatísticas", use_container_width=True, key="zerar_tudo_main"):
        if st.checkbox("Confirmar zerar TODAS as estatísticas", key="confirm_zerar"):
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

st.success("✅ Sistema totalmente funcional! A API está sendo monitorada.")
