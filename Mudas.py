import streamlit as st
import json
import os
import requests
import logging
import numpy as np
import pandas as pd
from collections import Counter, deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from streamlit_autorefresh import st_autorefresh

# =============================
# CONFIGURAÇÕES DE NOTIFICAÇÃO
# =============================
def enviar_previsao(mensagem):
    """Envia notificação de previsão"""
    try:
        st.toast(f"🎯 {mensagem}", icon="🔥")
        st.warning(f"🔔 NOVA PREVISÃO: {mensagem}")
        
        if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
            if st.session_state.telegram_token and st.session_state.telegram_chat_id:
                enviar_telegram(mensagem)
    except Exception as e:
        logging.error(f"Erro ao enviar previsão: {e}")

def enviar_resultado(mensagem):
    """Envia notificação de resultado"""
    try:
        st.toast(f"🎲 {mensagem}", icon="✅")
        st.success(f"📢 RESULTADO: {mensagem}")
        
        if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
            if st.session_state.telegram_token and st.session_state.telegram_chat_id:
                enviar_telegram(f"RESULTADO: {mensagem}")
    except Exception as e:
        logging.error(f"Erro ao enviar resultado: {e}")

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
HISTORICO_PATH = "historico_coluna_duzia.json"
ML_MODEL_PATH = "ml_roleta_model.pkl"
SCALER_PATH = "ml_scaler.pkl"
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =============================
# CLASSE PRINCIPAL DA ROLETA
# =============================
class RoletaInteligente:
    def __init__(self):
        self.race = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
        
    def get_vizinhos_zona(self, numero_central, quantidade=6):
        """Retorna vizinhos antes e depois do número central no race"""
        if numero_central not in self.race:
            return []
        
        posicao = self.race.index(numero_central)
        vizinhos = []
        
        for offset in range(-quantidade, quantidade + 1):
            vizinho = self.race[(posicao + offset) % len(self.race)]
            vizinhos.append(vizinho)
        
        return vizinhos

    def get_posicao_race(self, numero):
        """Retorna a posição física do número na roda"""
        return self.race.index(numero) if numero in self.race else -1

# =============================
# MÓDULO DE MACHINE LEARNING AVANÇADO
# =============================
class MLRoletaAvancado:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.roleta = RoletaInteligente()
        self.feature_names = []
        self.is_trained = False
        self.min_training_samples = 50  # Reduzido para treinar mais rápido
        self.model_loaded = False
        
    def extrair_features_avancadas(self, historico, numero_alvo=None):
        """Extrai features avançadas com padrões complexos"""
        if len(historico) < 8:
            return None, None
            
        try:
            features = []
            feature_names = []
            
            # Últimos números (sequência temporal)
            k = 8
            ultimos_numeros = list(historico)[-k:]
            
            # 1. Features básicas dos últimos números
            for i in range(min(k, len(ultimos_numeros))):
                features.append(ultimos_numeros[i])
                feature_names.append(f"ultimo_{i+1}")
            
            # 2. Estatísticas de janelamento
            features.extend([
                np.mean(ultimos_numeros),
                np.std(ultimos_numeros) if len(ultimos_numeros) > 1 else 0,
                np.median(ultimos_numeros),
                max(ultimos_numeros),
                min(ultimos_numeros)
            ])
            feature_names.extend(["media_janela", "desvio_janela", "mediana_janela", "max_janela", "min_janela"])
            
            # 3. Posições físicas na roda
            posicoes = [self.roleta.get_posicao_race(n) for n in ultimos_numeros if n is not None]
            if posicoes:
                features.extend([
                    np.mean(posicoes),
                    np.std(posicoes) if len(posicoes) > 1 else 0
                ])
            else:
                features.extend([0, 0])
            feature_names.extend(["media_posicoes", "desvio_posicoes"])
            
            # 4. Contagens por categorias
            vermelhos = [1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]
            pretos = [2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35]
            
            count_vermelhos = sum(1 for n in ultimos_numeros if n in vermelhos)
            count_pretos = sum(1 for n in ultimos_numeros if n in pretos)
            count_verde = sum(1 for n in ultimos_numeros if n == 0)
            
            features.extend([count_vermelhos, count_pretos, count_verde])
            feature_names.extend(["count_vermelhos", "count_pretos", "count_verde"])
            
            # 5. Padrões de repetição
            repetidos = len(ultimos_numeros) - len(set(ultimos_numeros))
            features.append(repetidos)
            feature_names.append("repetidos")
            
            # 6. Tendência de alta/baixa
            if len(ultimos_numeros) >= 2:
                diferencas = [ultimos_numeros[i] - ultimos_numeros[i-1] for i in range(1, len(ultimos_numeros))]
                tendencia = sum(1 for d in diferencas if d > 0) - sum(1 for d in diferencas if d < 0)
                features.append(tendencia)
            else:
                features.append(0)
            feature_names.append("tendencia")
            
            # 7. Frequência de quadrantes
            quadrantes = {
                'Q1': [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34],
                'Q2': [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
                'Q3': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
            }
            
            for quad, nums in quadrantes.items():
                count = sum(1 for n in ultimos_numeros if n in nums)
                features.append(count)
                feature_names.append(f"quad_{quad}")
            
            self.feature_names = feature_names
            return features, feature_names
            
        except Exception as e:
            logging.error(f"Erro ao extrair features: {e}")
            return None, None
    
    def preparar_dados_treinamento(self, historico_completo):
        """Prepara dados de treinamento do histórico completo"""
        X = []
        y = []
        
        for i in range(15, len(historico_completo)):
            janela = historico_completo[:i]
            features, _ = self.extrair_features_avancadas(janela)
            
            if features is not None and i < len(historico_completo):
                X.append(features)
                y.append(historico_completo[i])
        
        return np.array(X), np.array(y)
    
    def treinar_modelo(self, historico_completo):
        """Treina o modelo com o histórico disponível"""
        if len(historico_completo) < self.min_training_samples:
            return False, f"Necessário mínimo de {self.min_training_samples} amostras. Atual: {len(historico_completo)}"
        
        try:
            X, y = self.preparar_dados_treinamento(historico_completo)
            
            if len(X) < 30:
                return False, f"Dados insuficientes para treino: {len(X)} amostras"
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Normalizar features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Treinar modelo com parâmetros otimizados
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Avaliar
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            
            # Salvar modelo e scaler
            joblib.dump(self.model, ML_MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            
            return True, f"Modelo treinado com {len(X)} amostras. Acurácia: {accuracy:.2%}"
            
        except Exception as e:
            return False, f"Erro no treinamento: {str(e)}"
    
    def carregar_modelo(self):
        """Carrega modelo pré-treinado"""
        try:
            if os.path.exists(ML_MODEL_PATH) and os.path.exists(SCALER_PATH):
                self.model = joblib.load(ML_MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.is_trained = True
                self.model_loaded = True
                return True
        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {e}")
        return False
    
    def prever_proximo_numero(self, historico):
        """Faz previsão usando ML - FOCO EM POUCOS NÚMEROS MAIS PROVÁVEIS"""
        if not self.is_trained:
            return None, "Modelo não treinado"
        
        features, _ = self.extrair_features_avancadas(historico)
        if features is None:
            return None, "Features insuficientes"
        
        try:
            features_scaled = self.scaler.transform([features])
            probabilidades = self.model.predict_proba(features_scaled)[0]
            
            # FOCO: Top 8 números mais prováveis (reduzido para maior assertividade)
            top_8_indices = np.argsort(probabilidades)[-8:][::-1]
            top_8_numeros = [(idx, probabilidades[idx]) for idx in top_8_indices]
            
            return top_8_numeros, "Previsão ML realizada"
        except Exception as e:
            return None, f"Erro na previsão: {str(e)}"

# =============================
# ESTRATÉGIA DE PADRÕES DE REPETIÇÃO
# =============================
class EstrategiaPadroesRepeticao:
    def __init__(self):
        self.historico = deque(maxlen=20)
        self.nome = "Padrões de Repetição"
        
    def adicionar_numero(self, numero):
        self.historico.append(numero)
        
    def analisar_padroes(self):
        """Analisa padrões de repetição nos últimos números"""
        if len(self.historico) < 8:
            return None
            
        ultimos_8 = list(self.historico)[-8:]
        
        # Padrão 1: Números que se repetiram recentemente
        numeros_repetidos = []
        contador = Counter(ultimos_8)
        for num, count in contador.items():
            if count >= 2:  # Números que apareceram pelo menos 2x
                numeros_repetidos.append(num)
                
        if len(numeros_repetidos) >= 2:
            return {
                'nome': 'Repetição Recente',
                'numeros_apostar': numeros_repetidos[:4],  # Máximo 4 números
                'gatilho': f'Padrão repetição: {len(numeros_repetidos)} números',
                'confianca': 'Alta'
            }
            
        # Padrão 2: Sequência de não repetição (espera por repetição)
        if len(set(ultimos_8)) == len(ultimos_8):  # Todos diferentes
            # Apostar nos números que mais saíram no histórico completo
            contador_completo = Counter(list(self.historico))
            numeros_quentes = [num for num, count in contador_completo.most_common(4)]
            
            return {
                'nome': 'Sequência Única - Números Quentes',
                'numeros_apostar': numeros_quentes,
                'gatilho': '8 números diferentes consecutivos',
                'confianca': 'Média'
            }
            
        return None

# =============================
# ESTRATÉGIA DE VIZINHANÇA INTELIGENTE
# =============================
class EstrategiaVizinhancaInteligente:
    def __init__(self):
        self.roleta = RoletaInteligente()
        self.historico = deque(maxlen=15)
        self.nome = "Vizinhança Inteligente"
        
    def adicionar_numero(self, numero):
        self.historico.append(numero)
        
    def analisar_vizinhanca(self):
        """Analisa vizinhança dos últimos números sorteados"""
        if len(self.historico) < 5:
            return None
            
        ultimo_numero = self.historico[-1]
        
        # Obter vizinhos físicos do último número
        vizinhos = self.roleta.get_vizinhos_zona(ultimo_numero, 4)  # 4 de cada lado
        
        if vizinhos:
            # Filtrar apenas os 6 vizinhos mais próximos
            vizinhos_estrategia = vizinhos[4:10]  # Pegar os 6 do meio
            
            return {
                'nome': 'Vizinhança Física',
                'numeros_apostar': vizinhos_estrategia,
                'gatilho': f'Vizinhos de {ultimo_numero}',
                'confianca': 'Média-Alta'
            }
            
        return None

# =============================
# ESTRATÉGIA ML AVANÇADA - FOCO EM ASSERTIVIDADE
# =============================
class EstrategiaMLAssertivo:
    def __init__(self):
        self.ml = MLRoletaAvancado()
        self.historico = deque(maxlen=25)
        self.nome = "ML Assertivo"
        self.ml.carregar_modelo()
        self.roleta = RoletaInteligente()
        
        # Zonas reduzidas para maior precisão
        self.zonas_assertivas = {
            'Hot_1': 7,    # Zona mais quente
            'Hot_2': 10,   # Segunda zona
            'Hot_3': 2     # Terceira zona
        }
        
        self.numeros_zonas = {}
        for nome, central in self.zonas_assertivas.items():
            self.numeros_zonas[nome] = self.roleta.get_vizinhos_zona(central, 4)  # Apenas 4 de cada lado

    def adicionar_numero(self, numero):
        self.historico.append(numero)

    def analisar_ml_assertivo(self):
        """Estratégia ML focada em poucos números de alta probabilidade"""
        if len(self.historico) < 8:
            return None

        if not self.ml.is_trained:
            return None

        # Extrair números do histórico
        historico_numeros = []
        for item in list(self.historico):
            if isinstance(item, dict) and 'number' in item:
                historico_numeros.append(item['number'])
            elif isinstance(item, (int, float)):
                historico_numeros.append(int(item))

        if len(historico_numeros) < 8:
            return None

        previsao_ml, msg_ml = self.ml.prever_proximo_numero(historico_numeros)
        
        if previsao_ml:
            # FOCO: Top 8 números mais prováveis (reduzido para maior assertividade)
            top_8_numeros = [num for num, prob in previsao_ml[:8]]
            
            # Análise de concentração por zonas
            zona_recomendada = self.analisar_concentracao_zonas(top_8_numeros)
            
            if zona_recomendada:
                numeros_zona = self.numeros_zonas[zona_recomendada['zona']]
                
                # FILTRAGEM CRÍTICA: Manter apenas números que estão no top 8
                numeros_filtrados = [num for num in numeros_zona if num in top_8_numeros]
                
                if len(numeros_filtrados) >= 2:  # Mínimo 2 números
                    confianca = self.calcular_confianca_assertiva(zona_recomendada, len(numeros_filtrados))
                    
                    # ENVIAR ALERTA TELEGRAM
                    self.enviar_alerta_assertivo(zona_recomendada, numeros_filtrados, confianca, previsao_ml[:3])
                    
                    return {
                        'nome': 'ML Assertivo - Zona Filtrada',
                        'numeros_apostar': numeros_filtrados,
                        'gatilho': f'{zona_recomendada["zona"]} - {len(numeros_filtrados)}/8 números',
                        'confianca': confianca,
                        'previsao_ml': previsao_ml
                    }
            
            # Estratégia alternativa: Top 3 números do ML
            top_3_numeros = [num for num, prob in previsao_ml[:3]]
            if top_3_numeros:
                return {
                    'nome': 'ML Direto - Top 3',
                    'numeros_apostar': top_3_numeros,
                    'gatilho': 'Top 3 previsões ML',
                    'confianca': 'Muito Alta',
                    'previsao_ml': previsao_ml
                }
        
        return None

    def analisar_concentracao_zonas(self, top_numeros):
        """Analisa concentração nos top números"""
        contagem_zonas = {}
        
        for zona, numeros in self.numeros_zonas.items():
            count = sum(1 for num in top_numeros if num in numeros)
            contagem_zonas[zona] = count
        
        # Encontrar zona com maior concentração
        if contagem_zonas:
            zona_vencedora = max(contagem_zonas, key=contagem_zonas.get)
            contagem = contagem_zonas[zona_vencedora]
            
            # Threshold mais conservador: mínimo 3 números
            if contagem >= 3:
                return {
                    'zona': zona_vencedora,
                    'contagem': contagem,
                    'total': len(top_numeros)
                }
        
        return None

    def calcular_confianca_assertiva(self, zona_info, numeros_filtrados):
        """Calcula confiança de forma mais assertiva"""
        percentual = (zona_info['contagem'] / zona_info['total']) * 100
        
        if percentual >= 50:  # 4+ em 8
            return 'Excelente'
        elif percentual >= 37.5:  # 3 em 8
            return 'Muito Alta'
        else:
            return 'Alta'

    def enviar_alerta_assertivo(self, zona_info, numeros_aposta, confianca, top_3):
        """Envia alerta assertivo para Telegram"""
        try:
            if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
                token = st.session_state.telegram_token
                chat_id = st.session_state.telegram_chat_id
                
                if token and chat_id:
                    top_3_str = "\n".join([f"    {num}: {prob:.2%}" for num, prob in top_3])
                    
                    mensagem = f"""
🎯 <b>ALERTA ASSERTIVO - ML FILTRADO</b>

🏆 <b>Estratégia:</b> ML Assertivo (Top 8)
🎯 <b>Zona:</b> {zona_info['zona']}
📊 <b>Concentração:</b> {zona_info['contagem']}/{zona_info['total']} números
💎 <b>Confiança:</b> {confianca}

🎲 <b>NÚMEROS PARA APOSTAR ({len(numeros_aposta)}):</b>
{', '.join(map(str, sorted(numeros_aposta)))}

🤖 <b>Top 3 ML:</b>
{top_3_str}

⚡ <b>APOSTA ALTAMENTE ASSERTIVA - ENTRAR!</b>
"""
                    
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {
                        "chat_id": chat_id,
                        "text": mensagem,
                        "parse_mode": "HTML"
                    }
                    
                    requests.post(url, json=payload, timeout=10)
        except Exception as e:
            logging.error(f"Erro no alerta assertivo: {e}")

    def treinar_modelo_ml(self, historico_completo=None):
        """Treina o modelo de ML"""
        if historico_completo is not None:
            historico_numeros = []
            for item in historico_completo:
                if isinstance(item, dict) and 'number' in item:
                    historico_numeros.append(item['number'])
                elif isinstance(item, (int, float)):
                    historico_numeros.append(int(item))
        else:
            historico_numeros = []
            for item in list(self.historico):
                if isinstance(item, dict) and 'number' in item:
                    historico_numeros.append(item['number'])
                elif isinstance(item, (int, float)):
                    historico_numeros.append(int(item))
        
        if len(historico_numeros) >= self.ml.min_training_samples:
            success, message = self.ml.treinar_modelo(historico_numeros)
            return success, message
        else:
            return False, f"Histórico insuficiente: {len(historico_numeros)}/{self.ml.min_training_samples} números"

# =============================
# SISTEMA DE GESTÃO ASSERTIVO
# =============================
class SistemaRoletaAssertivo:
    def __init__(self):
        self.estrategia_ml = EstrategiaMLAssertivo()
        self.estrategia_padroes = EstrategiaPadroesRepeticao()
        self.estrategia_vizinhanca = EstrategiaVizinhancaInteligente()
        self.previsao_ativa = None
        self.historico_desempenho = []
        self.acertos = 0
        self.erros = 0
        self.estrategias_contador = {}
        self.estrategia_selecionada = "ML Assertivo"

    def set_estrategia(self, estrategia):
        """Define a estratégia a ser usada"""
        self.estrategia_selecionada = estrategia

    def treinar_modelo_ml(self, historico_completo=None):
        """Treina o modelo de ML"""
        return self.estrategia_ml.treinar_modelo_ml(historico_completo)

    def processar_novo_numero(self, numero):
        # Extrair número
        if isinstance(numero, dict) and 'number' in numero:
            numero_real = numero['number']
        else:
            numero_real = numero
            
        # Conferir previsão anterior
        if self.previsao_ativa:
            acerto = numero_real in self.previsao_ativa['numeros_apostar']
            
            nome_estrategia = self.previsao_ativa['nome']
            if nome_estrategia not in self.estrategias_contador:
                self.estrategias_contador[nome_estrategia] = {'acertos': 0, 'total': 0}
            
            self.estrategias_contador[nome_estrategia]['total'] += 1
            if acerto:
                self.estrategias_contador[nome_estrategia]['acertos'] += 1
                self.acertos += 1
                enviar_resultado(f"🎉 ACERTO! Número {numero_real} - {nome_estrategia}")
            else:
                self.erros += 1
                enviar_resultado(f"❌ ERRO! Número {numero_real} - {nome_estrategia}")
            
            self.historico_desempenho.append({
                'numero': numero_real,
                'acerto': acerto,
                'estrategia': nome_estrategia,
                'previsao': self.previsao_ativa['numeros_apostar']
            })
            
            self.previsao_ativa = None
        
        # Adicionar número às estratégias
        self.estrategia_ml.adicionar_numero(numero_real)
        self.estrategia_padroes.adicionar_numero(numero_real)
        self.estrategia_vizinhanca.adicionar_numero(numero_real)
        
        # Gerar nova previsão baseada na estratégia selecionada
        nova_estrategia = None
        
        if self.estrategia_selecionada == "ML Assertivo":
            nova_estrategia = self.estrategia_ml.analisar_ml_assertivo()
        elif self.estrategia_selecionada == "Padrões Repetição":
            nova_estrategia = self.estrategia_padroes.analisar_padroes()
        elif self.estrategia_selecionada == "Vizinhança":
            nova_estrategia = self.estrategia_vizinhanca.analisar_vizinhanca()
        
        # Prioridade: ML Assertivo > Padrões > Vizinhança
        if not nova_estrategia:
            nova_estrategia = self.estrategia_padroes.analisar_padroes()
        
        if not nova_estrategia:
            nova_estrategia = self.estrategia_vizinhanca.analisar_vizinhanca()
        
        if nova_estrategia:
            self.previsao_ativa = nova_estrategia
            msg = f"🎯 {nova_estrategia['nome']} - {nova_estrategia['confianca']}\n"
            msg += f"🎲 {nova_estrategia['gatilho']}\n"
            msg += f"🔢 Números: {', '.join(map(str, sorted(nova_estrategia['numeros_apostar'])))}"
            
            if 'previsao_ml' in nova_estrategia:
                numeros_ml = [num for num, prob in nova_estrategia['previsao_ml'][:2]]
                msg += f"\n🤖 ML Top: {numeros_ml}"
                
            enviar_previsao(msg)

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
# APLICAÇÃO STREAMLOT
# =============================
st.set_page_config(page_title="IA Roleta — Sistema Assertivo", layout="centered")
st.title("🎯 IA Roleta — Sistema Assertivo")

# Inicialização
if "sistema" not in st.session_state:
    st.session_state.sistema = SistemaRoletaAssertivo()

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
st.sidebar.title("⚙️ Configurações Assertivas")

# Configurações do Telegram
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
    
    if st.button("Salvar Telegram"):
        st.session_state.telegram_token = telegram_token
        st.session_state.telegram_chat_id = telegram_chat_id
        st.success("✅ Configurações salvas!")

# Seleção de Estratégia
estrategia = st.sidebar.selectbox(
    "🎯 Estratégia:",
    ["ML Assertivo", "Padrões Repetição", "Vizinhança"],
    key="estrategia_selecionada"
)

# Aplicar estratégia
if estrategia != st.session_state.sistema.estrategia_selecionada:
    st.session_state.sistema.set_estrategia(estrategia)
    st.toast(f"🔄 Estratégia: {estrategia}")

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
    st.write(f"🎯 **Mínimo necessário:** 50 números")
    
    if numeros_disponiveis >= 50:
        st.success("✨ **Pronto para treinar!**")
        
        if st.button("🚀 Treinar Modelo ML", type="primary"):
            with st.spinner("Treinando modelo..."):
                success, message = st.session_state.sistema.treinar_modelo_ml(st.session_state.historico)
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")
    else:
        st.warning(f"📥 Colete mais {50 - numeros_disponiveis} números")

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

# Interface principal
st.subheader("🔁 Últimos Números")
if st.session_state.historico:
    ultimos_10 = st.session_state.historico[-10:]
    numeros_str = " ".join(str(item['number'] if isinstance(item, dict) else item) for item in ultimos_10)
    st.write(numeros_str)
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
    
    if 'previsao_ml' in previsao and previsao['previsao_ml']:
        st.write("**🤖 Probabilidades ML (Top 3):**")
        for num, prob in previsao['previsao_ml'][:3]:
            st.write(f"  {num}: {prob:.2%}")
    
    st.info("⏳ Aguardando próximo sorteio...")
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

# Performance por estratégia
if sistema.estrategias_contador:
    st.write("**📊 Performance por Estratégia:**")
    for nome, dados in sistema.estrategias_contador.items():
        if dados['total'] > 0:
            taxa_estrategia = (dados['acertos'] / dados['total'] * 100)
            cor = "🟢" if taxa_estrategia >= 50 else "🟡" if taxa_estrategia >= 40 else "🔴"
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
