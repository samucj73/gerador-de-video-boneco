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

# Configurar logging
logging.basicConfig(level=logging.INFO)

# =============================
# CONFIGURAÇÕES
# =============================
HISTORICO_PATH = "historico_coluna_duzia.json"
ML_MODEL_PATH = "ml_roleta_model.pkl"
SCALER_PATH = "ml_scaler.pkl"

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
        
        for offset in range(-quantidade, quantidade + 1):
            vizinho = self.race[(posicao + offset) % len(self.race)]
            vizinhos.append(vizinho)
        
        return vizinhos

# =============================
# MÓDULO ML CORRIGIDO E FUNCIONAL
# =============================
class MLRoleta:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.roleta = RoletaInteligente()
        self.feature_names = []
        self.is_trained = False
        self.min_training_samples = 50
        self.model_loaded = False
        self.training_history = []
        
    def extrair_features_otimizadas(self, historico, lookback=15):
        if len(historico) < lookback:
            return None, None
            
        try:
            features = []
            feature_names = []
            
            ultimos_numeros = list(historico)[-lookback:]
            
            # 1. Últimos números
            for i in range(min(10, len(ultimos_numeros))):
                features.append(ultimos_numeros[i])
                feature_names.append(f"ultimo_{i+1}")
            
            while len(features) < 10:
                features.append(-1)
                feature_names.append(f"ultimo_{len(features)}")
            
            # 2. Estatísticas
            features.extend([
                np.mean(ultimos_numeros),
                np.std(ultimos_numeros) if len(ultimos_numeros) > 1 else 0,
                np.median(ultimos_numeros),
                max(ultimos_numeros),
                min(ultimos_numeros)
            ])
            feature_names.extend(["media_janela", "desvio_janela", "mediana_janela", "max_janela", "min_janela"])
            
            # 3. Cores
            vermelhos = [1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]
            pretos = [2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35]
            
            count_vermelhos = sum(1 for n in ultimos_numeros if n in vermelhos)
            count_pretos = sum(1 for n in ultimos_numeros if n in pretos)
            count_verde = sum(1 for n in ultimos_numeros if n == 0)
            
            features.extend([count_vermelhos, count_pretos, count_verde])
            feature_names.extend(["count_vermelhos", "count_pretos", "count_verde"])
            
            # 4. Dezenas
            count_duzia_1 = sum(1 for n in ultimos_numeros if 1 <= n <= 12)
            count_duzia_2 = sum(1 for n in ultimos_numeros if 13 <= n <= 24)
            count_duzia_3 = sum(1 for n in ultimos_numeros if 25 <= n <= 36)
            
            features.extend([count_duzia_1, count_duzia_2, count_duzia_3])
            feature_names.extend(["duzia_1", "duzia_2", "duzia_3"])
            
            self.feature_names = feature_names
            return np.array(features), feature_names
            
        except Exception as e:
            logging.error(f"Erro ao extrair features: {e}")
            return None, None
    
    def preparar_dados_treinamento(self, historico_completo, lookback=15):
        X = []
        y = []
        
        if len(historico_completo) < lookback + 5:
            return np.array(X), np.array(y)
        
        for i in range(lookback, len(historico_completo)):
            janela = historico_completo[i-lookback:i]
            features, _ = self.extrair_features_otimizadas(janela, lookback)
            
            if features is not None and len(features) > 0:
                X.append(features)
                y.append(historico_completo[i])
        
        return np.array(X), np.array(y)
    
    def treinar_modelo_otimizado(self, historico_completo):
        if len(historico_completo) < self.min_training_samples:
            return False, f"Necessário mínimo de {self.min_training_samples} amostras. Atual: {len(historico_completo)}"
        
        try:
            X, y = self.preparar_dados_treinamento(historico_completo)
            
            if len(X) < 30:
                return False, f"Dados insuficientes para treino: {len(X)} amostras"
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Normalizar
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Treinar modelo
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
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
            
            # Salvar modelo
            joblib.dump(self.model, ML_MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            
            return True, f"✅ Modelo treinado: {len(X)} amostras | Acurácia: {accuracy:.1%}"
            
        except Exception as e:
            logging.error(f"Erro no treinamento: {str(e)}")
            return False, f"❌ Erro no treinamento: {str(e)}"
    
    def carregar_modelo(self):
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
    
    def prever_top_12(self, historico):
        if not self.is_trained or self.model is None:
            return None, "Modelo não treinado"
        
        features, _ = self.extrair_features_otimizadas(historico)
        if features is None:
            return None, "Features insuficientes"
        
        try:
            features_reshaped = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features_reshaped)
            
            probabilidades = self.model.predict_proba(features_scaled)[0]
            
            # TOP 12 números mais prováveis
            top_indices = np.argsort(probabilidades)[-12:][::-1]
            
            top_numeros = []
            for idx in top_indices:
                if 0 <= idx <= 36:
                    prob = probabilidades[idx]
                    top_numeros.append((idx, prob))
            
            if top_numeros:
                return top_numeros, "Previsão Top 12 realizada"
            else:
                return None, "Nenhuma previsão confiável"
                
        except Exception as e:
            logging.error(f"Erro na previsão ML: {str(e)}")
            return None, f"Erro na previsão: {str(e)}"

    def get_modelo_info(self):
        if not self.is_trained:
            return "Modelo não treinado"
        
        info = {
            'treinado': self.is_trained,
            'carregado': self.model_loaded,
            'historico_treinamento': len(self.training_history)
        }
        return info

# =============================
# ESTRATÉGIA DAS ZONAS CORRIGIDA
# =============================
class EstrategiaZonasOtimizada:
    def __init__(self, usar_ml=False):
        self.roleta = RoletaInteligente()
        self.historico = deque(maxlen=50)
        self.nome = "Zonas + ML Top 12"
        self.usar_ml = usar_ml
        
        # Zonas
        self.zonas = {
            'Vermelha': 7,
            'Azul': 10,  
            'Amarela': 2
        }
        
        self.quantidade_zonas = {
            'Vermelha': 6,
            'Azul': 5,
            'Amarela': 4
        }
        
        # Pré-calcular zonas
        self.numeros_zonas = {}
        for nome, central in self.zonas.items():
            qtd = self.quantidade_zonas.get(nome, 6)
            self.numeros_zonas[nome] = self.roleta.get_vizinhos_zona(central, qtd)

        # Estatísticas
        self.stats_zonas = {zona: {
            'acertos': 0, 
            'tentativas': 0,
            'performance_media': 0
        } for zona in self.zonas.keys()}
        
        # ML
        self.ml = MLRoleta() if usar_ml else None
        if usar_ml:
            self.ml.carregar_modelo()

    def adicionar_numero(self, numero):
        self.historico.append(numero)
        return self.atualizar_stats(numero)

    def atualizar_stats(self, ultimo_numero):
        acertou_zona = None
        for zona, numeros in self.numeros_zonas.items():
            if ultimo_numero in numeros:
                self.stats_zonas[zona]['acertos'] += 1
                acertou_zona = zona
            self.stats_zonas[zona]['tentativas'] += 1
            
            if self.stats_zonas[zona]['tentativas'] > 0:
                self.stats_zonas[zona]['performance_media'] = (
                    self.stats_zonas[zona]['acertos'] / self.stats_zonas[zona]['tentativas'] * 100
                )
        
        return acertou_zona

    def get_zona_mais_quente(self):
        if len(self.historico) < 10:
            return None
            
        zonas_score = {}
        
        for zona in self.zonas.keys():
            score = 0
            
            # Frequência geral
            freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
            percentual_geral = freq_geral / len(self.historico)
            score += percentual_geral * 25
            
            # Frequência recente
            ultimos_15 = list(self.historico)[-15:] if len(self.historico) >= 15 else list(self.historico)
            freq_recente = sum(1 for n in ultimos_15 if n in self.numeros_zonas[zona])
            percentual_recente = freq_recente / len(ultimos_15)
            score += percentual_recente * 35
            
            # Performance histórica
            if self.stats_zonas[zona]['tentativas'] > 5:
                taxa_acerto = self.stats_zonas[zona]['performance_media']
                if taxa_acerto > 35: score += 25
                elif taxa_acerto > 25: score += 20
                else: score += 10
            else:
                score += 10
            
            zonas_score[zona] = score
        
        zona_vencedora = max(zonas_score, key=zonas_score.get) if zonas_score else None
        
        if zona_vencedora and zonas_score[zona_vencedora] >= 25:
            return zona_vencedora
        
        return None

    def analisar_zonas(self):
        if len(self.historico) < 10:
            return None

        # Extrair números para ML
        historico_numeros = list(self.historico)

        # Previsão ML - TOP 12
        previsao_ml = None
        if self.usar_ml and self.ml and self.ml.is_trained and len(historico_numeros) >= 15:
            previsao_ml, msg_ml = self.ml.prever_top_12(historico_numeros)

        # Análise tradicional de zonas
        zona_alvo = self.get_zona_mais_quente()
        
        if zona_alvo:
            numeros_apostar = self.numeros_zonas[zona_alvo]
            confianca = "Alta" if self.stats_zonas[zona_alvo]['performance_media'] > 30 else "Média"
            
            gatilho = f'Zona {zona_alvo} - Performance: {self.stats_zonas[zona_alvo]["performance_media"]:.1f}%'
            
            # Integração ML
            if previsao_ml and self.usar_ml:
                numeros_ml = [num for num, prob in previsao_ml]
                intersecao = set(numeros_ml) & set(numeros_apostar)
                if intersecao:
                    gatilho += f" | ML: {len(intersecao)} confirmações"
            
            return {
                'nome': f'Zona {zona_alvo}',
                'numeros_apostar': numeros_apostar,
                'gatilho': gatilho,
                'confianca': confianca,
                'previsao_ml': previsao_ml
            }
        
        # Se não há zona, usar ML puro
        elif previsao_ml and self.usar_ml:
            ml_top_numeros = [num for num, prob in previsao_ml[:6]]
            return {
                'nome': 'ML Top 12',
                'numeros_apostar': ml_top_numeros,
                'gatilho': 'Previsão ML pura - Top 6 números',
                'confianca': 'Média',
                'previsao_ml': previsao_ml
            }
        
        return None

    def treinar_modelo_ml(self, historico_completo=None):
        if not self.usar_ml or self.ml is None:
            return False, "ML não está ativado"
        
        try:
            historico_numeros = []
            if historico_completo is not None:
                for item in historico_completo:
                    if isinstance(item, dict) and 'number' in item:
                        historico_numeros.append(item['number'])
                    elif isinstance(item, (int, float)):
                        historico_numeros.append(int(item))
            else:
                historico_numeros = list(self.historico)
            
            if len(historico_numeros) < self.ml.min_training_samples:
                return False, f"Histórico insuficiente: {len(historico_numeros)}/{self.ml.min_training_samples}"
            
            success, message = self.ml.treinar_modelo_otimizado(historico_numeros)
            return success, message
            
        except Exception as e:
            return False, f"Erro no treinamento ML: {str(e)}"

    def get_analise_ml_detalhada(self):
        if not self.usar_ml or self.ml is None:
            return "🤖 ML: Não ativado"
        
        info = self.ml.get_modelo_info()
        analise = "🧠 MACHINE LEARNING - TOP 12 PREVISÕES\n"
        analise += "=" * 50 + "\n"
        
        if isinstance(info, dict) and info['treinado']:
            analise += "✅ Modelo Treinado\n"
        else:
            analise += "❌ Modelo Não Treinado\n"
        
        # Previsão atual se disponível
        historico_numeros = list(self.historico)
        if self.ml.is_trained and len(historico_numeros) >= 15:
            previsao, msg = self.ml.prever_top_12(historico_numeros)
            if previsao:
                analise += f"\n🔮 PREVISÃO TOP 12:\n"
                for i, (num, prob) in enumerate(previsao):
                    analise += f"{i+1:2d}. Número {num:2d}: {prob:6.2%}\n"
            else:
                analise += f"\n⚠️ {msg}\n"
        else:
            analise += f"\n📋 Dados: {len(historico_numeros)}/15 números\n"
        
        return analise

# =============================
# ESTRATÉGIA MIDAS (SIMPLIFICADA)
# =============================
class EstrategiaMidas:
    def __init__(self):
        self.historico = deque(maxlen=10)

    def adicionar_numero(self, numero):
        self.historico.append(numero)

    def analisar_midas(self):
        if len(self.historico) < 3:
            return None
        return None  # Simplificado para focar no ML

# =============================
# SISTEMA COMPLETO CORRIGIDO
# =============================
class SistemaRoletaCompleto:
    def __init__(self, usar_ml=False):
        self.estrategia_zonas = EstrategiaZonasOtimizada(usar_ml=usar_ml)
        self.estrategia_midas = EstrategiaMidas()
        self.previsao_ativa = None
        self.historico_desempenho = []
        self.acertos = 0
        self.erros = 0
        self.usar_ml = usar_ml

    def set_usar_ml(self, usar_ml):
        self.usar_ml = usar_ml
        self.estrategia_zonas.usar_ml = usar_ml
        self.estrategia_zonas.ml = MLRoleta() if usar_ml else None
        if usar_ml:
            self.estrategia_zonas.ml.carregar_modelo()

    def treinar_modelo_ml(self, historico_completo=None):
        if self.usar_ml:
            return self.estrategia_zonas.treinar_modelo_ml(historico_completo)
        return False, "ML não está ativado"

    def processar_novo_numero(self, numero):
        if isinstance(numero, dict) and 'number' in numero:
            numero_real = numero['number']
        else:
            numero_real = numero
            
        # Conferir previsão anterior
        if self.previsao_ativa:
            acerto = numero_real in self.previsao_ativa['numeros_apostar']
            if acerto:
                self.acertos += 1
            else:
                self.erros += 1
            
            self.historico_desempenho.append({
                'numero': numero_real,
                'acerto': acerto,
                'estrategia': self.previsao_ativa['nome']
            })
            
            self.previsao_ativa = None
        
        # Adicionar número às estratégias
        self.estrategia_zonas.adicionar_numero(numero_real)
        self.estrategia_midas.adicionar_numero(numero_real)
        
        # Nova estratégia
        nova_estrategia = self.estrategia_zonas.analisar_zonas()
        
        if nova_estrategia:
            self.previsao_ativa = nova_estrategia

# =============================
# APLICAÇÃO STREAMLIT CORRIGIDA
# =============================
st.set_page_config(page_title="IA Roleta — ML Top 12", layout="centered")
st.title("🎯 IA Roleta — Machine Learning Top 12")

# Inicialização
if "sistema" not in st.session_state:
    st.session_state.sistema = SistemaRoletaCompleto(usar_ml=False)

if "historico" not in st.session_state:
    try:
        if os.path.exists(HISTORICO_PATH):
            with open(HISTORICO_PATH, "r") as f:
                st.session_state.historico = json.load(f)
        else:
            st.session_state.historico = []
    except:
        st.session_state.historico = []

# Sidebar - Configurações
st.sidebar.title("⚙️ Configurações")

# Configuração ML
usar_ml = st.sidebar.checkbox("🤖 Ativar Machine Learning Top 12", value=False)
if usar_ml != st.session_state.sistema.usar_ml:
    st.session_state.sistema.set_usar_ml(usar_ml)

# Painel ML Expandido
if usar_ml:
    with st.sidebar.expander("🧠 Controle ML", expanded=True):
        # Calcular números disponíveis
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
            
            if st.button("🚀 Treinar Modelo ML", type="primary", use_container_width=True):
                with st.spinner("Treinando modelo ML..."):
                    success, message = st.session_state.sistema.treinar_modelo_ml(st.session_state.historico)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        else:
            st.warning(f"📥 Colete mais {50 - numeros_disponiveis} números")
        
        # Status do ML
        if st.session_state.sistema.estrategia_zonas.ml:
            if st.session_state.sistema.estrategia_zonas.ml.is_trained:
                st.success("✅ Modelo ML treinado")
                
                # Teste de previsão
                if st.button("🧪 Testar Previsão", use_container_width=True):
                    historico_numeros = list(st.session_state.sistema.estrategia_zonas.historico)
                    if len(historico_numeros) >= 15:
                        previsao, msg = st.session_state.sistema.estrategia_zonas.ml.prever_top_12(historico_numeros)
                        if previsao:
                            st.write("**Previsão Top 12:**")
                            df = pd.DataFrame(previsao, columns=['Número', 'Probabilidade'])
                            df['Probabilidade %'] = (df['Probabilidade'] * 100).round(2)
                            st.dataframe(df[['Número', 'Probabilidade %']].set_index('Número'))
            else:
                st.info("🤖 ML aguardando treinamento")

# Entrada manual
st.subheader("🎲 Inserir Números")
col1, col2 = st.columns([3, 1])
with col1:
    entrada = st.text_input("Números (0-36) separados por espaço:", placeholder="Ex: 7 15 23 32")
with col2:
    if st.button("Adicionar") and entrada:
        try:
            nums = [int(n) for n in entrada.split() if n.isdigit() and 0 <= int(n) <= 36]
            for n in nums:
                item = {"number": n, "timestamp": f"manual_{len(st.session_state.historico)}"}
                st.session_state.historico.append(item)
                st.session_state.sistema.processar_novo_numero(n)
            
            # Salvar histórico
            with open(HISTORICO_PATH, "w") as f:
                json.dump(st.session_state.historico, f, indent=2)
                
            st.success(f"✅ {len(nums)} números adicionados!")
            st.rerun()
        except Exception as e:
            st.error(f"Erro: {e}")

# Interface principal
st.subheader("📊 Últimos Números")
if st.session_state.historico:
    ultimos_10 = st.session_state.historico[-10:]
    numeros_str = " → ".join(str(item['number'] if isinstance(item, dict) else item) for item in ultimos_10)
    st.write(f"`{numeros_str}`")
else:
    st.info("Nenhum número registrado ainda")

st.subheader("🎯 Previsão Ativa")
sistema = st.session_state.sistema

if sistema.previsao_ativa:
    previsao = sistema.previsao_ativa
    st.success(f"**{previsao['nome']}**")
    st.write(f"**Confiança:** {previsao['confianca']}")
    st.write(f"**Gatilho:** {previsao['gatilho']}")
    st.write(f"**Números para apostar ({len(previsao['numeros_apostar'])}):**")
    st.write(f"`{', '.join(map(str, sorted(previsao['numeros_apostar'])))}`")
    
    # MOSTRAR PREVISÃO ML TOP 12
    if sistema.usar_ml and 'previsao_ml' in previsao and previsao['previsao_ml']:
        st.subheader("🤖 Previsão ML - Top 12")
        
        df_ml = pd.DataFrame(previsao['previsao_ml'], columns=['Número', 'Probabilidade'])
        df_ml['Rank'] = range(1, len(df_ml) + 1)
        df_ml['Probabilidade %'] = (df_ml['Probabilidade'] * 100).round(2)
        
        # Mostrar tabela
        st.dataframe(
            df_ml[['Rank', 'Número', 'Probabilidade %']].set_index('Rank'),
            use_container_width=True
        )
        
        # Estatísticas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Média", f"{df_ml['Probabilidade %'].mean():.2f}%")
        with col2:
            st.metric("🎯 Máxima", f"{df_ml['Probabilidade %'].max():.2f}%")
        with col3:
            st.metric("📊 Soma", f"{df_ml['Probabilidade %'].sum():.1f}%")
    
    st.info("⏳ Aguardando próximo número...")
else:
    st.info("🔍 Analisando padrões...")

# Desempenho
st.subheader("📈 Desempenho")
total = sistema.acertos + sistema.erros
taxa = (sistema.acertos / total * 100) if total > 0 else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("🎯 Acertos", sistema.acertos)
col2.metric("❌ Erros", sistema.erros)
col3.metric("✅ Taxa", f"{taxa:.1f}%")

# Análise ML Detalhada
if usar_ml:
    with st.expander("🔍 Análise ML Detalhada"):
        analise_ml = st.session_state.sistema.estrategia_zonas.get_analise_ml_detalhada()
        st.text(analise_ml)

# Auto refresh
st_autorefresh(interval=5000, key="refresh")

# Download histórico
if st.session_state.historico:
    st.download_button(
        "💾 Baixar Histórico",
        data=json.dumps(st.session_state.historico, indent=2),
        file_name="historico_roleta.json",
        mime="application/json"
    )
