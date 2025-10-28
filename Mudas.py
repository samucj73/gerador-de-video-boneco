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
from alertas import enviar_previsao, enviar_resultado
from streamlit_autorefresh import st_autorefresh

# =============================
# MÓDULO DE MACHINE LEARNING ATUALIZADO - TOP 12
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
        """Extrai features otimizadas e consistentes para ML"""
        if len(historico) < lookback:
            return None, None
            
        try:
            features = []
            feature_names = []
            
            # Usar apenas os últimos 'lookback' números
            ultimos_numeros = list(historico)[-lookback:]
            
            # 1. Features básicas dos últimos números
            for i in range(min(10, len(ultimos_numeros))):
                features.append(ultimos_numeros[i])
                feature_names.append(f"ultimo_{i+1}")
            
            # Preencher features faltantes com -1
            while len(features) < 10:
                features.append(-1)
                feature_names.append(f"ultimo_{len(features)}")
            
            # 2. Estatísticas básicas da janela
            features.extend([
                np.mean(ultimos_numeros),
                np.std(ultimos_numeros) if len(ultimos_numeros) > 1 else 0,
                np.median(ultimos_numeros),
                max(ultimos_numeros),
                min(ultimos_numeros)
            ])
            feature_names.extend(["media_janela", "desvio_janela", "mediana_janela", "max_janela", "min_janela"])
            
            # 3. Características da roleta
            vermelhos = [1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]
            pretos = [2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35]
            
            count_vermelhos = sum(1 for n in ultimos_numeros if n in vermelhos)
            count_pretos = sum(1 for n in ultimos_numeros if n in pretos)
            count_verde = sum(1 for n in ultimos_numeros if n == 0)
            
            features.extend([count_vermelhos, count_pretos, count_verde])
            feature_names.extend(["count_vermelhos", "count_pretos", "count_verde"])
            
            # 4. Distribuição por dezenas
            count_duzia_1 = sum(1 for n in ultimos_numeros if 1 <= n <= 12)
            count_duzia_2 = sum(1 for n in ultimos_numeros if 13 <= n <= 24)
            count_duzia_3 = sum(1 for n in ultimos_numeros if 25 <= n <= 36)
            
            features.extend([count_duzia_1, count_duzia_2, count_duzia_3])
            feature_names.extend(["duzia_1", "duzia_2", "duzia_3"])
            
            # 5. Padrões de transição
            if len(ultimos_numeros) > 1:
                transicoes = [abs(ultimos_numeros[i] - ultimos_numeros[i-1]) for i in range(1, len(ultimos_numeros))]
                features.extend([
                    np.mean(transicoes),
                    np.std(transicoes) if len(transicoes) > 1 else 0
                ])
            else:
                features.extend([0, 0])
            feature_names.extend(["media_transicoes", "desvio_transicoes"])
            
            # 6. Frequência de números específicos (últimos 5)
            if len(ultimos_numeros) >= 5:
                freq_ultimos_5 = Counter(ultimos_numeros[-5:])
                for num in [0, 7, 17, 27]:  # Números estratégicos
                    features.append(freq_ultimos_5.get(num, 0))
                    feature_names.append(f"freq_{num}")
            else:
                features.extend([0, 0, 0, 0])
                feature_names.extend(["freq_0", "freq_7", "freq_17", "freq_27"])
            
            self.feature_names = feature_names
            return np.array(features), feature_names
            
        except Exception as e:
            logging.error(f"Erro ao extrair features: {e}")
            return None, None
    
    def prever_proximo_numero(self, historico, top_k=12):  # ALTERADO: padrão 12 em vez de 5
        """Faz previsão usando ML com validação - RETORNA TOP 12"""
        if not self.is_trained or self.model is None:
            return None, "Modelo não treinado"
        
        features, _ = self.extrair_features_otimizadas(historico)
        if features is None:
            return None, "Features insuficientes"
        
        try:
            features_reshaped = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features_reshaped)
            
            # Obter probabilidades para TODOS os números (0-36)
            probabilidades = self.model.predict_proba(features_scaled)[0]
            
            # TOP 12 números mais prováveis - MAIOR para MENOR probabilidade
            top_indices = np.argsort(probabilidades)[-top_k:][::-1]  # Ordem decrescente
            
            top_numeros = []
            
            for idx in top_indices:
                # Verificar se o índice é válido (0-36)
                if 0 <= idx <= 36:  # Garantir que é um número válido da roleta
                    prob = probabilidades[idx]
                    if prob > 0.001:  # Threshold mínimo reduzido para incluir mais números
                        top_numeros.append((idx, prob))
            
            if top_numeros:
                return top_numeros, f"Previsão ML Top {len(top_numeros)} realizada"
            else:
                return None, "Nenhuma previsão confiável"
                
        except Exception as e:
            logging.error(f"Erro na previsão ML: {str(e)}")
            return None, f"Erro na previsão: {str(e)}"

    def prever_top_12_completo(self, historico):
        """Método dedicado para prever os 12 números mais prováveis"""
        return self.prever_proximo_numero(historico, top_k=12)

# =============================
# ESTRATÉGIA DAS ZONAS ATUALIZADA - TOP 12
# =============================
class EstrategiaZonasOtimizada:
    def __init__(self, usar_ml=False):
        self.roleta = RoletaInteligente()
        self.historico = deque(maxlen=35)
        self.nome = "Zonas Ultra Otimizada v4 + ML Top 12"
        self.usar_ml = usar_ml
        
        # Zonas otimizadas
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
            'sequencia_atual': 0,
            'sequencia_maxima': 0,
            'performance_media': 0
        } for zona in self.zonas.keys()}
        
        # ML
        self.ml = None
        if self.usar_ml:
            self.inicializar_ml()

    def inicializar_ml(self):
        """Inicializa o módulo ML de forma segura"""
        try:
            self.ml = MLRoleta()
            self.ml.carregar_modelo()
        except Exception as e:
            logging.error(f"Erro ao inicializar ML: {e}")
            self.ml = MLRoleta()

    def analisar_zonas_com_ml(self):
        """Análise integrada com ML - TOP 12"""
        if len(self.historico) < 10:
            return None

        # Extrair números para ML
        historico_numeros = []
        for item in list(self.historico):
            if isinstance(item, dict) and 'number' in item:
                historico_numeros.append(item['number'])
            elif isinstance(item, (int, float)):
                historico_numeros.append(int(item))

        # Previsão ML - TOP 12
        previsao_ml = None
        ml_info = ""
        
        if self.usar_ml and self.ml and self.ml.is_trained and len(historico_numeros) >= 15:
            previsao_ml, msg_ml = self.ml.prever_top_12_completo(historico_numeros)  # TOP 12
            ml_info = f"ML Top 12: {msg_ml}"

        # Análise tradicional de zonas
        zona_alvo = self.get_zona_mais_quente()
        
        if zona_alvo:
            numeros_apostar = self.numeros_zonas[zona_alvo]
            confianca = self.calcular_confianca_ultra(zona_alvo)
            score = self.get_zona_score(zona_alvo)
            
            gatilho = f'Zona {zona_alvo} - Score: {score:.1f} | Perf: {self.stats_zonas[zona_alvo]["performance_media"]:.1f}%'
            
            # INTEGRAÇÃO ML MELHORADA - TOP 12
            integracao_ml = ""
            if previsao_ml and self.usar_ml:
                # Usar TODOS os 12 números do ML para interseção
                numeros_ml = [num for num, prob in previsao_ml]  # Todos os 12 números
                intersecao = set(numeros_ml) & set(numeros_apostar)
                
                if intersecao:
                    # Aumentar confiança baseado no ML - mais interseções = mais confiança
                    intersecoes_count = len(intersecao)
                    if intersecoes_count >= 4:
                        confianca = 'Excelente'
                    elif intersecoes_count >= 3:
                        confianca = 'Muito Alta' 
                    elif intersecoes_count >= 2:
                        confianca = 'Alta'
                    elif intersecoes_count >= 1:
                        confianca = 'Média-Alta'
                    
                    integracao_ml = f" | 🤖 ML: {intersecoes_count} confirmações"
                    
                    # Info adicional sobre probabilidades
                    probabilidades_intersecao = []
                    for num, prob in previsao_ml:
                        if num in intersecao:
                            probabilidades_intersecao.append((num, prob))
                    
                    if probabilidades_intersecao:
                        # Ordenar por probabilidade
                        probabilidades_intersecao.sort(key=lambda x: x[1], reverse=True)
                        top_intersecao = probabilidades_intersecao[0]
                        integracao_ml += f" (Melhor: {top_intersecao[0]}@{top_intersecao[1]:.1%})"
            
            gatilho += integracao_ml
            if ml_info and self.usar_ml:
                gatilho += f" | {ml_info}"
            
            return {
                'nome': f'Zona {zona_alvo} + ML Top 12',
                'numeros_apostar': numeros_apostar,
                'gatilho': gatilho,
                'confianca': confianca,
                'zona': zona_alvo,
                'previsao_ml': previsao_ml if self.usar_ml else None,
                'numeros_ml_sugeridos': [num for num, prob in previsao_ml] if previsao_ml else []
            }
        
        # Se não há zona, mas ML tem previsão confiável - usar TOP 3 do ML
        elif previsao_ml and self.usar_ml:
            ml_top_numeros = [num for num, prob in previsao_ml[:6]]  # Top 6 para apostas focadas
            if len(ml_top_numeros) >= 3:
                return {
                    'nome': f'ML Top 12 Direto',
                    'numeros_apostar': ml_top_numeros,
                    'gatilho': f'ML Top 12 - Melhores {len(ml_top_numeros)} números',
                    'confianca': 'Média',
                    'zona': 'ML Puro',
                    'previsao_ml': previsao_ml,
                    'numeros_ml_sugeridos': [num for num, prob in previsao_ml]
                }
        
        return None

    def get_analise_ml_detalhada(self):
        """Análise detalhada do ML - TOP 12"""
        if not self.usar_ml or self.ml is None:
            return "🤖 ML: Não ativado"
        
        info = self.ml.get_modelo_info()
        analise = "🧠 MACHINE LEARNING - TOP 12 PREVISÕES\n"
        analise += "=" * 55 + "\n"
        
        if isinstance(info, str):
            analise += f"📊 Status: {info}\n"
        else:
            analise += f"✅ Treinado: {info['treinado']}\n"
            analise += f"📥 Carregado: {info['carregado']}\n"
            analise += f"📈 Treinamentos: {info['historico_treinamento']}\n"
            
            if info['ultimo_treinamento']:
                ultimo = info['ultimo_treinamento']
                analise += f"🕒 Último: {ultimo['timestamp'].strftime('%H:%M')}\n"
                analise += f"📊 Amostras: {ultimo['samples']}\n"
                analise += f"🎯 Acurácia: {ultimo['accuracy']:.1%}\n"
                analise += f"🏆 Top3 Accuracy: {ultimo['top3_accuracy']:.1%}\n"
        
        # Previsão atual TOP 12 se disponível
        historico_numeros = []
        for item in list(self.historico):
            if isinstance(item, dict) and 'number' in item:
                historico_numeros.append(item['number'])
            elif isinstance(item, (int, float)):
                historico_numeros.append(int(item))
                
        if self.ml.is_trained and len(historico_numeros) >= 15:
            previsao, msg = self.ml.prever_top_12_completo(historico_numeros)
            if previsao:
                analise += f"\n🔮 PREVISÃO TOP 12 (Probabilidades):\n"
                analise += "-" * 40 + "\n"
                for i, (num, prob) in enumerate(previsao):
                    barra = "█" * int(prob * 50)  # Barra visual
                    analise += f"{i+1:2d}. Número {num:2d}: {prob:6.2%} {barra}\n"
                
                # Estatísticas da previsão
                probs = [prob for _, prob in previsao]
                analise += f"\n📊 Estatísticas Top 12:\n"
                analise += f"📈 Probabilidade média: {np.mean(probs):.2%}\n"
                analise += f"📉 Probabilidade mínima: {min(probs):.2%}\n"
                analise += f"📈 Probabilidade máxima: {max(probs):.2%}\n"
                analise += f"📋 Soma probabilidades: {sum(probs):.1%}\n"
            else:
                analise += f"\n⚠️ Previsão: {msg}\n"
        else:
            analise += f"\n📋 Dados para previsão: {len(historico_numeros)}/15 números\n"
        
        return analise

    # ... (manter outros métodos existentes como get_zona_mais_quente, calcular_confianca_ultra, etc.)

# =============================
# ATUALIZAÇÃO DA INTERFACE STREAMLIT - TOP 12
# =============================

def mostrar_painel_ml_top_12():
    """Painel dedicado ao Machine Learning TOP 12"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 ML - Top 12 Previsões")
    
    usar_ml = st.sidebar.checkbox("Ativar ML (Random Forest Top 12)", value=False, key="usar_ml_top_12")
    
    if usar_ml:
        if not st.session_state.sistema.estrategia_zonas.ml or not st.session_state.sistema.estrategia_zonas.usar_ml:
            st.session_state.sistema.set_usar_ml(True)
            st.session_state.sistema.estrategia_zonas.inicializar_ml()
        
        with st.sidebar.expander("🔧 Configurações ML Top 12", expanded=True):
            # Informações do modelo
            if st.session_state.sistema.estrategia_zonas.ml:
                info = st.session_state.sistema.estrategia_zonas.ml.get_modelo_info()
                if isinstance(info, dict) and info['treinado']:
                    st.success("✅ Modelo ML Treinado - Top 12")
                    if info['ultimo_treinamento']:
                        st.write(f"📊 Acurácia: {info['ultimo_treinamento']['accuracy']:.1%}")
                        st.write(f"🏆 Top3: {info['ultimo_treinamento']['top3_accuracy']:.1%}")
                else:
                    st.warning("🤖 Modelo não treinado")
            
            # Treinamento
            st.write("**Treinamento Top 12:**")
            
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
            
            st.write(f"📊 Números disponíveis: **{numeros_disponiveis}**")
            st.write(f"🎯 Mínimo necessário: **50** números")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Treinar ML Top 12", type="primary", use_container_width=True, 
                           disabled=numeros_disponiveis < 50):
                    with st.spinner("Treinando modelo ML Top 12..."):
                        success, message = st.session_state.sistema.treinar_modelo_ml(st.session_state.historico)
                        if success:
                            st.success(message)
                            st.balloons()
                        else:
                            st.error(message)
            
            with col2:
                if st.button("🔄 Resetar ML", use_container_width=True):
                    if st.session_state.sistema.estrategia_zonas.ml:
                        st.session_state.sistema.estrategia_zonas.ml = None
                    st.session_state.sistema.estrategia_zonas.inicializar_ml()
                    st.rerun()
            
            # Previsão teste TOP 12
            if st.session_state.sistema.estrategia_zonas.ml and st.session_state.sistema.estrategia_zonas.ml.is_trained:
                if st.button("🧪 Testar Previsão Top 12", use_container_width=True):
                    historico_numeros = []
                    for item in list(st.session_state.sistema.estrategia_zonas.historico):
                        if isinstance(item, dict) and 'number' in item:
                            historico_numeros.append(item['number'])
                        elif isinstance(item, (int, float)):
                            historico_numeros.append(int(item))
                    
                    if len(historico_numeros) >= 15:
                        previsao, msg = st.session_state.sistema.estrategia_zonas.ml.prever_top_12_completo(historico_numeros)
                        if previsao:
                            st.write("**🔮 Previsão Top 12 (Teste):**")
                            
                            # Criar DataFrame para melhor visualização
                            df_previsao = pd.DataFrame(previsao, columns=['Número', 'Probabilidade'])
                            df_previsao['Rank'] = range(1, len(df_previsao) + 1)
                            df_previsao['Probabilidade %'] = (df_previsao['Probabilidade'] * 100).round(2)
                            
                            # Mostrar tabela formatada
                            st.dataframe(
                                df_previsao[['Rank', 'Número', 'Probabilidade %']].set_index('Rank'),
                                use_container_width=True
                            )
                            
                            # Estatísticas
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📈 Média", f"{df_previsao['Probabilidade %'].mean():.2f}%")
                            with col2:
                                st.metric("📉 Mínima", f"{df_previsao['Probabilidade %'].min():.2f}%")
                            with col3:
                                st.metric("📈 Máxima", f"{df_previsao['Probabilidade %'].max():.2f}%")
                                
                        else:
                            st.warning(f"Teste falhou: {msg}")
                    else:
                        st.warning("Necessário 15+ números para previsão Top 12")
    
    return usar_ml

# =============================
# ATUALIZAÇÃO DA INTERFACE PRINCIPAL - TOP 12
# =============================

# Na seção principal, substituir a exibição da previsão ML:
def main():
    # ... (código anterior de inicialização)
    
    # Configurações ML Top 12
    usar_ml = mostrar_painel_ml_top_12()
    st.session_state.sistema.set_usar_ml(usar_ml)
    
    # ... (resto do código)
    
    st.subheader("🎯 Previsão Ativa")
    sistema = st.session_state.sistema

    if sistema.previsao_ativa:
        previsao = sistema.previsao_ativa
        st.success(f"**{previsao['nome']}**")
        st.write(f"**Confiança:** {previsao['confianca']}")
        st.write(f"**Gatilho:** {previsao['gatilho']}")
        st.write(f"**Números para apostar ({len(previsao['numeros_apostar'])}):**")
        st.write(", ".join(map(str, sorted(previsao['numeros_apostar']))))
        
        # MOSTRAR PREVISÃO ML TOP 12 SE DISPONÍVEL
        if sistema.usar_ml and 'previsao_ml' in previsao and previsao['previsao_ml']:
            st.write("---")
            st.subheader("🤖 Previsão ML - Top 12 Números")
            
            # Criar DataFrame para melhor visualização
            df_ml = pd.DataFrame(previsao['previsao_ml'], columns=['Número', 'Probabilidade'])
            df_ml['Rank'] = range(1, len(df_ml) + 1)
            df_ml['Probabilidade %'] = (df_ml['Probabilidade'] * 100).round(2)
            
            # Mostrar tabela formatada
            st.dataframe(
                df_ml[['Rank', 'Número', 'Probabilidade %']].set_index('Rank'),
                use_container_width=True
            )
            
            # Gráfico de barras das probabilidades
            st.write("**📊 Distribuição de Probabilidades:**")
            chart_data = df_ml.set_index('Número')['Probabilidade %']
            st.bar_chart(chart_data)
            
            # Estatísticas resumidas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Média", f"{df_ml['Probabilidade %'].mean():.2f}%")
            with col2:
                st.metric("📈 Máxima", f"{df_ml['Probabilidade %'].max():.2f}%")
            with col3:
                st.metric("📉 Mínima", f"{df_ml['Probabilidade %'].min():.2f}%")
            with col4:
                st.metric("📋 Soma", f"{df_ml['Probabilidade %'].sum():.1f}%")
        
        st.info("⏳ Aguardando próximo sorteio para conferência...")
    else:
        st.info(f"🎲 Analisando padrões ({modo_estrategia})...")

    # Na análise detalhada ML
    with st.sidebar.expander("🤖 Análise ML Top 12", expanded=False):
        if usar_ml and st.session_state.sistema.estrategia_zonas.ml:
            analise_ml = st.session_state.sistema.estrategia_zonas.get_analise_ml_detalhada()
            st.text(analise_ml)
        else:
            st.info("Ative o ML para ver análises Top 12 detalhadas")

# =============================
# SISTEMA COMPLETO ATUALIZADO
# =============================
class SistemaRoletaCompleto:
    def __init__(self, usar_ml=False):
        self.estrategia_zonas = EstrategiaZonasOtimizada(usar_ml=usar_ml)
        self.estrategia_midas = EstrategiaMidas()
        self.previsao_ativa = None
        self.historico_desempenho = []
        self.acertos = 0
        self.erros = 0
        self.estrategias_contador = {}
        self.modo_estrategia = "Todas"
        self.usar_ml = usar_ml

    def set_usar_ml(self, usar_ml):
        """Atualiza configuração ML de forma segura"""
        self.usar_ml = usar_ml
        self.estrategia_zonas.usar_ml = usar_ml
        if usar_ml:
            self.estrategia_zonas.inicializar_ml()

    def treinar_modelo_ml(self, historico_completo=None):
        """Treina o modelo ML"""
        if self.usar_ml:
            return self.estrategia_zonas.treinar_modelo_ml(historico_completo)
        return False, "ML não está ativado"

    # ... (manter outros métodos existentes)

# =============================
# CLASSE RoletaInteligente (MANTIDA)
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
