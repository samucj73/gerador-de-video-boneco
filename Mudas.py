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
        """Retorna 6 vizinhos antes e 6 depois do número central no race"""
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
# MÓDULO DE MACHINE LEARNING CORRIGIDO
# =============================
class MLRoleta:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.roleta = RoletaInteligente()
        self.feature_names = []
        self.is_trained = False
        self.min_training_samples = 100
        self.model_loaded = False
        
    def extrair_features(self, historico, numero_alvo=None):
        """Extrai features avançadas do histórico para ML"""
        if len(historico) < 10:
            return None, None
            
        try:
            features = []
            feature_names = []
            
            # Últimos k números (sequência temporal)
            k = 10
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
                    np.std(posicoes) if len(posicoes) > 1 else 0,
                    (posicoes[-1] - posicoes[0]) % len(self.roleta.race) if len(posicoes) > 1 else 0
                ])
            else:
                features.extend([0, 0, 0])
            feature_names.extend(["media_posicoes", "desvio_posicoes", "distancia_percorrida"])
            
            # 4. Contagens por categorias
            vermelhos = [1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]
            pretos = [2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35]
            
            count_vermelhos = sum(1 for n in ultimos_numeros if n in vermelhos)
            count_pretos = sum(1 for n in ultimos_numeros if n in pretos)
            count_verde = sum(1 for n in ultimos_numeros if n == 0)
            
            features.extend([count_vermelhos, count_pretos, count_verde])
            feature_names.extend(["count_vermelhos", "count_pretos", "count_verde"])
            
            # 5. Duplas e colunas
            count_duzia_1 = sum(1 for n in ultimos_numeros if 1 <= n <= 12)
            count_duzia_2 = sum(1 for n in ultimos_numeros if 13 <= n <= 24)
            count_duzia_3 = sum(1 for n in ultimos_numeros if 25 <= n <= 36)
            
            features.extend([count_duzia_1, count_duzia_2, count_duzia_3])
            feature_names.extend(["duzia_1", "duzia_2", "duzia_3"])
            
            # 6. Transições e padrões
            transicoes = []
            for i in range(1, len(ultimos_numeros)):
                transicoes.append(abs(ultimos_numeros[i] - ultimos_numeros[i-1]))
            
            if transicoes:
                features.extend([
                    np.mean(transicoes),
                    np.std(transicoes) if len(transicoes) > 1 else 0
                ])
            else:
                features.extend([0, 0])
            feature_names.extend(["media_transicoes", "desvio_transicoes"])
            
            # 7. Tempo desde último zero
            tempo_zero = 0
            for i, num in enumerate(reversed(ultimos_numeros)):
                if num == 0:
                    tempo_zero = i + 1
                    break
            features.append(tempo_zero)
            feature_names.append("tempo_desde_zero")
            
            # 8. Frequência de zonas
            zonas = {
                'Amarela': self.roleta.get_vizinhos_zona(2, 6),
                'Vermelha': self.roleta.get_vizinhos_zona(7, 6),
                'Azul': self.roleta.get_vizinhos_zona(10, 6)
            }
            
            for zona, numeros in zonas.items():
                count = sum(1 for n in ultimos_numeros if n in numeros)
                features.append(count)
                feature_names.append(f"zona_{zona}")
            
            self.feature_names = feature_names
            return features, feature_names
            
        except Exception as e:
            logging.error(f"Erro ao extrair features: {e}")
            return None, None
    
    def preparar_dados_treinamento(self, historico_completo):
        """Prepara dados de treinamento do histórico completo"""
        X = []
        y = []
        
        for i in range(20, len(historico_completo)):
            janela = historico_completo[:i]
            features, _ = self.extrair_features(janela)
            
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
            
            if len(X) < 50:
                return False, f"Dados insuficientes para treino: {len(X)} amostras"
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Normalizar features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Treinar modelo
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
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
        """Faz previsão usando ML"""
        if not self.is_trained:
            return None, "Modelo não treinado"
        
        features, _ = self.extrair_features(historico)
        if features is None:
            return None, "Features insuficientes"
        
        try:
            features_scaled = self.scaler.transform([features])
            probabilidades = self.model.predict_proba(features_scaled)[0]
            
            # Top 5 números mais prováveis
            top_5_indices = np.argsort(probabilidades)[-5:][::-1]
            top_5_numeros = [(idx, probabilidades[idx]) for idx in top_5_indices]
            
            return top_5_numeros, "Previsão ML realizada"
        except Exception as e:
            return None, f"Erro na previsão: {str(e)}"

# =============================
# ESTRATÉGIA DAS ZONAS SUPER OTIMIZADA
# =============================
class EstrategiaZonasOtimizada:
    def __init__(self, usar_ml=False):
        self.roleta = RoletaInteligente()
        self.historico = deque(maxlen=30)  # Aumentado para 30
        self.nome = "Zonas Super Otimizada v3"
        self.usar_ml = usar_ml
        
        # Zonas otimizadas baseadas na performance real
        self.zonas = {
            'Vermelha': 7,   # Melhor performance: 36.4%
            'Amarela': 2,    # Performance: 20.0%  
            'Azul': 10       # Performance: 0.0% (precisa de ajuste)
        }
        
        # Ajustar quantidade de números por zona baseado na performance
        self.quantidade_zonas = {
            'Vermelha': 7,   # Mais números para zona quente
            'Amarela': 5,    # Menos números para zona fria
            'Azul': 5        # Menos números para zona fria
        }
        
        # Pré-calcular zonas
        self.numeros_zonas = {}
        for nome, central in self.zonas.items():
            qtd = self.quantidade_zonas.get(nome, 6)
            self.numeros_zonas[nome] = self.roleta.get_vizinhos_zona(central, qtd)

        # Estatísticas
        self.stats_zonas = {zona: {'acertos': 0, 'tentativas': 0, 'sequencia_atual': 0} for zona in self.zonas.keys()}
        
        # ML
        self.ml = None
        if self.usar_ml:
            self.ml = MLRoleta()
            self.ml.carregar_modelo()

    def adicionar_numero(self, numero):
        self.historico.append(numero)
        self.atualizar_stats(numero)

    def atualizar_stats(self, ultimo_numero):
        """Atualiza estatísticas de performance das zonas"""
        acertou = False
        for zona, numeros in self.numeros_zonas.items():
            if ultimo_numero in numeros:
                self.stats_zonas[zona]['acertos'] += 1
                self.stats_zonas[zona]['sequencia_atual'] += 1
                acertou = True
            else:
                self.stats_zonas[zona]['sequencia_atual'] = 0
            self.stats_zonas[zona]['tentativas'] += 1
        return acertou

    def get_zona_mais_quente(self):
        """Sistema de scoring super otimizado"""
        if len(self.historico) < 15:
            return None
            
        zonas_score = {}
        total_numeros = len(self.historico)
        
        for zona in self.zonas.keys():
            score = 0
            
            # CRITÉRIO 1: Frequência geral (30% do score)
            freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
            percentual_geral = freq_geral / total_numeros
            score += percentual_geral * 30
            
            # CRITÉRIO 2: Frequência recente (35% do score) - MAIS PESO
            ultimos_12 = list(self.historico)[-12:] if total_numeros >= 12 else list(self.historico)
            freq_recente = sum(1 for n in ultimos_12 if n in self.numeros_zonas[zona])
            percentual_recente = freq_recente / len(ultimos_12)
            score += percentual_recente * 35
            
            # CRITÉRIO 3: Performance histórica (25% do score)
            if self.stats_zonas[zona]['tentativas'] > 5:
                taxa_acerto = (self.stats_zonas[zona]['acertos'] / self.stats_zonas[zona]['tentativas']) * 100
                # Bônus para zonas com boa performance
                if taxa_acerto > 30:
                    score += 25
                elif taxa_acerto > 20:
                    score += 15
                else:
                    score += 5
            
            # CRITÉRIO 4: Sequência atual (10% do score)
            sequencia = self.stats_zonas[zona]['sequencia_atual']
            if sequencia >= 2:
                score += min(sequencia * 5, 10)  # Bônus por sequência
            
            zonas_score[zona] = score
        
        # Encontrar zona vencedora com threshold dinâmico
        zona_vencedora = max(zonas_score, key=zonas_score.get) if zonas_score else None
        
        # Threshold dinâmico baseado na performance da zona
        if zona_vencedora:
            threshold = 25
            # Reduzir threshold para zonas com boa performance histórica
            if self.stats_zonas[zona_vencedora]['tentativas'] > 10:
                taxa = (self.stats_zonas[zona_vencedora]['acertos'] / self.stats_zonas[zona_vencedora]['tentativas']) * 100
                if taxa > 35:
                    threshold = 20  # Mais sensível para zonas quentes
                elif taxa < 15:
                    threshold = 30  # Mais rigoroso para zonas frias
            
            return zona_vencedora if zonas_score[zona_vencedora] >= threshold else None
        
        return None

    def analisar_zonas(self):
        """Versão super otimizada"""
        if len(self.historico) < 15:
            return None

        # Previsão ML se disponível
        previsao_ml = None
        if self.usar_ml and self.ml and self.ml.is_trained:
            historico_numeros = []
            for item in list(self.historico):
                if isinstance(item, dict) and 'number' in item:
                    historico_numeros.append(item['number'])
                elif isinstance(item, (int, float)):
                    historico_numeros.append(int(item))
            
            if len(historico_numeros) >= 10:
                previsao_ml, msg_ml = self.ml.prever_proximo_numero(historico_numeros)
            
        zona_alvo = self.get_zona_mais_quente()
        
        if zona_alvo:
            numeros_apostar = self.numeros_zonas[zona_alvo]
            
            # Confiança super otimizada
            confianca = self.calcular_confianca_avancada(zona_alvo)
            score = self.get_zona_score(zona_alvo)
            
            gatilho = f'Zona {zona_alvo} - Score: {score:.1f}'
            
            # Integração ML melhorada
            if previsao_ml and self.usar_ml:
                numeros_ml = [num for num, prob in previsao_ml[:3]]
                intersecao = set(numeros_ml) & set(numeros_apostar)
                if intersecao:
                    if confianca == 'Alta':
                        confianca = 'Muito Alta'
                    gatilho += f" | ML: {len(intersecao)} números confirmados"
            
            return {
                'nome': f'Zona {zona_alvo}',
                'numeros_apostar': numeros_apostar,
                'gatilho': gatilho,
                'confianca': confianca,
                'zona': zona_alvo,
                'previsao_ml': previsao_ml if self.usar_ml else None
            }
        
        return None

    def calcular_confianca_avancada(self, zona):
        """Sistema de confiança avançado"""
        if len(self.historico) < 10:
            return 'Baixa'
            
        # Múltiplos fatores
        fatores = []
        
        # Fator 1: Frequência geral
        freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
        perc_geral = (freq_geral / len(self.historico)) * 100
        if perc_geral > 45: fatores.append(3)
        elif perc_geral > 35: fatores.append(2)
        else: fatores.append(1)
        
        # Fator 2: Frequência recente (últimos 15)
        ultimos_15 = list(self.historico)[-15:] if len(self.historico) >= 15 else list(self.historico)
        freq_recente = sum(1 for n in ultimos_15 if n in self.numeros_zonas[zona])
        perc_recente = (freq_recente / len(ultimos_15)) * 100
        if perc_recente > 60: fatores.append(3)
        elif perc_recente > 40: fatores.append(2)
        else: fatores.append(1)
        
        # Fator 3: Performance histórica
        if self.stats_zonas[zona]['tentativas'] > 8:
            taxa = (self.stats_zonas[zona]['acertos'] / self.stats_zonas[zona]['tentativas']) * 100
            if taxa > 40: fatores.append(3)
            elif taxa > 25: fatores.append(2)
            else: fatores.append(1)
        else:
            fatores.append(1)
        
        # Fator 4: Tendência (últimos 5 vs anteriores)
        if len(self.historico) >= 10:
            ultimos_5 = list(self.historico)[-5:]
            anteriores_5 = list(self.historico)[-10:-5]
            
            freq_ultimos = sum(1 for n in ultimos_5 if n in self.numeros_zonas[zona])
            freq_anteriores = sum(1 for n in anteriores_5 if n in self.numeros_zonas[zona])
            
            if freq_ultimos > freq_anteriores: fatores.append(3)  # Tendência positiva
            elif freq_ultimos == freq_anteriores: fatores.append(2)  # Estável
            else: fatores.append(1)  # Tendência negativa
        
        score_confianca = sum(fatores) / len(fatores)
        
        if score_confianca >= 2.5: return 'Muito Alta'
        elif score_confianca >= 2.0: return 'Alta'
        elif score_confianca >= 1.5: return 'Média'
        else: return 'Baixa'

    def get_zona_score(self, zona):
        """Score detalhado para debug"""
        if len(self.historico) < 10:
            return 0
            
        score = 0
        total_numeros = len(self.historico)
        
        # Frequência geral
        freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
        percentual_geral = freq_geral / total_numeros
        score += percentual_geral * 30
        
        # Frequência recente
        ultimos_12 = list(self.historico)[-12:] if total_numeros >= 12 else list(self.historico)
        freq_recente = sum(1 for n in ultimos_12 if n in self.numeros_zonas[zona])
        percentual_recente = freq_recente / len(ultimos_12)
        score += percentual_recente * 35
        
        # Performance histórica
        if self.stats_zonas[zona]['tentativas'] > 5:
            taxa_acerto = (self.stats_zonas[zona]['acertos'] / self.stats_zonas[zona]['tentativas']) * 100
            if taxa_acerto > 30: score += 25
            elif taxa_acerto > 20: score += 15
            else: score += 5
        
        # Sequência
        sequencia = self.stats_zonas[zona]['sequencia_atual']
        if sequencia >= 2:
            score += min(sequencia * 5, 10)
            
        return score

    def treinar_modelo_ml(self, historico_completo=None):
        """Treina o modelo de ML - VERSÃO CORRIGIDA"""
        if self.usar_ml and self.ml:
            # Usar histórico completo se fornecido, caso contrário usar self.historico
            if historico_completo is not None:
                # Extrair números do histórico completo
                historico_numeros = []
                for item in historico_completo:
                    if isinstance(item, dict) and 'number' in item:
                        historico_numeros.append(item['number'])
                    elif isinstance(item, (int, float)):
                        historico_numeros.append(int(item))
            else:
                # Usar apenas o histórico local (limitado)
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
        return False, "ML não está ativado ou inicializado"

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
        """Análise super detalhada"""
        if len(self.historico) == 0:
            return "Aguardando dados..."
        
        analise = "🎯 ANÁLISE SUPER OTIMIZADA - ZONAS v3\n"
        analise += "=" * 55 + "\n"
        
        # Status ML
        if self.usar_ml:
            status_ml = "✅ Treinado" if self.ml and self.ml.is_trained else "❌ Não treinado"
            analise += f"🤖 STATUS ML: {status_ml}\n"
            if self.ml and self.ml.is_trained and len(self.historico) >= 10:
                historico_numeros = []
                for item in list(self.historico):
                    if isinstance(item, dict) and 'number' in item:
                        historico_numeros.append(item['number'])
                    elif isinstance(item, (int, float)):
                        historico_numeros.append(int(item))
                
                previsao_ml, msg = self.ml.prever_proximo_numero(historico_numeros)
                if previsao_ml:
                    analise += "📊 PREVISÃO ML (Top 5):\n"
                    for num, prob in previsao_ml:
                        analise += f"  {num}: {prob:.2%}\n"
            analise += "---\n"
        
        # Performance detalhada
        analise += "📊 PERFORMANCE DETALHADA:\n"
        for zona in self.zonas.keys():
            tentativas = self.stats_zonas[zona]['tentativas']
            acertos = self.stats_zonas[zona]['acertos']
            taxa = (acertos / tentativas * 100) if tentativas > 0 else 0
            sequencia = self.stats_zonas[zona]['sequencia_atual']
            
            analise += f"📍 {zona}: {acertos}/{tentativas} → {taxa:.1f}% | Seq: {sequencia}\n"
        
        analise += "\n📈 FREQUÊNCIA ATUAL:\n"
        for zona in self.zonas.keys():
            freq = sum(1 for n in self.historico if isinstance(n, (int, float)) and n in self.numeros_zonas[zona])
            perc = (freq / len(self.historico)) * 100
            score = self.get_zona_score(zona)
            qtd_numeros = len(self.numeros_zonas[zona])
            analise += f"📍 {zona}: {freq}/{len(self.historico)} → {perc:.1f}% | Score: {score:.1f} | Números: {qtd_numeros}\n"
        
        # Tendências
        analise += "\n📊 TENDÊNCIAS:\n"
        if len(self.historico) >= 10:
            for zona in self.zonas.keys():
                ultimos_5 = list(self.historico)[-5:]
                anteriores_5 = list(self.historico)[-10:-5] if len(self.historico) >= 10 else []
                
                freq_ultimos = sum(1 for n in ultimos_5 if n in self.numeros_zonas[zona])
                freq_anteriores = sum(1 for n in anteriores_5 if n in self.numeros_zonas[zona]) if anteriores_5 else 0
                
                tendencia = "↗️" if freq_ultimos > freq_anteriores else "↘️" if freq_ultimos < freq_anteriores else "➡️"
                analise += f"📍 {zona}: {freq_ultimos}/5 vs {freq_anteriores}/5 {tendencia}\n"
        
        # Recomendações
        zona_recomendada = self.get_zona_mais_quente()
        if zona_recomendada:
            analise += f"\n💡 RECOMENDAÇÃO: Zona {zona_recomendada}\n"
            analise += f"🎯 Números: {sorted(self.numeros_zonas[zona_recomendada])}\n"
            analise += f"📈 Confiança: {self.calcular_confianca_avancada(zona_recomendada)}\n"
            analise += f"🔥 Score: {self.get_zona_score(zona_recomendada):.1f}\n"
            analise += f"🔢 Quantidade: {len(self.numeros_zonas[zona_recomendada])} números\n"
        else:
            analise += "\n⚠️  AGUARDAR: Nenhuma zona com confiança suficiente\n"
            analise += f"📋 Histórico atual: {len(self.historico)} números\n"
        
        return analise

    def get_analise_atual(self):
        """Mantido para compatibilidade"""
        return self.get_analise_detalhada()

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
# SISTEMA DE GESTÃO ATUALIZADO COM ML CORRIGIDO
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

    def set_modo_estrategia(self, modo):
        self.modo_estrategia = modo

    def set_usar_ml(self, usar_ml):
        self.usar_ml = usar_ml
        # Reinstanciar a estratégia de zonas com a nova configuração ML
        self.estrategia_zonas = EstrategiaZonasOtimizada(usar_ml=usar_ml)

    def treinar_modelo_ml(self, historico_completo=None):
        """Treina o modelo de ML - VERSÃO CORRIGIDA"""
        if self.usar_ml:
            return self.estrategia_zonas.treinar_modelo_ml(historico_completo)
        return False, "ML não está ativado"

    def processar_novo_numero(self, numero):
        # Extrair número se for um dicionário
        if isinstance(numero, dict) and 'number' in numero:
            numero_real = numero['number']
        else:
            numero_real = numero
            
        # Conferir previsão anterior se existir
        if self.previsao_ativa:
            acerto = numero_real in self.previsao_ativa['numeros_apostar']
            
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
                'numero': numero_real,
                'acerto': acerto,
                'estrategia': nome_estrategia,
                'previsao': self.previsao_ativa['numeros_apostar']
            })
            
            self.previsao_ativa = None
        
        # Adicionar número a todas as estratégias
        self.estrategia_zonas.adicionar_numero(numero_real)
        self.estrategia_midas.adicionar_numero(numero_real)
        
        # Verificar nova estratégia
        nova_estrategia = None
        
        if self.modo_estrategia == "Apenas Zonas":
            nova_estrategia = self.estrategia_zonas.analisar_zonas()
        elif self.modo_estrategia == "Apenas Midas":
            nova_estrategia = self.estrategia_midas.analisar_midas()
        else:  # Todas as estratégias
            nova_estrategia = self.estrategia_zonas.analisar_zonas()
            if not nova_estrategia:
                nova_estrategia = self.estrategia_midas.analisar_midas()
        
        if nova_estrategia:
            self.previsao_ativa = nova_estrategia
            # Enviar alerta
            msg = f"🎯 {nova_estrategia['nome']} - {nova_estrategia['confianca']}\n"
            msg += f"🎲 Gatilho: {nova_estrategia['gatilho']}\n"
            msg += f"🔢 Números: {', '.join(map(str, sorted(nova_estrategia['numeros_apostar'])))}"
            
            # Adicionar info ML se disponível
            if self.usar_ml and 'previsao_ml' in nova_estrategia and nova_estrategia['previsao_ml']:
                numeros_ml = [num for num, prob in nova_estrategia['previsao_ml'][:3]]
                msg += f"\n🤖 ML: {numeros_ml}"
                
            enviar_previsao(msg)

# =============================
# FUNÇÕES AUXILIARES (MANTIDAS)
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
# APLICAÇÃO STREAMLIT ATUALIZADA E CORRIGIDA
# =============================
st.set_page_config(page_title="IA Roleta — Zonas v3 Super Otimizada", layout="centered")
st.title("🎯 IA Roleta — Estratégia das Zonas v3 Super Otimizada + ML")

# Inicialização
if "sistema" not in st.session_state:
    st.session_state.sistema = SistemaRoletaCompleto(usar_ml=False)

if "historico" not in st.session_state:
    if os.path.exists(HISTORICO_PATH):
        try:
            with open(HISTORICO_PATH, "r") as f:
                st.session_state.historico = json.load(f)
        except:
            st.session_state.historico = []
    else:
        st.session_state.historico = []

# Sidebar - Configurações Avançadas
st.sidebar.title("⚙️ Configurações Avançadas")

# Configuração ML
usar_ml = st.sidebar.checkbox("🤖 Ativar Machine Learning", value=False)
if usar_ml != st.session_state.sistema.usar_ml:
    st.session_state.sistema.set_usar_ml(usar_ml)

modo_estrategia = st.sidebar.selectbox(
    "🎯 Estratégia:",
    ["Todas as Estratégias", "Apenas Zonas", "Apenas Midas"],
    key="modo_estrategia"
)

# Aplicar modo selecionado
st.session_state.sistema.set_modo_estrategia(modo_estrategia)

# Treinamento ML - APENAS se ML estiver ativado
if usar_ml:
    with st.sidebar.expander("🧠 Treinamento ML", expanded=True):
        # Calcular quantidade de números disponíveis de forma mais robusta
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
        st.write(f"✅ **Status:** {'Dados suficientes' if numeros_disponiveis >= 100 else 'Coletando dados...'}")
        
        # Informações adicionais sobre o treinamento
        if numeros_disponiveis >= 100:
            st.success("✨ **Pronto para treinar!**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Treinar Modelo ML", type="primary", use_container_width=True):
                    with st.spinner("Treinando modelo ML... Isso pode levar alguns segundos"):
                        try:
                            # CORREÇÃO: Passar o histórico completo para o treinamento
                            success, message = st.session_state.sistema.treinar_modelo_ml(st.session_state.historico)
                            if success:
                                st.success(f"✅ {message}")
                                st.balloons()
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"💥 Erro no treinamento: {str(e)}")
            
            with col2:
                # Botão para ver estatísticas dos dados
                if st.button("📈 Ver Dados", use_container_width=True):
                    st.write("**Estatísticas dos Dados:**")
                    st.write(f"- Total: {len(numeros_lista)} números")
                    if numeros_lista:
                        st.write(f"- Média: {np.mean(numeros_lista):.1f}")
                        st.write(f"- Desvio Padrão: {np.std(numeros_lista):.1f}")
                        st.write(f"- Últimos 10: {numeros_lista[-10:]}")
                    else:
                        st.write("- Nenhum dado disponível")
        
        else:
            st.warning(f"📥 Colete mais {100 - numeros_disponiveis} números para treinar o ML")
            
        # Mostrar status atual do ML
        st.write("---")
        st.write("**Status do ML:**")
        if st.session_state.sistema.estrategia_zonas.ml and st.session_state.sistema.estrategia_zonas.ml.is_trained:
            st.success("✅ Modelo ML treinado e ativo")
            # Tentar fazer uma previsão de exemplo
            if len(numeros_lista) >= 10:
                try:
                    previsao, msg = st.session_state.sistema.estrategia_zonas.ml.prever_proximo_numero(numeros_lista[-20:])  # Últimos 20 para previsão
                    if previsao:
                        st.write("**Última previsão ML (exemplo):**")
                        for i, (num, prob) in enumerate(previsao[:3]):  # Top 3
                            st.write(f"{i+1}. Número {num}: {prob:.2%}")
                except Exception as e:
                    st.write("🔍 Aguardando mais dados para previsão")
        else:
            st.info("🤖 ML aguardando treinamento")

# Informações sobre as Zonas
with st.sidebar.expander("📊 Informações das Zonas"):
    info_zonas = st.session_state.sistema.estrategia_zonas.get_info_zonas()
    for zona, dados in info_zonas.items():
        st.write(f"**Zona {zona}** (Núcleo: {dados['central']})")
        st.write(f"Números: {', '.join(map(str, dados['numeros']))}")
        st.write(f"Total: {dados['quantidade']} números")
        st.write("---")

# Análise detalhada
with st.sidebar.expander("🔍 Análise Detalhada - Zonas v3"):
    analise = st.session_state.sistema.estrategia_zonas.get_analise_detalhada()
    
    # Dividir a análise em seções para melhor legibilidade
    secoes = analise.split('---')
    for i, secao in enumerate(secoes):
        if 'STATUS ML' in secao or 'PREVISÃO ML' in secao:
            st.info(secao)
        elif 'RECOMENDAÇÃO' in secao:
            st.success(secao)
        elif 'AGUARDAR' in secao:
            st.warning(secao)
        else:
            st.text(secao)

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
    # Mostrar apenas os últimos 10 números
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
    
    # Mostrar previsão ML se disponível
    if sistema.usar_ml and 'previsao_ml' in previsao and previsao['previsao_ml']:
        st.write("**🤖 Previsão ML (Top 5):**")
        for num, prob in previsao['previsao_ml']:
            st.write(f"  {num}: {prob:.2%}")
    
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
