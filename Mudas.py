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
import joblib
from streamlit_autorefresh import st_autorefresh

# =============================
# CONFIGURAÇÕES DE NOTIFICAÇÃO
# =============================
def enviar_previsao(mensagem):
    """Envia notificação de previsão"""
    try:
        st.toast(f"🎯 {mensagem}", icon="🔥")
        # Também exibe como warning para maior visibilidade
        st.warning(f"🔔 NOVA PREVISÃO: {mensagem}")
        
        # Enviar para Telegram se configurado
        if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
            if st.session_state.telegram_token and st.session_state.telegram_chat_id:
                enviar_telegram(mensagem)
    except Exception as e:
        logging.error(f"Erro ao enviar previsão: {e}")

def enviar_resultado(mensagem):
    """Envia notificação de resultado"""
    try:
        st.toast(f"🎲 {mensagem}", icon="✅")
        # Exibe como success para resultados
        st.success(f"📢 RESULTADO: {mensagem}")
        
        # Enviar para Telegram se configurado
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
# MÓDULO DE MACHINE LEARNING ATUALIZADO COM CATBOOST
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
        """
        roleta_obj: instância de RoletaInteligente (deve ter .race, get_posicao_race(n) e get_vizinhos_zona(center, k) ou similar)
        min_training_samples: número mínimo de sorteios para treinar
        max_history: quantidade máxima de histórico usada para treinos incrementais
        retrain_every_n: retreina a cada N novos sorteios (se atingido min_training_samples)
        """
        self.roleta = roleta_obj
        self.min_training_samples = min_training_samples
        self.max_history = max_history
        self.retrain_every_n = retrain_every_n
        self.seed = seed

        # Modelos: ensemble simples com 2 instâncias (mesmo algoritmo, seeds diferentes).
        self.models = []  # lista de modelos treinados
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.contador_treinamento = 0
        self.meta = {}  # guarda metas como ultima_acuracia, historico de treinos etc.

        # Parâmetros de engenharia
        self.window_for_features = [5, 10, 20, 50]  # janelas multi-escala
        self.k_vizinhos = 2  # número de vizinhos físicos para formar blocos (antes/depois)
        self.numeros = list(range(37))  # 0..36

    # -------------------------
    # Helpers de vizinhança
    # -------------------------
    def get_neighbors(self, numero, k=None):
        """
        Retorna lista de vizinhos físicos do número (incluindo o central).
        Usa self.roleta.race (lista circular) para calcular vizinhança.
        """
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
            # fallback: retornar apenas o próprio número
            return [numero]

    # -------------------------
    # Feature engineering
    # -------------------------
    def extrair_features(self, historico, numero_alvo=None):
        """
        Extrai features avançadas a partir do histórico (lista ou deque) de números.
        Retorna: features (lista float), feature_names (lista str)
        """
        try:
            historico = list(historico)
            N = len(historico)
            if N < 5:  # requer ao menos 5 para várias features
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

            # --- 2) Estatísticas da janela (para várias janelas)
            for w in self.window_for_features:
                janela = historico[-w:] if N >= w else historico[:]
                arr = np.array(janela, dtype=float)
                features.append(arr.mean() if len(arr) > 0 else 0.0); names.append(f"media_{w}")
                features.append(arr.std() if len(arr) > 1 else 0.0); names.append(f"std_{w}")
                features.append(np.median(arr) if len(arr) > 0 else 0.0); names.append(f"mediana_{w}")

            # --- 3) Frequência por janela e indicadores "quente/frio" relativos
            counter_full = Counter(historico)
            for w in self.window_for_features:
                janela = historico[-w:] if N >= w else historico[:]
                c = Counter(janela)
                # proporção de números distintos (diversidade)
                features.append(len(c) / (w if w>0 else 1)); names.append(f"diversidade_{w}")
                # proporção do top1 nessa janela
                top1_count = c.most_common(1)[0][1] if len(c)>0 else 0
                features.append(top1_count / (w if w>0 else 1)); names.append(f"top1_prop_{w}")

            # --- 4) Tempo desde último para cada número (37 features)
            # Se número não ocorreu, assume len(historico)+1 para indicar "muito tempo"
            for num in self.numeros:
                try:
                    rev_idx = historico[::-1].index(num)
                    tempo = rev_idx  # 0 significa saiu no último sorteio
                except ValueError:
                    tempo = N + 1
                features.append(tempo)
                names.append(f"tempo_desde_{num}")

            # --- 5) Contagens por cor e dúzia e coluna (última janela 50)
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

            # --- 6) Vizinhos físicos: quantos dos últimos K_seq estão nos vizinhos do último número
            ultimo_num = historico[-1]
            vizinhos_k = self.get_neighbors(ultimo_num, k=6)  # mais amplo
            count_in_vizinhos = sum(1 for x in ultimos if x in vizinhos_k) / len(ultimos)
            features.append(count_in_vizinhos); names.append("prop_ultimos_em_vizinhos_6")

            # --- 7) Repetições e padrões binários
            features.append(1 if N>=2 and historico[-1] == historico[-2] else 0); names.append("repetiu_ultimo")
            features.append(1 if N>=2 and (historico[-1] % 2) == (historico[-2] % 2) else 0); names.append("repetiu_paridade")
            features.append(1 if N>=2 and duzia_of(historico[-1]) == duzia_of(historico[-2]) else 0); names.append("repetiu_duzia")

            # --- 8) Diferenças entre janelas (indicador de mudança de tendência)
            if N >= max(self.window_for_features):
                small = np.mean(historico[-self.window_for_features[0]:])
                large = np.mean(historico[-self.window_for_features[-1]:])
                features.append(small - large); names.append("delta_media_small_large")
            else:
                features.append(0.0); names.append("delta_media_small_large")

            # --- 9) Estatísticas de transição (média e std das diferenças absolutas)
            diffs = [abs(historico[i] - historico[i-1]) for i in range(1, len(historico))]
            features.append(np.mean(diffs) if len(diffs)>0 else 0.0); names.append("media_transicoes")
            features.append(np.std(diffs) if len(diffs)>1 else 0.0); names.append("std_transicoes")

            # finalize
            self.feature_names = names
            return features, names

        except Exception as e:
            logging.error(f"[extrair_features] Erro: {e}")
            return None, None

    # -------------------------
    # Preparar dados para treino
    # -------------------------
    def preparar_dados_treinamento(self, historico_completo):
        """
        Gera X, y a partir do histórico completo.
        Usa janelas deslizantes: para cada i, features de historico[:i] -> target historico[i].
        Limita a self.max_history (últimos N).
        """
        historico_completo = list(historico_completo)
        # Limitar ao max_history mais recentes para evitar crescimento infinito
        if len(historico_completo) > self.max_history:
            historico_completo = historico_completo[-self.max_history:]

        X = []
        y = []
        # Começa em 20 pra ter janela inicial com informações
        for i in range(20, len(historico_completo)):
            janela = historico_completo[:i]
            feats, _ = self.extrair_features(janela)
            if feats is None:
                continue
            X.append(feats)
            y.append(historico_completo[i])  # próximo número
        if len(X) == 0:
            return np.array([]), np.array([])
        return np.array(X), np.array(y)

    # -------------------------
    # Treinamento (CatBoost ou RF fallback)
    # -------------------------
    def _build_and_train_model(self, X_train, y_train, X_val=None, y_val=None, seed=0):
        """
        Tenta treinar CatBoost (com early stopping), senão RandomForest.
        Retorna modelo treinado e string com nome.
        """
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                iterations=1500,
                learning_rate=0.05,
                depth=10,
                l2_leaf_reg=5,
                bagging_temperature=0.8,
                random_strength=1.0,
                loss_function='MultiClass',
                eval_metric='MultiClass',
                random_seed=seed,
                use_best_model=True,
                early_stopping_rounds=100,
                verbose=False
            )
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            else:
                model.fit(X_train, y_train, verbose=False)
            return model, "CatBoost"
        except Exception as e:
            logging.warning(f"CatBoost não disponível ou falha ({e}). Usando RandomForest como fallback.")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=400,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            return model, "RandomForest"

    def treinar_modelo(self, historico_completo, force_retrain: bool = False, balance: bool = True):
        """
        Treina ensemble (2 modelos) usando o histórico.
        balance: se True faz resampling estratificado para mitigar classes raras.
        """
        try:
            if len(historico_completo) < self.min_training_samples and not force_retrain:
                return False, f"Necessário mínimo de {self.min_training_samples} amostras. Atual: {len(historico_completo)}"

            X, y = self.preparar_dados_treinamento(historico_completo)
            if X.size == 0 or len(X) < 50:
                return False, f"Dados insuficientes para treino: {len(X)} amostras"

            # Normalize
            X_scaled = self.scaler.fit_transform(X)

            # Split train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=self.seed, stratify=y if len(set(y))>1 else None
            )

            # Opcional: balancear com resample per class
            if balance:
                # Cria concat para reamostragem
                df_train = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
                df_train['y'] = y_train
                # encontra classe majoritária e resample outras para equilibrar
                max_count = df_train['y'].value_counts().max()
                frames = []
                for cls, grp in df_train.groupby('y'):
                    if len(grp) < max_count:
                        grp_up = resample(grp, replace=True, n_samples=max_count, random_state=self.seed)
                        frames.append(grp_up)
                    else:
                        frames.append(grp)
                df_bal = pd.concat(frames)
                y_train = df_bal['y'].values
                X_train = df_bal.drop(columns=['y']).values

            # Treinar 2 modelos com seeds diferentes e armazenar ensemble
            models = []
            model_names = []
            for s in [self.seed, self.seed + 7]:
                model, name = self._build_and_train_model(X_train, y_train, X_val, y_val, seed=s)
                models.append(model)
                model_names.append(name)

            # Avaliar (média das probabilidades)
            probs = []
            for m in models:
                if hasattr(m, 'predict_proba'):
                    probs.append(m.predict_proba(X_val))
                else:
                    preds = m.predict(X_val)
                    # transforma em one-hot probabilidades
                    prob = np.zeros((len(preds), len(self.numeros)))
                    for i, p in enumerate(preds):
                        prob[i, p] = 1.0
                    probs.append(prob)
            avg_prob = np.mean(probs, axis=0)
            y_pred = np.argmax(avg_prob, axis=1)
            acc = accuracy_score(y_val, y_pred)

            # Save ensemble
            self.models = models
            self.is_trained = True
            self.contador_treinamento += 1
            self.meta['last_accuracy'] = acc
            self.meta['trained_on'] = len(historico_completo)

            # persistir
            try:
                joblib.dump({'models': self.models}, ML_MODEL_PATH)
                joblib.dump(self.scaler, SCALER_PATH)
                joblib.dump(self.meta, META_PATH)
            except Exception as e:
                logging.warning(f"Falha ao salvar modelos: {e}")

            return True, f"Ensemble treinado ({', '.join(model_names)}) com {len(X)} amostras. Acurácia validação: {acc:.2%}"

        except Exception as e:
            logging.error(f"[treinar_modelo] Erro: {e}", exc_info=True)
            return False, f"Erro no treinamento: {str(e)}"

    # -------------------------
    # Carregar modelos salvos
    # -------------------------
    def carregar_modelo(self):
        try:
            if os.path.exists(ML_MODEL_PATH) and os.path.exists(SCALER_PATH):
                data = joblib.load(ML_MODEL_PATH)
                self.models = data.get('models', [])
                self.scaler = joblib.load(SCALER_PATH)
                if os.path.exists(META_PATH):
                    self.meta = joblib.load(META_PATH)
                self.is_trained = len(self.models) > 0
                return True
            return False
        except Exception as e:
            logging.error(f"[carregar_modelo] Erro: {e}")
            return False

    # -------------------------
    # Previsão: números + blocos (vizinhanca)
    # -------------------------
    def _ensemble_predict_proba(self, X_scaled):
        """
        Retorna média de probabilidades dos modelos do ensemble para X_scaled.
        """
        if not self.models:
            raise RuntimeError("Modelos não carregados/treinados")

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

    def prever_proximo_numero(self, historico, top_k: int = 20):
        """
        Retorna top_k números mais prováveis com probabilidades.
        """
        if not self.is_trained:
            return None, "Modelo não treinado"

        feats, _ = self.extrair_features(historico)
        if feats is None:
            return None, "Features insuficientes"

        Xs = np.array([feats])
        Xs_scaled = self.scaler.transform(Xs)
        try:
            probs = self._ensemble_predict_proba(Xs_scaled)[0]  # probs para cada número 0..36
            top_idx = np.argsort(probs)[-top_k:][::-1]
            top = [(int(idx), float(probs[idx])) for idx in top_idx]
            return top, "Previsão ML realizada"
        except Exception as e:
            return None, f"Erro na previsão: {str(e)}"

    def prever_blocos_vizinhos(self, historico, k_neighbors: int = 2, top_blocks: int = 5):
        """
        Retorna os top_blocks blocos (vizinhos físicos) com probabilidade agregada.
        Cada bloco é um número central + k_neighbors antes/depois.
        """
        pred, msg = self.prever_proximo_numero(historico, top_k=37)
        if pred is None:
            return None, msg
        # prob dict
        prob = {num: p for num, p in pred}
        blocks = []
        # gerar blocos para cada número 0..36
        for num in range(37):
            neigh = self.get_neighbors(num, k=k_neighbors)
            agg_prob = sum(prob.get(n, 0.0) for n in neigh)
            blocks.append((num, tuple(neigh), agg_prob))
        blocks_sorted = sorted(blocks, key=lambda x: x[2], reverse=True)[:top_blocks]
        # formatar para retorno
        formatted = [{"central": b[0], "vizinhos": list(b[1]), "prob": float(b[2])} for b in blocks_sorted]
        return formatted, "Previsão de blocos realizada"

    # -------------------------
    # Feedback / atualização on-line
    # -------------------------
    def registrar_resultado(self, historico, previsao_top, resultado_real):
        """
        Recebe feedback (resultado_real int), e atualiza meta/registro.
        previsao_top: lista de números previstos (ex: top3 ou top20)
        Guarda logs e, opcionalmente, faz um retreinamento incremental leve.
        """
        try:
            hit = resultado_real in [p for p,_ in previsao_top] if isinstance(previsao_top[0], tuple) else resultado_real in previsao_top
            log_entry = {
                'prev_top': previsao_top,
                'resultado': resultado_real,
                'hit': bool(hit)
            }
            # Guardar em meta history
            self.meta.setdefault('history_feedback', []).append(log_entry)
            # Ajuste simples: se 3 erros seguidos aumentar prioridade de retreino
            recent = self.meta['history_feedback'][-10:]
            hits = sum(1 for r in recent if r['hit'])
            if len(recent) >= 5 and hits / len(recent) < 0.25:
                # força retreinamento com dados atuais (se tiver suficientes)
                logging.info("[feedback] Baixa performance detectada — forçando retreinamento incremental")
                self.treinar_modelo(historico, force_retrain=True, balance=True)
            return True
        except Exception as e:
            logging.error(f"[registrar_resultado] Erro: {e}")
            return False

    # -------------------------
    # Verificar re-treinamento automático
    # -------------------------
    def verificar_treinamento_automatico(self, historico_completo):
        """
        Verifica se é hora de retreinar: se len(historico) cresceu em múltiplos de retrain_every_n
        e se temos dados mínimos.
        """
        try:
            n = len(historico_completo)
            if n >= self.min_training_samples:
                # padrão: retreina quando n % retrain_every_n == 0
                if n % self.retrain_every_n == 0:
                    return self.treinar_modelo(historico_completo)
            return False, "Aguardando próximo ciclo de treinamento"
        except Exception as e:
            return False, f"Erro ao verificar retrain: {e}"

    # -------------------------
    # Utilitários
    # -------------------------
    def resumo_meta(self):
        """
        Retorna informações rápidas sobre o estado do modelo.
        """
        return {
            "is_trained": self.is_trained,
            "contador_treinamento": self.contador_treinamento,
            "meta": self.meta
        }

# =============================
# ESTRATÉGIA DAS ZONAS ATUALIZADA - 6 VIZINHOS ANTES E 6 DEPOIS
# =============================
class EstrategiaZonasOtimizada:
    def __init__(self):
        self.roleta = RoletaInteligente()
        self.historico = deque(maxlen=35)
        self.nome = "Zonas Ultra Otimizada v5"
        
        # Zonas com 6 vizinhos antes e 6 depois (total 13 números por zona)
        self.zonas = {
            'Vermelha': 7,   # Performance: 38.4% 🏆
            'Azul': 10,      # Performance: 27.8% 📈  
            'Amarela': 2     # Performance: 26.5% 📈
        }
        
        # TODAS as zonas agora com 6 vizinhos antes e 6 depois
        self.quantidade_zonas = {
            'Vermelha': 6,   # 6 antes + 6 depois + central = 13 números
            'Azul': 6,       # 6 antes + 6 depois + central = 13 números
            'Amarela': 6     # 6 antes + 6 depois + central = 13 números
        }
        
        # Pré-calcular zonas com 13 números cada
        self.numeros_zonas = {}
        for nome, central in self.zonas.items():
            qtd = self.quantidade_zonas.get(nome, 6)
            self.numeros_zonas[nome] = self.roleta.get_vizinhos_zona(central, qtd)

        # Estatísticas avançadas
        self.stats_zonas = {zona: {
            'acertos': 0, 
            'tentativas': 0, 
            'sequencia_atual': 0,
            'sequencia_maxima': 0,
            'performance_media': 0
        } for zona in self.zonas.keys()}

    def adicionar_numero(self, numero):
        self.historico.append(numero)
        return self.atualizar_stats(numero)

    def atualizar_stats(self, ultimo_numero):
        """Atualiza estatísticas de performance das zonas"""
        acertou_zona = None
        for zona, numeros in self.numeros_zonas.items():
            if ultimo_numero in numeros:
                self.stats_zonas[zona]['acertos'] += 1
                self.stats_zonas[zona]['sequencia_atual'] += 1
                # Atualizar sequência máxima
                if self.stats_zonas[zona]['sequencia_atual'] > self.stats_zonas[zona]['sequencia_maxima']:
                    self.stats_zonas[zona]['sequencia_maxima'] = self.stats_zonas[zona]['sequencia_atual']
                acertou_zona = zona
            else:
                self.stats_zonas[zona]['sequencia_atual'] = 0
            self.stats_zonas[zona]['tentativas'] += 1
            
            # Atualizar performance média
            if self.stats_zonas[zona]['tentativas'] > 0:
                self.stats_zonas[zona]['performance_media'] = (
                    self.stats_zonas[zona]['acertos'] / self.stats_zonas[zona]['tentativas'] * 100
                )
        
        return acertou_zona

    def get_zona_mais_quente(self):
        """Sistema de scoring ULTRA otimizado com performance real"""
        if len(self.historico) < 15:
            return None
            
        zonas_score = {}
        total_numeros = len(self.historico)
        
        for zona in self.zonas.keys():
            score = 0
            
            # CRITÉRIO 1: Frequência geral (25% do score)
            freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
            percentual_geral = freq_geral / total_numeros
            score += percentual_geral * 25
            
            # CRITÉRIO 2: Frequência recente (35% do score)
            ultimos_15 = list(self.historico)[-15:] if total_numeros >= 15 else list(self.historico)
            freq_recente = sum(1 for n in ultimos_15 if n in self.numeros_zonas[zona])
            percentual_recente = freq_recente / len(ultimos_15)
            score += percentual_recente * 35
            
            # CRITÉRIO 3: Performance histórica REAL (30% do score)
            if self.stats_zonas[zona]['tentativas'] > 10:
                taxa_acerto = self.stats_zonas[zona]['performance_media']
                # Bônus PROGRESSIVO baseado na performance real
                if taxa_acerto > 40: 
                    score += 30  # Máximo para zonas excelentes
                elif taxa_acerto > 35:
                    score += 25
                elif taxa_acerto > 30:
                    score += 20
                elif taxa_acerto > 25:
                    score += 15
                else:
                    score += 10
            else:
                score += 10  # Default para zonas novas
            
            # CRITÉRIO 4: Sequência e momentum (10% do score)
            sequencia = self.stats_zonas[zona]['sequencia_atual']
            if sequencia >= 2:
                score += min(sequencia * 3, 10)  # Bônus mais conservador
            
            zonas_score[zona] = score
        
        # Encontrar zona vencedora com threshold MAIS INTELIGENTE
        zona_vencedora = max(zonas_score, key=zonas_score.get) if zonas_score else None
        
        # Threshold DINÂMICO baseado na performance REAL
        if zona_vencedora:
            threshold = 28  # Aumentado ligeiramente
            
            # Ajuste baseado na performance histórica
            if self.stats_zonas[zona_vencedora]['tentativas'] > 15:
                taxa = self.stats_zonas[zona_vencedora]['performance_media']
                if taxa > 38:  # Zonas excelentes
                    threshold = 25  # Mais sensível
                elif taxa < 25:  # Zonas com baixa performance
                    threshold = 32  # Mais rigoroso
            
            # Bônus adicional se estiver em sequência
            if self.stats_zonas[zona_vencedora]['sequencia_atual'] >= 2:
                threshold -= 2  # Mais sensível durante sequências
            
            return zona_vencedora if zonas_score[zona_vencedora] >= threshold else None
        
        return None

    def analisar_zonas(self):
        """Versão ULTRA otimizada com 13 números por zona"""
        if len(self.historico) < 15:
            return None
            
        zona_alvo = self.get_zona_mais_quente()
        
        if zona_alvo:
            numeros_apostar = self.numeros_zonas[zona_alvo]
            
            # Confiança ULTRA inteligente
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
        """Sistema de confiança ULTRA inteligente"""
        if len(self.historico) < 10:
            return 'Baixa'
            
        # Múltiplos fatores PONDERADOS
        fatores = []
        pesos = []
        
        # Fator 1: Performance histórica (PESO 4)
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
        
        # Fator 2: Frequência recente (PESO 3)
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
        
        # Fator 3: Sequência atual (PESO 2)
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
        
        # Fator 4: Tendência (PESO 2)
        if len(self.historico) >= 10:
            ultimos_5 = list(self.historico)[-5:]
            anteriores_5 = list(self.historico)[-10:-5]
            
            freq_ultimos = sum(1 for n in ultimos_5 if n in self.numeros_zonas[zona])
            freq_anteriores = sum(1 for n in anteriores_5 if n in self.numeros_zonas[zona])
            
            if freq_ultimos > freq_anteriores: 
                fatores.append(3)  # Tendência positiva
                pesos.append(2)
            elif freq_ultimos == freq_anteriores: 
                fatores.append(2)  # Estável
                pesos.append(2)
            else: 
                fatores.append(1)  # Tendência negativa
                pesos.append(2)
        
        # Cálculo PONDERADO
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
        """Score detalhado para debug"""
        if len(self.historico) < 10:
            return 0
            
        score = 0
        total_numeros = len(self.historico)
        
        # Frequência geral
        freq_geral = sum(1 for n in self.historico if n in self.numeros_zonas[zona])
        percentual_geral = freq_geral / total_numeros
        score += percentual_geral * 25
        
        # Frequência recente
        ultimos_15 = list(self.historico)[-15:] if total_numeros >= 15 else list(self.historico)
        freq_recente = sum(1 for n in ultimos_15 if n in self.numeros_zonas[zona])
        percentual_recente = freq_recente / len(ultimos_15)
        score += percentual_recente * 35
        
        # Performance histórica
        if self.stats_zonas[zona]['tentativas'] > 10:
            taxa_acerto = self.stats_zonas[zona]['performance_media']
            if taxa_acerto > 40: score += 30
            elif taxa_acerto > 35: score += 25
            elif taxa_acerto > 30: score += 20
            elif taxa_acerto > 25: score += 15
            else: score += 10
        else:
            score += 10
        
        # Sequência
        sequencia = self.stats_zonas[zona]['sequencia_atual']
        if sequencia >= 2:
            score += min(sequencia * 3, 10)
            
        return score

    def get_info_zonas(self):
        """Retorna informações sobre as zonas para display"""
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
        """Análise ULTRA detalhada com 13 números por zona"""
        if len(self.historico) == 0:
            return "Aguardando dados..."
        
        analise = "🎯 ANÁLISE ULTRA OTIMIZADA - ZONAS v5\n"
        analise += "=" * 55 + "\n"
        analise += "🔧 CONFIGURAÇÃO: 6 antes + 6 depois (13 números/zona)\n"
        analise += "=" * 55 + "\n"
        
        # Performance ULTRA detalhada
        analise += "📊 PERFORMANCE AVANÇADA:\n"
        for zona in self.zonas.keys():
            tentativas = self.stats_zonas[zona]['tentativas']
            acertos = self.stats_zonas[zona]['acertos']
            taxa = self.stats_zonas[zona]['performance_media']
            sequencia = self.stats_zonas[zona]['sequencia_atual']
            seq_maxima = self.stats_zonas[zona]['sequencia_maxima']
            
            analise += f"📍 {zona}: {acertos}/{tentativas} → {taxa:.1f}% | Seq: {sequencia} | Máx: {seq_maxima}\n"
        
        analise += "\n📈 FREQUÊNCIA ATUAL:\n"
        for zona in self.zonas.keys():
            freq = sum(1 for n in self.historico if isinstance(n, (int, float)) and n in self.numeros_zonas[zona])
            perc = (freq / len(self.historico)) * 100
            score = self.get_zona_score(zona)
            qtd_numeros = len(self.numeros_zonas[zona])
            analise += f"📍 {zona}: {freq}/{len(self.historico)} → {perc:.1f}% | Score: {score:.1f} | Números: {qtd_numeros}\n"
        
        # Tendências avançadas
        analise += "\n📊 TENDÊNCIAS AVANÇADAS:\n"
        if len(self.historico) >= 10:
            for zona in self.zonas.keys():
                ultimos_5 = list(self.historico)[-5:]
                anteriores_5 = list(self.historico)[-10:-5]
                
                freq_ultimos = sum(1 for n in ultimos_5 if n in self.numeros_zonas[zona])
                freq_anteriores = sum(1 for n in anteriores_5 if n in self.numeros_zonas[zona]) if anteriores_5 else 0
                
                tendencia = "↗️" if freq_ultimos > freq_anteriores else "↘️" if freq_ultimos < freq_anteriores else "➡️"
                variacao = freq_ultimos - freq_anteriores
                analise += f"📍 {zona}: {freq_ultimos}/5 vs {freq_anteriores}/5 {tendencia} (Δ: {variacao:+d})\n"
        
        # Recomendações ULTRA inteligentes
        zona_recomendada = self.get_zona_mais_quente()
        if zona_recomendada:
            analise += f"\n💡 RECOMENDAÇÃO ULTRA: Zona {zona_recomendada}\n"
            analise += f"🎯 Números: {sorted(self.numeros_zonas[zona_recomendada])}\n"
            analise += f"📈 Confiança: {self.calcular_confianca_ultra(zona_recomendada)}\n"
            analise += f"🔥 Score: {self.get_zona_score(zona_recomendada):.1f}\n"
            analise += f"🔢 Quantidade: {len(self.numeros_zonas[zona_recomendada])} números\n"
            analise += f"📊 Performance: {self.stats_zonas[zona_recomendada]['performance_media']:.1f}%\n"
            
            # Dica estratégica baseada na performance
            perf = self.stats_zonas[zona_recomendada]['performance_media']
            if perf > 35:
                analise += f"💎 ESTRATÉGIA: Zona de ALTA performance - Aposta forte recomendada!\n"
            elif perf > 25:
                analise += f"🎯 ESTRATÉGIA: Zona de performance sólida - Aposta moderada\n"
            else:
                analise += f"⚡ ESTRATÉGIA: Zona em desenvolvimento - Aposta conservadora\n"
        else:
            analise += "\n⚠️  AGUARDAR: Nenhuma zona com confiança suficiente\n"
            analise += f"📋 Histórico atual: {len(self.historico)} números\n"
            analise += f"🎯 Threshold mínimo: Score 28+ | Performance >25%\n"
        
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
# ESTRATÉGIA ML ATUALIZADA - CATBOOST E TREINAMENTO AUTOMÁTICO
# =============================
class EstrategiaML:
    def __init__(self):
        self.ml = MLRoleta()
        self.historico = deque(maxlen=30)
        self.nome = "Machine Learning (CatBoost)"
        self.ml.carregar_modelo()
        self.roleta = RoletaInteligente()
        self.contador_sorteios = 0
        
        # Definir as zonas para análise (ATUALIZADO - 6 antes e 6 depois)
        self.zonas_ml = {
            'Vermelha': 7,
            'Azul': 10,  
            'Amarela': 2
        }
        
        # TODAS as zonas agora com 6 vizinhos antes e 6 depois
        self.quantidade_zonas_ml = {
            'Vermelha': 6,
            'Azul': 6,
            'Amarela': 6
        }
        
        # Pré-calcular números das zonas (13 números cada)
        self.numeros_zonas_ml = {}
        for nome, central in self.zonas_ml.items():
            qtd = self.quantidade_zonas_ml.get(nome, 6)
            self.numeros_zonas_ml[nome] = self.roleta.get_vizinhos_zona(central, qtd)

    def adicionar_numero(self, numero):
        self.historico.append(numero)
        self.contador_sorteios += 1
        
        # Verificar treinamento automático a cada 10 sorteios
        if self.contador_sorteios >= 10:
            self.contador_sorteios = 0
            self.treinar_automatico()

    def treinar_automatico(self):
        """Treinamento automático a cada 10 sorteios"""
        historico_numeros = self.extrair_numeros_historico()
        
        if len(historico_numeros) >= self.ml.min_training_samples:
            try:
                success, message = self.ml.treinar_modelo(historico_numeros)
                if success:
                    logging.info(f"✅ Treinamento automático ML: {message}")
                    enviar_resultado(f"🤖 ML retreinado automaticamente: {message}")
                else:
                    logging.warning(f"⚠️ Treinamento automático falhou: {message}")
            except Exception as e:
                logging.error(f"❌ Erro no treinamento automático: {e}")

    def extrair_numeros_historico(self):
        """Extrai números do histórico para treinamento"""
        historico_numeros = []
        for item in list(self.historico):
            if isinstance(item, dict) and 'number' in item:
                historico_numeros.append(item['number'])
            elif isinstance(item, (int, float)):
                historico_numeros.append(int(item))
        return historico_numeros

    def analisar_ml(self):
        """Estratégia ML com CatBoost e treinamento automático"""
        if len(self.historico) < 10:
            return None

        if not self.ml.is_trained:
            return None

        historico_numeros = self.extrair_numeros_historico()

        if len(historico_numeros) < 10:
            return None

        previsao_ml, msg_ml = self.ml.prever_proximo_numero(historico_numeros)
        
        if previsao_ml:
            # Pegar os top 20 números mais prováveis
            top_20_numeros = [num for num, prob in previsao_ml[:20]]
            
            # Analisar distribuição por zonas
            distribuicao_zonas = self.analisar_distribuicao_zonas(top_20_numeros)
            
            if distribuicao_zonas:
                zona_vencedora = distribuicao_zonas['zona_vencedora']
                numeros_zona = self.numeros_zonas_ml[zona_vencedora]
                contagem = distribuicao_zonas['contagem']
                total_zonas = distribuicao_zonas['total_zonas']
                
                # Calcular confiança baseada na distribuição
                confianca = self.calcular_confianca_zona_ml(distribuicao_zonas)
                
                # ENVIAR ALERTA TELEGRAM PARA ENTRADA
                self.enviar_alerta_entrada_telegram(zona_vencedora, contagem, total_zonas, numeros_zona, confianca)
                
                return {
                    'nome': 'Machine Learning - CatBoost',
                    'numeros_apostar': numeros_zona,
                    'gatilho': f'ML CatBoost - Zona {zona_vencedora} ({contagem}/{total_zonas} números)',
                    'confianca': confianca,
                    'previsao_ml': previsao_ml,
                    'zona_ml': zona_vencedora,
                    'distribuicao': distribuicao_zonas
                }
        
        return None

    def analisar_distribuicao_zonas(self, top_20_numeros):
        """Analisa a distribuição dos top 20 números pelas zonas"""
        contagem_zonas = {}
        
        # Contar quantos números de cada zona estão no top 20
        for zona, numeros in self.numeros_zonas_ml.items():
            count = sum(1 for num in top_20_numeros if num in numeros)
            contagem_zonas[zona] = count
        
        # Encontrar a zona com maior contagem
        if contagem_zonas:
            zona_vencedora = max(contagem_zonas, key=contagem_zonas.get)
            contagem_vencedora = contagem_zonas[zona_vencedora]
            
            # Threshold aumentado para 6 (13 números por zona)
            if contagem_vencedora >= 6:
                return {
                    'zona_vencedora': zona_vencedora,
                    'contagem': contagem_vencedora,
                    'total_zonas': len(top_20_numeros),
                    'distribuicao_completa': contagem_zonas
                }
        
        return None

    def calcular_confianca_zona_ml(self, distribuicao):
        """Calcula confiança baseada na distribuição por zonas"""
        contagem = distribuicao['contagem']
        total = distribuicao['total_zonas']
        percentual = (contagem / total) * 100
        
        # Ajustado para 20 números
        if percentual >= 50:  # 10+ números
            return 'Muito Alta'
        elif percentual >= 40:  # 8-9 números
            return 'Alta'
        elif percentual >= 30:  # 6-7 números
            return 'Média'
        elif percentual >= 25:  # 5 números
            return 'Baixa'
        else:
            return 'Muito Baixa'

    def enviar_alerta_entrada_telegram(self, zona, contagem, total, numeros_zona, confianca):
        """Envia alerta específico de entrada para o Telegram"""
        try:
            if 'telegram_token' in st.session_state and 'telegram_chat_id' in st.session_state:
                token = st.session_state.telegram_token
                chat_id = st.session_state.telegram_chat_id
                
                if token and chat_id:
                    mensagem = f"""
🎯 <b>ALERTA DE ENTRADA - ML CATBOOST</b>

🏆 <b>Zona Recomendada:</b> {zona}
📊 <b>Confiança:</b> {confianca}
🔢 <b>Números na zona:</b> {contagem}/{total}

🎲 <b>Números para apostar (13):</b>
{', '.join(map(str, sorted(numeros_zona)))}

💡 <b>Estratégia:</b> Machine Learning - CatBoost
🔄 <b>Treinamento:</b> Automático a cada 10 sorteios
🕒 <b>Horário:</b> {pd.Timestamp.now().strftime('%H:%M:%S')}

⚡ <b>ENTRAR AGORA!</b>
"""
                    
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {
                        "chat_id": chat_id,
                        "text": mensagem,
                        "parse_mode": "HTML"
                    }
                    
                    response = requests.post(url, json=payload, timeout=10)
                    if response.status_code == 200:
                        logging.info("Alerta de entrada enviado para Telegram")
                    else:
                        logging.error(f"Erro ao enviar alerta: {response.status_code}")
        except Exception as e:
            logging.error(f"Erro no alerta Telegram: {e}")

    def treinar_modelo_ml(self, historico_completo=None):
        """Treina o modelo de ML"""
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
        """Retorna análise do ML com informações de treinamento automático"""
        if not self.ml.is_trained:
            return "🤖 ML: Modelo não treinado"
        
        if len(self.historico) < 10:
            return "🤖 ML: Aguardando mais dados para análise"
        
        historico_numeros = self.extrair_numeros_historico()

        previsao_ml, msg = self.ml.prever_proximo_numero(historico_numeros)
        
        if previsao_ml:
            # Detectar qual modelo está sendo usado
            modelo_tipo = "CatBoost" if hasattr(self.ml.model, 'iterations') else "RandomForest"
            
            analise = f"🤖 ANÁLISE ML - {modelo_tipo.upper()} (TOP 20):\n"
            analise += f"🔄 Treinamentos realizados: {self.ml.contador_treinamento}\n"
            analise += f"📊 Próximo treinamento: {10 - self.contador_sorteios} sorteios\n"
            analise += "🎯 Previsões (Top 10):\n"
            for i, (num, prob) in enumerate(previsao_ml[:10]):
                analise += f"  {i+1}. Número {num}: {prob:.2%}\n"
            
            # Análise de distribuição por zonas
            top_20_numeros = [num for num, prob in previsao_ml[:20]]
            distribuicao = self.analisar_distribuicao_zonas(top_20_numeros)
            
            if distribuicao:
                analise += f"\n🎯 DISTRIBUIÇÃO POR ZONAS (20 números):\n"
                for zona, count in distribuicao['distribuicao_completa'].items():
                    analise += f"  📍 {zona}: {count}/20 números\n"
                
                analise += f"\n💡 ZONA RECOMENDADA: {distribuicao['zona_vencedora']}\n"
                analise += f"🎯 Confiança: {self.calcular_confianca_zona_ml(distribuicao)}\n"
                analise += f"🔢 Números da zona: {sorted(self.numeros_zonas_ml[distribuicao['zona_vencedora']])}\n"
                analise += f"📈 Percentual: {(distribuicao['contagem']/20)*100:.1f}%\n"
            else:
                analise += "\n⚠️  Nenhuma zona com predominância suficiente (mínimo 6 números)\n"
            
            return analise
        else:
            return "🤖 ML: Erro na previsão"

    def get_info_zonas_ml(self):
        """Retorna informações sobre as zonas usadas no ML"""
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
# SISTEMA DE GESTÃO ATUALIZADO COM CATBOOST
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
        self.estrategia_selecionada = "Zonas"  # Padrão: Zonas
        self.contador_sorteios_global = 0

    def set_estrategia(self, estrategia):
        """Define a estratégia a ser usada"""
        self.estrategia_selecionada = estrategia

    def treinar_modelo_ml(self, historico_completo=None):
        """Treina o modelo de ML"""
        return self.estrategia_ml.treinar_modelo_ml(historico_completo)

    def processar_novo_numero(self, numero):
        # Extrair número se for um dicionário
        if isinstance(numero, dict) and 'number' in numero:
            numero_real = numero['number']
        else:
            numero_real = numero
            
        # Incrementar contador global
        self.contador_sorteios_global += 1
            
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
                # Notificação de ACERTO
                enviar_resultado(f"🎉 ACERTO! Número {numero_real} - Estratégia: {nome_estrategia}")
            else:
                self.erros += 1
                # Notificação de ERRO
                enviar_resultado(f"❌ ERRO! Número {numero_real} - Estratégia: {nome_estrategia}")
            
            self.historico_desempenho.append({
                'numero': numero_real,
                'acerto': acerto,
                'estrategia': nome_estrategia,
                'previsao': self.previsao_ativa['numeros_apostar']
            })
            
            self.previsao_ativa = None
        
        # Adicionar número a todas as estratégias (para manter estatísticas)
        self.estrategia_zonas.adicionar_numero(numero_real)
        self.estrategia_midas.adicionar_numero(numero_real)
        self.estrategia_ml.adicionar_numero(numero_real)
        
        # Verificar nova estratégia baseada na seleção
        nova_estrategia = None
        
        if self.estrategia_selecionada == "Zonas":
            nova_estrategia = self.estrategia_zonas.analisar_zonas()
        elif self.estrategia_selecionada == "Midas":
            nova_estrategia = self.estrategia_midas.analisar_midas()
        elif self.estrategia_selecionada == "ML":
            nova_estrategia = self.estrategia_ml.analisar_ml()
        
        if nova_estrategia:
            self.previsao_ativa = nova_estrategia
            # Enviar alerta
            msg = f"🎯 {nova_estrategia['nome']} - {nova_estrategia['confianca']}\n"
            msg += f"🎲 Gatilho: {nova_estrategia['gatilho']}\n"
            msg += f"🔢 Números: {', '.join(map(str, sorted(nova_estrategia['numeros_apostar'])))}"
            
            # Adicionar info ML se disponível
            if 'previsao_ml' in nova_estrategia and nova_estrategia['previsao_ml']:
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
# APLICAÇÃO STREAMLIT ATUALIZADA
# =============================
st.set_page_config(page_title="IA Roleta — Multi-Estratégias", layout="centered")
st.title("🎯 IA Roleta — Sistema Multi-Estratégias")

# Inicialização
if "sistema" not in st.session_state:
    st.session_state.sistema = SistemaRoletaCompleto()

if "historico" not in st.session_state:
    if os.path.exists(HISTORICO_PATH):
        try:
            with open(HISTORICO_PATH, "r") as f:
                st.session_state.historico = json.load(f)
        except:
            st.session_state.historico = []
    else:
        st.session_state.historico = []

# Inicializar configurações do Telegram
if "telegram_token" not in st.session_state:
    st.session_state.telegram_token = ""
if "telegram_chat_id" not in st.session_state:
    st.session_state.telegram_chat_id = ""

# Sidebar - Configurações Avançadas
st.sidebar.title("⚙️ Configurações")

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

# Treinamento ML
with st.sidebar.expander("🧠 Treinamento ML", expanded=False):
    # Calcular quantidade de números disponíveis
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
    st.write(f"✅ **Status:** {'Dados suficientes' if numeros_disponiveis >= 100 else 'Coletando dados...'}")
    
    # Informações adicionais sobre o treinamento
    if numeros_disponiveis >= 100:
        st.success("✨ **Pronto para treinar!**")
        
        if st.button("🚀 Treinar Modelo ML", type="primary", use_container_width=True):
            with st.spinner("Treinando modelo ML com CatBoost... Isso pode levar alguns segundos"):
                try:
                    success, message = st.session_state.sistema.treinar_modelo_ml(numeros_lista)
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                        enviar_resultado(f"🤖 Modelo ML treinado com sucesso! {message}")
                    else:
                        st.error(f"❌ {message}")
                        enviar_resultado(f"❌ Falha no treinamento ML: {message}")
                except Exception as e:
                    st.error(f"💥 Erro no treinamento: {str(e)}")
                    enviar_resultado(f"💥 Erro no treinamento ML: {str(e)}")
    
    else:
        st.warning(f"📥 Colete mais {100 - numeros_disponiveis} números para treinar o ML")
        
    # Mostrar status atual do ML
    st.write("---")
    st.write("**Status do ML:**")
    if st.session_state.sistema.estrategia_ml.ml.is_trained:
        # Detectar qual modelo está sendo usado
        if hasattr(st.session_state.sistema.estrategia_ml.ml.model, 'iterations'):
            modelo_tipo = "CatBoost"
        else:
            modelo_tipo = "RandomForest"
            
        st.success(f"✅ Modelo {modelo_tipo} treinado ({st.session_state.sistema.estrategia_ml.ml.contador_treinamento} vezes)")
        st.info(f"🔄 Próximo treinamento automático em: {10 - st.session_state.sistema.estrategia_ml.contador_sorteios} sorteios")
    else:
        st.info("🤖 ML aguardando treinamento")

# Informações sobre as Estratégias ATUALIZADAS
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
        st.write("**🤖 Estratégia Machine Learning - CATBOOST:**")
        st.write("- **Modelo**: CatBoost (Gradient Boosting)")
        st.write("- **Vantagem**: Mais preciso para dados sequenciais")
        st.write("- **Análise**: Top 20 números previstos")
        st.write("- **Treinamento**: Automático a cada 10 sorteios")
        st.write("- **Zonas**: 6 antes + 6 depois (13 números/zona)")
        st.write("- **Threshold**: Mínimo 6 números na mesma zona")
        st.write("- **Saída**: Zona com maior concentração")
        st.write("- **Telegram**: Alertas automáticos de entrada")
        
        # Mostrar zonas do ML
        info_zonas_ml = st.session_state.sistema.estrategia_ml.get_info_zonas_ml()
        for zona, dados in info_zonas_ml.items():
            st.write(f"**Zona {zona}** (Núcleo: {dados['central']})")
            st.write(f"Descrição: {dados['descricao']}")
            st.write(f"Números: {', '.join(map(str, dados['numeros']))}")
            st.write(f"Total: {dados['quantidade']} números")
            st.write("---")

# Análise detalhada baseada na estratégia selecionada
with st.sidebar.expander(f"🔍 Análise - {estrategia}", expanded=False):
    if estrategia == "Zonas":
        analise = st.session_state.sistema.estrategia_zonas.get_analise_detalhada()
    elif estrategia == "ML":
        analise = st.session_state.sistema.estrategia_ml.get_analise_ml()
    else:  # Midas
        analise = "🎯 Estratégia Midas ativa\nAnalisando padrões de terminais..."
    
    st.text(analise)

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
        enviar_resultado(f"📝 {len(nums)} números adicionados manualmente")
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
        enviar_resultado(f"🔄 Novo número da API: {numero_atual}")

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
    
    # Mostrar previsão ML se disponível (especialmente para estratégia ML)
    if 'previsao_ml' in previsao and previsao['previsao_ml']:
        st.write("**🤖 Probabilidades ML (Top 5):**")
        for num, prob in previsao['previsao_ml'][:5]:
            st.write(f"  {num}: {prob:.2%}")
    
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
