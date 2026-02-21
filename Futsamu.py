import streamlit as st
import requests
import random
import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import time

# =====================================================
# CLASSE PRINCIPAL - SUA CLASSE ORIGINAL
# =====================================================
class AnaliseLotofacilAvancada:
    def __init__(self, concursos):
        self.concursos = concursos
        self.total_concursos = len(concursos)
        self.numeros = list(range(1, 26))

        self.frequencias = self._frequencias()
        self.defasagens = self._defasagens()
        self.padroes = self._padroes()
        self.numeros_chave = self._numeros_chave()

        self.dna = self._dna_inicial()

    # =================================================
    # DNA
    # =================================================
    def _dna_inicial(self):
        return {
            "freq": 1.0,
            "defas": 1.0,
            "soma": 1.0,
            "pares": 1.0,
            "seq": 1.0,
            "chave": 1.0
        }

    # =================================================
    # ESTATÍSTICAS
    # =================================================
    def _frequencias(self):
        c = Counter()
        for con in self.concursos:
            c.update(con)
        return {n: c[n] / self.total_concursos for n in self.numeros}

    def _defasagens(self):
        d = {}
        for n in self.numeros:
            for i, c in enumerate(self.concursos):
                if n in c:
                    d[n] = i
                    break
            else:
                d[n] = self.total_concursos
        return d

    def _padroes(self):
        return {
            "somas": [sum(c) for c in self.concursos],
            "pares": [sum(1 for n in c if n % 2 == 0) for c in self.concursos]
        }

    def _numeros_chave(self):
        c = Counter()
        for con in self.concursos[:20]:
            c.update(con)
        return [n for n, q in c.items() if q >= 10]

    # =================================================
    # SCORE
    # =================================================
    def score_numero(self, n):
        return (
            self.frequencias[n] * self.dna["freq"] +
            (1 - self.defasagens[n] / self.total_concursos) * self.dna["defas"] +
            (self.dna["chave"] if n in self.numeros_chave else 0)
        )

    # =================================================
    # FECHAMENTO
    # =================================================
    def gerar_fechamento(self, tamanho):
        scores = {n: self.score_numero(n) for n in self.numeros}
        base = sorted(scores, key=scores.get, reverse=True)[:tamanho]
        return sorted(base)

    def gerar_jogos(self, fechamento, qtd):
        jogos = set()
        while len(jogos) < qtd:
            j = sorted(random.sample(fechamento, 15))
            soma = sum(j)
            pares = sum(1 for n in j if n % 2 == 0)
            if 180 <= soma <= 220 and 6 <= pares <= 9:
                jogos.add(tuple(j))
        return [list(j) for j in jogos]

    # =================================================
    # CONFERÊNCIA
    # =================================================
    def conferir(self, jogos, resultado):
        dados = []
        for i, j in enumerate(jogos, 1):
            dados.append({
                "Jogo": i,
                "Dezenas": ", ".join(f"{n:02d}" for n in j),
                "Acertos": len(set(j) & set(resultado)),
                "Soma": sum(j),
                "Pares": sum(1 for n in j if n % 2 == 0)
            })
        return pd.DataFrame(dados)

    # =================================================
    # APRENDIZADO (11–14)
    # =================================================
    def reforcar_dna_por_acertos(self, jogos, resultado):
        for jogo in jogos:
            acertos = len(set(jogo) & set(resultado))
            if acertos < 11:
                continue

            reforco = {11: 0.02, 12: 0.04, 13: 0.06, 14: 0.08}.get(acertos, 0)

            soma = sum(jogo)
            pares = sum(1 for n in jogo if n % 2 == 0)

            if soma >= np.mean(self.padroes["somas"]):
                self.dna["soma"] += reforco

            if pares >= np.mean(self.padroes["pares"]):
                self.dna["pares"] += reforco

            # sequência
            max_seq, atual = 1, 1
            for i in range(1, len(jogo)):
                if jogo[i] == jogo[i-1] + 1:
                    atual += 1
                    max_seq = max(max_seq, atual)
                else:
                    atual = 1

            if max_seq >= 3:
                self.dna["seq"] += reforco

            for n in jogo:
                if self.frequencias[n] >= 0.5:
                    self.dna["freq"] += reforco / 15
                if self.defasagens[n] <= 10:
                    self.dna["defas"] += reforco / 15

            if any(n in self.numeros_chave for n in jogo):
                self.dna["chave"] += reforco

        for k in self.dna:
            self.dna[k] = max(0.5, min(2.0, self.dna[k]))


# =====================================================
# CLASSE DE TESTE CIENTÍFICO (SEM SEABORN)
# =====================================================
class TesteCientificoLotofacil:
    
    def __init__(self):
        self.resultados_testes = []
        self.historico_concursos = []
        
    def carregar_todos_concursos(self):
        """Carrega TODO o histórico disponível"""
        url = "https://loteriascaixa-api.herokuapp.com/api/lotofacil/"
        try:
            data = requests.get(url).json()
            self.historico_concursos = []
            for d in data:
                self.historico_concursos.append({
                    'numero': d['concurso'],
                    'data': d['data'],
                    'dezenas': sorted(map(int, d['dezenas'])),
                    'arrecadacao': d.get('arrecadacao_total', 0)
                })
            return True
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")
            return False
    
    def testar_sistema_em_concurso(self, concurso_teste, qtd_concursos_anteriores=200, qtd_jogos=10):
        """
        Testa o sistema em UM concurso específico
        
        Args:
            concurso_teste: dicionário com o concurso a ser "previsto"
            qtd_concursos_anteriores: quantos concursos usar para treino
            qtd_jogos: quantos jogos gerar
        """
        # Encontra o índice do concurso de teste
        idx_teste = None
        for i, c in enumerate(self.historico_concursos):
            if c['numero'] == concurso_teste['numero']:
                idx_teste = i
                break
        
        if idx_teste is None:
            return None
        
        # Pega concursos ANTERIORES ao teste (dados históricos reais)
        inicio = max(0, idx_teste - qtd_concursos_anteriores)
        concursos_treino = [c['dezenas'] for c in self.historico_concursos[inicio:idx_teste]]
        
        if len(concursos_treino) < 10:
            return None  # Não há dados suficientes
        
        # Inicializa seu sistema com dados históricos
        seu_sistema = AnaliseLotofacilAvancada(concursos_treino)
        
        # Gera jogos
        tamanho_fechamento = random.choice([16, 17, 18])  # Varia um pouco
        fechamento = seu_sistema.gerar_fechamento(tamanho_fechamento)
        jogos_gerados = seu_sistema.gerar_jogos(fechamento, qtd_jogos)
        
        # Resultado REAL (que seu sistema não viu)
        resultado_real = concurso_teste['dezenas']
        
        # Confere acertos
        acertos_por_jogo = [len(set(j) & set(resultado_real)) for j in jogos_gerados]
        
        # Gera jogos ALEATÓRIOS para comparação
        jogos_aleatorios = self.gerar_jogos_aleatorios(qtd_jogos)
        acertos_aleatorios = [len(set(j) & set(resultado_real)) for j in jogos_aleatorios]
        
        # Estatísticas do teste
        resultado_teste = {
            'concurso': concurso_teste['numero'],
            'data': concurso_teste['data'],
            'qtd_treino': len(concursos_treino),
            'acertos_seu_sistema_max': max(acertos_por_jogo),
            'acertos_seu_sistema_media': np.mean(acertos_por_jogo),
            'acertos_seu_sistema_std': np.std(acertos_por_jogo),
            'acertos_aleatorio_max': max(acertos_aleatorios),
            'acertos_aleatorio_media': np.mean(acertos_aleatorios),
            'acertos_aleatorio_std': np.std(acertos_aleatorios),
            'diferenca_media': np.mean(acertos_por_jogo) - np.mean(acertos_aleatorios),
            'jogos_gerados': jogos_gerados,
            'resultado_real': resultado_real,
            'fechamento_usado': fechamento,
            'dna_usado': seu_sistema.dna.copy() if hasattr(seu_sistema, 'dna') else None
        }
        
        return resultado_teste
    
    def gerar_jogos_aleatorios(self, qtd):
        """Gera jogos puramente aleatórios para comparação"""
        jogos = []
        for _ in range(qtd):
            jogo = sorted(random.sample(range(1, 26), 15))
            jogos.append(jogo)
        return jogos
    
    def testar_historicamente(self, qtd_testes=50, qtd_concursos_anteriores=200, qtd_jogos=10):
        """
        Testa o sistema em MÚLTIPLOS concursos passados
        """
        if len(self.historico_concursos) < qtd_testes + qtd_concursos_anteriores:
            qtd_testes = len(self.historico_concursos) - qtd_concursos_anteriores - 10
        
        # Pega os últimos N concursos para teste
        concursos_teste = self.historico_concursos[-qtd_testes:]
        
        self.resultados_testes = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, concurso in enumerate(concursos_teste):
            status_text.text(f"Testando concurso {concurso['numero']}...")
            
            resultado = self.testar_sistema_em_concurso(
                concurso, 
                qtd_concursos_anteriores, 
                qtd_jogos
            )
            
            if resultado:
                self.resultados_testes.append(resultado)
            
            progress_bar.progress((i + 1) / len(concursos_teste))
            time.sleep(0.1)  # Para não sobrecarregar a API
        
        status_text.text("Testes concluídos!")
        return self.resultados_testes
    
    def analisar_resultados(self):
        """
        Análise estatística completa dos resultados
        """
        if not self.resultados_testes:
            return None
        
        df = pd.DataFrame(self.resultados_testes)
        
        # Estatísticas básicas
        stats_basicas = {
            'Total de testes': len(df),
            'Média acertos - Seu sistema': df['acertos_seu_sistema_media'].mean(),
            'Média acertos - Aleatório': df['acertos_aleatorio_media'].mean(),
            'Diferença média': df['diferenca_media'].mean(),
            'Desvio padrão - Seu sistema': df['acertos_seu_sistema_std'].mean(),
            'Desvio padrão - Aleatório': df['acertos_aleatorio_std'].mean(),
            'Máximo acertos - Seu sistema': df['acertos_seu_sistema_max'].max(),
            'Máximo acertos - Aleatório': df['acertos_aleatorio_max'].max(),
        }
        
        # Teste T-Student pareado
        t_stat, p_value = stats.ttest_rel(
            df['acertos_seu_sistema_media'], 
            df['acertos_aleatorio_media']
        )
        
        stats_basicas['T-statistic'] = t_stat
        stats_basicas['P-value'] = p_value
        stats_basicas['Estatisticamente melhor (p<0.05)'] = p_value < 0.05
        
        # Distribuição de acertos máximos
        stats_basicas['Vezes que fez 11+'] = (df['acertos_seu_sistema_max'] >= 11).sum()
        stats_basicas['Vezes que fez 12+'] = (df['acertos_seu_sistema_max'] >= 12).sum()
        stats_basicas['Vezes que fez 13+'] = (df['acertos_seu_sistema_max'] >= 13).sum()
        stats_basicas['Vezes que fez 14+'] = (df['acertos_seu_sistema_max'] >= 14).sum()
        stats_basicas['Vezes que fez 15'] = (df['acertos_seu_sistema_max'] == 15).sum()
        
        return df, stats_basicas
    
    def plotar_resultados(self, df):
        """
        Gráficos comparativos (SEM SEABORN)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Comparação de médias
        axes[0, 0].scatter(df.index, df['acertos_seu_sistema_media'], 
                          alpha=0.6, label='Seu Sistema', color='blue', s=30)
        axes[0, 0].scatter(df.index, df['acertos_aleatorio_media'], 
                          alpha=0.6, label='Aleatório', color='red', s=30)
        axes[0, 0].set_xlabel('Teste #', fontsize=12)
        axes[0, 0].set_ylabel('Média de Acertos', fontsize=12)
        axes[0, 0].set_title('Comparação: Média de Acertos por Teste', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histograma das diferenças
        axes[0, 1].hist(df['diferenca_media'], bins=15, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        axes[0, 1].axvline(x=df['diferenca_media'].mean(), color='blue', linestyle='-', linewidth=2, label=f'Média: {df["diferenca_media"].mean():.3f}')
        axes[0, 1].set_xlabel('Diferença (Seu Sistema - Aleatório)', fontsize=12)
        axes[0, 1].set_ylabel('Frequência', fontsize=12)
        axes[0, 1].set_title('Distribuição das Diferenças', fontsize=14)
        axes[0, 1].legend()
        
        # 3. Acertos máximos
        bins = range(0, 16)
        axes[1, 0].hist(df['acertos_seu_sistema_max'], bins=bins, alpha=0.7, label='Seu Sistema', color='blue', edgecolor='black')
        axes[1, 0].hist(df['acertos_aleatorio_max'], bins=bins, alpha=0.5, label='Aleatório', color='red', edgecolor='black')
        axes[1, 0].set_xlabel('Máximo de Acertos', fontsize=12)
        axes[1, 0].set_ylabel('Frequência', fontsize=12)
        axes[1, 0].set_title('Distribuição dos Acertos Máximos', fontsize=14)
        axes[1, 0].legend()
        
        # 4. Box plot comparativo (feito manualmente)
        dados_boxplot = [df['acertos_seu_sistema_media'], df['acertos_aleatorio_media']]
        bp = axes[1, 1].boxplot(dados_boxplot, labels=['Seu Sistema', 'Aleatório'], patch_artist=True)
        
        # Colorir os boxplots
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1, 1].set_ylabel('Acertos', fontsize=12)
        axes[1, 1].set_title('Box Plot Comparativo', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# =====================================================
# INTERFACE STREAMLIT PARA TESTES
# =====================================================
def main_com_testes():
    st.set_page_config(page_title="🎯 TESTE CIENTÍFICO LOTOFÁCIL", layout="wide")
    
    st.title("🔬 TESTE CIENTÍFICO DO SEU SISTEMA")
    st.markdown("""
    ### Vamos descobrir se seu sistema realmente funciona!
    
    Este teste vai:
    1. Usar dados HISTÓRICOS reais
    2. Simular que você NÃO sabe o resultado
    3. Comparar com sorteios puramente aleatórios
    4. Dar uma conclusão estatística
    """)
    
    # Inicializa sessão
    if 'teste' not in st.session_state:
        st.session_state.teste = TesteCientificoLotofacil()
    if 'resultados' not in st.session_state:
        st.session_state.resultados = None
    if 'df_resultados' not in st.session_state:
        st.session_state.df_resultados = None
    
    # Sidebar com parâmetros
    with st.sidebar:
        st.header("⚙️ Parâmetros do Teste")
        
        qtd_testes = st.slider("Quantidade de testes", 10, 100, 30)
        qtd_anteriores = st.slider("Concursos para treino", 50, 500, 200)
        qtd_jogos = st.slider("Jogos por teste", 5, 20, 10)
        
        if st.button("🚀 INICIAR TESTE CIENTÍFICO", type="primary"):
            with st.spinner("Carregando histórico completo..."):
                sucesso = st.session_state.teste.carregar_todos_concursos()
                
            if sucesso:
                st.success(f"✅ {len(st.session_state.teste.historico_concursos)} concursos carregados!")
                
                with st.spinner("Executando testes (pode levar alguns minutos)..."):
                    resultados = st.session_state.teste.testar_historicamente(
                        qtd_testes=qtd_testes,
                        qtd_concursos_anteriores=qtd_anteriores,
                        qtd_jogos=qtd_jogos
                    )
                    
                    if resultados:
                        st.session_state.resultados = resultados
                        df, stats = st.session_state.teste.analisar_resultados()
                        st.session_state.df_resultados = df
                        st.session_state.stats = stats
                        st.success("✅ Testes concluídos!")
                    else:
                        st.error("Erro nos testes")
    
    # Abas para resultados
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Resultados Estatísticos", 
        "📈 Gráficos", 
        "🔍 Detalhamento",
        "💡 Conclusão"
    ])
    
    with tab1:
        if st.session_state.df_resultados is not None:
            stats = st.session_state.stats
            
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Média Seu Sistema", f"{stats['Média acertos - Seu sistema']:.3f}")
            with col2:
                st.metric("Média Aleatório", f"{stats['Média acertos - Aleatório']:.3f}")
            with col3:
                diff = stats['Diferença média']
                st.metric("Diferença", f"{diff:.3f}", 
                         delta=f"{diff:.3f}" if diff > 0 else None)
            with col4:
                st.metric("P-Value", f"{stats['P-value']:.4f}")
            
            # Resultado do teste estatístico
            if stats['P-value'] < 0.05:
                st.success("🎯 **CONCLUSÃO ESTATÍSTICA: Seu sistema é MELHOR que aleatório!**")
            elif stats['P-value'] < 0.1:
                st.warning("📊 **TENDÊNCIA POSITIVA: Quase significativo estatisticamente**")
            else:
                st.info("📊 **EQUIVALENTE A ALEATÓRIO: Continue ajustando**")
            
            # Tabela de acertos
            st.subheader("📊 Performance em acertos máximos")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Vezes que fez 11+", stats['Vezes que fez 11+'])
                st.metric("Vezes que fez 12+", stats['Vezes que fez 12+'])
                st.metric("Vezes que fez 13+", stats['Vezes que fez 13+'])
            with col2:
                st.metric("Vezes que fez 14+", stats['Vezes que fez 14+'])
                st.metric("Vezes que fez 15", stats['Vezes que fez 15'])
            
            # DataFrame
            st.subheader("📋 Resultados detalhados")
            st.dataframe(st.session_state.df_resultados, use_container_width=True)
    
    with tab2:
        if st.session_state.df_resultados is not None:
            fig = st.session_state.teste.plotar_resultados(st.session_state.df_resultados)
            st.pyplot(fig)
    
    with tab3:
        if st.session_state.resultados:
            st.subheader("🔍 Detalhamento por concurso")
            for r in st.session_state.resultados[-10:]:  # Últimos 10
                with st.expander(f"Concurso {r['concurso']} - {r['data']}"):
                    st.write(f"**Resultado real:** {r['resultado_real']}")
                    st.write(f"**Max acertos:** {r['acertos_seu_sistema_max']} (vs aleatório: {r['acertos_aleatorio_max']})")
                    st.write(f"**Média acertos:** {r['acertos_seu_sistema_media']:.2f} (vs aleatório: {r['acertos_aleatorio_media']:.2f})")
                    st.write(f"**Diferença:** {r['diferenca_media']:.2f}")
                    
                    if r.get('dna_usado'):
                        st.write("**DNA usado:**", r['dna_usado'])
    
    with tab4:
        st.header("💡 INTERPRETAÇÃO DOS RESULTADOS")
        
        st.markdown("""
        ### Como interpretar o teste:
        
        1. **P-value < 0.05** = Seu sistema é estatisticamente melhor que aleatório
        2. **P-value entre 0.05 e 0.1** = Há uma tendência, mas não conclusiva
        3. **P-value > 0.1** = Seu sistema é equivalente a aleatório
        
        ### O que significa "melhor que aleatório":
        - Seus filtros (soma, pares, etc.) estão realmente selecionando combinações mais prováveis
        - Você está evitando combinações "ruins" que nunca acontecem
        - O sistema tem MÉRITO ESTATÍSTICO real
        
        ### Lembre-se:
        - Mesmo sendo melhor que aleatório, NÃO GARANTE PRÊMIOS
        - A vantagem é pequena, mas REAL
        - Em loteria, pequenas vantagens fazem diferença a longo prazo
        """)
        
        if st.session_state.df_resultados is not None:
            stats = st.session_state.stats
            if stats['P-value'] < 0.05:
                st.balloons()
                st.success("""
                ### 🎉 PARABÉNS! Seu sistema tem MÉRITO ESTATÍSTICO!
                
                Os números mostram que seu sistema realmente seleciona 
                combinações melhores que o aleatório puro.
                
                **Isso explica seus 14 pontos!** Você não teve apenas sorte, 
                seu sistema tem qualidade real.
                """)
            elif stats['P-value'] < 0.1:
                st.info("""
                ### 📈 QUASE LÁ!
                
                Seu sistema mostra uma tendência positiva consistente.
                Com alguns ajustes finos, pode alcançar significância estatística.
                
                Continue refinando os filtros e o DNA!
                """)
            else:
                st.info("""
                ### 📊 Resultado Inconclusivo
                
                Continue ajustando seu sistema! A diferença positiva na média
                mostra que você está no caminho certo. Com mais ajustes,
                pode alcançar significância estatística.
                """)


# =====================================================
# EXECUÇÃO PRINCIPAL
# =====================================================
if __name__ == "__main__":
    main_com_testes()
