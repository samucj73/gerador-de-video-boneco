import streamlit as st
import json
import os
import requests
import logging
from collections import Counter, deque
from alertas import enviar_previsao, enviar_resultado
from streamlit_autorefresh import st_autorefresh

# =============================
# CONFIGURAÇÕES GLOBAIS
# =============================
HISTORICO_PATH = "historico_coluna_duzia.json"
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =============================
# CLASSES DE DOMÍNIO
# =============================

class Configuracao:
    """Classe para gerenciar configurações do sistema"""
    def __init__(self):
        self.historico_path = HISTORICO_PATH
        self.api_url = API_URL
        self.headers = HEADERS
        self.janela_terminais = 12
        self.max_numeros_aposta = 8

class HistoricoManager:
    """Classe para gerenciar o histórico de resultados"""
    
    def __init__(self, caminho_arquivo):
        self.caminho_arquivo = caminho_arquivo
        self.dados = self._carregar_historico()
    
    def _carregar_historico(self):
        """Carrega o histórico do arquivo JSON"""
        try:
            if os.path.exists(self.caminho_arquivo):
                with open(self.caminho_arquivo, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logging.error(f"Erro ao carregar histórico: {e}")
            return []
    
    def adicionar_resultado(self, numero, timestamp):
        """Adiciona um novo resultado ao histórico"""
        resultado = {"number": numero, "timestamp": timestamp}
        self.dados.append(resultado)
        self._salvar_historico()
    
    def _salvar_historico(self):
        """Salva o histórico no arquivo JSON"""
        try:
            with open(self.caminho_arquivo, 'w') as f:
                json.dump(self.dados, f, indent=2)
        except Exception as e:
            logging.error(f"Erro ao salvar histórico: {e}")
    
    def get_ultimos_numeros(self, quantidade=10):
        """Retorna os últimos números do histórico"""
        return [h["number"] for h in self.dados[-quantidade:]]
    
    def get_timestamp_mais_recente(self):
        """Retorna o timestamp do resultado mais recente"""
        if self.dados:
            return self.dados[-1]["timestamp"]
        return None

class APIClient:
    """Classe para comunicação com a API de resultados"""
    
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
    
    def buscar_ultimo_resultado(self):
        """Busca o último resultado da API"""
        try:
            response = requests.get(self.url, headers=self.headers, timeout=5)
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

class RoletaFisica:
    """Classe que representa a roleta física e suas propriedades"""
    
    def __init__(self):
        self.numeros_roleta = list(range(37))  # 0-36
        self.race = [
            0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27,
            13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33,
            1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12,
            35, 3, 26
        ]
        self.terminais = self._criar_terminais()
        self.vizinhos_race = self._criar_vizinhos_race()
    
    def _criar_terminais(self):
        """Cria o mapeamento de terminais"""
        return {
            '0': [0, 10, 20, 30],
            '1': [1, 11, 21, 31],
            '2': [2, 12, 22, 32],
            '3': [3, 13, 23, 33],
            '4': [4, 14, 24, 34],
            '5': [5, 15, 25, 35],
            '6': [6, 16, 26, 36],
            '7': [7, 17, 27],
            '8': [8, 18, 28],
            '9': [9, 19, 29]
        }
    
    def _criar_vizinhos_race(self):
        """Cria o mapeamento de vizinhos no race"""
        return {
            '33': [1, 20, 36, 11, 30, 4, 21, 2, 14, 31, 9],
            '21': [2, 25, 28, 12, 35, 9, 22, 18, 0, 32, 15],
            '35': [3, 26, 27, 13, 36, 8, 23, 10, 16, 33, 1],
            '19': [4, 21, 20, 14, 31, 5, 24, 16, 17, 34, 6],
            '10': [5, 24, 32, 15, 19, 2, 25, 17, 12, 35, 3],
            '34': [6, 27, 24, 16, 33, 3, 26, 0, 13, 36, 11],
            '29': [7, 28, 25, 17, 34, 6, 27, 13, 26, 0, 32],
            '30': [8, 23, 22, 18, 29, 7, 28, 12],
            '31': [9, 22, 15, 19, 4, 18, 29, 7],
            '23': [10, 5, 1, 20, 14, 11, 30, 8]
        }
    
    def extrair_terminal(self, numero):
        """Extrai o terminal de um número"""
        return numero % 10
    
    def get_vizinhos_race(self, numero):
        """Retorna os vizinhos de um número no race"""
        return self.vizinhos_race.get(str(numero), [])
    
    def get_posicao_race(self, numero):
        """Retorna a posição de um número no race"""
        if numero in self.race:
            return self.race.index(numero)
        return -1
    
    def get_vizinhos_fisicos(self, numero, quantidade_vizinhos=2):
        """Retorna vizinhos físicos no race"""
        posicao = self.get_posicao_race(numero)
        if posicao == -1:
            return []
        
        vizinhos = []
        for offset in range(-quantidade_vizinhos, quantidade_vizinhos + 1):
            if offset == 0:
                continue
            vizinho = self.race[(posicao + offset) % len(self.race)]
            vizinhos.append(vizinho)
        
        return vizinhos

# =============================
# ESTRATÉGIAS
# =============================

class EstrategiaBase:
    """Classe base para todas as estratégias"""
    
    def __init__(self, nome, roleta_fisica):
        self.nome = nome
        self.roleta = roleta_fisica
        self.historico = deque(maxlen=20)
    
    def adicionar_numero(self, numero):
        """Adiciona um número ao histórico da estratégia"""
        self.historico.append(numero)
    
    def verificar_gatilho(self, ultimo_numero):
        """Método abstrato para verificar gatilho - deve ser implementado pelas subclasses"""
        raise NotImplementedError("Método verificar_gatilho deve ser implementado pela subclass")
    
    def _filtrar_numeros_recentes(self, numeros_candidatos):
        """Filtra números que saíram recentemente"""
        numeros_recentes = set(list(self.historico)[-5:])
        return [n for n in numeros_candidatos if n not in numeros_recentes]

class EstrategiaMidas(EstrategiaBase):
    """Implementa as estratégias Midas do PDF"""
    
    def __init__(self, roleta_fisica):
        super().__init__("Estratégias Midas", roleta_fisica)
        self.estrategias = {
            'zero': {'terminal': [0, 10, 20, 30], 'nome': 'Padrão do Zero', 'min_aparicoes': 2},
            'sete': {'terminal': [7, 17, 27], 'nome': 'Padrão do Sete', 'min_aparicoes': 2},
            'cinco': {'terminal': [5, 15, 25, 35], 'nome': 'Padrão do Cinco', 'min_aparicoes': 2},
            'gemeos': {'terminal': [11, 22, 33], 'nome': 'Padrão Gêmeos', 'min_aparicoes': 2}
        }
    
    def verificar_gatilho(self, ultimo_numero):
        """Verifica se algum gatilho Midas foi ativado"""
        for key, estrategia in self.estrategias.items():
            if ultimo_numero in estrategia['terminal']:
                # Verifica se apareceu o mínimo de vezes necessário
                count_aparicoes = sum(1 for n in self.historico[-8:] if n in estrategia['terminal'])
                
                if count_aparicoes >= estrategia['min_aparicoes']:
                    numeros_apostar = self._gerar_numeros_aposta(estrategia['terminal'])
                    return {
                        'nome': estrategia['nome'],
                        'numeros_apostar': numeros_apostar,
                        'gatilho': f'{estrategia["nome"]} ativado ({count_aparicoes}x)'
                    }
        return None
    
    def _gerar_numeros_aposta(self, terminal):
        """Gera números para apostar baseado no terminal"""
        numeros_apostar = terminal.copy()
        for num in terminal:
            numeros_apostar.extend(self.roleta.get_vizinhos_race(num))
        return list(set(numeros_apostar))[:8]

class EstrategiaTerminaisDominantes(EstrategiaBase):
    """Implementa a estratégia de Terminais Dominantes"""
    
    def __init__(self, roleta_fisica, janela=12):
        super().__init__("Terminais Dominantes", roleta_fisica)
        self.janela = janela
        self.historico = deque(maxlen=janela + 1)
    
    def adicionar_numero(self, numero):
        """Adiciona número mantendo o tamanho da janela"""
        self.historico.append(numero)
    
    def verificar_gatilho(self, ultimo_numero):
        """Verifica gatilho para Terminais Dominantes"""
        if len(self.historico) < self.janela + 1:
            return None
        
        ultimos_12 = list(self.historico)[:-1]
        numero_13 = self.historico[-1]
        dominantes = self._calcular_terminais_dominantes()
        terminal_13 = self.roleta.extrair_terminal(numero_13)
        
        condicao_a = numero_13 in ultimos_12
        condicao_b = terminal_13 in [self.roleta.extrair_terminal(n) for n in ultimos_12]
        
        if condicao_a or condicao_b:
            numeros_apostar = self._selecionar_numeros_estrategicos(dominantes, numero_13)
            return {
                'nome': self.nome,
                'numeros_apostar': numeros_apostar,
                'gatilho': f"Critério {'A' if condicao_a else 'B'} - Número {numero_13}"
            }
        return None
    
    def _calcular_terminais_dominantes(self):
        """Calcula os terminais dominantes nos últimos números"""
        if len(self.historico) < self.janela:
            return []
        
        ultimos_12 = list(self.historico)[-self.janela:-1] if len(self.historico) >= self.janela + 1 else list(self.historico)[:-1]
        terminais = [self.roleta.extrair_terminal(n) for n in ultimos_12]
        contagem = Counter(terminais)
        return [t for t, _ in contagem.most_common(2)]
    
    def _selecionar_numeros_estrategicos(self, terminais_dominantes, numero_gatilho):
        """Seleciona números estratégicos para apostar"""
        numeros_candidatos = set()
        
        # Adicionar números dos terminais dominantes
        for terminal in terminais_dominantes[:2]:
            base = [n for n in range(37) if self.roleta.extrair_terminal(n) == terminal]
            numeros_frios = self._filtrar_numeros_recentes(base)
            numeros_candidatos.update(numeros_frios[:3] if numeros_frios else base[:3])
        
        # Adicionar vizinhos do número gatilho
        vizinhos = self.roleta.get_vizinhos_fisicos(numero_gatilho, 2)
        numeros_candidatos.update(vizinhos)
        
        return list(numeros_candidatos)[:8]

# =============================
# GESTÃO DE ESTRATÉGIAS
# =============================

class GestorEstrategias:
    """Gerencia todas as estratégias e previsões"""
    
    def __init__(self, config):
        self.config = config
        self.roleta = RoletaFisica()
        self.estrategias = [
            EstrategiaTerminaisDominantes(self.roleta, config.janela_terminais),
            EstrategiaMidas(self.roleta)
        ]
        self.previsao_ativa = None
        self.desempenho = DesempenhoManager()
    
    def processar_novo_numero(self, numero):
        """Processa um novo número sorteado"""
        # Atualiza todas as estratégias
        for estrategia in self.estrategias:
            estrategia.adicionar_numero(numero)
        
        # Conferir previsão anterior se existir
        resultado_conferencia = self._conferir_previsao_anterior(numero)
        
        # Verificar nova estratégia
        nova_estrategia = self._verificar_estrategias_ativas()
        if nova_estrategia:
            self._definir_nova_previsao(nova_estrategia)
        
        return resultado_conferencia
    
    def _verificar_estrategias_ativas(self):
        """Verifica se alguma estratégia foi ativada"""
        if self.previsao_ativa:
            return None  # Já existe previsão ativa
        
        historico_recente = []
        for estrategia in self.estrategias:
            if estrategia.historico:
                historico_recente.extend(list(estrategia.historico)[-5:])
        
        ultimo_numero = historico_recente[-1] if historico_recente else None
        
        if ultimo_numero is None:
            return None
        
        # Prioridade: Terminais Dominantes primeiro
        for estrategia in self.estrategias:
            gatilho = estrategia.verificar_gatilho(ultimo_numero)
            if gatilho:
                return gatilho
        
        return None
    
    def _definir_nova_previsao(self, estrategia):
        """Define uma nova previsão ativa"""
        self.previsao_ativa = {
            'estrategia': estrategia['nome'],
            'numeros_apostar': estrategia['numeros_apostar'],
            'gatilho': estrategia['gatilho'],
            'timestamp': len(self.estrategias[0].historico)  # Usa histórico da primeira estratégia como referência
        }
        
        # Enviar alerta
        msg = f"🎯 {estrategia['nome']}\n"
        msg += f"Gatilho: {estrategia['gatilho']}\n"
        msg += f"Números: {', '.join(map(str, sorted(estrategia['numeros_apostar'])))}"
        enviar_previsao(msg)
    
    def _conferir_previsao_anterior(self, numero_sorteado):
        """Conferir se a previsão anterior acertou"""
        if not self.previsao_ativa:
            return None
        
        acerto = numero_sorteado in self.previsao_ativa['numeros_apostar']
        resultado = {
            'acerto': acerto,
            'numero_sorteado': numero_sorteado,
            'estrategia': self.previsao_ativa['estrategia'],
            'previsao': self.previsao_ativa['numeros_apostar']
        }
        
        # Atualizar desempenho
        self.desempenho.registrar_resultado(resultado)
        
        # Limpar previsão
        self.previsao_ativa = None
        
        return resultado

class DesempenhoManager:
    """Gerencia o desempenho das estratégias"""
    
    def __init__(self):
        self.acertos = 0
        self.erros = 0
        self.historico = []
        self.estatisticas_estrategias = {}
    
    def registrar_resultado(self, resultado):
        """Registra um resultado no desempenho"""
        if resultado['acerto']:
            self.acertos += 1
        else:
            self.erros += 1
        
        self.historico.append(resultado)
        
        # Atualizar estatísticas por estratégia
        estrategia = resultado['estrategia']
        if estrategia not in self.estatisticas_estrategias:
            self.estatisticas_estrategias[estrategia] = {'acertos': 0, 'tentativas': 0}
        
        self.estatisticas_estrategias[estrategia]['tentativas'] += 1
        if resultado['acerto']:
            self.estatisticas_estrategias[estrategia]['acertos'] += 1
    
    def get_taxa_acerto(self):
        """Retorna a taxa de acerto geral"""
        total = self.acertos + self.erros
        return (self.acertos / total * 100) if total > 0 else 0.0
    
    def get_taxa_acerto_estrategia(self, estrategia):
        """Retorna a taxa de acerto de uma estratégia específica"""
        if estrategia not in self.estatisticas_estrategias:
            return 0.0
        
        dados = self.estatisticas_estrategias[estrategia]
        if dados['tentativas'] == 0:
            return 0.0
        
        return (dados['acertos'] / dados['tentativas']) * 100

# =============================
# INTERFACE DO USUÁRIO
# =============================

class InterfaceUsuario:
    """Classe para gerenciar a interface do Streamlit"""
    
    def __init__(self, gestor, historico_manager, api_client):
        self.gestor = gestor
        self.historico_manager = historico_manager
        self.api_client = api_client
        self._inicializar_session_state()
    
    def _inicializar_session_state(self):
        """Inicializa o estado da sessão"""
        if 'ultima_conferencia' not in st.session_state:
            st.session_state.ultima_conferencia = None
    
    def exibir_interface(self):
        """Exibe a interface principal"""
        st.set_page_config(page_title="IA Roleta — Sistema POO", layout="centered")
        st.title("🎯 IA Roleta — Sistema com Programação Orientada a Objetos")
        
        self._exibir_entrada_manual()
        self._processar_atualizacao_automatica()
        self._exibir_ultimos_numeros()
        self._exibir_previsao_ativa()
        self._exibir_ultima_conferencia()
        self._exibir_desempenho()
        self._exibir_informacoes_estrategias()
        self._exibir_download_historico()
    
    def _exibir_entrada_manual(self):
        """Exibe a seção de entrada manual"""
        st.subheader("✍️ Inserir Sorteios Manualmente")
        entrada = st.text_input("Digite números (0-36) separados por espaço:")
        if st.button("Adicionar") and entrada:
            self._processar_entrada_manual(entrada)
    
    def _processar_entrada_manual(self, entrada):
        """Processa a entrada manual de números"""
        try:
            nums = [int(n) for n in entrada.split() if n.isdigit() and 0 <= int(n) <= 36]
            for n in nums:
                self._processar_novo_numero(n, f"manual_{len(self.historico_manager.dados)}")
            st.success(f"{len(nums)} números adicionados!")
            st.rerun()
        except Exception as e:
            st.error(f"Erro: {e}")
    
    def _processar_atualizacao_automatica(self):
        """Processa a atualização automática via API"""
        st_autorefresh(interval=3000, key="refresh")
        
        resultado = self.api_client.buscar_ultimo_resultado()
        ultimo_ts = self.historico_manager.get_timestamp_mais_recente()
        
        if resultado and resultado.get("timestamp") and resultado["timestamp"] != ultimo_ts:
            numero_atual = resultado.get("number")
            if numero_atual is not None:
                self._processar_novo_numero(numero_atual, resultado["timestamp"])
    
    def _processar_novo_numero(self, numero, timestamp):
        """Processa um novo número (manual ou automático)"""
        # Adicionar ao histórico
        self.historico_manager.adicionar_resultado(numero, timestamp)
        
        # Processar no gestor de estratégias
        resultado_conferencia = self.gestor.processar_novo_numero(numero)
        
        # Atualizar interface se houve conferência
        if resultado_conferencia:
            st.session_state.ultima_conferencia = resultado_conferencia
    
    def _exibir_ultimos_numeros(self):
        """Exibe os últimos números sorteados"""
        st.subheader("🔁 Últimos Números")
        ultimos_numeros = self.historico_manager.get_ultimos_numeros(10)
        if ultimos_numeros:
            st.write(" ".join(map(str, ultimos_numeros)))
        else:
            st.write("Nenhum número registrado")
    
    def _exibir_previsao_ativa(self):
        """Exibe a previsão ativa"""
        st.subheader("🎯 Previsão Ativa")
        
        if self.gestor.previsao_ativa:
            previsao = self.gestor.previsao_ativa
            st.success(f"**{previsao['estrategia']}**")
            st.write(f"**Gatilho:** {previsao['gatilho']}")
            st.write(f"**Números para apostar ({len(previsao['numeros_apostar'])}):**")
            st.write(", ".join(map(str, sorted(previsao['numeros_apostar']))))
            st.info("⏳ Aguardando próximo sorteio para conferência...")
        else:
            st.info("⏳ Aguardando gatilho para nova previsão...")
    
    def _exibir_ultima_conferencia(self):
        """Exibe o resultado da última conferência"""
        if st.session_state.ultima_conferencia:
            st.subheader("📊 Última Conferência")
            conferencia = st.session_state.ultima_conferencia
            if conferencia['acerto']:
                st.success(f"🎉 **ACERTOU!** Número {conferencia['numero_sorteado']} estava na previsão!")
            else:
                st.error(f"❌ **ERROU!** Número {conferencia['numero_sorteado']} não estava na previsão.")
            st.write(f"Estratégia: {conferencia['estrategia']}")
    
    def _exibir_desempenho(self):
        """Exibe o desempenho detalhado"""
        st.subheader("📈 Desempenho Detalhado")
        
        desempenho = self.gestor.desempenho
        total = desempenho.acertos + desempenho.erros
        taxa = desempenho.get_taxa_acerto()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🟢 Acertos", desempenho.acertos)
        col2.metric("🔴 Erros", desempenho.erros)
        col3.metric("📊 Total", total)
        col4.metric("✅ Taxa", f"{taxa:.1f}%")
        
        # Estatísticas por estratégia
        st.write("**📊 Desempenho por Estratégia:**")
        for estrategia, dados in desempenho.estatisticas_estrategias.items():
            taxa_estrategia = desempenho.get_taxa_acerto_estrategia(estrategia)
            cor = "🟢" if taxa_estrategia >= 40 else "🟡" if taxa_estrategia >= 25 else "🔴"
            st.write(f"{cor} {estrategia}: {dados['acertos']}/{dados['tentativas']} ({taxa_estrategia:.1f}%)")
        
        # Histórico recente
        if desempenho.historico:
            st.write("**Últimas 5 conferências:**")
            for i, conf in enumerate(desempenho.historico[-5:]):
                resultado_str = "🟢" if conf['acerto'] else "🔴"
                st.write(f"{resultado_str} {conf['estrategia']}: Número {conf['numero_sorteado']}")
    
    def _exibir_informacoes_estrategias(self):
        """Exibe informações sobre as estratégias"""
        with st.expander("📚 Estratégias Disponíveis"):
            st.write("""
            **🎯 Terminais Dominantes (Inteligente)**
            - Gatilho: Critério A (número repetido) ou B (terminal repetido)
            - **Máximo 8 números**: Terminais dominantes + vizinhos estratégicos
            - Foco em números frios (não sorteados recentemente)
            
            **🎯 Estratégias Midas**
            - **Padrão do Zero**: Terminal 0 aparecer ≥2x nos últimos 8 sorteios
            - **Padrão do Sete**: Terminal 7 aparecer ≥2x nos últimos 8 sorteios  
            - **Padrão do Cinco**: Terminal 5 aparecer ≥2x nos últimos 8 sorteios
            - **Padrão Gêmeos**: Gêmeos (11,22,33) aparecer ≥2x nos últimos 8 sorteios
            - **Máximo 8 números** por estratégia
            """)
    
    def _exibir_download_historico(self):
        """Exibe botão para download do histórico"""
        if os.path.exists(self.historico_manager.caminho_arquivo):
            with open(self.historico_manager.caminho_arquivo, "r") as f:
                conteudo = f.read()
            st.download_button("📥 Baixar histórico", data=conteudo, file_name="historico_roleta.json")

# =============================
# FUNÇÃO PRINCIPAL
# =============================

def tocar_som_moeda():
    """Toca som de moeda (mantido para compatibilidade)"""
    st.markdown("""<audio autoplay><source src="" type="audio/mp3"></audio>""", unsafe_allow_html=True)

def main():
    """Função principal da aplicação"""
    # Inicializar componentes
    config = Configuracao()
    historico_manager = HistoricoManager(config.historico_path)
    api_client = APIClient(config.api_url, config.headers)
    gestor = GestorEstrategias(config)
    
    # Inicializar interface
    interface = InterfaceUsuario(gestor, historico_manager, api_client)
    
    # Exibir interface
    interface.exibir_interface()

if __name__ == "__main__":
    main()
