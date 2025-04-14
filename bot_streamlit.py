import streamlit as st
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from streamlit_autorefresh import st_autorefresh
from collections import Counter

# --- Funções auxiliares ---

def inicializa_driver(url):
    """Inicializa o Selenium WebDriver em modo headless e abre a URL."""
    if "driver" not in st.session_state:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        st.session_state.driver = driver
    return st.session_state.driver

def captura_numeros(url, seletor):
    """Usa o driver para capturar os números utilizando o seletor CSS informado."""
    driver = inicializa_driver(url)
    # Dá um tempo para a página carregar completamente
    time.sleep(3)
    elementos = driver.find_elements(By.CSS_SELECTOR, seletor)
    numeros = [el.text.strip() for el in elementos if el.text.strip() != ""]
    return numeros

def auto_detect_css(url):
    """
    Percorre todos os elementos da página e tenta identificar candidatos a seletor CSS.
    Procura por elementos cujo texto seja numérico e coleta as classes associadas.
    Retorna uma lista de seletores (no formato ".classe") ordenados por frequência.
    """
    driver = inicializa_driver(url)
    time.sleep(3)
    # Busca todos os elementos na página
    elementos = driver.find_elements(By.XPATH, "//*")
    contagem_classes = Counter()

    for el in elementos:
        texto = el.text.strip()
        # Verifica se o texto é numérico (pode ajustar para aceitar também espaços ou símbolos se necessário)
        if texto.isdigit():
            classes = el.get_attribute("class")
            if classes:
                # Considera cada classe individualmente
                for cls in classes.split():
                    contagem_classes[cls] += 1

    # Filtra candidatos: por exemplo, considere somente classes que apareceram mais de uma vez
    candidatos = [f".{cls}" for cls, freq in contagem_classes.items() if freq > 1]

    # Ordena pelos mais frequentes
    candidatos = sorted(candidatos, key=lambda x: contagem_classes[x[1:]], reverse=True)
    return candidatos

# --- Setup da aplicação Streamlit ---

st.set_page_config(page_title="Bot de Captura - Roleta", layout="wide")
st.title("Bot de Captura - Roleta Betdair")

# Entradas do usuário: URL e seletor CSS
url = st.text_input("Informe a URL do site:", value="https://www.betdair.com.br/roleta")
seletor_manual = st.text_input("Informe o seletor CSS para capturar os números (ou deixe em branco para usar a detecção automática):", value="")

# Variáveis de controle no session_state
if "capturando" not in st.session_state:
    st.session_state.capturando = False
if "captured_data" not in st.session_state:
    st.session_state.captured_data = []
if "detected_candidates" not in st.session_state:
    st.session_state.detected_candidates = []

# Botão para Auto Detect CSS
if st.button("Auto Detect CSS"):
    st.info("Detectando seletores CSS automaticamente...")
    candidatos = auto_detect_css(url)
    if candidatos:
        st.session_state.detected_candidates = candidatos
        st.success(f"{len(candidatos)} candidatos encontrados.")
    else:
        st.warning("Nenhum candidato detectado. Verifique se há elementos com texto numérico na página.")
        
# Se candidatos foram detectados, permite a seleção via dropdown
if st.session_state.detected_candidates:
    seletor_detectado = st.selectbox("Selecione um seletor dentre os candidatos:", st.session_state.detected_candidates)
else:
    seletor_detectado = ""

# Define o seletor efetivo:
# Se o usuário digitou algo, usa o manual; caso contrário, se houver detecção, usa o selecionado;
# Caso nenhum deles esteja definido, usa o seletor padrão baseado na nossa conclusão:
seletor = seletor_manual.strip() or seletor_detectado.strip()
if not seletor:
    seletor = ".result-wrapper .result-number"
    st.info("Nenhum seletor foi definido manualmente ou detectado. Utilizando seletor padrão: '.result-wrapper .result-number'")

st.write("Seletor usado:", seletor)

# Botões para iniciar e parar a captura
col1, col2 = st.columns(2)
if col1.button("Iniciar Captura"):
    st.session_state.capturando = True
if col2.button("Parar Captura"):
    st.session_state.capturando = False
    # Se o driver estiver ativo, feche-o e remova do state
    if "driver" in st.session_state:
        st.session_state.driver.quit()
        del st.session_state.driver

# Se estiver em modo de captura e se um seletor válido estiver definido, realiza a captura periodicamente
if st.session_state.capturando:
    if seletor:
        st.info("Captura em andamento. A página será atualizada automaticamente a cada 5 segundos.")
        numeros = captura_numeros(url, seletor)
        # Adiciona os dados à lista de histórico se for uma nova captura
        if numeros and (not st.session_state.captured_data or numeros != st.session_state.captured_data[-1]["numeros"]):
            registro = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "numeros": numeros}
            st.session_state.captured_data.append(registro)
        st.write("Última captura:")
        st.write(numeros)
        # Atualização automática a cada 5000ms (5 segundos)
        st_autorefresh(interval=5000, key="roleta_autorefresh")
    else:
        st.error("Por favor, defina um seletor CSS (manual ou detectado) para capturar os números.")

# Painel de exibição do histórico das capturas
st.subheader("Histórico dos Números Capturados")
if st.session_state.captured_data:
    for registro in st.session_state.captured_data:
        st.write(f"[{registro['timestamp']}] → {registro['numeros']}")
else:
    st.write("Nenhum dado capturado ainda.")
