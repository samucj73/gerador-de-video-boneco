import streamlit as st
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

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
    """Usa o driver para capturar os números com o seletor CSS informado."""
    driver = inicializa_driver(url)
    # Dá um tempo para a página carregar completamente
    time.sleep(3)
    elementos = driver.find_elements(By.CSS_SELECTOR, seletor)
    numeros = [el.text.strip() for el in elementos if el.text.strip() != ""]
    return numeros

# --- Setup da aplicação Streamlit ---

st.set_page_config(page_title="Bot de Captura - Roleta", layout="wide")
st.title("Bot de Captura - Roleta Betdair")

# Entradas do usuário: URL e Seletor CSS
url = st.text_input("Informe a URL do site:", value="https://www.betdair.com.br/roleta")
seletor = st.text_input("Informe o seletor CSS para capturar os números:", value=".numero")

# Inicializa variáveis de controle no session_state
if "capturando" not in st.session_state:
    st.session_state.capturando = False
if "captured_data" not in st.session_state:
    st.session_state.captured_data = []

# Botões para iniciar e parar a captura
col1, col2 = st.columns(2)
if col1.button("Iniciar Captura"):
    st.session_state.capturando = True
    st.experimental_rerun()

if col2.button("Parar Captura"):
    st.session_state.capturando = False
    # Se o driver estiver ativo, feche-o e remova do state
    if "driver" in st.session_state:
        st.session_state.driver.quit()
        del st.session_state.driver
    st.experimental_rerun()

# Se estiver em modo de captura, realiza a captura periodicamente
if st.session_state.capturando:
    st.info("Captura em andamento. A página será atualizada a cada 5 segundos.")
    numeros = captura_numeros(url, seletor)
    # Adiciona os dados à lista de histórico se for uma nova captura
    if numeros and (not st.session_state.captured_data or numeros != st.session_state.captured_data[-1]["numeros"]):
        registro = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "numeros": numeros}
        st.session_state.captured_data.append(registro)
    st.write("Última captura:")
    st.write(numeros)
    # Aguarda 5 segundos e recarrega (para atualizar os dados)
    time.sleep(5)
    st.experimental_rerun()

# Painel de exibição do histórico das capturas
st.subheader("Histórico dos Números Capturados")
if st.session_state.captured_data:
    for registro in st.session_state.captured_data:
        st.write(f"[{registro['timestamp']}] → {registro['numeros']}")
else:
    st.write("Nenhum dado capturado ainda.")
