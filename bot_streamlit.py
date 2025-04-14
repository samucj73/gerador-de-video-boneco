import streamlit as st
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from streamlit_autorefresh import st_autorefresh
from collections import Counter
from PIL import Image
import streamlit.components.v1 as components
import os

# --- Funções auxiliares ---

def inicializa_driver(url):
    """Inicializa o Selenium WebDriver e abre a URL."""
    if "driver" not in st.session_state:
        chrome_options = Options()
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1280,800")
        chrome_options.add_argument("--headless=new")  # usa headless novo
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        st.session_state.driver = driver
    return st.session_state.driver

def captura_numeros(url, seletor):
    """Captura os números da roleta via seletor CSS."""
    driver = inicializa_driver(url)
    time.sleep(3)
    elementos = driver.find_elements(By.CSS_SELECTOR, seletor)
    numeros = [el.text.strip() for el in elementos if el.text.strip()]
    return numeros

def auto_detect_css(url):
    """Detecta seletores CSS que provavelmente representam números."""
    driver = inicializa_driver(url)
    time.sleep(3)
    elementos = driver.find_elements(By.XPATH, "//*")
    contagem_classes = Counter()

    for el in elementos:
        texto = el.text.strip()
        if texto.isdigit():
            classes = el.get_attribute("class")
            if classes:
                for cls in classes.split():
                    contagem_classes[cls] += 1

    candidatos = [f".{cls}" for cls, freq in contagem_classes.items() if freq > 1]
    candidatos = sorted(candidatos, key=lambda x: contagem_classes[x[1:]], reverse=True)
    return candidatos

def tirar_screenshot(driver):
    """Tira e exibe um screenshot da página carregada."""
    screenshot_path = "screenshot.png"
    driver.save_screenshot(screenshot_path)
    img = Image.open(screenshot_path)
    return img

# --- Setup da aplicação ---

st.set_page_config(page_title="Bot de Captura - Roleta", layout="wide")
st.title("Bot de Captura - Roleta Betfair")

# Entrada de URL e seletor CSS
url = st.text_input("Informe a URL do site:", value="https://www.betdair.com.br/roleta")
seletor_manual = st.text_input("Informe o seletor CSS (ou deixe em branco para auto):", value="")

# Exibição da página no iframe (se permitido) ou screenshot
st.markdown("### Visualização da Página Capturada")
try:
    components.iframe(url, height=500, scrolling=True)
except:
    try:
        driver = inicializa_driver(url)
        img = tirar_screenshot(driver)
        st.image(img, caption="Screenshot da página", use_column_width=True)
    except:
        st.warning("Não foi possível exibir a página nem carregar screenshot.")

# Session state
if "capturando" not in st.session_state:
    st.session_state.capturando = False
if "captured_data" not in st.session_state:
    st.session_state.captured_data = []
if "detected_candidates" not in st.session_state:
    st.session_state.detected_candidates = []

# Botão para detectar seletor CSS automaticamente
if st.button("Auto Detect CSS"):
    st.info("Detectando seletores CSS automaticamente...")
    candidatos = auto_detect_css(url)
    if candidatos:
        st.session_state.detected_candidates = candidatos
        st.success(f"{len(candidatos)} candidatos encontrados.")
    else:
        st.warning("Nenhum candidato detectado.")

# Seleção de seletor
if st.session_state.detected_candidates:
    seletor_detectado = st.selectbox("Selecione um seletor detectado:", st.session_state.detected_candidates)
else:
    seletor_detectado = ""

# Escolha final do seletor
seletor = seletor_manual.strip() or seletor_detectado.strip()
if not seletor:
    seletor = ".result-wrapper .result-number"
    st.info("Nenhum seletor informado. Usando seletor padrão: '.result-wrapper .result-number'")

st.write("**Seletor em uso:**", seletor)

# Botões de controle
col1, col2 = st.columns(2)
if col1.button("Iniciar Captura"):
    st.session_state.capturando = True
if col2.button("Parar Captura"):
    st.session_state.capturando = False
    if "driver" in st.session_state:
        st.session_state.driver.quit()
        del st.session_state.driver

# Execução da captura automática
if st.session_state.capturando:
    if seletor:
        st.info("Captura ativa. Atualizando a cada 5 segundos.")
        numeros = captura_numeros(url, seletor)
        if numeros and (not st.session_state.captured_data or numeros != st.session_state.captured_data[-1]["numeros"]):
            registro = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "numeros": numeros}
            st.session_state.captured_data.append(registro)
        st.write("Última captura:")
        st.write(numeros)
        st_autorefresh(interval=5000, key="auto_refresh")
    else:
        st.error("Defina um seletor CSS válido para capturar os números.")

# Histórico
st.subheader("Histórico das Capturas")
if st.session_state.captured_data:
    for registro in st.session_state.captured_data:
        st.write(f"[{registro['timestamp']}] → {registro['numeros']}")
else:
    st.write("Nenhum dado capturado ainda.")
