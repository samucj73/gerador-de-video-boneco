import streamlit as st
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from streamlit_autorefresh import st_autorefresh
from collections import Counter
from PIL import Image

# --- Funções auxiliares ---

def inicializa_driver(url):
    """Inicializa o Selenium WebDriver com configurações específicas para Render."""
    if "driver" not in st.session_state:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_options.add_argument("--remote-debugging-port=9222")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        st.session_state.driver = driver
    return st.session_state.driver

def captura_numeros(url, seletor):
    """Usa o driver para capturar os números utilizando o seletor CSS informado."""
    driver = inicializa_driver(url)
    time.sleep(3)
    elementos = driver.find_elements(By.CSS_SELECTOR, seletor)
    numeros = [el.text.strip() for el in elementos if el.text.strip() != ""]
    return numeros

def auto_detect_css(url):
    """Detecta candidatos a seletor CSS com base em texto numérico."""
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

def mostrar_screenshot(driver):
    """Captura e mostra a imagem da página como pré-visualização."""
    screenshot_path = "/tmp/preview.png"
    driver.save_screenshot(screenshot_path)
    st.image(Image.open(screenshot_path), caption="Pré-visualização da página", use_column_width=True)

# --- Setup da aplicação Streamlit ---

st.set_page_config(page_title="Bot de Captura - Roleta", layout="wide")
st.title("Bot de Captura - Roleta Betfair")

url = st.text_input("Informe a URL do site:", value="https://www.betdair.com.br/roleta")
seletor_manual = st.text_input("Informe o seletor CSS:", value="")

if "capturando" not in st.session_state:
    st.session_state.capturando = False
if "captured_data" not in st.session_state:
    st.session_state.captured_data = []
if "detected_candidates" not in st.session_state:
    st.session_state.detected_candidates = []

if st.button("Auto Detect CSS"):
    st.info("Detectando seletores CSS automaticamente...")
    candidatos = auto_detect_css(url)
    if candidatos:
        st.session_state.detected_candidates = candidatos
        st.success(f"{len(candidatos)} candidatos encontrados.")
    else:
        st.warning("Nenhum candidato detectado.")
        
if st.session_state.detected_candidates:
    seletor_detectado = st.selectbox("Selecione um seletor:", st.session_state.detected_candidates)
else:
    seletor_detectado = ""

seletor = seletor_manual.strip() or seletor_detectado.strip()
if not seletor:
    seletor = ".result-wrapper .result-number"
    st.info("Usando seletor padrão: '.result-wrapper .result-number'")

st.write("Seletor usado:", seletor)

col1, col2 = st.columns(2)
if col1.button("Iniciar Captura"):
    st.session_state.capturando = True
if col2.button("Parar Captura"):
    st.session_state.capturando = False
    if "driver" in st.session_state:
        st.session_state.driver.quit()
        del st.session_state.driver

if st.session_state.capturando:
    if seletor:
        st.info("Captura em andamento. Atualizando a cada 5 segundos.")
        numeros = captura_numeros(url, seletor)
        if numeros and (not st.session_state.captured_data or numeros != st.session_state.captured_data[-1]["numeros"]):
            registro = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "numeros": numeros}
            st.session_state.captured_data.append(registro)
        st.write("Última captura:")
        st.write(numeros)
        mostrar_screenshot(st.session_state.driver)
        st_autorefresh(interval=5000, key="roleta_autorefresh")
    else:
        st.error("Seletor não definido!")

st.subheader("Histórico")
if st.session_state.captured_data:
    for registro in st.session_state.captured_data:
        st.write(f"[{registro['timestamp']}] → {registro['numeros']}")
else:
    st.write("Nenhum dado capturado ainda.")
