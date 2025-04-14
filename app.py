import os
import time
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def inicializa_driver():
    chrome_options = Options()
    chrome_options.binary_location = "/usr/bin/google-chrome"
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service("/usr/local/bin/chromedriver")

    # Verifica se os arquivos existem (debug útil)
    assert os.path.exists(chrome_options.binary_location), "Chrome não encontrado!"
    assert os.path.exists(service.path), "Chromedriver não encontrado!"

    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def captura_numeros(url, seletor):
    driver = inicializa_driver()
    driver.get(url)
    time.sleep(3)
    elementos = driver.find_elements(By.CSS_SELECTOR, seletor)
    numeros = [e.text for e in elementos]
    driver.quit()
    return numeros

# Streamlit UI
st.title("Bot de Coleta de Números da Roleta")

url = st.text_input("URL da Roleta", "https://example.com")
seletor = st.text_input("Seletor CSS dos Números", ".roulette-history .number")

if st.button("Coletar Números"):
    try:
        numeros = captura_numeros(url, seletor)
        st.success("Números coletados com sucesso!")
        st.write(numeros)
    except Exception as e:
        st.error(f"Erro ao capturar os números: {e}")
