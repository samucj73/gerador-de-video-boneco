import requests
import logging

# =============================
# Configurações do Telegram
# =============================
TELEGRAM_TOKEN = "SEU_TOKEN_AQUI"
CHAT_ID = "SEU_CHAT_ID_AQUI"
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# =============================
# Enviar previsão
# =============================
def enviar_previsao(mensagem: str):
    try:
        payload = {
            "chat_id": CHAT_ID,
            "text": mensagem,
            "parse_mode": "HTML"
        }
        response = requests.post(BASE_URL, data=payload, timeout=5)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Erro ao enviar previsão: {e}")

# =============================
# Enviar resultado
# =============================
def enviar_resultado(mensagem: str):
    try:
        payload = {
            "chat_id": CHAT_ID,
            "text": mensagem,
            "parse_mode": "HTML"
        }
        response = requests.post(BASE_URL, data=payload, timeout=5)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Erro ao enviar resultado: {e}")
