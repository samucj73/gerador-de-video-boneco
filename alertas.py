import requests
import logging

# =============================
# Configurações do Telegram
# =============================
TELEGRAM_TOKEN = "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY"
TELEGRAM_CHAT_ID = "5121457416"
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# =============================
# Enviar previsão
# =============================
def enviar_previsao(mensagem: str):
    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
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
            "chat_id": TELEGRAM_CHAT_ID,
            "text": mensagem,
            "parse_mode": "HTML"
        }
        response = requests.post(BASE_URL, data=payload, timeout=5)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Erro ao enviar resultado: {e}")
