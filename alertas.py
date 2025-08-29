# alertas.py
import requests
import logging
from typing import Optional

# =============================
# Configurações do Telegram
# =============================
TELEGRAM_TOKEN = "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY"
TELEGRAM_CHAT_ID = "5121457416"
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# =============================
# Controle de alertas repetidos
# =============================
ultimo_alerta_prev: Optional[str] = None
ultimo_alerta_res: Optional[str] = None

# =============================
# Função genérica de envio
# =============================
def _enviar_telegram(mensagem: str):
    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": mensagem,
            "parse_mode": "HTML"
        }
        response = requests.post(BASE_URL, data=payload, timeout=5)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Erro ao enviar mensagem Telegram: {e}")

# =============================
# Enviar previsão
# =============================
def enviar_previsao(mensagem: str, forcar: bool = False):
    """
    Envia previsão para o Telegram.
    :param mensagem: texto da previsão
    :param forcar: se True, envia mesmo que seja igual ao último alerta
    """
    global ultimo_alerta_prev
    if mensagem != ultimo_alerta_prev or forcar:
        _enviar_telegram(mensagem)
        ultimo_alerta_prev = mensagem
    else:
        logging.info("Alerta de previsão repetido ignorado.")

# =============================
# Enviar resultado
# =============================
def enviar_resultado(mensagem: str, forcar: bool = False):
    """
    Envia resultado para o Telegram.
    :param mensagem: texto do resultado
    :param forcar: se True, envia mesmo que seja igual ao último alerta
    """
    global ultimo_alerta_res
    if mensagem != ultimo_alerta_res or forcar:
        _enviar_telegram(mensagem)
        ultimo_alerta_res = mensagem
    else:
        logging.info("Alerta de resultado repetido ignorado.")
