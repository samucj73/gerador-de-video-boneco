import streamlit as st
from datetime import datetime, timedelta
import requests
import json
import os
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

# =============================
# Configurações e Segurança
# =============================

API_KEY = os.getenv("FOOTBALL_API_KEY", "9058de85e3324bdb969adc005b5d918a")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003073115320")
TELEGRAM_CHAT_ID_ALT2 = os.getenv("TELEGRAM_CHAT_ID_ALT2", "-1002754276285")

HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL_FD = "https://api.football-data.org/v4"
BASE_URL_TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

ALERTAS_PATH = "alertas.json"
CACHE_JOGOS = "cache_jogos.json"
CACHE_CLASSIFICACAO = "cache_classificacao.json"
CACHE_TIMEOUT = 3600  # 1 hora

# =============================
# Ligas
# =============================
LIGA_DICT = {
    "FIFA World Cup": "WC",
    "UEFA Champions League": "CL", 
    "Bundesliga": "BL1",
    "Eredivisie": "DED",
    "Campeonato Brasileiro Série A": "BSA",
    "Primera Division": "PD",
    "Ligue 1": "FL1",
    "Championship (Inglaterra)": "ELC",
    "Primeira Liga (Portugal)": "PPL",
    "European Championship": "EC",
    "Serie A (Itália)": "SA",
    "Premier League (Inglaterra)": "PL"
}

# =============================
# Funções de Cache
# =============================
def carregar_json(caminho: str) -> dict:
    try:
        if os.path.exists(caminho):
            with open(caminho, "r", encoding='utf-8') as f:
                dados = json.load(f)
            if caminho in [CACHE_JOGOS, CACHE_CLASSIFICACAO]:
                agora = datetime.now().timestamp()
                for key in list(dados.keys()):
                    if isinstance(dados[key], dict) and '_timestamp' in dados[key]:
                        if agora - dados[key]['_timestamp'] > CACHE_TIMEOUT:
                            del dados[key]
                    elif agora - os.path.getmtime(caminho) > CACHE_TIMEOUT:
                        return {}
            return dados
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"Erro ao carregar {caminho}: {e}")
    return {}

def salvar_json(caminho: str, dados: dict):
    try:
        if caminho in [CACHE_JOGOS, CACHE_CLASSIFICACAO]:
            dados['_timestamp'] = datetime.now().timestamp()
        with open(caminho, "w", encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
    except IOError as e:
        st.error(f"Erro ao salvar {caminho}: {e}")

def carregar_alertas() -> dict:
    return carregar_json(ALERTAS_PATH)

def salvar_alertas(alertas: dict):
    salvar_json(ALERTAS_PATH, alertas)

def carregar_cache_jogos() -> dict:
    return carregar_json(CACHE_JOGOS)

def salvar_cache_jogos(dados: dict):
    salvar_json(CACHE_JOGOS, dados)

def carregar_cache_classificacao() -> dict:
    return carregar_json(CACHE_CLASSIFICACAO)

def salvar_cache_classificacao(dados: dict):
    salvar_json(CACHE_CLASSIFICACAO, dados)

# =============================
# Funções de Utilidade
# =============================
def formatar_data_iso(data_iso: str) -> tuple[str, str]:
    try:
        data_jogo = datetime.fromisoformat(data_iso.replace("Z", "+00:00")) - timedelta(hours=3)
        return data_jogo.strftime("%d/%m/%Y"), data_jogo.strftime("%H:%M")
    except ValueError:
        return "Data inválida", "Hora inválida"

def abreviar_nome(nome: str, max_len: int = 15) -> str:
    if len(nome) <= max_len:
        return nome
    palavras = nome.split()
    abreviado = " ".join([p[0] + "." if len(p) > 2 else p for p in palavras])
    return abreviado[:max_len-3] + "..." if len(abreviado) > max_len else abreviado

# =============================
# Comunicação API / Telegram
# =============================
def enviar_telegram(msg: str, chat_id: str = TELEGRAM_CHAT_ID) -> bool:
    try:
        response = requests.get(
            BASE_URL_TG, 
            params={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
        return response.status_code == 200
    except requests.RequestException as e:
        st.error(f"Erro ao enviar para Telegram: {e}")
        return False

def obter_dados_api(url: str, timeout: int = 10) -> dict | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Erro na requisição API: {e}")
        return None

def obter_classificacao(liga_id: str) -> dict:
    cache = carregar_cache_classificacao()
    if liga_id in cache:
        return cache[liga_id]

    url = f"{BASE_URL_FD}/competitions/{liga_id}/standings"
    data = obter_dados_api(url)
    if not data:
        return {}

    standings = {}
    for s in data.get("standings", []):
        if s["type"] != "TOTAL":
            continue
        for t in s["table"]:
            name = t["team"]["name"]
            standings[name] = {
                "scored": t.get("goalsFor", 0),
                "against": t.get("goalsAgainst", 0),
                "played": t.get("playedGames", 1)
            }

    cache[liga_id] = standings
    salvar_cache_classificacao(cache)
    return standings

def obter_jogos(liga_id: str, data: str) -> list:
    cache = carregar_cache_jogos()
    key = f"{liga_id}_{data}"
    if key in cache:
        return cache[key]

    url = f"{BASE_URL_FD}/competitions/{liga_id}/matches?dateFrom={data}&dateTo={data}"
    data_api = obter_dados_api(url)
    jogos = data_api.get("matches", []) if data_api else []
    cache[key] = jogos
    salvar_cache_jogos(cache)
    return jogos

# =============================
# Cálculo de Tendência
# =============================
def calcular_tendencia(home: str, away: str, classificacao: dict) -> tuple[float, float, str]:
    dados_home = classificacao.get(home, {"scored": 0, "against": 0, "played": 1})
    dados_away = classificacao.get(away, {"scored": 0, "against": 0, "played": 1})

    played_home = max(dados_home["played"], 1)
    played_away = max(dados_away["played"], 1)

    media_home_feitos = dados_home["scored"] / played_home
    media_home_sofridos = dados_home["against"] / played_home
    media_away_feitos = dados_away["scored"] / played_away
    media_away_sofridos = dados_away["against"] / played_away

    estimativa = ((media_home_feitos + media_away_sofridos) / 2 +
                  (media_away_feitos + media_home_sofridos) / 2)

    if estimativa >= 3.0:
        tendencia = "Mais 2.5"
        confianca = min(95, 70 + (estimativa - 3.0) * 10)
    elif estimativa >= 2.0:
        tendencia = "Mais 1.5"
        confianca = min(90, 60 + (estimativa - 2.0) * 10)
    else:
        tendencia = "Menos 2.5"
        confianca = min(85, 55 + (2.0 - estimativa) * 10)

    return estimativa, confianca, tendencia

def enviar_alerta_telegram(fixture: dict, tendencia: str, estimativa: float, confianca: float):
    home = fixture.get("homeTeam", {}).get("name", "Desconhecido")
    away = fixture.get("awayTeam", {}).get("name", "Desconhecido")
    data_formatada, hora_formatada = formatar_data_iso(fixture.get("utcDate", ""))
    competicao = fixture.get("competition", {}).get("name", "Desconhecido")

    status = fixture.get("status", "DESCONHECIDO")
    gols_home = fixture.get("score", {}).get("fullTime", {}).get("home")
    gols_away = fixture.get("score", {}).get("fullTime", {}).get("away")
    
    placar = f"{gols_home} x {gols_away}" if gols_home is not None and gols_away is not None else None

    msg = (
        f"⚽ <b>Alerta de Gols!</b>\n"
        f"🏟️ {home} vs {away}\n"
        f"📅 {data_formatada} ⏰ {hora_formatada} (BRT)\n"
        f"📌 Status: {status}\n"
    )
    
    if placar:
        msg += f"📊 Placar: <b>{placar}</b>\n"
        
    msg += (
        f"📈 Tendência: <b>{tendencia}</b>\n"
        f"🎯 Estimativa: <b>{estimativa:.2f} gols</b>\n"
        f"💯 Confiança: <b>{confianca:.0f}%</b>\n"
        f"🏆 Liga: {competicao}"
    )
    
    enviar_telegram(msg)

def verificar_enviar_alerta(fixture: dict, tendencia: str, estimativa: float, confianca: float):
    alertas = carregar_alertas()
    fixture_id = str(fixture.get("id"))
    
    if fixture_id not in alertas:
        alertas[fixture_id] = {
            "tendencia": tendencia,
            "estimativa": estimativa,
            "confianca": confianca,
            "conferido": False
        }
        enviar_alerta_telegram(fixture, tendencia, estimativa, confianca)
        salvar_alertas(alertas)

# =============================
# Conferência de Resultados (robusta)
# =============================
def conferir_resultados():
    alertas = carregar_alertas()
    jogos_cache = carregar_cache_jogos()
    
    if not alertas:
        st.info("ℹ️ Nenhum alerta para conferir.")
        return

    jogos_conferidos = []
    mudou = False

    for fixture_id, info in alertas.items():
        if info.get("conferido"):
            continue

        jogo_dado = None
        for key, matches in jogos_cache.items():
            if key == "_timestamp" or not isinstance(matches, list):
                continue

            for match in matches:
                if not isinstance(match, dict):
                    continue

                match_id = None
                if "id" in match:
                    match_id = match["id"]
                elif "fixture" in match and isinstance(match["fixture"], dict):
                    match_id = match["fixture"].get("id")
                elif "match" in match and isinstance(match["match"], dict):
                    match_id = match["match"].get("id")

                if match_id and str(match_id) == str(fixture_id):
                    jogo_dado = match
                    break
            if jogo_dado:
                break

        if not jogo_dado:
            continue

        resultado_info = processar_resultado_jogo(jogo_dado, info)
        if resultado_info:
            exibir_resultado_streamlit(resultado_info)
            
            if resultado_info["status"] == "FINISHED":
                enviar_resultado_telegram(resultado_info)
                info["conferido"] = True
                mudou = True

        jogos_conferidos.append(preparar_dados_pdf(jogo_dado, info, resultado_info))

    if mudou:
        salvar_alertas(alertas)
        st.success("✅ Resultados conferidos e atualizados!")

    if jogos_conferidos:
        gerar_pdf_jogos(jogos_conferidos)

# =============================
# Processamento de Resultado
# =============================
def processar_resultado_jogo(jogo: dict, info: dict) -> dict | None:
    home = jogo.get("homeTeam", {}).get("name", "Desconhecido")
    away = jogo.get("awayTeam", {}).get("name", "Desconhecido")
    status = jogo.get("status", "DESCONHECIDO")
    gols_home = jogo.get("score", {}).get("fullTime", {}).get("home")
    gols_away = jogo.get("score", {}).get("fullTime", {}).get("away")
    
    placar = f"{gols_home} x {gols_away}" if gols_home is not None and gols_away is not None else "-"
    total_gols = (gols_home or 0) + (gols_away or 0)

    resultado = "⏳ Aguardando"
    if status == "FINISHED":
        tendencia = info.get("tendencia", "")
        if "Mais 2.5" in tendencia:
            resultado = "🟢 GREEN" if total_gols > 2 else "🔴 RED"
        elif "Mais 1.5" in tendencia:
            resultado = "🟢 GREEN" if total_gols > 1 else "🔴 RED"
        elif "Menos 2.5" in tendencia:
            resultado = "🟢 GREEN" if total_gols < 3 else "🔴 RED"
        else:
            resultado = "⚪ INDEFINIDO"

    return {
        "home": home,
        "away": away,
        "status": status,
        "placar": placar,
        "tendencia": info.get("tendencia", "-"),
        "estimativa": info.get("estimativa", 0),
        "confianca": info.get("confianca", 0),
        "resultado": resultado,
        "total_gols": total_gols
    }

# =============================
# Streamlit e PDF
# =============================
def exibir_resultado_streamlit(resultado: dict):
    bg_color = "#1e4620" if resultado["resultado"] == "🟢 GREEN" else \
               "#5a1e1e" if resultado["resultado"] == "🔴 RED" else "#2c2c2c"
    
    st.markdown(f"""
    <div style="border:1px solid #444; border-radius:10px; padding:12px; margin-bottom:10px;
                background-color:{bg_color}; font-size:15px; color:#f1f1f1;">
        <b>🏟️ {resultado['home']} vs {resultado['away']}</b><br>
        📌 Status: <b>{resultado['status']}</b><br>
        ⚽ Tendência: <b>{resultado['tendencia']}</b> | Estim.: {resultado['estimativa']:.2f} | Conf.: {resultado['confianca']:.0f}%<br>
        📊 Placar: <b>{resultado['placar']}</b><br>
        ✅ Resultado: {resultado['resultado']}
    </div>
    """, unsafe_allow_html=True)

def enviar_resultado_telegram(resultado: dict):
    msg = (
        f"📊 <b>Resultado Conferido</b>\n"
        f"🏟️ {resultado['home']} vs {resultado['away']}\n"
        f"⚽ Tendência: {resultado['tendencia']} | Estim.: {resultado['estimativa']:.2f} | Conf.: {resultado['confianca']:.0f}%\n"
        f"📊 Placar Final: <b>{resultado['placar']}</b>\n"
        f"✅ Resultado: <b>{resultado['resultado']}</b>"
    )
    enviar_telegram(msg, TELEGRAM_CHAT_ID_ALT2)

def preparar_dados_pdf(jogo: dict, info: dict, resultado: dict) -> list:
    home = abreviar_nome(jogo.get("homeTeam", {}).get("name", "Desconhecido"))
    away = abreviar_nome(jogo.get("awayTeam", {}).get("name", "Desconhecido"))
    hora = datetime.fromisoformat(jogo.get("utcDate", "").replace("Z", "+00:00")) - timedelta(hours=3)
    
    return [
        f"{home} vs {away}",
        info.get("tendencia", "-"),
        f"{info.get('estimativa', 0):.2f}",
        f"{info.get('confianca', 0):.0f}%",
        resultado["placar"] if resultado else "-",
        jogo.get("status", "DESCONHECIDO"),
        resultado["resultado"] if resultado else "⏳ Aguardando",
        hora.strftime("%d/%m %H:%M")
    ]

def gerar_relatorio_pdf(jogos_conferidos: list) -> io.BytesIO:
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter, 
                          rightMargin=20, leftMargin=20, 
                          topMargin=20, bottomMargin=20)

    data = [["Jogo", "Tendência", "Estimativa", "Confiança", 
             "Placar", "Status", "Resultado", "Hora"]] + jogos_conferidos

    table = Table(data, repeatRows=1, 
                 colWidths=[120, 70, 60, 60, 50, 70, 60, 70])
    
    style = TableStyle([
        ('BACKGROUND', (0,0),
        (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ])
    table.setStyle(style)
    pdf.build([table])
    buffer.seek(0)
    return buffer

def gerar_pdf_jogos(jogos_conferidos: list):
    if not jogos_conferidos:
        return
    pdf_buffer = gerar_relatorio_pdf(jogos_conferidos)
    st.download_button(
        label="📥 Baixar Relatório PDF",
        data=pdf_buffer,
        file_name=f"resultados_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )

# =============================
# Função Principal Streamlit
# =============================
def main():
    st.set_page_config(page_title="⚽ Sistema Unificado de Alertas", layout="wide")
    st.title("⚽ Sistema Unificado de Alertas (Football-Data + Telegram)")

    # Escolha de data
    data_selecionada = st.date_input("📅 Escolha a data para os jogos:", value=datetime.today())
    data_str = data_selecionada.strftime("%Y-%m-%d")

    ligas_selecionadas = st.multiselect(
        "🏆 Selecione as ligas:", options=list(LIGA_DICT.keys()),
        default=["UEFA Champions League", "Campeonato Brasileiro Série A"]
    )

    for liga_nome in ligas_selecionadas:
        liga_id = LIGA_DICT[liga_nome]
        st.subheader(f"⚽ {liga_nome}")

        jogos = obter_jogos(liga_id, data_str)
        if not jogos:
            st.info(f"ℹ️ Nenhum jogo encontrado para {liga_nome} na data {data_str}")
            continue

        classificacao = obter_classificacao(liga_id)
        for fixture in jogos:
            home = fixture.get("homeTeam", {}).get("name", "Desconhecido")
            away = fixture.get("awayTeam", {}).get("name", "Desconhecido")

            estimativa, confianca, tendencia = calcular_tendencia(home, away, classificacao)
            verificar_enviar_alerta(fixture, tendencia, estimativa, confianca)

    # Conferência de resultados
    st.subheader("📊 Conferência de Resultados")
    conferir_resultados()

if __name__ == "__main__":
    main()
    
