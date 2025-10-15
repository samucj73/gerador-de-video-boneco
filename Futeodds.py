# Futebol_Alertas_Oddstop.py
import streamlit as st
from datetime import datetime, timedelta, date
import requests
import os
import json
import math
import time
import logging
from functools import lru_cache
from dotenv import load_dotenv

# =============================
# Configuração Inicial
# =============================
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================
# Configurações Odds API + Telegram
# =============================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "7ad63ba7ce77a6f31c33acc766f3e9fb")
ODDS_BASE = "https://api.the-odds-api.com/v4"

# Mapa de ligas: chave -> nome amigável
ALL_LIGAS = {
    # Europe + Global (principais)
    "soccer_epl": "England - Premier League (EPL)",
    "soccer_spain_la_liga": "Spain - La Liga",
    "soccer_italy_serie_a": "Italy - Serie A",
    "soccer_germany_bundesliga": "Germany - Bundesliga",
    "soccer_france_ligue_one": "France - Ligue 1",
    "soccer_brazil_campeonato": "Brazil - Série A (Brasileirão)",
    "soccer_uefa_champs_league": "UEFA Champions League",
    "soccer_uefa_europa_league": "UEFA Europa League",
    "soccer_uefa_europa_conference_league": "UEFA Europa Conference League",
    "soccer_portugal_primeira_liga": "Portugal - Primeira Liga",
    "soccer_netherlands_eredivisie": "Netherlands - Eredivisie",
    "soccer_mexico_ligamx": "Mexico - Liga MX",
    "soccer_turkey_super_league": "Turkey - Super Lig",
    "soccer_argentina_primera_division": "Argentina - Primera División",
    # American leagues (US/CONCACAF)
    "soccer_usa_mls": "USA - Major League Soccer (MLS)",
    "soccer_usa_usl_championship": "USA - USL Championship",
    # Outras
    "soccer_china_superleague": "China - Super League",
    "soccer_japan_j_league": "Japan - J League",
    "soccer_korea_kleague1": "Korea - K League 1",
    "soccer_belgium_first_div": "Belgium - First Division A",
}

# Default principais
DEFAULT_PRINCIPAIS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_brazil_campeonato",
    "soccer_uefa_champs_league",
    "soccer_usa_mls",
]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003073115320")
TELEGRAM_CHAT_ID_ALT2 = os.getenv("TELEGRAM_CHAT_ID_ALT2", "-1002932611974")
BASE_URL_TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

ALERTAS_PATH = "alertas.json"
TOP3_PATH = "top3.json"

# =============================
# Rate Limiter
# =============================
class RateLimitedAPI:
    def __init__(self, calls_per_minute=30):
        self.calls_per_minute = calls_per_minute
        self.last_calls = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove chamadas antigas
        self.last_calls = [t for t in self.last_calls if now - t < 60]
        
        if len(self.last_calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.last_calls[0])
            if sleep_time > 0:
                logger.info(f"Rate limit atingido. Aguardando {sleep_time:.1f} segundos")
                time.sleep(sleep_time)
        
        self.last_calls.append(now)

rate_limiter = RateLimitedAPI(30)

# =============================
# Persistência
# =============================
def carregar_alertas():
    try:
        if os.path.exists(ALERTAS_PATH):
            with open(ALERTAS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar alertas: {e}")
    return {}

def salvar_alertas(alertas):
    try:
        with open(ALERTAS_PATH, "w", encoding="utf-8") as f:
            json.dump(alertas, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Erro ao salvar alertas: {e}")

def carregar_top3():
    try:
        if os.path.exists(TOP3_PATH):
            with open(TOP3_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar top3: {e}")
    return []

def salvar_top3(lista):
    try:
        with open(TOP3_PATH, "w", encoding="utf-8") as f:
            json.dump(lista, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Erro ao salvar top3: {e}")

# =============================
# Envio Telegram (Melhorado)
# =============================
def enviar_telegram_seguro(msg, chat_id=TELEGRAM_CHAT_ID, max_retries=3):
    """Versão mais robusta para envio no Telegram"""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                BASE_URL_TG, 
                params={
                    "chat_id": chat_id, 
                    "text": msg, 
                    "parse_mode": "Markdown"
                }, 
                timeout=15
            )
            if response.status_code == 200:
                logger.info(f"Mensagem enviada com sucesso para Telegram (tentativa {attempt + 1})")
                return True
            else:
                logger.warning(f"Telegram retornou status {response.status_code} na tentativa {attempt + 1}")
        except Exception as e:
            logger.warning(f"Tentativa {attempt + 1} falhou: {e}")
            time.sleep(2)
    
    logger.error(f"Falha ao enviar para Telegram após {max_retries} tentativas")
    return False

def enviar_telegram(msg, chat_id=TELEGRAM_CHAT_ID):
    """Função wrapper para manter compatibilidade"""
    return enviar_telegram_seguro(msg, chat_id)

# =============================
# Validação de Dados
# =============================
def validar_resposta_odds_api(dados):
    """Valida a estrutura dos dados da Odds API"""
    if not isinstance(dados, list):
        logger.error("Resposta inválida da Odds API - não é uma lista")
        return False
    
    valid_count = 0
    for evento in dados:
        required_fields = ['id', 'commence_time', 'home_team', 'away_team', 'bookmakers']
        if all(field in evento for field in required_fields):
            valid_count += 1
        else:
            logger.warning(f"Evento com campos faltando: {evento.get('id', 'unknown')}")
    
    logger.info(f"Validação: {valid_count}/{len(dados)} eventos válidos")
    return valid_count > 0

# =============================
# Helpers Odds API (Com Cache e Rate Limiting)
# =============================
@lru_cache(maxsize=100)
def obter_odds_para_liga_cached(liga_key, regions="eu,us,au", date_str=None):
    """Versão com cache para evitar requisições repetidas"""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    return obter_odds_para_liga(liga_key, regions)

def obter_odds_para_liga(liga_key, regions="eu,us,au", markets="totals", odds_format="decimal"):
    """
    Consulta a Odds API para uma liga (sport_key) retornando eventos com mercado 'totals'.
    """
    rate_limiter.wait_if_needed()
    
    url = f"{ODDS_BASE}/sports/{liga_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso"
    }
    try:
        logger.info(f"Consultando Odds API para liga: {liga_key}")
        r = requests.get(url, params=params, timeout=20)
        
        if r.status_code == 200:
            dados = r.json()
            if validar_resposta_odds_api(dados):
                logger.info(f"✅ {len(dados)} eventos obtidos para {liga_key}")
                return dados
            else:
                logger.warning(f"⚠️ Dados inválidos para {liga_key}")
                return []
        elif r.status_code == 401:
            logger.error("❌ Erro de autenticação na Odds API - verifique a API key")
            st.error("❌ Erro de autenticação na Odds API - verifique a API key")
            return []
        elif r.status_code == 429:
            logger.warning("⚠️ Rate limit excedido na Odds API")
            st.warning("⚠️ Rate limit excedido - aguarde um momento")
            return []
        else:
            logger.warning(f"⚠️ Status code {r.status_code} para {liga_key}")
            return []
    except Exception as e:
        logger.error(f"❌ Erro na requisição para {liga_key}: {e}")
        st.warning(f"Erro Odds API {liga_key}: {e}")
        return []

def parse_iso_to_datetime(s):
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s2 = s.replace("Z", "+00:00")
        else:
            s2 = s
        return datetime.fromisoformat(s2)
    except Exception:
        try:
            return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None

def extrair_markets_totals(event):
    """
    Retorna um dicionário com os mercados totals encontrados no evento,
    mapeado por ponto (ex: 2.5 => {'over': odds, 'under': odds})
    """
    totals_map = {}
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes", []):
                try:
                    point = float(outcome.get("point")) if outcome.get("point") is not None else None
                except Exception:
                    try:
                        point = float(str(outcome.get("point")))
                    except Exception:
                        point = None
                name = outcome.get("name", "").strip().lower()
                price = outcome.get("price")
                if point is None or price is None:
                    continue
                key = str(point)
                totals_map.setdefault(key, {"overs": [], "unders": []})
                if "over" in name:
                    totals_map[key]["overs"].append(price)
                elif "under" in name:
                    totals_map[key]["unders"].append(price)
    consolidated = {}
    for k, v in totals_map.items():
        if not v["overs"] or not v["unders"]:
            continue
        avg_over = sum(v["overs"]) / len(v["overs"])
        avg_under = sum(v["unders"]) / len(v["unders"])
        consolidated[k] = {"over": avg_over, "under": avg_under}
    return consolidated

def implied_prob_from_over_under(over_odds, under_odds):
    """
    Calcula probabilidade implícita de Over usando over_odds e under_odds (decimal).
    Faz normalização para retirar o vigorish (bookmaker margin).
    """
    try:
        inv_over = 1.0 / float(over_odds)
        inv_under = 1.0 / float(under_odds)
        total = inv_over + inv_under
        if total <= 0:
            return 0.5
        prob_over = inv_over / total
        return max(0.0, min(1.0, prob_over))
    except Exception:
        return 0.5

# =============================
# Estatística / Poisson helpers
# =============================
def poisson_cdf(k, lam):
    try:
        s = 0.0
        for i in range(0, k+1):
            s += (lam**i) / math.factorial(i)
        return math.exp(-lam) * s
    except Exception:
        return 0.5

def prob_over_k(estimativa, threshold):
    try:
        if threshold == 1.5:
            k = 1
        elif threshold == 2.5:
            k = 2
        elif threshold == 3.5:
            k = 3
        else:
            k = int(math.floor(threshold))
        p = 1 - poisson_cdf(k, estimativa)
        return max(0.0, min(1.0, p))
    except Exception:
        return 0.5

def confidence_from_prob(prob):
    try:
        conf = 50 + (prob - 0.5) * 100
        conf = max(30, min(95, conf))
        return round(conf, 0)
    except Exception:
        return 50

# =============================
# Funções de cálculo (usando odds quando possível)
# =============================
def calcular_estimativas_e_probs_por_jogo_from_odds(event):
    """Calcula estimativas e probabilidades baseadas nas odds"""
    try:
        totals = extrair_markets_totals(event)
        estimativa = 2.5
        probs = {"1.5": None, "2.5": None, "3.5": None}
        confs = {"1.5": 30, "2.5": 30, "3.5": 30}

        if "2.5" in totals:
            estimativa = 2.5

        for point in ["1.5", "2.5", "3.5"]:
            if point in totals:
                over_odds = totals[point]["over"]
                under_odds = totals[point]["under"]
                prob = implied_prob_from_over_under(over_odds, under_odds)
                probs[point] = prob
                confs[point] = confidence_from_prob(prob)

        # Estimativas para pontos faltantes
        if probs["1.5"] is None and probs["2.5"] is not None:
            p25 = probs["2.5"]
            p15_est = min(0.99, p25 + 0.18)
            probs["1.5"] = p15_est
            confs["1.5"] = confidence_from_prob(p15_est)
        if probs["3.5"] is None and probs["2.5"] is not None:
            p25 = probs["2.5"]
            p35_est = max(0.01, p25 - 0.28)
            probs["3.5"] = p35_est
            confs["3.5"] = confidence_from_prob(p35_est)

        # Fallback para valores None
        for k in probs:
            if probs[k] is None:
                probs[k] = 0.5
                confs[k] = confidence_from_prob(0.5)

        return {
            "estimativa": round(estimativa, 2),
            "prob_1_5": round(probs["1.5"] * 100, 1),
            "prob_2_5": round(probs["2.5"] * 100, 1),
            "prob_3_5": round(probs["3.5"] * 100, 1),
            "conf_1_5": confs["1.5"],
            "conf_2_5": confs["2.5"],
            "conf_3_5": confs["3.5"],
        }
    except Exception as e:
        logger.error(f"Erro no cálculo de probabilidades: {e}")
        # Retorna valores padrão em caso de erro
        return {
            "estimativa": 2.5,
            "prob_1_5": 50.0,
            "prob_2_5": 50.0,
            "prob_3_5": 50.0,
            "conf_1_5": 50,
            "conf_2_5": 50,
            "conf_3_5": 50,
        }

# =============================
# Seleção Top3
# =============================
def selecionar_top3_distintos(partidas_info, max_por_faixa=3, prefer_best_fit=True):
    """Seleciona os top 3 jogos distintos para cada faixa de gols"""
    if not partidas_info:
        return [], [], []

    base = list(partidas_info)

    def get_num(d, k):
        v = d.get(k, 0)
        try:
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    def sort_key(match, prob_key):
        prob = get_num(match, prob_key)
        conf = get_num(match, prob_key.replace("prob", "conf"))
        est = get_num(match, "estimativa")
        return (prob, conf, est)

    selected_ids = set()
    selected_teams = set()

    def safe_team_names(m):
        return str(m.get("home", "")).strip(), str(m.get("away", "")).strip()

    def allocate(prefix, other_prefixes):
        nonlocal base, selected_ids, selected_teams
        prob_key = f"prob_{prefix}"
        candidatos = [m for m in base if str(m.get("fixture_id")) not in selected_ids]

        preferred = []
        if prefer_best_fit:
            for m in candidatos:
                cur = get_num(m, prob_key)
                others = [get_num(m, f"prob_{o}") for o in other_prefixes]
                if cur >= max(others):
                    preferred.append(m)

        preferred_sorted = sorted(preferred, key=lambda x: sort_key(x, prob_key), reverse=True)
        remaining = [m for m in candidatos if m not in preferred_sorted]
        remaining_sorted = sorted(remaining, key=lambda x: sort_key(x, prob_key), reverse=True)

        chosen = []

        def try_add_list(lst, respect_teams=True):
            nonlocal chosen, selected_ids, selected_teams
            for m in lst:
                if len(chosen) >= max_por_faixa:
                    break
                fid = str(m.get("fixture_id"))
                if fid in selected_ids:
                    continue
                home, away = safe_team_names(m)
                if respect_teams and (home in selected_teams or away in selected_teams):
                    continue
                chosen.append(m)
                selected_ids.add(fid)
                selected_teams.add(home)
                selected_teams.add(away)

        try_add_list(preferred_sorted, respect_teams=True)
        if len(chosen) < max_por_faixa:
            try_add_list(remaining_sorted, respect_teams=True)
        if len(chosen) < max_por_faixa:
            try_add_list(preferred_sorted + remaining_sorted, respect_teams=False)

        return chosen

    # Ordem: +2.5 primeiro
    top_25 = allocate("2_5", other_prefixes=["1_5", "3_5"])
    top_15 = allocate("1_5", other_prefixes=["2_5", "3_5"])
    top_35 = allocate("3_5", other_prefixes=["2_5", "1_5"])

    logger.info(f"Seleção Top3: +1.5={len(top_15)}, +2.5={len(top_25)}, +3.5={len(top_35)}")
    return top_15, top_25, top_35

# =============================
# Coleta eventos / transformação
# =============================
def coletar_jogos_do_dia_por_ligas(ligas, data_obj: date, regions="eu,us,au"):
    """Coleta jogos do dia para as ligas selecionadas"""
    partidas = []
    total_ligas = len(ligas)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, liga in enumerate(ligas):
        status_text.text(f"Consultando {ALL_LIGAS.get(liga, liga)}... ({idx+1}/{total_ligas})")
        progress_bar.progress((idx + 1) / total_ligas)
        
        eventos = obter_odds_para_liga(liga, regions=regions, markets="totals", odds_format="decimal")
        if not eventos:
            continue
            
        for ev in eventos:
            commence = ev.get("commence_time") or ev.get("commence_time_zone")
            dt = parse_iso_to_datetime(commence)
            if not dt:
                continue
            if dt.date() != data_obj:
                continue
                
            # extrair nomes
            home = ev.get("home_team") or (ev.get("teams") and ev.get("teams")[0])
            away = ev.get("away_team") or (ev.get("teams") and ev.get("teams")[1])
            hora_formatada = dt.strftime("%H:%M") if dt else "??:??"
            
            calc = calcular_estimativas_e_probs_por_jogo_from_odds(ev)
            partidas.append({
                "fixture_id": ev.get("id") or ev.get("id") or f"{liga}_{home}_{away}_{commence}",
                "home": home,
                "away": away,
                "hora": hora_formatada,
                "competicao": ALL_LIGAS.get(liga, liga),
                "estimativa": calc.get("estimativa"),
                "prob_1_5": calc.get("prob_1_5"),
                "prob_2_5": calc.get("prob_2_5"),
                "prob_3_5": calc.get("prob_3_5"),
                "conf_1_5": calc.get("conf_1_5"),
                "conf_2_5": calc.get("conf_2_5"),
                "conf_3_5": calc.get("conf_3_5"),
                "liga_key": liga,
                "raw_event": ev
            })
    
    progress_bar.empty()
    status_text.empty()
    
    logger.info(f"Coletados {len(partidas)} jogos para {data_obj}")
    return partidas

# =============================
# UI Streamlit
# =============================
st.set_page_config(
    page_title="Oddstop - ⚽ Alertas Top3 (Odds API)", 
    layout="wide",
    page_icon="⚽"
)

st.title("⚽ Oddstop — Alertas Top3 por Faixa (+1.5 / +2.5 / +3.5) — Odds API")

# Sidebar: seleção de ligas e configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Configurações de API
    st.subheader("Configurações de API")
    regions_config = st.multiselect(
        "Regiões para odds",
        ["eu", "us", "au", "uk"],
        default=["eu", "us", "au"]
    )
    
    # Limite de requisições
    max_requests = st.slider("Máx. requisições/minuto", 10, 100, 30)
    rate_limiter.calls_per_minute = max_requests
    
    # Modo debug
    debug_mode = st.checkbox("Modo Debug", value=False)
    
    st.markdown("---")
    st.markdown("**Selecione as ligas**")
    liga_options = list(ALL_LIGAS.keys())
    
    # manter seleção em session_state
    if "selected_ligas" not in st.session_state:
        st.session_state.selected_ligas = DEFAULT_PRINCIPAIS.copy()
    
    selected = st.multiselect(
        "Ligas", 
        options=liga_options, 
        format_func=lambda k: ALL_LIGAS[k], 
        default=st.session_state.selected_ligas
    )
    
    # botão para salvar seleção atual na session_state
    if st.button("💾 Salvar seleção de ligas"):
        st.session_state.selected_ligas = selected.copy()
        st.success("Seleção salva para esta sessão.")
    
    st.markdown("---")
    
    # Estatísticas
    st.subheader("📊 Estatísticas")
    top3_salvos = carregar_top3()
    st.write(f"Envios registrados: **{len(top3_salvos)}**")
    
    if debug_mode:
        st.write("🔍 Modo Debug Ativo")
        st.write(f"Ligas selecionadas: {len(selected)}")
        st.write(f"Rate Limit: {max_requests}/min")

# Seleção final usada pelo app
selected_ligas = selected if selected else (st.session_state.get("selected_ligas") or DEFAULT_PRINCIPAIS)

# Abas principais
aba = st.tabs(["⚡ Gerar & Enviar Top3", "📊 Jogos (Odds)", "🎯 Conferência Top3", "🔧 Configurações"])

# ---------- ABA 1: Gerar & Enviar Top3 ----------
with aba[0]:
    st.subheader("🔎 Buscar jogos do dia e enviar Top3 por faixa")
    data_selecionada = st.date_input("📅 Data dos jogos:", value=datetime.today().date())
    hoje_str = data_selecionada.strftime("%Y-%m-%d")

    st.markdown("**Obs:** uso das odds para estimar P(+1.5/+2.5/+3.5). Ajuste as ligas no *sidebar*.")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Buscar jogos e calcular Top3", type="primary", use_container_width=True):
            with st.spinner("Buscando jogos e calculando probabilidades via Odds API..."):
                partidas_info = coletar_jogos_do_dia_por_ligas(selected_ligas, data_selecionada, regions=",".join(regions_config))

                if not partidas_info:
                    st.info("❌ Nenhum jogo encontrado para essa data nas ligas selecionadas (Odds API).")
                else:
                    top_15, top_25, top_35 = selecionar_top3_distintos(partidas_info, max_por_faixa=3)

                    # Exibir resultados
                    st.success(f"✅ {len(partidas_info)} jogos processados. Top3 calculados!")
                    
                    col1a, col2a, col3a = st.columns(3)
                    
                    with col1a:
                        st.write("### 🥇 Top 3 +1.5")
                        if top_15:
                            for t in top_15:
                                st.write(f"**{t['home']} x {t['away']}**")
                                st.write(f"P(+1.5): {t['prob_1_5']}% | Conf: {t['conf_1_5']}%")
                                st.write("---")
                        else:
                            st.info("Nenhum jogo selecionado")
                    
                    with col2a:
                        st.write("### 🥈 Top 3 +2.5")
                        if top_25:
                            for t in top_25:
                                st.write(f"**{t['home']} x {t['away']}**")
                                st.write(f"P(+2.5): {t['prob_2_5']}% | Conf: {t['conf_2_5']}%")
                                st.write("---")
                        else:
                            st.info("Nenhum jogo selecionado")
                    
                    with col3a:
                        st.write("### 🥉 Top 3 +3.5")
                        if top_35:
                            for t in top_35:
                                st.write(f"**{t['home']} x {t['away']}**")
                                st.write(f"P(+3.5): {t['prob_3_5']}% | Conf: {t['conf_3_5']}%")
                                st.write("---")
                        else:
                            st.info("Nenhum jogo selecionado")

                    # Salvar em session_state para possível envio
                    st.session_state.last_top3 = {
                        "top_15": top_15,
                        "top_25": top_25, 
                        "top_35": top_35,
                        "data": hoje_str
                    }
                    
                    with col2:
                        if st.button("📤 Enviar Top3 para Telegram", type="secondary", use_container_width=True):
                        if "last_top3" not in st.session_state:
                            st.warning("⚠️ Primeiro busque os jogos para gerar o Top3")
            else:
                top_15 = st.session_state.last_top3["top_15"]
                top_25 = st.session_state.last_top3["top_25"]
                top_35 = st.session_state.last_top3["top_35"]
                hoje_str = st.session_state.last_top3["data"]

                with st.spinner("Enviando mensagens para Telegram..."):
                    success_count = 0

    

                    # Mensagem +1.5
                    if top_15:
                        msg = f"🔔 *TOP 3 +1.5 GOLS — {hoje_str}*\n\n"
                        for idx, j in enumerate(top_15, start=1):
                            msg += (f"{idx}️⃣ *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n"
                                    f"   • Est: {j['estimativa']:.2f} gols | P(+1.5): *{j['prob_1_5']:.1f}%* | Conf: *{j['conf_1_5']:.0f}%*\n")
                        if enviar_telegram_seguro(msg, TELEGRAM_CHAT_ID) and enviar_telegram_seguro(msg, TELEGRAM_CHAT_ID_ALT2):
                            success_count += 1

                    # Mensagem +2.5
                    if top_25:
                        msg = f"🔔 *TOP 3 +2.5 GOLS — {hoje_str}*\n\n"
                        for idx, j in enumerate(top_25, start=1):
                            msg += (f"{idx}️⃣ *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n"
                                    f"   • Est: {j['estimativa']:.2f} gols | P(+2.5): *{j['prob_2_5']:.1f}%* | Conf: *{j['conf_2_5']:.0f}%*\n")
                        if enviar_telegram_seguro(msg, TELEGRAM_CHAT_ID) and enviar_telegram_seguro(msg, TELEGRAM_CHAT_ID_ALT2):
                            success_count += 1

                    # Mensagem +3.5
                    if top_35:
                        msg = f"🔔 *TOP 3 +3.5 GOLS — {hoje_str}*\n\n"
                        for idx, j in enumerate(top_35, start=1):
                            msg += (f"{idx}️⃣ *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n"
                                    f"   • Est: {j['estimativa']:.2f} gols | P(+3.5): *{j['prob_3_5']:.1f}%* | Conf: *{j['conf_3_5']:.0f}%*\n")
                        if enviar_telegram_seguro(msg, TELEGRAM_CHAT_ID) and enviar_telegram_seguro(msg, TELEGRAM_CHAT_ID_ALT2):
                            success_count += 1

                    # salva o lote Top3 (persistente)
                    if success_count > 0:
                        top3_list = carregar_top3()
                        novo_top = {
                            "data_envio": hoje_str,
                            "hora_envio": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "selected_ligas": selected_ligas,
                            "top_1_5": top_15,
                            "top_2_5": top_25,
                            "top_3_5": top_35
                        }
                        top3_list.append(novo_top)
                        salvar_top3(top3_list)
                        st.success(f"✅ {success_count} mensagens enviadas com sucesso!")
                    else:
                        st.error("❌ Falha ao enviar mensagens para Telegram")

# ---------- ABA 2: Jogos (Odds) ----------
with aba[1]:
    st.subheader("📊 Jogos do dia com Odds (Odds API)")
    data_selecionada2 = st.date_input("📅 Data dos jogos para listar:", value=datetime.today().date(), key="lista_odds")
    
    if st.button("🔍 Listar jogos e odds do dia", key="listar_odds"):
        with st.spinner("Consultando Odds API para listar jogos..."):
            partidas = coletar_jogos_do_dia_por_ligas(selected_ligas, data_selecionada2, regions=",".join(regions_config))
            if not partidas:
                st.info("ℹ️ Nenhum jogo/odds encontrado para essa data nas ligas selecionadas.")
            else:
                st.success(f"✅ {len(partidas)} jogos encontrados")
                
                # Filtros
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    min_prob_15 = st.slider("Mín. P(+1.5)%", 0, 100, 50)
                with col_f2:
                    min_prob_25 = st.slider("Mín. P(+2.5)%", 0, 100, 50)
                with col_f3:
                    min_prob_35 = st.slider("Mín. P(+3.5)%", 0, 100, 50)
                
                partidas_filtradas = [
                    p for p in partidas 
                    if p['prob_1_5'] >= min_prob_15 
                    and p['prob_2_5'] >= min_prob_25 
                    and p['prob_3_5'] >= min_prob_35
                ]
                
                st.write(f"**Jogos filtrados:** {len(partidas_filtradas)}")
                
                for p in partidas_filtradas:
                    with st.expander(f"🏟️ {p['home']} x {p['away']} — {p['competicao']} — {p['hora']} BRT"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("P(+1.5)", f"{p['prob_1_5']}%", f"Conf: {p['conf_1_5']}%")
                        with col2:
                            st.metric("P(+2.5)", f"{p['prob_2_5']}%", f"Conf: {p['conf_2_5']}%")
                        with col3:
                            st.metric("P(+3.5)", f"{p['prob_3_5']}%", f"Conf: {p['conf_3_5']}%")
                        st.write(f"**Estimativa:** {p['estimativa']} gols")

# ---------- ABA 3: Conferência Top 3 ----------
with aba[2]:
    st.subheader("🎯 Conferência dos Top 3 enviados")
    top3_salvos = carregar_top3()

    if not top3_salvos:
        st.info("ℹ️ Nenhum Top 3 registrado ainda. Gere e envie um Top 3 na aba 'Gerar & Enviar Top3'.")
    else:
        st.write(f"✅ Total de envios registrados: {len(top3_salvos)}")
        options = [f"{idx+1} - {t['data_envio']} ({t['hora_envio']})" for idx, t in enumerate(top3_salvos)]
        seletor = st.selectbox("Selecione o lote Top3 para conferir:", options, index=len(options)-1)
        idx_selecionado = options.index(seletor)
        lote = top3_salvos[idx_selecionado]
        
        st.markdown(f"### Lote selecionado — Envio: **{lote['data_envio']}** às **{lote['hora_envio']}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("+1.5 Gols", len(lote.get('top_1_5', [])))
        with col2:
            st.metric("+2.5 Gols", len(lote.get('top_2_5', [])))
        with col3:
            st.metric("+3.5 Gols", len(lote.get('top_3_5', [])))
        
        st.markdown("---")

        if st.button("🔄 Rechecar resultados e enviar conferência"):
            with st.spinner("Conferindo resultados via Odds API..."):
                def processar_lista_e_mandar(lista_top, threshold_label):
                    detalhes_local = []
                    lines_for_msg = []
                    for j in lista_top:
                        liga = j.get("liga_key") or j.get("liga_id")
                        eventos = obter_odds_para_liga(liga, regions=",".join(regions_config), markets="totals")
                        found = None
                        for ev in eventos:
                            ht = ev.get("home_team") or (ev.get("teams") and ev.get("teams")[0])
                            at = ev.get("away_team") or (ev.get("teams") and ev.get("teams")[1])
                            if ht == j.get("home") and at == j.get("away"):
                                found = ev
                                break
                        if not found:
                            lines_for_msg.append(f"🏟️ {j.get('home')} x {j.get('away')} — _sem resultado disponível_")
                            detalhes_local.append({"home": j.get("home"), "away": j.get("away"), "aposta": f"+{threshold_label}", "status": "Não encontrado / sem resultado"})
                            continue
                        lines_for_msg.append(f"🏟️ {found.get('home_team')} x {found.get('away_team')} — _Odds confirmadas — sem placar via Odds API_")
                        detalhes_local.append({"home": found.get("home_team"), "away": found.get("away_team"), "aposta": f"+{threshold_label}", "status": "Odds encontradas — sem placar"})
                    
                    header = f"✅ RESULTADOS - CONFERÊNCIA +{threshold_label}\n(Lote: {lote['data_envio']})\n\n"
                    body = "\n".join(lines_for_msg) if lines_for_msg else "_Nenhum jogo para conferir nesta faixa no lote selecionado._"
                    msg = header + body
                    
                    success = enviar_telegram_seguro(msg, TELEGRAM_CHAT_ID) and enviar_telegram_seguro(msg, TELEGRAM_CHAT_ID_ALT2)
                    return detalhes_local, success

                resultados = []
                for label, lista in [("1.5", lote.get("top_1_5", [])), ("2.5", lote.get("top_2_5", [])), ("3.5", lote.get("top_3_5", []))]:
                    detalhes, success = processar_lista_e_mandar(lista, label)
                    resultados.append((label, detalhes, success))

                st.success("✅ Mensagens de conferência processadas!")
                for label, detalhes, success in resultados:
                    st.write(f"**+{label} Gols:** {'✅ Enviado' if success else '❌ Falha'} ({len(detalhes)} jogos)")

        if st.button("🔎 Rechecar odds aqui (sem enviar Telegram)"):
            with st.spinner("Conferindo odds localmente..."):
                for label, lista in [("1.5", lote.get("top_1_5", [])), ("2.5", lote.get("top_2_5", [])), ("3.5", lote.get("top_3_5", []))]:
                    st.write(f"### Conferência +{label}")
                    if not lista:
                        st.info("Nenhum jogo nesta faixa")
                        continue
                    
                    for j in lista:
                        liga = j.get("liga_key") or j.get("liga_id")
                        eventos = obter_odds_para_liga(liga, regions=",".join(regions_config), markets="totals")
                        found = None
                        for ev in eventos:
                            ht = ev.get("home_team") or (ev.get("teams") and ev.get("teams")[0])
                            at = ev.get("away_team") or (ev.get("teams") and ev.get("teams")[1])
                            if ht == j.get("home") and at == j.get("away"):
                                found = ev
                                break
                        if not found:
                            st.warning(f"❌ {j.get('home')} x {j.get('away')} — Odds/Evento não encontrado")
                            continue
                        
                        calc = calcular_estimativas_e_probs_por_jogo_from_odds(found)
                        st.success(f"✅ {found.get('home_team')} x {found.get('away_team')}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"P(+1.5): {calc.get('prob_1_5')}%")
                        with col2:
                            st.write(f"P(+2.5): {calc.get('prob_2_5')}%")
                        with col3:
                            st.write(f"P(+3.5): {calc.get('prob_3_5')}%")

        if st.button("📥 Exportar lote selecionado (.json)"):
            nome_arquivo = f"relatorio_top3_{lote['data_envio']}_{lote['hora_envio'].replace(':','-').replace(' ','_')}.json"
            try:
                with open(nome_arquivo, "w", encoding="utf-8") as f:
                    json.dump(lote, f, ensure_ascii=False, indent=2)
                st.success(f"✅ Lote exportado: {nome_arquivo}")
            except Exception as e:
                st.error(f"❌ Erro ao exportar: {e}")

# ---------- ABA 4: Configurações ----------
with aba[3]:
    st.subheader("🔧 Configurações Avançadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Configurações de API**")
        st.info(f"Odds API Key: {'✅ Configurada' if ODDS_API_KEY else '❌ Não configurada'}")
        st.info(f"Telegram Token: {'✅ Configurado' if TELEGRAM_TOKEN else '❌ Não configurado'}")
        
        if st.button("🔄 Testar Conexão Odds API"):
            with st.spinner("Testando conexão..."):
                test_events = obter_odds_para_liga("soccer_epl", regions="eu", markets="totals")
                if test_events:
                    st.success("✅ Conexão com Odds API funcionando!")
                    st.write(f"Eventos de teste: {len(test_events)}")
                else:
                    st.error("❌ Falha na conexão com Odds API")
    
    with col2:
        st.write("**Gerenciamento de Dados**")
        
        if st.button("🗑️ Limpar Cache de Dados"):
            try:
                if os.path.exists(TOP3_PATH):
                    os.remove(TOP3_PATH)
                if os.path.exists(ALERTAS_PATH):
                    os.remove(ALERTAS_PATH)
                st.success("✅ Cache limpo com sucesso!")
            except Exception as e:
                st.error(f"❌ Erro ao limpar cache: {e}")
        
        if st.button("📊 Estatísticas do Sistema"):
            st.write("**Arquivos de Dados:**")
            st.write(f"- top3.json: {os.path.exists(TOP3_PATH)}")
            st.write(f"- alertas.json: {os.path.exists(ALERTAS_PATH)}")
            st.write(f"- app.log: {os.path.exists('app.log')}")
            
            if os.path.exists('app.log'):
                with open('app.log', 'r') as f:
                    lines = f.readlines()
                    st.write(f"Linhas no log: {len(lines)}")
    
    st.write("---")
    st.write("**Logs Recentes**")
    if st.button("📋 Mostrar Últimos Logs"):
        if os.path.exists('app.log'):
            with open('app.log', 'r') as f:
                lines = f.readlines()
                last_lines = lines[-20:]  # Últimas 20 linhas
                st.text_area("Logs", "\n".join(last_lines), height=300)
        else:
            st.info("Arquivo de log não encontrado")

# Footer
st.markdown("---")
st.markdown(
    "⚽ **Oddstop Alertas** - Desenvolvido com Streamlit e Odds API | "
    "[The Odds API](https://the-odds-api.com/)"
)
