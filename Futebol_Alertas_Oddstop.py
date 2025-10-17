# Futebol_Alertas_Oddstop.py
# Versão: 1.1 - Cache persistente + Rate limiter + Modo "economia máxima"
# Uso exclusivo da Football Highlights API (RapidAPI)
# Rodar: streamlit run Futebol_Alertas_Oddstop.py

import streamlit as st
import requests
import os
from datetime import datetime, timedelta
import time
import math
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional
import threading
import pickle
from pathlib import Path

load_dotenv()

# =========================
# CONFIGURAÇÕES (coloque no seu .env)
# =========================
RAPID_API_KEY = os.getenv("RAPID_API_KEY", "COLE_SUA_CHAVE_AQUI")
RAPID_API_HOST = os.getenv("RAPID_API_HOST", "football-highlights-api.p.rapidapi.com")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "COLE_SEU_BOT_TOKEN_AQUI")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "COLE_SEU_CHAT_ID_AQUI")

# Cache / Rate limit configs (padrões que você pode ajustar no .env)
CACHE_FILE = os.getenv("CACHE_FILE", "api_cache.pickle")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))  # 10 minutos default
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))  # 60 req/min default
RATE_BURST = int(os.getenv("RATE_BURST", "10"))  # burst permitido
MODE_ECONOMIA_MAXIMA = True  # modo solicitado: usa cache expirado em caso de rate limit / erro

# Arquivo histórico simples
HISTORY_CSV = os.getenv("HISTORY_CSV", "alert_history.csv")

# =========================
# CONSTANTES / HEADERS
# =========================
BASE_URL = f"https://{RAPID_API_HOST}"
HEADERS = {
    "x-rapidapi-key": RAPID_API_KEY,
    "x-rapidapi-host": RAPID_API_HOST,
    "Content-Type": "application/json"
}

# =========================
# CACHE EM DISCO (thread-safe) - suporta "stale" (dados expirados)
# =========================
class DiskCache:
    def __init__(self, path: str, default_ttl: int = 600):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.default_ttl = default_ttl
        self._data = {}  # key -> (expire_ts, value)
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with self.lock:
                    with open(self.path, "rb") as f:
                        self._data = pickle.load(f)
            except Exception:
                self._data = {}

    def _save(self):
        try:
            with self.lock:
                with open(self.path, "wb") as f:
                    pickle.dump(self._data, f)
        except Exception as e:
            print("Cache save error:", e)

    def make_key(self, url: str, params: Optional[dict]):
        key = url
        if params:
            items = tuple(sorted((k, json.dumps(v, sort_keys=True, default=str)) for k, v in params.items()))
            key = f"{url}|{items}"
        return key

    def get(self, url: str, params: Optional[dict] = None, allow_stale: bool = False):
        key = self.make_key(url, params)
        with self.lock:
            entry = self._data.get(key)
            if not entry:
                return None
            expire_ts, value = entry
            now = time.time()
            if expire_ts is None:
                return value
            if expire_ts > now:
                return value
            # expirado
            if allow_stale:
                # devolve o valor mesmo expirado (modo economia máxima)
                return value
            else:
                # remove expirado
                del self._data[key]
                self._save()
                return None

    def set(self, url: str, params: Optional[dict], value: Any, ttl: Optional[int] = None):
        key = self.make_key(url, params)
        ttl = ttl if ttl is not None else self.default_ttl
        expire_ts = None if ttl <= 0 else time.time() + ttl
        with self.lock:
            self._data[key] = (expire_ts, value)
            # salvar imediato para persistência
            self._save()

    def invalidate(self, url: str, params: Optional[dict] = None):
        key = self.make_key(url, params)
        with self.lock:
            if key in self._data:
                del self._data[key]
                self._save()

# inicializar cache global
disk_cache = DiskCache(CACHE_FILE, default_ttl=CACHE_TTL_SECONDS)

# =========================
# RATE LIMITER (Token Bucket)
# =========================
class TokenBucket:
    def __init__(self, rate_per_minute: int, burst: int):
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.fill_rate = float(rate_per_minute) / 60.0  # tokens per second
        self.timestamp = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.timestamp
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.timestamp = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

_rate_buckets = {}
_rate_lock = threading.Lock()

def get_bucket_for(key: str) -> TokenBucket:
    with _rate_lock:
        if key not in _rate_buckets:
            _rate_buckets[key] = TokenBucket(RATE_LIMIT_PER_MINUTE, RATE_BURST)
        return _rate_buckets[key]

# =========================
# safe_request integrado: cache + rate limit + modo economia máxima
# =========================
def safe_request(path: str, params: dict = None, timeout: int = 10, cache_ttl: Optional[int] = None, use_cache: bool = True) -> Optional[dict]:
    """
    Faz requisição com:
      - cache em disco com TTL (persistente)
      - token-bucket rate limiter
      - em modo 'economia máxima' usa cache expirado se rate limit/pedido falhar
    Parâmetros:
      - cache_ttl: se None, usa CACHE_TTL_SECONDS; se <=0, não cacheia
      - use_cache: False para desabilitar cache nesta chamada
    """
    url = BASE_URL.rstrip("/") + path
    params_safe = params or {}

    # Se cache ativado, tentar retornar do cache (válido)
    if use_cache and (cache_ttl is None or cache_ttl > 0):
        cached = disk_cache.get(url, params_safe, allow_stale=False)
        if cached is not None:
            return cached

    # Rate limiting por host (pode mudar para por-path se desejar)
    bucket_key = RAPID_API_HOST
    bucket = get_bucket_for(bucket_key)

    # Tentar consumir token; se não houver token, no modo "economia máxima" devolve cache expired (se disponível)
    if not bucket.consume(1.0):
        # sem token agora
        if MODE_ECONOMIA_MAXIMA and use_cache:
            stale = disk_cache.get(url, params_safe, allow_stale=True)
            if stale is not None:
                # log leve
                print(f"[safe_request] Rate limit ativo. Usando cache expirado para {path}")
                return stale
        # se não temos cache expirado, devolve None (evita request)
        print(f"[safe_request] Rate limit ativo e sem cache stale para {path}. Abortando request.")
        return None

    # Se token disponível, fazer requisição (com tentativa e fallback)
    try:
        resp = requests.get(url, headers=HEADERS, params=params_safe, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # salvar no cache se aplicável
        if use_cache and (cache_ttl is None or cache_ttl > 0):
            ttl_to_use = cache_ttl if cache_ttl is not None else CACHE_TTL_SECONDS
            disk_cache.set(url, params_safe, data, ttl=ttl_to_use)
        return data
    except Exception as e:
        print(f"[safe_request] Erro na requisição {path}: {e}")
        # Em erro, retornar cache expirado se modo economia máxima estiver ativo
        if MODE_ECONOMIA_MAXIMA and use_cache:
            stale = disk_cache.get(url, params_safe, allow_stale=True)
            if stale is not None:
                print(f"[safe_request] Erro na requisição; retornando cache expirado para {path}")
                return stale
        return None

# =========================
# FUNÇÕES DA API (usam safe_request)
# =========================
def get_matches(date_str: str, limit: int = 500, countryCode: Optional[str]=None) -> List[dict]:
    params = {"date": date_str, "limit": limit}
    if countryCode:
        params["countryCode"] = countryCode
    data = safe_request("/matches", params, cache_ttl=CACHE_TTL_SECONDS, use_cache=True)
    if not data:
        return []
    return data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []

@st.cache_data(ttl=3600)
def get_leagues_cached(limit: int = 200, countryName: Optional[str]=None) -> List[dict]:
    params = {"limit": limit}
    if countryName:
        params["countryName"] = countryName
    data = safe_request("/leagues", params, cache_ttl=3600, use_cache=True)
    if not data:
        return []
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    return data if isinstance(data, list) else []

def get_team_statistics(team_id: int, from_date: str) -> List[dict]:
    path = f"/teams/statistics/{team_id}"
    params = {"fromDate": from_date}
    data = safe_request(path, params, cache_ttl=CACHE_TTL_SECONDS, use_cache=True)
    if not data:
        return []
    return data if isinstance(data, list) else []

def get_last_five_games(team_id: int) -> List[dict]:
    params = {"teamId": team_id}
    data = safe_request("/last-five-games", params, cache_ttl=CACHE_TTL_SECONDS, use_cache=True)
    if not data:
        return []
    return data if isinstance(data, list) else []

def get_head2head(teamA: int, teamB: int) -> List[dict]:
    params = {"teamIdOne": teamA, "teamIdTwo": teamB}
    data = safe_request("/head-2-head", params, cache_ttl=CACHE_TTL_SECONDS, use_cache=True)
    if not data:
        return []
    return data if isinstance(data, list) else []

def get_match_odds(match_id: int, limit: int = 5) -> List[dict]:
    params = {"matchId": match_id, "limit": limit}
    # odds são dinâmicos — não cacheamos por padrão
    data = safe_request("/odds", params, cache_ttl=0, use_cache=False)
    if not data:
        return []
    return data if isinstance(data, list) else []

def get_match_by_id(match_id: int) -> Optional[dict]:
    data = safe_request(f"/matches/{match_id}", {}, cache_ttl=CACHE_TTL_SECONDS, use_cache=True)
    if not data:
        return None
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return None

# =========================
# Funções de cálculo de tendência +1.5
# =========================
def average_goals_from_matches(matches: List[dict], team_id: int) -> Tuple[float, int]:
    scored = []
    for m in matches:
        try:
            # adaptações várias estruturas possíveis
            home = m.get("homeTeam", {}).get("id") or m.get("home", {}).get("id") or m.get("homeTeam", {}).get("teamId")
            away = m.get("awayTeam", {}).get("id") or m.get("away", {}).get("id") or m.get("awayTeam", {}).get("teamId")
            score = m.get("score") or {}
            full = score.get("fullTime") or {}
            home_goals = full.get("home")
            away_goals = full.get("away")
            if home_goals is None or away_goals is None:
                home_goals = m.get("homeScore", home_goals)
                away_goals = m.get("awayScore", away_goals)
            if home is None or away is None or home_goals is None or away_goals is None:
                continue
            if int(team_id) == int(home):
                scored.append(home_goals)
            elif int(team_id) == int(away):
                scored.append(away_goals)
        except Exception:
            continue
    if not scored:
        return 0.0, 0
    return sum(scored) / len(scored), len(scored)

def average_conceded_from_matches(matches: List[dict], team_id: int) -> Tuple[float, int]:
    conceded = []
    for m in matches:
        try:
            home = m.get("homeTeam", {}).get("id") or m.get("home", {}).get("id")
            away = m.get("awayTeam", {}).get("id") or m.get("away", {}).get("id")
            score = m.get("score") or {}
            full = score.get("fullTime") or {}
            home_goals = full.get("home")
            away_goals = full.get("away")
            if home_goals is None or away_goals is None:
                home_goals = m.get("homeScore", home_goals)
                away_goals = m.get("awayScore", away_goals)
            if home is None or away is None or home_goals is None or away_goals is None:
                continue
            if int(team_id) == int(home):
                conceded.append(away_goals)
            elif int(team_id) == int(away):
                conceded.append(home_goals)
        except Exception:
            continue
    if not conceded:
        return 0.0, 0
    return sum(conceded) / len(conceded), len(conceded)

def estimate_total_goals(home_team_id: int, away_team_id: int, date_from_for_stats: str) -> Tuple[float, float]:
    # Puxa last five e h2h (essas chamadas usam cache por padrão)
    last_home = get_last_five_games(home_team_id)
    last_away = get_last_five_games(away_team_id)

    home_scored_avg, n_hs = average_goals_from_matches(last_home, home_team_id)
    home_conceded_avg, n_hc = average_conceded_from_matches(last_home, home_team_id)

    away_scored_avg, n_as = average_goals_from_matches(last_away, away_team_id)
    away_conceded_avg, n_ac = average_conceded_from_matches(last_away, away_team_id)

    h2h = get_head2head(home_team_id, away_team_id)
    h2h_home_scored, n_h2h_hs = average_goals_from_matches(h2h, home_team_id)
    h2h_away_scored, n_h2h_as = average_goals_from_matches(h2h, away_team_id)

    try:
        part_home = (home_scored_avg + (away_conceded_avg if away_conceded_avg else h2h_away_scored)) / (2 if (home_scored_avg and (away_conceded_avg or h2h_away_scored)) else 1)
    except Exception:
        part_home = home_scored_avg or h2h_home_scored or 0.0

    try:
        part_away = (away_scored_avg + (home_conceded_avg if home_conceded_avg else h2h_home_scored)) / (2 if (away_scored_avg and (home_conceded_avg or h2h_home_scored)) else 1)
    except Exception:
        part_away = away_scored_avg or h2h_away_scored or 0.0

    h2h_total_avg = 0.0
    if isinstance(h2h, list) and h2h:
        totals = []
        for m in h2h:
            full = m.get("score", {}).get("fullTime", {})
            h = full.get("home"); a = full.get("away")
            if h is not None and a is not None:
                totals.append(h + a)
        if totals:
            h2h_total_avg = sum(totals)/len(totals)

    est_from_last5 = part_home + part_away
    est_from_h2h = h2h_total_avg or 0.0
    baseline = 0.0
    count_baseline = 0
    for v in [home_scored_avg, away_scored_avg, home_conceded_avg, away_conceded_avg]:
        if v:
            baseline += v
            count_baseline += 1
    baseline = baseline / count_baseline if count_baseline else 0.0

    estimated_total = 0.5 * est_from_last5 + 0.35 * est_from_h2h + 0.15 * baseline

    data_points = (n_hs + n_hc + n_as + n_ac + n_h2h_hs + n_h2h_as)
    confidence = min(95, 30 + data_points * 8)
    if h2h_total_avg:
        diff = abs(est_from_last5 - h2h_total_avg)
        if diff < 0.4:
            confidence += 8
        elif diff < 0.8:
            confidence += 5

    confidence = max(5, min(95, confidence))
    return round(estimated_total, 2), round(confidence, 0)

# =========================
# TELEGRAM
# =========================
def send_telegram_message(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or "COLE" in TELEGRAM_BOT_TOKEN:
        st.warning("Token do Telegram não configurado. Mensagem NÃO enviada.")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Erro ao enviar Telegram: {e}")
        return False

# =========================
# HISTÓRICO SIMPLES
# =========================
def append_history(row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(HISTORY_CSV):
        df.to_csv(HISTORY_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(HISTORY_CSV, index=False)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Futebol Alertas - Oddstop (RapidAPI)", layout="wide")

st.title("⚽ Futebol Alertas — Oddstop (com Cache & Rate Limiter)")
st.markdown(
    """
    App de alertas com Streamlit. Usa **somente** a Football Highlights API (RapidAPI).
    - Cache persistente em disco com TTL configurável (padrão 10 min).
    - Rate limiter token-bucket (padrão 60 req/min, burst 10).
    - Modo 'economia máxima' ativo: se o rate limit impedir requisição, o app usa dados expirados do cache (se houver).
    """
)

# Sidebar: configuração
with st.sidebar:
    st.header("Configurações")
    api_key_show = st.text_input("RAPID API Key (override .env)", value="" , type="password")
    if api_key_show:
        HEADERS["x-rapidapi-key"] = api_key_show
    st.write("Host:", RAPID_API_HOST)
    TELEGRAM_OK = st.checkbox("Habilitar envio Telegram (use .env para credenciais)", value=True)
    st.markdown("---")
    st.write("Histórico salvo em:", HISTORY_CSV)
    st.markdown("")
    st.write("Cache file:", CACHE_FILE)
    st.write(f"Cache TTL (s): {CACHE_TTL_SECONDS}")
    st.write(f"Rate limit: {RATE_LIMIT_PER_MINUTE} req/min (burst {RATE_BURST})")
    st.write("Modo economia máxima: ON (usa cache expirado se necessário)")

# Main controls
col1, col2, col3 = st.columns([2,2,1])
with col1:
    chosen_date = st.date_input("Escolha a data das partidas", value=datetime.utcnow().date())
with col2:
    country_filter = st.text_input("Filtrar por país (ISO code, ex: BR) - opcional", value="")
with col3:
    limit_matches = st.number_input("Máx jogos a buscar", min_value=10, max_value=1000, value=300, step=10)

st.markdown("### Filtros de liga")
all_leagues = get_leagues_cached(limit=200)
league_names = []
league_map = {}
for l in all_leagues:
    name = l.get("name") or l.get("leagueName") or l.get("title") or "Unknown"
    league_names.append(name)
    league_map[name] = l

selected_leagues = st.multiselect("Selecione ligas (vazio = todas)", options=sorted(set(league_names)), default=[])

if st.button("Buscar partidas"):
    with st.spinner("Buscando partidas na API (usando cache/rate limiter)..."):
        date_str = chosen_date.strftime("%Y-%m-%d")
        matches = get_matches(date_str, limit=int(limit_matches), countryCode=country_filter or None)
        if not matches:
            st.warning("Nenhuma partida retornada (verifique filtros / cache).")
        else:
            st.success(f"{len(matches)} partidas encontradas (brutas).")
            # Filtrar por ligas se necessário
            if selected_leagues:
                filtered = []
                for m in matches:
                    league_info = m.get("league") or m.get("competition") or {}
                    lname = league_info.get("name") or league_info.get("leagueName") or league_info.get("title")
                    if lname in selected_leagues:
                        filtered.append(m)
                matches = filtered
                st.info(f"{len(matches)} partidas após filtro de ligas.")
        # Mostrar e analisar
        if matches:
            rows = []
            top_alerts = []
            for m in matches:
                match_id = m.get("id") or m.get("matchId") or m.get("match_id")
                league = m.get("league") or m.get("competition") or {}
                league_name = league.get("name") or league.get("leagueName") or league.get("title") or "Liga"
                home = m.get("homeTeam") or m.get("home") or {}
                away = m.get("awayTeam") or m.get("away") or {}
                home_name = home.get("name") or home.get("teamName") or home.get("title") or "Home"
                away_name = away.get("name") or away.get("teamName") or away.get("title") or "Away"
                home_id = home.get("id") or home.get("teamId") or None
                away_id = away.get("id") or away.get("teamId") or None
                kickoff_raw = m.get("commence_time") or m.get("utcDate") or m.get("startDate") or m.get("start_time")
                kickoff_display = kickoff_raw or "Unknown"

                estimated_total, confidence = 1.4, 30
                if home_id and away_id:
                    est = estimate_total_goals(int(home_id), int(away_id), (chosen_date - timedelta(days=365)).strftime("%Y-%m-%d"))
                    if est:
                        estimated_total, confidence = est

                trend_flag = estimated_total > 1.5
                rows.append({
                    "match_id": match_id,
                    "league": league_name,
                    "home": home_name,
                    "away": away_name,
                    "kickoff": kickoff_display,
                    "est_total": estimated_total,
                    "confidence": confidence,
                    "trend_+1.5": "SIM" if trend_flag else "NÃO"
                })
                if trend_flag and confidence >= 50:
                    top_alerts.append(rows[-1])

            df = pd.DataFrame(rows)
            st.subheader("Partidas encontradas")
            st.dataframe(df[["kickoff","league","home","away","est_total","confidence","trend_+1.5"]].sort_values(by=["confidence"], ascending=False))
            st.markdown("---")
            st.subheader("Top alertas (+1.5) — Alta confiança")
            if top_alerts:
                top_df = pd.DataFrame(top_alerts).sort_values(by="confidence", ascending=False)
                st.dataframe(top_df[["kickoff","league","home","away","est_total","confidence"]])
            else:
                st.info("Nenhuma partida com confiança alta encontrada.")

            if st.button("Enviar alertas (Telegram)"):
                sent = 0
                for a in top_alerts:
                    text = (
                        f"📊 TOP JOGO\n\n"
                        f"🏟️ <b>{a['home']} vs {a['away']}</b>\n"
                        f"⚽ Tendência: +1.5 Gols | Est.: {a['est_total']} | Conf.: {int(a['confidence'])}%\n"
                        f"🕒 {a['kickoff']} | {a['league']}\n"
                    )
                    ok = send_telegram_message(text) if TELEGRAM_OK else False
                    append_history({
                        "timestamp": datetime.utcnow().isoformat(),
                        "match_id": a.get("match_id"),
                        "home": a.get("home"),
                        "away": a.get("away"),
                        "est_total": a.get("est_total"),
                        "confidence": a.get("confidence"),
                        "sent": ok
                    })
                    if ok:
                        sent += 1
                        time.sleep(0.4)
                st.success(f"{sent} alertas enviados para o Telegram." if sent else "Nenhum alerta enviado (verifique credenciais Telegram).")

# Mostrar histórico
st.markdown("---")
st.header("Histórico de alertas (local)")
if os.path.exists(HISTORY_CSV):
    try:
        hist = pd.read_csv(HISTORY_CSV)
        st.dataframe(hist.sort_values(by="timestamp", ascending=False).head(200))
    except Exception as e:
        st.error(f"Erro ao ler histórico: {e}")
else:
    st.info("Nenhum histórico encontrado ainda.")

# UTILIDADES ADICIONAIS
st.markdown("---")
st.write("### Utilitários")
colA, colB = st.columns(2)
with colA:
    match_id_input = st.text_input("Conferir partida por ID (match_id) - opcional")
    if st.button("Checar resultado desta partida"):
        if not match_id_input:
            st.warning("Informe o match_id")
        else:
            match_info = get_match_by_id(match_id_input)
            if not match_info:
                st.error("Partida não encontrada.")
            else:
                st.json(match_info)
with colB:
    st.write("Exportar histórico")
    if st.button("Exportar CSV completo"):
        if os.path.exists(HISTORY_CSV):
            with open(HISTORY_CSV, "rb") as f:
                st.download_button("Download histórico CSV", data=f, file_name=HISTORY_CSV, mime="text/csv")
        else:
            st.info("Sem histórico para exportar.")

st.markdown("**Observações técnicas:** O cálculo de tendência +1.5 usa heurísticas combinando últimos 5 jogos, H2H e estatísticas de time. Ajuste pesos e thresholds na função `estimate_total_goals` se desejar mais agressividade/recorte.")

st.markdown("---")
st.caption("Desenvolvido para usar exclusivamente a Football Highlights API (RapidAPI). Insira RAPID_API_KEY e credenciais do Telegram no arquivo .env antes de usar o envio de mensagens.")

# FIM DO ARQUIVO
