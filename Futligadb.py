# Futebol_Alertas_EliteMaster.py
"""
Elite Master — Futebol Alertas (Clean Professional)
Fonte: OpenLigaDB
Mensagens: Estilo B (Premium / VIP)
Estrutura: abas (Dashboard, Jogos do Dia, Top 3 Alertas, Histórico/Conferência, Configurações)
Uso: streamlit run Futebol_Alertas_EliteMaster.py
"""

import streamlit as st
from datetime import datetime, timedelta, date
import requests
import os
import json
import math
import time
import logging
from functools import lru_cache, wraps
from typing import List, Dict, Any, Optional, Tuple

# ----------------------------
# Logging & page
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EliteMaster")

st.set_page_config(page_title="🏆 ELITE MASTER — Alertas Top3", layout="wide")

# ----------------------------
# CSS - Clean Professional theme
# ----------------------------
st.markdown(
    """
    <style>
    /* Background / cards */
    .reportview-container, .main, .block-container {
        background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 100%);
        color: #0b2545;
    }
    .card {
        background: #ffffff;
        padding: 14px;
        border-radius: 10px;
        border: 1px solid rgba(11,37,69,0.06);
        box-shadow: 0 4px 20px rgba(11,37,69,0.04);
        margin-bottom: 12px;
    }
    .header-title { font-size:20px; font-weight:700; color:#072146; }
    .muted { color: #6b7a90; font-size:13px; }
    .btn-primary > button { background-color: #0b66c3; color: white; border-radius:6px; }
    .small { font-size:13px; color:#44596b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Constants & paths
# ----------------------------
OPENLIGA_BASE = "https://api.openligadb.de"

# Map of leagues (you can expand)
LEAGAS_MAP = {
    "Bundesliga (Alemanha)": "bl1",
    "2. Bundesliga (Alemanha)": "bl2",
    "DFB-Pokal (Alemanha)": "dfb",
    "Brasileirão Série A": "br1",  # placeholder key for mapping; OpenLiga may not have BR - kept for UI
    "Brasileirão Série B": "br2",
    "LaLiga (Espanha)": "laliga",
    "Libertadores": "libertadores"
}

ALERTAS_PATH = "alertas.json"
TOP3_PATH = "top3.json"

# ----------------------------
# Persistence helpers
# ----------------------------
def carregar_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Erro ler {path}: {e}")
            return default
    return default

def salvar_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def carregar_alertas():
    return carregar_json(ALERTAS_PATH, {})

def salvar_alertas(obj):
    salvar_json(ALERTAS_PATH, obj)

def carregar_top3():
    return carregar_json(TOP3_PATH, [])

def salvar_top3(lista):
    salvar_json(TOP3_PATH, lista)

# ----------------------------
# Retries decorator
# ----------------------------
def with_retries(max_attempts=3, backoff=0.6):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts+1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logger.warning(f"Attempt {attempt}/{max_attempts} failed for {fn.__name__}: {e}")
                    time.sleep(backoff * attempt)
            logger.error(f"All attempts failed for {fn.__name__}: {last_exc}")
            raise last_exc
        return wrapper
    return deco

# ----------------------------
# Telegram formatting (Modelo B - Premium / VIP)
# ----------------------------
def montar_mensagem_alerta_premium(home: str, away: str, faixa: str, prob_pct: float, estimativa: float, liga: str, hora_brt: str) -> str:
    faixa_text = faixa if faixa.startswith("+") else f"+{faixa}"
    msg = (
        "🔥 *ALERTA ELITE MASTER* 🔥\n\n"
        f"🏟️ *{home} x {away}*\n"
        f"⚽ Tendência: *{faixa_text} Gols*\n"
        f"📈 Probabilidade: *{prob_pct:.0f}%*\n"
        f"💰 Estimativa: *{estimativa:.2f}*  |  {liga}  |  {hora_brt} BRT\n"
    )
    return msg

def montar_mensagem_resultado_premium(home: str, away: str, faixa: str, score: str, resultado: str) -> str:
    msg = (
        "📊 *RESULTADO CONFERIDO*\n\n"
        f"🏟️ *{home} x {away}*\n"
        f"⚽ Tendência: *+{faixa} Gols*\n"
        f"📊 Placar Final: *{score}*\n"
        f"{resultado}\n"
    )
    return msg

@with_retries(max_attempts=3, backoff=0.7)
def enviar_telegram_raw(token: str, chat_id: str, text: str, parse_mode: str="Markdown") -> dict:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}
    r = requests.post(url, data=payload, timeout=12)
    r.raise_for_status()
    logger.info(f"Sent telegram msg to {chat_id} ({len(text)} chars)")
    return r.json()

def enviar_para_chats(token: str, chat_ids: List[str], text: str) -> List[Tuple[str,str]]:
    results = []
    for cid in chat_ids:
        if not cid:
            continue
        try:
            enviar_telegram_raw(token, cid, text)
            results.append((cid, "ok"))
            time.sleep(0.4)
        except Exception as e:
            logger.warning(f"Erro enviar TG para {cid}: {e}")
            results.append((cid, str(e)))
    return results

# ----------------------------
# OpenLiga helpers (cache)
# ----------------------------
@lru_cache(maxsize=64)
@with_retries(max_attempts=2, backoff=0.4)
def obter_jogos_liga_temporada(liga_id: str, temporada: str) -> List[dict]:
    url = f"{OPENLIGA_BASE}/getmatchdata/{liga_id}/{temporada}"
    logger.info(f"Consultando OpenLiga: {url}")
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        return r.json()
    r.raise_for_status()
    return []

def parse_data_openliga(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
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

def filtrar_jogos_por_data(jogos: List[dict], data_obj: date) -> List[dict]:
    out = []
    for j in jogos:
        date_str = j.get("matchDateTimeUTC") or j.get("matchDateTime")
        dt = parse_data_openliga(date_str)
        if not dt: continue
        if dt.date() == data_obj:
            out.append(j)
    return out

# ----------------------------
# Estatística / Poisson
# ----------------------------
def calcular_media_gols_times(jogos_hist: List[dict]) -> Dict[str, dict]:
    stats = {}
    for j in jogos_hist:
        home = j.get("team1", {}).get("teamName")
        away = j.get("team2", {}).get("teamName")
        placar = None
        for r in j.get("matchResults", []):
            if r.get("resultTypeID") == 2:
                placar = (r.get("pointsTeam1", 0), r.get("pointsTeam2", 0))
                break
        if not placar: continue
        stats.setdefault(home, {"marcados": [], "sofridos": []})
        stats.setdefault(away, {"marcados": [], "sofridos": []})
        stats[home]["marcados"].append(placar[0])
        stats[home]["sofridos"].append(placar[1])
        stats[away]["marcados"].append(placar[1])
        stats[away]["sofridos"].append(placar[0])
    medias = {}
    for t, g in stats.items():
        media_m = sum(g["marcados"]) / len(g["marcados"]) if g["marcados"] else 1.5
        media_s = sum(g["sofridos"]) / len(g["sofridos"]) if g["sofridos"] else 1.2
        medias[t] = {"media_gols_marcados": round(media_m,2), "media_gols_sofridos": round(media_s,2)}
    return medias

def media_gols_h2h(home: str, away: str, jogos_hist: List[dict], max_jogos=5):
    confrontos = []
    for j in jogos_hist:
        t1 = j.get("team1", {}).get("teamName")
        t2 = j.get("team2", {}).get("teamName")
        if {t1, t2} == {home, away}:
            for r in j.get("matchResults", []):
                if r.get("resultTypeID") == 2:
                    gols = (r.get("pointsTeam1", 0), r.get("pointsTeam2", 0))
                    confrontos.append((j.get("matchDateTimeUTC") or j.get("matchDateTime"), gols[0]+gols[1]))
                    break
    if not confrontos:
        return {"media_gols": 0, "total_jogos": 0}
    confrontos = sorted(confrontos, key=lambda x: x[0] or "", reverse=True)[:max_jogos]
    total_p = 0
    total_w = 0
    for idx, (_, total) in enumerate(confrontos):
        peso = max_jogos - idx
        total_p += total * peso
        total_w += peso
    media = round(total_p / total_w, 2) if total_w else 0
    return {"media_gols": media, "total_jogos": len(confrontos)}

def calcular_estimativa(media_h2h, media_casa, media_fora, peso_h2h=0.3):
    mc = media_casa.get("media_gols_marcados", 1.5)
    ms = media_casa.get("media_gols_sofridos", 1.2)
    fc = media_fora.get("media_gols_marcados", 1.4)
    fs = media_fora.get("media_gols_sofridos", 1.1)
    time_casa = mc + fs
    time_fora = fc + ms
    base = (time_casa + time_fora) / 2
    h2h = media_h2h.get("media_gols", base) if media_h2h.get("total_jogos",0)>0 else base
    final = (1 - peso_h2h) * base + peso_h2h * h2h
    return round(final, 2)

def poisson_cdf(k: int, lam: float) -> float:
    s = 0.0
    for i in range(0, k+1):
        s += (lam**i) / math.factorial(i)
    return math.exp(-lam) * s

def prob_over_k(estimativa: float, threshold: float) -> float:
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

def prob_to_conf(prob: float) -> float:
    conf = 50 + (prob - 0.5) * 100
    conf = max(30, min(95, conf))
    return round(conf, 0)

# ----------------------------
# Selection logic
# ----------------------------
def selecionar_top3_distintos(partidas_info: List[dict], max_por_faixa=3):
    """
    mantem prioridade: +1.5 -> +2.5 -> +3.5
    evita repetir fixture_id entre faixas
    """
    if not partidas_info:
        return [], [], []

    def get_num(d, k):
        v = d.get(k, 0)
        try:
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    selected_ids = set()
    selected_teams = set()

    def safe_names(m):
        return str(m.get("home","")).strip(), str(m.get("away","")).strip()

    def allocate(key_prob, other_probs):
        candidatos = [m for m in partidas_info if str(m.get("fixture_id")) not in selected_ids]
        # prefer candidates where this prob is >= other probs
        preferred = [m for m in candidatos if get_num(m, key_prob) >= max([get_num(m, p) for p in other_probs])]
        def sort_key(x):
            return (get_num(x, key_prob), get_num(x, key_prob.replace("prob","conf")), x.get("estimativa",0))
        preferred_sorted = sorted(preferred, key=sort_key, reverse=True)
        remaining = [m for m in candidatos if m not in preferred_sorted]
        remaining_sorted = sorted(remaining, key=sort_key, reverse=True)

        chosen = []
        def try_add(arr, respect_teams=True):
            nonlocal chosen
            for m in arr:
                if len(chosen) >= max_por_faixa: break
                fid = str(m.get("fixture_id"))
                if fid in selected_ids: continue
                home, away = safe_names(m)
                if respect_teams and (home in selected_teams or away in selected_teams): continue
                chosen.append(m)
                selected_ids.add(fid)
                selected_teams.add(home); selected_teams.add(away)
        try_add(preferred_sorted, True)
        if len(chosen) < max_por_faixa:
            try_add(remaining_sorted, True)
        if len(chosen) < max_por_faixa:
            try_add(preferred_sorted + remaining_sorted, False)
        return chosen

    top_15 = allocate("prob_1_5", ["prob_2_5","prob_3_5"])
    top_25 = allocate("prob_2_5", ["prob_1_5","prob_3_5"])
    top_35 = allocate("prob_3_5", ["prob_1_5","prob_2_5"])

    return top_15, top_25, top_35

# ----------------------------
# Conferência reconsulta OpenLiga
# ----------------------------
def conferir_jogo_openliga(fixture_id: Any, liga_id: str, temporada: str, tipo_threshold: str):
    try:
        jogos = obter_jogos_liga_temporada(liga_id, temporada)
        match = None
        for j in jogos:
            if str(j.get("matchID")) == str(fixture_id):
                match = j; break
        if not match:
            return None
        home = match.get("team1",{}).get("teamName")
        away = match.get("team2",{}).get("teamName")
        final = None
        for r in match.get("matchResults", []):
            if r.get("resultTypeID") == 2:
                final = (r.get("pointsTeam1",0), r.get("pointsTeam2",0))
                break
        if final is None:
            return {"home": home, "away": away, "total_gols": None, "aposta": f"+{tipo_threshold}", "resultado": "Em andamento / sem resultado"}
        total = final[0] + final[1]
        if tipo_threshold == "1.5":
            green = total >= 2
        elif tipo_threshold == "2.5":
            green = total >= 3
        else:
            green = total >= 4
        return {"home": home, "away": away, "total_gols": total, "aposta": f"+{tipo_threshold}", "resultado": "🟢 GREEN" if green else "🔴 RED", "score": f"{final[0]} x {final[1]}"}
    except Exception as e:
        logger.exception("Erro conferir jogo")
        return None

# ----------------------------
# UI: Tabs & main flow
# ----------------------------
st.markdown('<div class="header-title">🏆 ELITE MASTER — Alertas Top3</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Análises com OpenLigaDB · Mensagens Premium · Layout Clean</div>', unsafe_allow_html=True)
st.markdown("---")

tabs = st.tabs(["🏠 Dashboard", "⚽ Jogos do Dia", "🔥 Top 3 Alertas", "📊 Histórico / Conferência", "⚙️ Configurações"])

# ----------------------------
# Configurações tab
# ----------------------------
with tabs[4]:
    st.markdown('<div class="card"><b>Configurações</b></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        st.info("Insira token/chat para testes ou configure variáveis de ambiente (recomendado).")
        token_input = st.text_input("Telegram Bot Token", value=os.getenv("TELEGRAM_TOKEN",""), placeholder="123456:ABC-DEF...")
        chat_main = st.text_input("Telegram Chat ID (principal)", value=os.getenv("TELEGRAM_CHAT_ID",""), placeholder="-1001234567890")
        chat_alt = st.text_input("Telegram Chat ID (alternativo, opcional)", value=os.getenv("TELEGRAM_CHAT_ID_ALT2",""), placeholder="-100987654321")
        temporada_padrao = st.selectbox("Temporada padrão (para médias)", ["2022","2023","2024","2025"], index=2)
        envio_auto_checkbox = st.checkbox("Ativar envio automático diário (Top3)", value=False)
        envio_min_prob = st.slider("Enviar apenas se Prob >= (apenas para envio automático)", 30, 95, 45, 5)
    with col2:
        st.markdown("### Ações rápidas")
        if st.button("Salvar configurações na sessão"):
            st.session_state["TG_TOKEN"] = token_input.strip()
            st.session_state["TG_CHAT"] = chat_main.strip()
            st.session_state["TG_CHAT_ALT"] = chat_alt.strip()
            st.session_state["TEMP_PADRAO"] = temporada_padrao
            st.session_state["AUTO_SEND"] = envio_auto_checkbox
            st.session_state["AUTO_SEND_MIN"] = envio_min_prob
            st.success("Configurações salvas na sessão.")
        if st.button("Testar envio Telegram (mensagem de teste)"):
            token = token_input.strip() or os.getenv("TELEGRAM_TOKEN","")
            c_main = chat_main.strip() or os.getenv("TELEGRAM_CHAT_ID","")
            if not token or not c_main:
                st.error("Token ou Chat ID principal ausente. Configure primeiro.")
            else:
                try:
                    msg = "🔥 *ELITE MASTER - TESTE* 🔥\n\nMensagem de teste enviada pelo painel."
                    res = enviar_para_chats(token, [c_main, chat_alt.strip() if chat_alt.strip() else None], msg)
                    st.write(res)
                    st.success("Teste enviado. Verifique o chat.")
                except Exception as e:
                    st.error(f"Erro envio teste: {e}")

    st.markdown("---")
    st.markdown("**Arquivo(s) de histórico:**")
    st.code(f"{TOP3_PATH}  (histórico de envios)\n{ALERTAS_PATH}  (log de alertas)")

# ----------------------------
# Dashboard tab
# ----------------------------
with tabs[0]:
    st.markdown('<div class="card"><b>Dashboard</b></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    top3_salvos = carregar_top3()
    total_envios = len(top3_salvos)
    log_alertas = carregar_alertas().get("log", [])
    total_alerts = len(log_alertas)
    greens = reds = 0
    for lote in top3_salvos:
        for k in ("top_1_5","top_2_5","top_3_5"):
            for j in lote.get(k, []):
                if j.get("resultado"):
                    if "GREEN" in j.get("resultado"):
                        greens += 1
                    else:
                        reds += 1
    taxa = f"{(greens/(greens+reds)*100):.1f}%" if (greens+reds)>0 else "N/A"
    with col1: st.metric("Top3 enviados", total_envios)
    with col2: st.metric("Alertas (log)", total_alerts)
    with col3: st.metric("Taxa de acerto (conferidos)", taxa)
    st.markdown("### Últimos envios")
    preview = []
    for lote in reversed(top3_salvos[-6:]):
        preview.append({"Envio": f"{lote.get('data_envio')} {lote.get('hora_envio')}", "Itens": sum(len(lote.get(k,[])) for k in lote if k.startswith("top_"))})
    if preview:
        st.table(preview)
    else:
        st.info("Nenhum envio registrado ainda.")

# ----------------------------
# Jogos do Dia tab
# ----------------------------
with tabs[1]:
    st.markdown('<div class="card"><b>Jogos do Dia</b></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        data_selecionada = st.date_input("Data", value=datetime.utcnow().date())
        liga_selecionada = st.selectbox("Escolha a liga (OpenLiga)", list(LEAGAS_MAP.keys()))
    with col2:
        temporada_input = st.selectbox("Temporada (para médias)", ["2022","2023","2024","2025"], index=2)
    with col3:
        if st.button("Carregar jogos & Analisar"):
            pass

    if st.button("Buscar jogos (OpenLigaDB)"):
        liga_id = LEAGAS_MAP.get(liga_selecionada)
        try:
            jogos_hist = obter_jogos_liga_temporada(liga_id, temporada_input)
            jogos_dia = filtrar_jogos_por_data(jogos_hist, data_selecionada)
            if not jogos_dia:
                st.info("Nenhum jogo encontrado para a data/ligas selecionadas.")
            else:
                medias_liga = calcular_media_gols_times(jogos_hist)
                rows = []
                for m in jogos_dia:
                    home = m.get("team1",{}).get("teamName")
                    away = m.get("team2",{}).get("teamName")
                    hora_dt = parse_data_openliga(m.get("matchDateTimeUTC") or m.get("matchDateTime"))
                    hora_brt = (hora_dt - timedelta(hours=3)).strftime("%H:%M") if hora_dt else "??:??"
                    media_h2h = media_gols_h2h(home, away, jogos_hist, max_jogos=5)
                    media_home = medias_liga.get(home, {"media_gols_marcados":1.5,"media_gols_sofridos":1.2})
                    media_away = medias_liga.get(away, {"media_gols_marcados":1.4,"media_gols_sofridos":1.1})
                    estim = calcular_estimativa(media_h2h, media_home, media_away, peso_h2h=0.3)
                    p15 = prob_over_k(estim, 1.5); p25 = prob_over_k(estim, 2.5); p35 = prob_over_k(estim, 3.5)
                    rows.append({
                        "fixture_id": m.get("matchID"), "home": home, "away": away,
                        "hora": hora_brt, "estimativa": estim,
                        "prob_1_5": round(p15*100,1), "prob_2_5": round(p25*100,1), "prob_3_5": round(p35*100,1),
                        "liga_id": liga_id, "temporada": temporada_input
                    })
                st.session_state["JOGOS_DO_DIA"] = rows
                st.success(f"{len(rows)} jogos carregados e analisados.")
        except Exception as e:
            st.error(f"Erro ao buscar jogos: {e}")

    jogos = st.session_state.get("JOGOS_DO_DIA", [])
    if jogos:
        st.markdown("### Jogos analisados")
        tabela = []
        for j in jogos:
            tabela.append({"Jogo": f"{j['home']} x {j['away']}", "Hora (BRT)": j["hora"], "Estimativa": j["estimativa"], "P(+1.5)": f"{j['prob_1_5']}%", "P(+2.5)": f"{j['prob_2_5']}%", "P(+3.5)": f"{j['prob_3_5']}%"})
        st.table(tabela)
        st.info("Vá em 'Top 3 Alertas' para selecionar e enviar os melhores jogos.")

# ----------------------------
# Top 3 Alertas tab
# ----------------------------
with tabs[2]:
    st.markdown('<div class="card"><b>Top 3 Alertas — Seleção & Envio</b></div>', unsafe_allow_html=True)
    jogos = st.session_state.get("JOGOS_DO_DIA", [])
    if not jogos:
        st.info("Nenhum jogo carregado. Use 'Jogos do Dia' primeiro.")
    else:
        faixa = st.selectbox("Prioridade de faixa", ["1.5","2.5","3.5"], index=0)
        top_n = st.slider("Quantos por faixa (Top N)", 1, 5, 3)
        top_15, top_25, top_35 = selecionar_top3_distintos(jogos, max_por_faixa=top_n)
        st.markdown("#### Top selecionados por faixa (prioridade: +1.5 → +2.5 → +3.5)")
        def mostrar_lista(lst, label):
            if not lst:
                st.write(f"— nenhum Top para +{label}")
                return
            for i, j in enumerate(lst, start=1):
                st.write(f"{i}. {j['home']} x {j['away']}  |  P(+{label}) {j[f'prob_{label.replace('.','_')}']}%  |  Est: {j['estimativa']}  |  {j['hora']} BRT")
        mostrar_lista(top_15, "1.5")
        mostrar_lista(top_25, "2.5")
        mostrar_lista(top_35, "3.5")

        st.markdown("---")
        col_send1, col_send2 = st.columns([1,1])
        with col_send1:
            if st.button("Enviar TOPs separados (uma mensagem por faixa)"):
                token = st.session_state.get("TG_TOKEN") or os.getenv("TELEGRAM_TOKEN","")
                chat_main = st.session_state.get("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID","")
                chat_alt = st.session_state.get("TG_CHAT_ALT") or os.getenv("TELEGRAM_CHAT_ID_ALT2","")
                if not token or not chat_main:
                    st.error("Configure token/chat em Configurações.")
                else:
                    # mensagens por faixa
                    def enviar_lista(lista, label):
                        if not lista: return
                        msg = f"🔥 *TOP {len(lista)} +{label} — ELITE MASTER* 🔥\n\n"
                        for idx, j in enumerate(lista, start=1):
                            msg += f"{idx}. *{j['home']} x {j['away']}*  |  P:+{label} *{j[f'prob_{label.replace('.','_')}']}%*  |  Est: *{j['estimativa']}*  |  {j['hora']} BRT\n"
                        return enviar_para_chats(token, [chat_main, chat_alt], msg)
                    res1 = enviar_lista(top_15, "1.5")
                    res2 = enviar_lista(top_25, "2.5")
                    res3 = enviar_lista(top_35, "3.5")
                    # log
                    log = carregar_alertas()
                    log.setdefault("log",[]).append({"when": datetime.now().isoformat(), "tops": {"1.5":[x["fixture_id"] for x in top_15], "2.5":[x["fixture_id"] for x in top_25], "3.5":[x["fixture_id"] for x in top_35]}})
                    salvar_alertas(log)
                    # salvar histórico compacto
                    hist = carregar_top3()
                    novo = {"data_envio": date.today().isoformat(), "hora_envio": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "temporada": jogos[0].get("temporada",""), "top_1_5": top_15, "top_2_5": top_25, "top_3_5": top_35}
                    hist.append(novo); salvar_top3(hist)
                    st.success("Mensagens enviadas. Veja o log ou o chat do Telegram.")
        with col_send2:
            if st.button("Enviar Mensagem consolidada (1 mensagem)"):
                token = st.session_state.get("TG_TOKEN") or os.getenv("TELEGRAM_TOKEN","")
                chat_main = st.session_state.get("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID","")
                chat_alt = st.session_state.get("TG_CHAT_ALT") or os.getenv("TELEGRAM_CHAT_ID_ALT2","")
                if not token or not chat_main:
                    st.error("Configure token/chat em Configurações.")
                else:
                    # consolidar top de todas faixas (concatenar)
                    consolidated = f"🔥 *ELITE MASTER - TOPS Consolidados* 🔥\n\n"
                    for label, lst in [("1.5", top_15), ("2.5", top_25), ("3.5", top_35)]:
                        if not lst: continue
                        consolidated += f"• +{label}\n"
                        for i, j in enumerate(lst, start=1):
                            consolidated += f"  {i}. *{j['home']} x {j['away']}*  |  P:+{label} *{j[f'prob_{label.replace('.','_')}']}%*  |  Est: *{j['estimativa']}*  |  {j['hora']} BRT\n"
                    res = enviar_para_chats(token, [chat_main, chat_alt], consolidated)
                    # log + histórico
                    log = carregar_alertas(); log.setdefault("log",[]).append({"when": datetime.now().isoformat(), "consolidated_top": [x["fixture_id"] for x in (top_15+top_25+top_35)]}); salvar_alertas(log)
                    hist = carregar_top3(); novo = {"data_envio": date.today().isoformat(), "hora_envio": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "temporada": jogos[0].get("temporada",""), "top_all": top_15+top_25+top_35}; hist.append(novo); salvar_top3(hist)
                    st.success("Mensagem consolidada enviada.")

# ----------------------------
# Histórico / Conferência tab
# ----------------------------
with tabs[3]:
    st.markdown('<div class="card"><b>Histórico e Conferência</b></div>', unsafe_allow_html=True)
    top3_salvos = carregar_top3()
    if not top3_salvos:
        st.info("Nenhum histórico encontrado. Gere e envie Top3 primeiro.")
    else:
        options = [f"{i+1} - {t['data_envio']} ({t['hora_envio']})" for i,t in enumerate(top3_salvos)]
        selecionado = st.selectbox("Selecione lote para conferência:", options)
        idx = options.index(selecionado)
        lote = top3_salvos[idx]
        st.write("Lote:", lote.get("data_envio"), lote.get("hora_envio"))
        # mostrar partidas
        for k in [k for k in lote.keys() if k.startswith("top_")]:
            st.write(f"### {k.replace('_',' ')}")
            for j in lote.get(k, []):
                st.write(f"- {j.get('home')} x {j.get('away')}  |  Est: {j.get('estimativa')}  |  Resultado: {j.get('resultado','-')}")
        st.markdown("---")
        if st.button("Rechecar resultados e enviar conferência (Telegram)"):
            token = st.session_state.get("TG_TOKEN") or os.getenv("TELEGRAM_TOKEN","")
            chat_main = st.session_state.get("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID","")
            chat_alt = st.session_state.get("TG_CHAT_ALT") or os.getenv("TELEGRAM_CHAT_ID_ALT2","")
            if not token or not chat_main:
                st.error("Configure token/chat em Configurações.")
            else:
                resumo_total = {}
                for k in [k for k in lote.keys() if k.startswith("top_")]:
                    label = k.split("_",1)[1].replace("_",".")
                    lista = lote.get(k, [])
                    lines = []
                    greens = reds = 0
                    for j in lista:
                        info = conferir_jogo_openliga(j.get("fixture_id"), j.get("liga_id"), lote.get("temporada",""), label)
                        if not info:
                            lines.append(f"🏟️ {j.get('home')} x {j.get('away')} — sem resultado")
                            continue
                        if info.get("total_gols") is None:
                            lines.append(f"🏟️ {info['home']} — Em andamento / sem resultado")
                            continue
                        lines.append(f"🏟️ {info['home']} {info.get('score','')} {info['away']} → {info['resultado']}")
                        if "GREEN" in info["resultado"]:
                            greens += 1
                        else:
                            reds += 1
                        # atualizar lote com resultado localmente (não sobrescreve histórico original, só mostra)
                    header = f"✅ RESULTADOS - CONFERÊNCIA +{label}\n(Lote: {lote['data_envio']})\n\n"
                    body = "\n".join(lines) if lines else "_Nenhum jogo nesta faixa_"
                    resumo = f"\n\nResumo: 🟢 {greens} GREEN | 🔴 {reds} RED"
                    msg = header + body + resumo
                    enviar_para_chats(token, [chat_main, chat_alt], msg)
                    resumo_total[label] = {"greens": greens, "reds": reds}
                st.success("Conferência enviada ao Telegram.")
                st.json(resumo_total)

        if st.button("Rechecar resultados local (sem enviar)"):
            for k in [k for k in lote.keys() if k.startswith("top_")]:
                label = k.split("_",1)[1].replace("_",".")
                st.write(f"## Conferência +{label}")
                for j in lote.get(k, []):
                    info = conferir_jogo_openliga(j.get("fixture_id"), j.get("liga_id"), lote.get("temporada",""), label)
                    if not info:
                        st.warning(f"🏟️ {j.get('home')} x {j.get('away')} — sem resultado")
                        continue
                    if info.get("total_gols") is None:
                        st.info(f"🏟️ {info['home']} — Em andamento / sem resultado")
                        continue
                    if "GREEN" in info["resultado"]:
                        st.success(f"🏟️ {info['home']} {info.get('score','')} {info['away']} → {info['resultado']}")
                    else:
                        st.error(f"🏟️ {info['home']} {info.get('score','')} {info['away']} → {info['resultado']}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown('<div class="muted">Elite Master • Fonte: OpenLigaDB • Configure tokens em Configurações ou via variáveis de ambiente.</div>', unsafe_allow_html=True)
