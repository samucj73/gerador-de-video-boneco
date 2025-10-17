# Futebol_Alertas_OpenLiga_Top3_Sports.py
"""
App Streamlit - Futebol Alertas Top3 (OpenLigaDB)
Tema: Sports Style (Dark) + Mensagens Premium / VIP (modelo B)
Fonte de dados: OpenLigaDB (https://api.openligadb.de)

Como usar:
- Defina variáveis de ambiente opcionais: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_CHAT_ID_ALT2
- Ou preencha os campos na aba Configurações do app (recomendado para testes locais).
- Rode: streamlit run Futebol_Alertas_OpenLiga_Top3_Sports.py
"""

import streamlit as st
from datetime import datetime, timedelta, date
import requests
import json
import os
import math
import time
import logging
from functools import lru_cache, wraps
from typing import List, Dict, Any, Optional, Tuple

# ----------------------------
# Config básica & logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FutebolAlertasSports")

st.set_page_config(page_title="⚽ Alertas Sports — Top3 (OpenLigaDB)", layout="wide")

# ----------------------------
# CSS - Tema Sports (escuro com acentos)
# ----------------------------
# Atenção: esse CSS usa hacks simples para o Streamlit atual — personalize conforme preferir.
st.markdown(
    """
    <style>
    /* Fundo geral */
    .reportview-container, .main, .block-container {
        background: linear-gradient(180deg, #0b1220 0%, #0f1724 100%);
        color: #e6eef6;
    }
    /* Card-like containers */
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.08));
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.4);
        margin-bottom: 10px;
    }
    .stButton>button {
        background: linear-gradient(90deg,#2bb673,#1f9d4a);
        color: white;
        border: none;
    }
    .small-muted { color: #9fb0a0; font-size: 13px; }
    .big-title { font-size:22px; font-weight:700; color:#f7f9fb; }
    .accent { color: #ffd166; font-weight:700; } /* amarelo sports */
    .green { color: #2bb673; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Constantes e paths
# ----------------------------
OPENLIGA_BASE = "https://api.openligadb.de"

# Ligas (exemplo: Alemanha) - você pode adicionar outras.
LIGAS_OPENLIGA = {
    "Bundesliga (Alemanha)": "bl1",
    "2. Bundesliga (Alemanha)": "bl2",
    "DFB-Pokal (Alemanha)": "dfb"
}

ALERTAS_PATH = "alertas.json"
TOP3_PATH = "top3.json"

# ----------------------------
# Utilitários: persistência simples
# ----------------------------
def carregar_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Erro ao ler {path}: {e}")
            return default
    return default

def salvar_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def carregar_alertas():
    return carregar_json(ALERTAS_PATH, {})

def salvar_alertas(obj):
    salvar_json(ALERTAS_PATH, obj)

def carregar_top3():
    return carregar_json(TOP3_PATH, [])

def salvar_top3(lista):
    salvar_json(TOP3_PATH, lista)

# ----------------------------
# Retries decorator simples
# ----------------------------
def with_retries(max_attempts=3, backoff=0.6):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
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
# Telegram - mensagem estilo B (premium/vip)
# ----------------------------
def montar_mensagem_alerta_vip(home: str, away: str, faixa: str, prob_pct: float, estimativa: float, liga: str, hora_brt: str) -> str:
    # faixa ex: "1.5" ou "+1.5"
    faixa_text = faixa if faixa.startswith("+") else f"+{faixa}"
    msg = (
        "🔥 *ALERTA ELITE MASTER* 🔥\n\n"
        f"🏟️ *{home} x {away}*\n"
        f"⚽ Tendência: *{faixa_text} Gols*\n"
        f"📈 Probabilidade: *{prob_pct:.0f}%*\n"
        f"💰 Estimativa: *{estimativa:.2f}*  |  {liga}  |  {hora_brt} BRT\n"
    )
    return msg

def montar_mensagem_conferencia(home: str, away: str, faixa: str, score: str, resultado: str) -> str:
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
    base = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}
    r = requests.post(base, data=payload, timeout=12)
    r.raise_for_status()
    logger.info(f"Telegram send OK to {chat_id} (len {len(text)})")
    return r.json()

def enviar_para_config(token: str, chat_ids: List[str], text: str) -> List[Tuple[str, Optional[str]]]:
    results = []
    for cid in chat_ids:
        if not cid:
            continue
        try:
            enviar_telegram_raw(token, cid, text)
            results.append((cid, "ok"))
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Erro envio TG {cid}: {e}")
            results.append((cid, str(e)))
    return results

# ----------------------------
# OpenLiga helpers (com cache simples)
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

def filtrar_jogos_por_data(jogos: List[dict], data_obj: date) -> List[dict]:
    out = []
    for j in jogos:
        date_str = j.get("matchDateTimeUTC") or j.get("matchDateTime")
        dt = parse_data_openliga(date_str)
        if not dt:
            continue
        if dt.date() == data_obj:
            out.append(j)
    return out

# ----------------------------
# Estatística / Poisson (mesma lógica)
# ----------------------------
def calcular_media_gols_times(jogos_hist: List[dict]) -> Dict[str, dict]:
    stats = {}
    for j in jogos_hist:
        home = j.get("team1", {}).get("teamName")
        away = j.get("team2", {}).get("teamName")
        final = None
        for r in j.get("matchResults", []):
            if r.get("resultTypeID") == 2:
                final = (r.get("pointsTeam1", 0), r.get("pointsTeam2", 0))
                break
        if not final:
            continue
        stats.setdefault(home, {"marcados": [], "sofridos": []})
        stats.setdefault(away, {"marcados": [], "sofridos": []})
        stats[home]["marcados"].append(final[0])
        stats[home]["sofridos"].append(final[1])
        stats[away]["marcados"].append(final[1])
        stats[away]["sofridos"].append(final[0])
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
    total_pontos = 0
    total_peso = 0
    for idx, (_, total) in enumerate(confrontos):
        peso = max_jogos - idx
        total_pontos += total * peso
        total_peso += peso
    media = round(total_pontos / total_peso,2) if total_peso else 0
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
    return round(final,2)

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
    return round(conf,0)

# ----------------------------
# Seleção Top (mais simples e forte: Top por prob +1.5)
# ----------------------------
def selecionar_top_por_faixa(partidas: List[dict], faixa: str, top_n=3) -> List[dict]:
    # faixa: "1.5" | "2.5" | "3.5"
    key_prob = f"prob_{faixa.replace('.','_')}"
    sorted_ = sorted(partidas, key=lambda x: x.get(key_prob, 0), reverse=True)
    return sorted_[:top_n]

# ----------------------------
# Conferência de resultado (reconsulta OpenLiga)
# ----------------------------
def conferir_jogo_openliga(fixture_id: Any, liga_id: str, temporada: str, threshold_label: str):
    try:
        jogos = obter_jogos_liga_temporada(liga_id, temporada)
        match = None
        for j in jogos:
            if str(j.get("matchID")) == str(fixture_id):
                match = j
                break
        if not match:
            return None
        home = match.get("team1", {}).get("teamName")
        away = match.get("team2", {}).get("teamName")
        final = None
        for r in match.get("matchResults", []):
            if r.get("resultTypeID") == 2:
                final = (r.get("pointsTeam1", 0), r.get("pointsTeam2", 0))
                break
        if final is None:
            return {"home": home, "away": away, "total_gols": None, "aposta": f"+{threshold_label}", "resultado": "Em andamento / sem resultado"}
        total = final[0] + final[1]
        if threshold_label == "1.5":
            green = total >= 2
        elif threshold_label == "2.5":
            green = total >= 3
        else:
            green = total >= 4
        return {"home": home, "away": away, "total_gols": total, "aposta": f"+{threshold_label}", "resultado": "🟢 GREEN" if green else "🔴 RED", "score": f"{final[0]} x {final[1]}"}
    except Exception as e:
        logger.exception("Erro na conferência OpenLiga")
        return None

# ----------------------------
# UI: Layout com abas
# ----------------------------
st.markdown('<div class="big-title">⚽ Alertas Sports — Top3 (OpenLigaDB)</div>', unsafe_allow_html=True)
st.markdown('<div class="small-muted">Visual: Sports • Mensagens: Premium / VIP • Fonte: OpenLigaDB</div>', unsafe_allow_html=True)
st.markdown("---")

tabs = st.tabs(["🏠 Dashboard", "⚽ Jogos do Dia", "🔥 Top 3 Alertas", "📊 Histórico / Conferência", "⚙️ Configurações"])

# --- Configurações (aba) ---
with tabs[4]:
    st.markdown('<div class="card"><b>Configurações & Tokens</b></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        st.info("Prefira usar VARIÁVEIS DE AMBIENTE em produção. Aqui você pode preencher para testes.")
        token_input = st.text_input("Telegram Bot Token (ou deixe vazio para usar env)", value=os.getenv("TELEGRAM_TOKEN",""), placeholder="123456:ABC-DEF...") 
        chat_main = st.text_input("Telegram Chat ID principal (chat_id)", value=os.getenv("TELEGRAM_CHAT_ID",""), placeholder="-1001234567890")
        chat_alt = st.text_input("Telegram Chat ID alternativo (opcional)", value=os.getenv("TELEGRAM_CHAT_ID_ALT2",""), placeholder="-100987654321")
        temporada_padrao = st.selectbox("Temporada padrão (para médias)", ["2022","2023","2024","2025"], index=2)
        ligar_envio_limite = st.slider("Enviar apenas se Prob >= (apenas para envio automático)", 30, 95, 45, 5)
    with col2:
        st.markdown("### Ações")
        if st.button("Salvar (temporário) configurações nesta sessão"):
            st.session_state["TG_TOKEN"] = token_input.strip()
            st.session_state["TG_CHAT"] = chat_main.strip()
            st.session_state["TG_CHAT_ALT"] = chat_alt.strip()
            st.session_state["TEMP_PADRAO"] = temporada_padrao
            st.session_state["ENVIO_LIMITE"] = ligar_envio_limite
            st.success("Configurações salvas na sessão atual.")
        if st.button("Testar envio Telegram (mensagem de teste)"):
            token = token_input.strip() or os.getenv("TELEGRAM_TOKEN","")
            c_main = chat_main.strip() or os.getenv("TELEGRAM_CHAT_ID","")
            if not token or not c_main:
                st.error("Token ou Chat ID principal ausente. Preencha ou configure env vars.")
            else:
                try:
                    txt = "🔥 *TESTE DE ALERTA* 🔥\n\nMensagem de teste enviada pelo app Sports."
                    res = enviar_para_config(token, [c_main, chat_alt.strip() if chat_alt.strip() else None], txt)
                    st.write(res)
                    st.success("Teste executado. Verifique o(s) chat(s).")
                except Exception as e:
                    st.error(f"Erro no envio de teste: {e}")

    st.markdown("---")
    st.markdown("**Paths usados para histórico:**")
    st.code(f"{TOP3_PATH} (Top3 enviados)\n{ALERTAS_PATH} (alertas e log)")

# --- Dashboard (aba) ---
with tabs[0]:
    st.markdown('<div class="card"><b>Dashboard</b></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    top3_salvos = carregar_top3()
    total_envios = len(top3_salvos)
    total_alerts = len(carregar_alertas().get("log", []))
    # calcular taxa de acerto simples nos históricos (se houver)
    greens = 0
    reds = 0
    for lote in top3_salvos:
        for key in ("top_1_5","top_2_5","top_3_5"):
            for j in lote.get(key, []):
                # se existirem resultados marcados
                if "resultado" in j and j.get("resultado"):
                    if "GREEN" in j.get("resultado"):
                        greens += 1
                    else:
                        reds += 1
    taxa_acerto = f"{(greens/(greens+reds)*100):.1f}%" if (greens+reds)>0 else "N/A"
    with col1:
        st.metric("Total Top3 enviados", total_envios)
    with col2:
        st.metric("Total alertas (log)", total_alerts)
    with col3:
        st.metric("Taxa de acerto (conferidos)", taxa_acerto)

    st.markdown("### Últimos resultados conferidos")
    # mostrar até últimos 8 itens
    preview = []
    for lote in reversed(top3_salvos[-8:]):
        data = lote.get("data_envio")
        hora = lote.get("hora_envio")
        resumo = []
        for key in ("top_1_5","top_2_5","top_3_5"):
            for j in lote.get(key, []):
                res = j.get("resultado") or ""
                if res:
                    resumo.append(f"{j.get('home')} {res}")
        preview.append({"Envio": f"{data} {hora}", "Resumo": " | ".join(resumo)[:150]})
    if preview:
        st.table(preview)
    else:
        st.info("Nenhum histórico para exibir ainda.")

# --- Jogos do Dia (aba) ---
with tabs[1]:
    st.markdown('<div class="card"><b>Jogos do Dia</b></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        data_selecionada = st.date_input("Selecione a data", value=datetime.utcnow().date())
        liga_selecionada = st.selectbox("Liga", list(LIGAS_OPENLIGA.keys()))
    with col2:
        temporada = st.selectbox("Temporada (para médias)", ["2022","2023","2024","2025"], index=2)
    with col3:
        if st.button("Buscar jogos & Analisar"):
            pass

    # ação de busca
    if st.button("Carregar jogos do dia (OpenLiga)"):
        liga_id = LIGAS_OPENLIGA.get(liga_selecionada)
        try:
            jogos_hist = obter_jogos_liga_temporada(liga_id, temporada)
            jogos_dia = filtrar_jogos_por_data(jogos_hist, data_selecionada)
            if not jogos_dia:
                st.info("Nenhum jogo encontrado para essa data/ligas.")
            else:
                # montar dados com estimativas
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
                        "fixture_id": m.get("matchID"),
                        "home": home, "away": away,
                        "hora": hora_brt,
                        "estimativa": estim,
                        "prob_1_5": round(p15*100,1),
                        "prob_2_5": round(p25*100,1),
                        "prob_3_5": round(p35*100,1),
                        "liga_id": liga_id,
                        "temporada": temporada
                    })
                st.session_state["JOGOS_DO_DIA"] = rows
                st.success(f"{len(rows)} jogos carregados.")
        except Exception as e:
            st.error(f"Erro ao carregar jogos: {e}")

    # Exibição tabela com estilo
    jogos = st.session_state.get("JOGOS_DO_DIA", [])
    if jogos:
        st.markdown("### Tabela de Jogos (análise rápida)")
        # constrói tabela simples
        tabela = []
        for j in jogos:
            tabela.append({
                "Jogo": f"{j['home']} x {j['away']}",
                "Hora (BRT)": j["hora"],
                "Estimativa": j["estimativa"],
                "P(+1.5)": f"{j['prob_1_5']}%",
                "P(+2.5)": f"{j['prob_2_5']}%",
                "P(+3.5)": f"{j['prob_3_5']}%"
            })
        st.table(tabela)
        st.markdown("Você pode ir para a aba *Top 3 Alertas* para selecionar/enviar os melhores jogos automaticamente.")

# --- Top 3 Alertas (aba) ---
with tabs[2]:
    st.markdown('<div class="card"><b>Top 3 Alertas — Seleção & Envio</b></div>', unsafe_allow_html=True)
    jogos = st.session_state.get("JOGOS_DO_DIA", [])
    if not jogos:
        st.info("Nenhum jogo carregado. Vá em 'Jogos do Dia' e carregue os jogos primeiro.")
    else:
        faixa = st.selectbox("Escolher faixa para Top (prioridade)", ["1.5","2.5","3.5"], index=0)
        top_n = st.slider("Quantos top (n)", 1, 5, 3)
        top_sel = selecionar_top_por_faixa(jogos, faixa, top_n)
        st.markdown("#### Selecionados")
        for i, j in enumerate(top_sel, start=1):
            st.markdown(f"**{i}** — {j['home']} x {j['away']}  |  P(+{faixa}): *{j[f'prob_{faixa.replace('.','_')}']}%*  |  Est: *{j['estimativa']}*  |  {j['hora']} BRT")

        st.markdown("---")
        # envio
        col_send1, col_send2 = st.columns([1,1])
        with col_send1:
            if st.button("Enviar TOP selecionados (Premium VIP)"):
                # pegar token/chat da sessão ou env
                token = st.session_state.get("TG_TOKEN") or os.getenv("TELEGRAM_TOKEN","")
                chat_main = st.session_state.get("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID","")
                chat_alt = st.session_state.get("TG_CHAT_ALT") or os.getenv("TELEGRAM_CHAT_ID_ALT2","")
                if not token or not chat_main:
                    st.error("Token ou Chat ID principal não configurado. Vá em Configurações.")
                else:
                    envio_res = []
                    for j in top_sel:
                        msg = montar_mensagem_alerta_vip(j['home'], j['away'], faixa, j[f'prob_{faixa.replace(".","_")}'], j['estimativa'], j.get('liga_id',''), j.get('hora','??:??'))
                        res = enviar_para_config(token, [chat_main, chat_alt], msg)
                        envio_res.append({"fixture": j["fixture_id"], "res": res})
                        # marcar log local
                        log = carregar_alertas()
                        log.setdefault("log", [])
                        log["log"].append({"when": datetime.now().isoformat(), "fixture": j["fixture_id"], "message": msg})
                        salvar_alertas(log)
                        time.sleep(0.4)
                    # salvar top3 no histórico
                    top3_list = carregar_top3()
                    novo = {"data_envio": date.today().isoformat(), "hora_envio": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "temporada": jogos[0].get("temporada",""), f"top_{faixa.replace('.','_')}": top_sel}
                    top3_list.append(novo)
                    salvar_top3(top3_list)
                    st.success("Envio executado. Verifique os chats. Resultado resumido abaixo.")
                    st.json(envio_res)
        with col_send2:
            if st.button("Enviar Mensagem Consolidada (1 msg)"):
                token = st.session_state.get("TG_TOKEN") or os.getenv("TELEGRAM_TOKEN","")
                chat_main = st.session_state.get("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID","")
                chat_alt = st.session_state.get("TG_CHAT_ALT") or os.getenv("TELEGRAM_CHAT_ID_ALT2","")
                if not token or not chat_main:
                    st.error("Token ou Chat ID principal não configurado. Vá em Configurações.")
                else:
                    consolidated = f"🔥 *TOP {top_n} - Consolidados (faixa +{faixa})* 🔥\n\n"
                    for i, j in enumerate(top_sel, start=1):
                        consolidated += f"{i}. *{j['home']} x {j['away']}* — P:+{faixa} *{j[f'prob_{faixa.replace('.','_')}']}%* | Est: *{j['estimativa']}* | {j['hora']} BRT\n"
                    res = enviar_para_config(token, [chat_main, chat_alt], consolidated)
                    # salvar log e histórico (como acima)
                    log = carregar_alertas()
                    log.setdefault("log", []).append({"when": datetime.now().isoformat(), "consolidated_top": [x["fixture_id"] for x in top_sel]})
                    salvar_alertas(log)
                    top3_list = carregar_top3()
                    novo = {"data_envio": date.today().isoformat(), "hora_envio": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "temporada": jogos[0].get("temporada",""), f"top_{faixa.replace('.','_')}": top_sel}
                    top3_list.append(novo)
                    salvar_top3(top3_list)
                    st.success("Mensagem consolidada enviada.")
                    st.write(res)

# --- Histórico / Conferência (aba) ---
with tabs[3]:
    st.markdown('<div class="card"><b>Histórico e Conferência</b></div>', unsafe_allow_html=True)
    top3_salvos = carregar_top3()
    if not top3_salvos:
        st.info("Nenhum Top3 salvo ainda. Gere e envie Top3 primeiro.")
    else:
        options = [f"{i+1} - {t['data_envio']} ({t['hora_envio']})" for i,t in enumerate(top3_salvos)]
        selecionado = st.selectbox("Escolha lote para conferir:", options)
        idx = options.index(selecionado)
        lote = top3_salvos[idx]
        st.write("Lote selecionado:", lote["data_envio"], lote["hora_envio"])
        st.markdown("### Partidas no lote")
        # list all found keys like top_1_5 top_2_5 top_3_5
        for k in [k for k in lote.keys() if k.startswith("top_")]:
            st.write(f"#### {k.replace('_',' ')}")
            for j in lote.get(k, []):
                st.write(f"- {j.get('home')} x {j.get('away')} — Est: {j.get('estimativa')} — Resultado: {j.get('resultado','-')}")
        st.markdown("---")
        if st.button("Rechecar resultados e enviar conferência (uma mensagem por faixa)"):
            token = st.session_state.get("TG_TOKEN") or os.getenv("TELEGRAM_TOKEN","")
            chat_main = st.session_state.get("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID","")
            chat_alt = st.session_state.get("TG_CHAT_ALT") or os.getenv("TELEGRAM_CHAT_ID_ALT2","")
            if not token or not chat_main:
                st.error("Configure token/chat na aba Configurações.")
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
                    header = f"✅ RESULTADOS - CONFERÊNCIA +{label}\n(Lote: {lote['data_envio']})\n\n"
                    body = "\n".join(lines) if lines else "_Nenhum jogo nesta faixa_"
                    resumo = f"\n\nResumo: 🟢 {greens} GREEN | 🔴 {reds} RED"
                    msg = header + body + resumo
                    enviar_para_config(token, [chat_main, chat_alt], msg)
                    resumo_total[label] = {"greens": greens, "reds": reds}
                st.success("Conferência enviada para Telegram.")
                st.json(resumo_total)

        if st.button("Rechecar resultados local (sem enviar)"):
            for k in [k for k in lote.keys() if k.startswith("top_")]:
                label = k.split("_",1)[1].replace("_",".")
                st.write(f"## Conferência +{label}")
                for j in lote.get(k, []):
                    info = conferir_jogo_openliga(j.get("fixture_id"), j.get("liga_id"), lote.get("temporada",""), label)
                    if not info:
                        st.warning(f"🏟️ {j.get('home')} x {j.get('away')} — resultado não encontrado")
                        continue
                    if info.get("total_gols") is None:
                        st.info(f"🏟️ {info['home']} — Em andamento / sem resultado")
                        continue
                    if "GREEN" in info["resultado"]:
                        st.success(f"🏟️ {info['home']} {info.get('score','')} {info['away']} → {info['resultado']}")
                    else:
                        st.error(f"🏟️ {info['home']} {info.get('score','')} {info['away']} → {info['resultado']}")

# ----------------------------
# Footer nota
# ----------------------------
st.markdown("---")
st.markdown('<div class="small-muted">Desenvolvido para uso com OpenLigaDB. Configure tokens na aba <b>Configurações</b>. Em produção, mova tokens para variáveis de ambiente.</div>', unsafe_allow_html=True)
