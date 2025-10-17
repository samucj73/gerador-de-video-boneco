# Futebol_Alertas_EliteMaster_DarkAuto.py
"""
Elite Master - Dark Premium - Envio AUTOMÁTICO
Fonte: OpenLigaDB
Modo: Dark, responsivo, envio automático dos Top3 e conferência automática.
Atenção: para testes locais, coloque seu TELEGRAM_TOKEN e TELEGRAM_CHAT_ID abaixo.
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
# CONFIGURAÇÃO (coloque seus tokens aqui para testes)
# ----------------------------
# >>> SUBSTITUA ESTES VALORES PARA TESTES LOCAIS <<<
TELEGRAM_TOKEN = "SEU_TOKEN_DE_TESTE_AQUI"            # ex.: "123456789:AAE...XYZ"
TELEGRAM_CHAT_ID = "@EliteMaster"                     # ex.: "@EliteMaster" ou "-1001234567890"
TELEGRAM_CHAT_ID_ALT2 = ""                            # opcional (deixe vazio se não quiser)
# -------------------------------------------------------------------------

OPENLIGA_BASE = "https://api.openligadb.de"

# Mapa de ligas. Note: OpenLiga tem ligas específicas (o mapeamento é um exemplo).
LEAGAS_MAP = {
    "Bundesliga (Alemanha)": "bl1",
    "2. Bundesliga (Alemanha)": "bl2",
    "DFB-Pokal (Alemanha)": "dfb"
    # Você pode adicionar mais mapeamentos compatíveis com OpenLiga
}

# arquivos de persistência
ALERTAS_PATH = "alertas.json"   # log de envios
TOP3_PATH = "top3.json"         # histórico de top3 enviados

# ----------------------------
# Logging & page config
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EliteMasterDarkAuto")

st.set_page_config(page_title="🏆 ELITE MASTER DARK — Alertas Automáticos", layout="wide")

# ----------------------------
# CSS Dark responsivo (limpo)
# ----------------------------
st.markdown(
    """
    <style>
    /* Dark background and clean cards */
    .reportview-container, .main, .block-container {
        background: linear-gradient(180deg,#0e1116 0%, #0b0f14 100%);
        color: #e6eef6;
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.16));
        padding: 14px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.03);
        box-shadow: 0 6px 22px rgba(0,0,0,0.6);
        margin-bottom: 12px;
    }
    .header { font-size:20px; color:#ffd86b; font-weight:700; }
    .muted { color: #9fb0c4; font-size:13px; }
    .small { font-size:13px; color:#9fb0c4; }
    .btn-primary > button { background: linear-gradient(90deg,#ffd86b,#ffb84d); color: #0b0f14; border-radius:8px; }
    @media (max-width: 600px) {
        .header { font-size:18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers: persistência
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
# Decorator retries simples
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
# Telegram: Montagem e envio (Modelo B - Premium/VIP)
# ----------------------------
def montar_mensagem_alerta_vip(home: str, away: str, faixa: str, prob_pct: float, estimativa: float, liga: str, hora_brt: str) -> str:
    faixa_text = faixa if str(faixa).startswith("+") else f"+{faixa}"
    msg = (
        "🔥 *ALERTA ELITE MASTER* 🔥\n\n"
        f"🏟️ *{home} x {away}*\n"
        f"⚽ Tendência: *{faixa_text} Gols*\n"
        f"📈 Probabilidade: *{prob_pct:.0f}%*\n"
        f"💰 Estimativa: *{estimativa:.2f}*  |  {liga}  |  {hora_brt} BRT\n"
    )
    return msg

def montar_mensagem_resultado_vip(home: str, away: str, faixa: str, score: str, resultado: str) -> str:
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

def enviar_para_todos(token: str, chat_ids: List[str], text: str) -> List[Tuple[str, Optional[str]]]:
    results = []
    for cid in chat_ids:
        if not cid:
            continue
        try:
            enviar_telegram_raw(token, cid, text)
            results.append((cid, "ok"))
            time.sleep(0.4)
        except Exception as e:
            logger.warning(f"Erro ao enviar para {cid}: {e}")
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
        final = None
        for r in j.get("matchResults", []):
            if r.get("resultTypeID") == 2:
                final = (r.get("pointsTeam1",0), r.get("pointsTeam2",0))
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
        t1 = j.get("team1",{}).get("teamName")
        t2 = j.get("team2",{}).get("teamName")
        if {t1, t2} == {home, away}:
            for r in j.get("matchResults", []):
                if r.get("resultTypeID") == 2:
                    gols = (r.get("pointsTeam1",0), r.get("pointsTeam2",0))
                    confrontos.append((j.get("matchDateTimeUTC") or j.get("matchDateTime"), gols[0]+gols[1]))
                    break
    if not confrontos:
        return {"media_gols": 0, "total_jogos": 0}
    confrontos = sorted(confrontos, key=lambda x: x[0] or "", reverse=True)[:max_jogos]
    total_p = 0; total_w = 0
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
# Seleção Top3 distinta (prioridade +1.5 -> +2.5 -> +3.5)
# ----------------------------
def selecionar_top3_distintos(partidas_info: List[dict], max_por_faixa=3):
    if not partidas_info:
        return [], [], []
    def get_num(d,k):
        v = d.get(k, 0)
        try: return float(v) if v is not None else 0.0
        except: return 0.0
    selected_ids = set()
    selected_teams = set()
    def safe_names(m):
        return str(m.get("home","")).strip(), str(m.get("away","")).strip()
    def allocate(prob_key, other_keys):
        candidatos = [m for m in partidas_info if str(m.get("fixture_id")) not in selected_ids]
        preferred = [m for m in candidatos if get_num(m, prob_key) >= max([get_num(m, o) for o in other_keys])]
        def sort_key(x):
            return (get_num(x, prob_key), get_num(x, prob_key.replace("prob","conf")), x.get("estimativa",0))
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
# Conferência OpenLiga (reconsulta)
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
        logger.exception("Erro na conferência OpenLiga")
        return None

# ----------------------------
# Lógica automática: calcula -> envia -> registra -> depois confere resultados
# ----------------------------
def executar_fluxo_automatico(data_alvo: date, ligas_map: Dict[str,str], temporada: str, top_n_per_faixa=3, token:str=None, chats:List[str]=None):
    """
    - busca jogos do dia para todas ligas configuradas
    - calcula estimativas/probs
    - seleciona top por faixa (não repete fixtures entre faixas)
    - envia automaticamente (apenas se ainda não foi enviado no dia)
    - registra histórico em top3.json e alertas.json
    """
    if token is None or not chats:
        logger.warning("Token ou chats não configurados -> não envia automaticamente.")
        return {"sent": False, "reason": "token_or_chats_missing"}

    # carregar histórico para evitar reenvio
    top3_hist = carregar_top3()
    alert_log = carregar_alertas()
    alert_log.setdefault("log", [])

    todos_jogos = []
    medias_por_liga = {}
    for liga_nome, liga_id in ligas_map.items():
        try:
            jogos_hist = obter_jogos_liga_temporada(liga_id, temporada)
            medias_por_liga[liga_id] = calcular_media_gols_times(jogos_hist)
            jogos_do_dia = filtrar_jogos_por_data(jogos_hist, data_alvo)
            for j in jogos_do_dia:
                j["_liga_id"] = liga_id; j["_liga_nome"]=liga_nome; j["_temporada"]=temporada
                todos_jogos.append(j)
        except Exception as e:
            logger.warning(f"Erro ao coletar liga {liga_nome}: {e}")

    if not todos_jogos:
        logger.info("Nenhum jogo no dia para as ligas selecionadas.")
        return {"sent": False, "reason": "no_games"}

    partidas_info = []
    for match in todos_jogos:
        home = match.get("team1",{}).get("teamName")
        away = match.get("team2",{}).get("teamName")
        hora_dt = parse_data_openliga(match.get("matchDateTimeUTC") or match.get("matchDateTime"))
        hora_brt = (hora_dt - timedelta(hours=3)).strftime("%H:%M") if hora_dt else "??:??"
        liga_id = match.get("_liga_id")
        jogos_hist_liga = obter_jogos_liga_temporada(liga_id, temporada)
        medias_liga = medias_por_liga.get(liga_id, {})
        media_h2h = media_gols_h2h(home, away, jogos_hist_liga, max_jogos=5)
        media_casa = medias_liga.get(home, {"media_gols_marcados":1.5,"media_gols_sofridos":1.2})
        media_fora = medias_liga.get(away, {"media_gols_marcados":1.4,"media_gols_sofridos":1.1})
        estimativa = calcular_estimativa(media_h2h, media_casa, media_fora, peso_h2h=0.3)
        p15 = prob_over_k(estimativa, 1.5); p25 = prob_over_k(estimativa, 2.5); p35 = prob_over_k(estimativa, 3.5)
        partidas_info.append({
            "fixture_id": match.get("matchID"),
            "home": home, "away": away,
            "hora": hora_brt,
            "competicao": match.get("_liga_nome"),
            "estimativa": estimativa,
            "prob_1_5": round(p15*100,1),
            "prob_2_5": round(p25*100,1),
            "prob_3_5": round(p35*100,1),
            "conf_1_5": prob_to_conf(p15),
            "conf_2_5": prob_to_conf(p25),
            "conf_3_5": prob_to_conf(p35),
            "liga_id": liga_id,
            "temporada": temporada
        })

    # Seleciona Tops distintos
    top_15, top_25, top_35 = selecionar_top3_distintos(partidas_info, max_por_faixa=top_n_per_faixa)

    # Verificar se já foi enviado hoje (baseado em data_envio) para evitar duplicidade
    hoje_str = data_alvo.isoformat()
    ja_enviado_hoje = False
    for lote in top3_hist:
        if lote.get("data_envio") == hoje_str:
            ja_enviado_hoje = True
            break

    sent_summary = {"sent_batches": [], "skipped_if_already_sent": ja_enviado_hoje}

    if ja_enviado_hoje:
        logger.info("Top3 já enviado hoje, não envia novamente automaticamente.")
    else:
        # montar e enviar mensagens (uma por faixa)
        mensagens = []
        if top_15:
            msg = f"🔥 *TOP {len(top_15)} +1.5 — ELITE MASTER* 🔥\n\n"
            for idx, j in enumerate(top_15, start=1):
                msg += f"{idx}. *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n   • Est: {j['estimativa']:.2f} | P(+1.5): *{j['prob_1_5']:.1f}%* | Conf: *{j['conf_1_5']:.0f}%*\n"
            mensagens.append(("1.5", msg))
        if top_25:
            msg = f"🔥 *TOP {len(top_25)} +2.5 — ELITE MASTER* 🔥\n\n"
            for idx, j in enumerate(top_25, start=1):
                msg += f"{idx}. *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n   • Est: {j['estimativa']:.2f} | P(+2.5): *{j['prob_2_5']:.1f}%* | Conf: *{j['conf_2_5']:.0f}%*\n"
            mensagens.append(("2.5", msg))
        if top_35:
            msg = f"🔥 *TOP {len(top_35)} +3.5 — ELITE MASTER* 🔥\n\n"
            for idx, j in enumerate(top_35, start=1):
                msg += f"{idx}. *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n   • Est: {j['estimativa']:.2f} | P(+3.5): *{j['prob_3_5']:.1f}%* | Conf: *{j['conf_3_5']:.0f}%*\n"
            mensagens.append(("3.5", msg))

        for faixa_label, texto in mensagens:
            try:
                res = enviar_para_todos(token, chats, texto)
                sent_summary["sent_batches"].append({"faixa": faixa_label, "result": res})
                # pequeno delay
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Erro ao enviar faixa {faixa_label}: {e}")

        # registrar no histórico top3
        novo_top = {
            "data_envio": hoje_str,
            "hora_envio": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temporada": temporada,
            "top_1_5": top_15,
            "top_2_5": top_25,
            "top_3_5": top_35,
            "conferido": False  # será atualizado depois
        }
        top3_hist.append(novo_top)
        salvar_top3(top3_hist)
        # registrar log de alertas
        alert_log["log"].append({"when": datetime.now().isoformat(), "action": "auto_send", "data_envio": hoje_str, "tops": {"1.5":[x["fixture_id"] for x in top_15], "2.5":[x["fixture_id"] for x in top_25], "3.5":[x["fixture_id"] for x in top_35]}})
        salvar_alertas(alert_log)
        logger.info("Top3 enviado automaticamente e registrado no histórico.")

    # Retornar resumo
    return {"sent_summary": sent_summary, "tot_matches": len(partidas_info)}

# ----------------------------
# Conferência automática de envios anteriores (para enviar resultados)
# ----------------------------
def executar_conferencia_automatica(token: str, chats: List[str]):
    """
    Reconsulta itens de top3.json que não foram conferidos (conferido == False)
    e envia conferência para Telegram. Marca 'conferido': True quando processado.
    """
    top3_hist = carregar_top3()
    updated = False
    resumo_global = []
    for lote in top3_hist:
        if lote.get("conferido"):
            continue  # já conferido
        # para cada faixa no lote
        data_envio = lote.get("data_envio")
        temporada = lote.get("temporada", "")
        detalhes_msg = []
        total_g = total_r = 0
        for faixa_key in ("top_1_5","top_2_5","top_3_5"):
            lista = lote.get(faixa_key, [])
            label = faixa_key.split("_",1)[1].replace("_",".")
            if not lista:
                continue
            for j in lista:
                info = conferir_jogo_openliga(j.get("fixture_id"), j.get("liga_id"), temporada, label)
                if not info:
                    detalhes_msg.append(f"🏟️ {j.get('home')} x {j.get('away')} — sem resultado")
                    continue
                if info.get("total_gols") is None:
                    detalhes_msg.append(f"🏟️ {info['home']} — Em andamento / sem resultado")
                    continue
                # enviar resultado por partida (ou agrupar por faixa)
                detalhes_msg.append(f"🏟️ {info['home']} {info.get('score','')} {info['away']} → {info['resultado']}")
                if "GREEN" in info["resultado"]:
                    total_g += 1
                else:
                    total_r += 1
        if not detalhes_msg:
            continue
        header = f"✅ RESULTADOS - CONFERÊNCIA (Lote: {data_envio})\n\n"
        body = "\n".join(detalhes_msg)
        resumo = f"\n\nResumo: 🟢 {total_g} GREEN | 🔴 {total_r} RED"
        msg = header + body + resumo
        try:
            enviar_para_todos(token, chats, msg)
            # marcar lote como conferido
            lote["conferido"] = True
            updated = True
            resumo_global.append({"lote": data_envio, "greens": total_g, "reds": total_r})
            # pequeno delay
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Erro ao enviar conferência do lote {data_envio}: {e}")
    if updated:
        salvar_top3(top3_hist)
    return resumo_global

# ----------------------------
# UI: Abas e execução automática ao carregar
# ----------------------------
st.markdown('<div class="header">🏆 ELITE MASTER — Dark Premium (Auto Alerts)</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Envio automático de Top3 + conferência automática de resultados (teste local)</div>', unsafe_allow_html=True)
st.markdown("---")

tabs = st.tabs(["🏠 Dashboard", "⚽ Jogos do Dia", "🔥 Top 3 (Auto)", "📊 Histórico / Conferência", "⚙️ Configurações"])

# ----------------------------
# Aba Configurações (mostra/permite editar tokens no app)
# ----------------------------
with tabs[4]:
    st.markdown('<div class="card"><b>Configurações (teste local)</b></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        # mostrar os tokens atuais e permitir alterar para sessão
        token_sess = st.text_input("Telegram Token (coloque seu token aqui para testes)", value=TELEGRAM_TOKEN)
        chat_sess = st.text_input("Telegram Chat ID (ex: @EliteMaster ou -100...)", value=TELEGRAM_CHAT_ID)
        chat_alt_sess = st.text_input("Telegram Chat ID alt (opcional)", value=TELEGRAM_CHAT_ID_ALT2)
        temporada_padrao = st.selectbox("Temporada padrão:", ["2022","2023","2024","2025"], index=2)
        top_n_conf = st.slider("Top N por faixa (ao enviar)", 1, 5, 3)
    with col2:
        if st.button("Salvar configurações nesta sessão"):
            st.session_state["TG_TOKEN"] = token_sess.strip()
            st.session_state["TG_CHAT"] = chat_sess.strip()
            st.session_state["TG_CHAT_ALT"] = chat_alt_sess.strip()
            st.session_state["TEMP_PADRAO"] = temporada_padrao
            st.session_state["TOP_N"] = top_n_conf
            st.success("Configurações salvas na sessão (temporário).")
        if st.button("Testar envio (mensagem de teste)"):
            tok = token_sess.strip()
            chats = [chat_sess.strip()] + ([chat_alt_sess.strip()] if chat_alt_sess.strip() else [])
            if not tok or not chats[0]:
                st.error("Token ou chat principal ausente.")
            else:
                try:
                    txt = "🔥 *ELITE MASTER - TESTE DE ENVIO* 🔥\n\nTeste automático do app."
                    res = enviar_para_todos(tok, chats, txt)
                    st.write(res)
                    st.success("Teste enviado.")
                except Exception as e:
                    st.error(f"Erro envio teste: {e}")

    st.markdown("---")
    st.markdown("**Nota:** Para produção mova os tokens para variáveis de ambiente e não mantenha no código.")
    st.markdown("**Arquivos:**")
    st.code(f"{TOP3_PATH}  (histórico de envios)\n{ALERTAS_PATH}  (log de envios)")

# ----------------------------
# Aba Dashboard
# ----------------------------
with tabs[0]:
    st.markdown('<div class="card"><b>Dashboard</b></div>', unsafe_allow_html=True)
    top3_hist = carregar_top3()
    log = carregar_alertas().get("log", [])
    total_envios = len(top3_hist)
    total_alerts = len(log)
    greens = reds = 0
    for lote in top3_hist:
        for k in ("top_1_5","top_2_5","top_3_5"):
            for j in lote.get(k, []):
                if j.get("resultado"):
                    if "GREEN" in j.get("resultado"):
                        greens += 1
                    else:
                        reds += 1
    taxa = f"{(greens/(greens+reds)*100):.1f}%" if (greens+reds)>0 else "N/A"
    c1, c2, c3 = st.columns(3)
    c1.metric("Top3 enviados (lotes)", total_envios)
    c2.metric("Entradas no log", total_alerts)
    c3.metric("Taxa acerto (conferidos)", taxa)
    st.markdown("### Últimos envios (preview)")
    preview = []
    for lote in reversed(top3_hist[-8:]):
        preview.append({"Envio": f"{lote.get('data_envio')} {lote.get('hora_envio')}", "Itens": sum(len(lote.get(k,[])) for k in lote if k.startswith("top_"))})
    if preview:
        st.table(preview)
    else:
        st.info("Nenhum envio registrado ainda.")

# ----------------------------
# Aba Jogos do Dia
# ----------------------------
with tabs[1]:
    st.markdown('<div class="card"><b>Jogos do Dia</b></div>', unsafe_allow_html=True)
    colA, colB = st.columns([2,1])
    with colA:
        data_selecionada = st.date_input("Data", value=datetime.utcnow().date())
        liga_selecionada = st.selectbox("Liga (OpenLiga)", list(LEAGAS_MAP.keys()))
    with colB:
        temporada_input = st.selectbox("Temporada (para médias)", ["2022","2023","2024","2025"], index=2)
    if st.button("Carregar jogos e analisar (manual)"):
        liga_id = LEAGAS_MAP.get(liga_selecionada)
        try:
            jogos_hist = obter_jogos_liga_temporada(liga_id, temporada_input)
            jogos_dia = filtrar_jogos_por_data(jogos_hist, data_selecionada)
            if not jogos_dia:
                st.info("Nenhum jogo encontrado para a data/league.")
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
                    rows.append({"fixture_id": m.get("matchID"), "home": home, "away": away, "hora": hora_brt, "estimativa": estim, "prob_1_5": round(p15*100,1), "prob_2_5": round(p25*100,1), "prob_3_5": round(p35*100,1), "liga_id": liga_id, "temporada": temporada_input})
                st.session_state["JOGOS_DO_DIA"] = rows
                st.success(f"{len(rows)} jogos carregados.")
        except Exception as e:
            st.error(f"Erro: {e}")

    jogos = st.session_state.get("JOGOS_DO_DIA", [])
    if jogos:
        st.markdown("### Jogos carregados")
        display = []
        for j in jogos:
            display.append({"Jogo": f"{j['home']} x {j['away']}", "Hora (BRT)": j["hora"], "Estimativa": j["estimativa"], "P(+1.5)": f"{j['prob_1_5']}%", "P(+2.5)": f"{j['prob_2_5']}%", "P(+3.5)": f"{j['prob_3_5']}%"})
        st.table(display)
        st.info("O envio automático ocorrerá ao carregar a página (se tokens estiverem configurados).")

# ----------------------------
# Aba Top 3 (Auto)
# ----------------------------
with tabs[2]:
    st.markdown('<div class="card"><b>Top 3 (Automático)</b></div>', unsafe_allow_html=True)
    st.markdown("O sistema calcula e envia automaticamente os Top3 do dia (uma vez por dia). Para testes, insira token/chat na aba Configurações e recarregue a página.")
    # Mostrar resumo do último envio (se houver)
    top3_hist = carregar_top3()
    if top3_hist:
        ultimo = top3_hist[-1]
        st.write("Último envio:", ultimo.get("data_envio"), ultimo.get("hora_envio"))
        for k in ("top_1_5","top_2_5","top_3_5"):
            lst = ultimo.get(k, [])
            if lst:
                st.write(f"### {k.replace('_',' ')}")
                for j in lst:
                    st.write(f"- {j['home']} x {j['away']}  |  P(+{k.split('_')[1].replace('_','.')}): {j.get(f'prob_{k.split('_')[1]}','-')}%  |  Est: {j.get('estimativa')}")
    else:
        st.info("Nenhum envio automático registrado ainda.")

# ----------------------------
# Aba Histórico / Conferência
# ----------------------------
with tabs[3]:
    st.markdown('<div class="card"><b>Histórico e Conferência</b></div>', unsafe_allow_html=True)
    top3_hist = carregar_top3()
    if not top3_hist:
        st.info("Sem histórico.")
    else:
        options = [f"{i+1} - {t['data_envio']} ({t['hora_envio']})" for i,t in enumerate(top3_hist)]
        sel = st.selectbox("Selecione lote:", options)
        idx = options.index(sel)
        lote = top3_hist[idx]
        st.write("Lote:", lote.get("data_envio"), lote.get("hora_envio"))
        for k in [k for k in lote.keys() if k.startswith("top_")]:
            st.write(f"## {k.replace('_',' ')}")
            for j in lote.get(k, []):
                st.write(f"- {j.get('home')} x {j.get('away')}  |  Est: {j.get('estimativa')}  | Resultado: {j.get('resultado','-')}")
        if st.button("Forçar conferir este lote e enviar resultados (Telegram)"):
            token = st.session_state.get("TG_TOKEN") or TELEGRAM_TOKEN
            chat_main = st.session_state.get("TG_CHAT") or TELEGRAM_CHAT_ID
            chat_alt = st.session_state.get("TG_CHAT_ALT") or TELEGRAM_CHAT_ID_ALT2
            chats = [chat_main] + ([chat_alt] if chat_alt else [])
            resumo = []
            for k in [k for k in lote.keys() if k.startswith("top_")]:
                label = k.split("_",1)[1].replace("_",".")
                for j in lote.get(k, []):
                    info = conferir_jogo_openliga(j.get("fixture_id"), j.get("liga_id"), lote.get("temporada",""), label)
                    if not info:
                        resumo.append({"fixture": j.get("fixture_id"), "status": "no_result"})
                        continue
                    # enviar resultado por partida
                    msg = montar_mensagem_resultado_vip(info["home"], info["away"], label, info.get("score","-"), info.get("resultado","-"))
                    enviar_para_todos(token, chats, msg)
                    resumo.append({"fixture": j.get("fixture_id"), "resultado": info.get("resultado","-")})
            st.json(resumo)

# ----------------------------
# Execução automática ao carregar a página
# ----------------------------
# Obter token/chat da sessão (salvo pela UI) ou dos valores no topo
token_use = st.session_state.get("TG_TOKEN") or TELEGRAM_TOKEN
chat_use = st.session_state.get("TG_CHAT") or TELEGRAM_CHAT_ID
chat_alt_use = st.session_state.get("TG_CHAT_ALT") or TELEGRAM_CHAT_ID_ALT2
chats_list = [chat_use] + ([chat_alt_use] if chat_alt_use else [])

# Executar envio automático do dia atual (somente se token/chat estiverem presentes)
try:
    resultado_auto = executar_fluxo_automatico(date.today(), LEAGAS_MAP, st.session_state.get("TEMP_PADRAO","2024"), top_n_per_faixa=st.session_state.get("TOP_N",3), token=token_use, chats=chats_list)
    # Em seguida, executar conferência automática de envios anteriores (que ainda não foram conferidos)
    resultado_conf = executar_conferencia_automatica(token_use, chats_list)
    # Mostrar um resumo discreto no footer
    st.sidebar.markdown("### Auto: status")
    st.sidebar.write(resultado_auto.get("sent_summary", {}).get("sent_batches", []) if isinstance(resultado_auto, dict) else resultado_auto)
    st.sidebar.write({"conferencias": resultado_conf})
except Exception as e:
    logger.warning(f"Erro no fluxo automático: {e}")
    st.sidebar.error("Erro no fluxo automático. Veja logs.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown('<div class="small">Elite Master • Dark Premium • Envio Automático • Use apenas para testes com token no código</div>', unsafe_allow_html=True)
