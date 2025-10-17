# Futebol_Alertas_OpenLiga_Top3.py
import streamlit as st
from datetime import datetime, timedelta, date, timezone
import requests
import os
import json
import math
import time
import logging
from functools import lru_cache, wraps
from typing import List, Dict, Any, Tuple, Optional

# =============================
# Configuração / Logging
# =============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FutebolAlertas")

st.set_page_config(page_title="⚽ Alertas Top3 (OpenLigaDB) - Alemanha", layout="wide")
st.title("⚽ Alertas Top3 por Faixa (+1.5 / +2.5 / +3.5) — OpenLigaDB (Alemanha)")

# =============================
# Configurações (ajuste para produção)
# =============================
OPENLIGA_BASE = "https://api.openligadb.de"

ligas_openliga = {
    "Bundesliga (Alemanha)": "bl1",
    "2. Bundesliga (Alemanha)": "bl2",
    "DFB-Pokal (Alemanha)": "dfb"
}

# Recomendado: definir TELEGRAM_TOKEN e CHAT_IDs via variáveis de ambiente.
# Para compatibilidade, se não existir env var, usaremos os valores antigos (mas é melhor mover para env).
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003073115320")
TELEGRAM_CHAT_ID_ALT2 = os.getenv("TELEGRAM_CHAT_ID_ALT2", "-1002932611974")
BASE_URL_TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

ALERTAS_PATH = "alertas.json"
TOP3_PATH = "top3.json"

# =============================
# Utilitários: retries simples
# =============================
def with_retries(max_attempts=3, backoff=1.0):
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
            logger.error(f"All {max_attempts} attempts failed for {fn.__name__}: {last_exc}")
            raise last_exc
        return wrapper
    return deco

# =============================
# Persistência (JSON)
# =============================
def carregar_alertas() -> Dict[str, Any]:
    if os.path.exists(ALERTAS_PATH):
        with open(ALERTAS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def salvar_alertas(alertas: Dict[str, Any]):
    with open(ALERTAS_PATH, "w", encoding="utf-8") as f:
        json.dump(alertas, f, ensure_ascii=False, indent=2)

def carregar_top3() -> List[Dict[str, Any]]:
    if os.path.exists(TOP3_PATH):
        with open(TOP3_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def salvar_top3(lista: List[Dict[str, Any]]):
    with open(TOP3_PATH, "w", encoding="utf-8") as f:
        json.dump(lista, f, ensure_ascii=False, indent=2)

# =============================
# Envio Telegram (padronizado)
# =============================
@with_retries(max_attempts=3, backoff=0.7)
def enviar_telegram(msg: str, chat_id: str = TELEGRAM_CHAT_ID, parse_mode: str = "Markdown") -> dict:
    """
    Envia mensagem ao Telegram. Lança exceção se falhar depois das tentativas.
    """
    payload = {
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }
    r = requests.post(BASE_URL_TG, data=payload, timeout=10)
    r.raise_for_status()
    logger.info(f"Mensagem enviada ao chat {chat_id} (tamanho {len(msg)} chars)")
    return r.json()

def enviar_para_todos(msg: str):
    """Envia para os chat ids configurados (se existirem)."""
    errors = []
    for cid in {TELEGRAM_CHAT_ID, TELEGRAM_CHAT_ID_ALT2}:
        if not cid:
            continue
        try:
            enviar_telegram(msg, chat_id=cid)
        except Exception as e:
            errors.append((cid, str(e)))
            logger.warning(f"Erro ao enviar para {cid}: {e}")
    return errors

# =============================
# OpenLigaDB helpers (com cache)
# =============================
@lru_cache(maxsize=32)
@with_retries(max_attempts=2, backoff=0.5)
def obter_jogos_liga_temporada(liga_id: str, temporada: str) -> List[dict]:
    url = f"{OPENLIGA_BASE}/getmatchdata/{liga_id}/{temporada}"
    logger.info(f"Buscando {url}")
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        return r.json()
    r.raise_for_status()
    return []

def parse_data_openliga_to_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # normalizar Z -> +00:00 para fromisoformat
        if s.endswith("Z"):
            s2 = s.replace("Z", "+00:00")
        else:
            s2 = s
        dt = datetime.fromisoformat(s2)
        return dt
    except Exception:
        try:
            return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None

def filtrar_jogos_por_data(jogos_all: List[dict], data_obj: date) -> List[dict]:
    out = []
    for j in jogos_all:
        date_str = j.get("matchDateTimeUTC") or j.get("matchDateTime")
        dt = parse_data_openliga_to_datetime(date_str)
        if not dt:
            continue
        if dt.date() == data_obj:
            out.append(j)
    return out

# =============================
# Estatística / Poisson (mesma lógica com pequenas marcas)
# =============================
def calcular_media_gols_times(jogos_hist: List[dict]) -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    for j in jogos_hist:
        home = j.get("team1", {}).get("teamName")
        away = j.get("team2", {}).get("teamName")
        placar = None
        for r in j.get("matchResults", []):
            if r.get("resultTypeID") == 2:
                placar = (r.get("pointsTeam1", 0), r.get("pointsTeam2", 0))
                break
        if not placar:
            continue
        stats.setdefault(home, {"marcados": [], "sofridos": []})
        stats.setdefault(away, {"marcados": [], "sofridos": []})
        stats[home]["marcados"].append(placar[0])
        stats[home]["sofridos"].append(placar[1])
        stats[away]["marcados"].append(placar[1])
        stats[away]["sofridos"].append(placar[0])

    medias = {}
    for time, gols in stats.items():
        media_marcados = sum(gols["marcados"]) / len(gols["marcados"]) if gols["marcados"] else 1.5
        media_sofridos = sum(gols["sofridos"]) / len(gols["sofridos"]) if gols["sofridos"] else 1.2
        medias[time] = {"media_gols_marcados": round(media_marcados, 2), "media_gols_sofridos": round(media_sofridos, 2)}
    return medias

def media_gols_confrontos_diretos_openliga(home: str, away: str, jogos_hist: List[dict], max_jogos=5) -> dict:
    confrontos = []
    for j in jogos_hist:
        t1 = j.get("team1", {}).get("teamName")
        t2 = j.get("team2", {}).get("teamName")
        if {t1, t2} == {home, away}:
            for r in j.get("matchResults", []):
                if r.get("resultTypeID") == 2:
                    gols = (r.get("pointsTeam1", 0), r.get("pointsTeam2", 0))
                    total = gols[0] + gols[1]
                    data_str = j.get("matchDateTimeUTC") or j.get("matchDateTime")
                    confrontos.append((data_str, total))
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
    media_ponderada = round(total_pontos / total_peso, 2) if total_peso else 0
    return {"media_gols": media_ponderada, "total_jogos": len(confrontos)}

def calcular_estimativa_consolidada(media_h2h: dict, media_casa: dict, media_fora: dict, peso_h2h=0.3) -> float:
    media_casa_marcados = media_casa.get("media_gols_marcados", 1.5)
    media_casa_sofridos = media_casa.get("media_gols_sofridos", 1.2)
    media_fora_marcados = media_fora.get("media_gols_marcados", 1.4)
    media_fora_sofridos = media_fora.get("media_gols_sofridos", 1.1)
    media_time_casa = media_casa_marcados + media_fora_sofridos
    media_time_fora = media_fora_marcados + media_casa_sofridos
    estimativa_base = (media_time_casa + media_time_fora) / 2
    h2h_media = media_h2h.get("media_gols", estimativa_base) if media_h2h.get("total_jogos", 0) > 0 else estimativa_base
    estimativa_final = (1 - peso_h2h) * estimativa_base + peso_h2h * h2h_media
    return round(estimativa_final, 2)

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

def confidence_from_prob(prob: float) -> float:
    conf = 50 + (prob - 0.5) * 100
    conf = max(30, min(95, conf))
    return round(conf, 0)

# =============================
# Conferência via OpenLigaDB (reconsulta)
# =============================
def conferir_jogo_openliga(fixture_id: Any, liga_id: str, temporada: str, tipo_threshold: str) -> Optional[dict]:
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
            return {
                "home": home,
                "away": away,
                "total_gols": None,
                "aposta": f"+{tipo_threshold}",
                "resultado": "Em andamento / sem resultado"
            }
        total = final[0] + final[1]
        if tipo_threshold == "1.5":
            green = total >= 2
        elif tipo_threshold == "2.5":
            green = total >= 3
        else:
            green = total >= 4
        return {
            "home": home,
            "away": away,
            "total_gols": total,
            "aposta": f"+{tipo_threshold}",
            "resultado": "🟢 GREEN" if green else "🔴 RED",
            "score": f"{final[0]} x {final[1]}"
        }
    except Exception as e:
        logger.exception("Erro ao conferir jogo")
        return None

# =============================
# Seleção Top3 distintos (prioridade: +1.5 -> +2.5 -> +3.5)
# =============================
def selecionar_top3_distintos(partidas_info: List[dict], max_por_faixa=3, prefer_best_fit=True) -> Tuple[List[dict], List[dict], List[dict]]:
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

    def safe_team_names(m):
        return str(m.get("home", "")).strip(), str(m.get("away", "")).strip()

    def allocate(prefix, other_prefixes):
        prob_key = f"prob_{prefix}"
        candidatos = [m for m in partidas_info if str(m.get("fixture_id")) not in selected_ids]

        preferred = []
        if prefer_best_fit:
            for m in candidatos:
                cur = get_num(m, prob_key)
                others = [get_num(m, f"prob_{o}") for o in other_prefixes]
                if cur >= max(others):
                    preferred.append(m)

        def sort_key(match):
            prob = get_num(match, prob_key)
            conf = get_num(match, prob_key.replace("prob", "conf"))
            est = get_num(match, "estimativa")
            return (prob, conf, est)

        preferred_sorted = sorted(preferred, key=sort_key, reverse=True)
        remaining = [m for m in candidatos if m not in preferred_sorted]
        remaining_sorted = sorted(remaining, key=sort_key, reverse=True)

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

    # Alocação seguindo a prioridade pedida pelo UI: +1.5 primeiro
    top_15 = allocate("1_5", other_prefixes=["2_5", "3_5"])
    top_25 = allocate("2_5", other_prefixes=["1_5", "3_5"])
    top_35 = allocate("3_5", other_prefixes=["1_5", "2_5"])

    return top_15, top_25, top_35

# =============================
# UI Streamlit
# =============================
# --- Painel esquerdo: opções ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configuração")
    temporada_hist = st.selectbox("Temporada (para médias):", ["2022", "2023", "2024", "2025"], index=2)
    data_selecionada = st.date_input("Data dos jogos:", value=datetime.utcnow().date())
    hoje_str = data_selecionada.strftime("%Y-%m-%d")

    st.markdown("**Observação:** listas distintas — *um jogo não será repetido entre faixas*. Prioridade: **+1.5 → +2.5 → +3.5**.")
    envio_opcao = st.radio("Como enviar as mensagens ao Telegram?", ("Uma mensagem por faixa (separadas)", "Mensagem consolidada (todas faixas em 1)"), index=0)
    btn_buscar_enviar = st.button("Buscar jogos do dia e gerar Top3")

with col2:
    st.subheader("Últimos Top3 enviados")
    top3_salvos = carregar_top3()
    if top3_salvos:
        st.write(f"Total envios registrados: {len(top3_salvos)}")
        ultimos = top3_salvos[-10:][::-1]  # últimos 10
        for idx, t in enumerate(ultimos, start=1):
            st.markdown(f"**{idx}** — {t['data_envio']} ({t['hora_envio']}) — Temporada: {t.get('temporada')}")
    else:
        st.info("Nenhum Top3 registrado ainda.")

# ---------- Ação principal: buscar, calcular, selecionar e enviar ----------
if btn_buscar_enviar:
    with st.spinner("Buscando jogos e calculando probabilidades..."):
        try:
            # 1) coletar jogos e médias por liga
            jogos_por_liga = {}
            medias_por_liga = {}
            for liga_nome, liga_id in ligas_openliga.items():
                jogos_hist = obter_jogos_liga_temporada(liga_id, temporada_hist)
                jogos_por_liga[liga_id] = jogos_hist
                medias_por_liga[liga_id] = calcular_media_gols_times(jogos_hist)

            # 2) agregar jogos do dia
            jogos_do_dia = []
            for liga_nome, liga_id in ligas_openliga.items():
                jogos_hist = jogos_por_liga.get(liga_id, [])
                filtrados = filtrar_jogos_por_data(jogos_hist, data_selecionada)
                for j in filtrados:
                    j["_liga_id"] = liga_id
                    j["_liga_nome"] = liga_nome
                    j["_temporada"] = temporada_hist
                    jogos_do_dia.append(j)

            if not jogos_do_dia:
                st.info("Nenhum jogo encontrado para essa data nas ligas selecionadas.")
            else:
                partidas_info = []
                for match in jogos_do_dia:
                    home = match.get("team1", {}).get("teamName")
                    away = match.get("team2", {}).get("teamName")
                    hora_dt = parse_data_openliga_to_datetime(match.get("matchDateTimeUTC") or match.get("matchDateTime"))
                    # Ajuste simples para exibir horário local aproximado (BRT = UTC-3)
                    hora_formatada = (hora_dt - timedelta(hours=3)).strftime("%H:%M") if hora_dt else "??:??"
                    liga_id = match.get("_liga_id")
                    jogos_hist_liga = jogos_por_liga.get(liga_id, [])
                    medias_liga = medias_por_liga.get(liga_id, {})

                    media_h2h = media_gols_confrontos_diretos_openliga(home, away, jogos_hist_liga, max_jogos=5)
                    media_casa = medias_liga.get(home, {"media_gols_marcados":1.5, "media_gols_sofridos":1.2})
                    media_fora = medias_liga.get(away, {"media_gols_marcados":1.4, "media_gols_sofridos":1.1})

                    estimativa = calcular_estimativa_consolidada(media_h2h, media_casa, media_fora, peso_h2h=0.3)

                    p15 = prob_over_k(estimativa, 1.5)
                    p25 = prob_over_k(estimativa, 2.5)
                    p35 = prob_over_k(estimativa, 3.5)
                    c15 = confidence_from_prob(p15)
                    c25 = confidence_from_prob(p25)
                    c35 = confidence_from_prob(p35)

                    partidas_info.append({
                        "fixture_id": match.get("matchID"),
                        "home": home, "away": away,
                        "hora": hora_formatada,
                        "competicao": match.get("_liga_nome"),
                        "estimativa": estimativa,
                        "prob_1_5": round(p15*100,1),
                        "prob_2_5": round(p25*100,1),
                        "prob_3_5": round(p35*100,1),
                        "conf_1_5": c15,
                        "conf_2_5": c25,
                        "conf_3_5": c35,
                        "liga_id": liga_id,
                        "temporada": match.get("_temporada")
                    })

                # Seleciona Top3 distintos
                top_15, top_25, top_35 = selecionar_top3_distintos(partidas_info, max_por_faixa=3)

                # --- montar mensagens padronizadas ---
                def montar_bloco_faixa(lista, faixa_label):
                    if not lista:
                        return f"*TOP {faixa_label} - Nenhuma partida selecionada*\n\n"
                    header = f"*TOP {faixa_label} — {hoje_str}*\n"
                    lines = []
                    for i, j in enumerate(lista, start=1):
                        lines.append(f"{i}. *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n   › Est: {j['estimativa']:.2f} | P(+{faixa_label}): *{j[f'prob_{faixa_label.replace('.','_')}']}%* | Conf: *{j[f'conf_{faixa_label.replace('.','_')}']}%*")
                    return header + "\n".join(lines) + "\n"

                # Mais compacto: criar 1 mensagem por faixa ou consolidada
                if envio_opcao.startswith("Uma mensagem por faixa"):
                    msgs = []
                    if top_15:
                        msg1 = f"🔔 *TOP 3 +1.5 GOLS — {hoje_str}*\n\n"
                        for idx, j in enumerate(top_15, start=1):
                            msg1 += (f"{idx}. *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n"
                                     f"   • Est: {j['estimativa']:.2f} | P(+1.5): *{j['prob_1_5']:.1f}%* | Conf: *{j['conf_1_5']:.0f}%*\n")
                        msgs.append(msg1)
                    if top_25:
                        msg2 = f"🔔 *TOP 3 +2.5 GOLS — {hoje_str}*\n\n"
                        for idx, j in enumerate(top_25, start=1):
                            msg2 += (f"{idx}. *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n"
                                     f"   • Est: {j['estimativa']:.2f} | P(+2.5): *{j['prob_2_5']:.1f}%* | Conf: *{j['conf_2_5']:.0f}%*\n")
                        msgs.append(msg2)
                    if top_35:
                        msg3 = f"🔔 *TOP 3 +3.5 GOLS — {hoje_str}*\n\n"
                        for idx, j in enumerate(top_35, start=1):
                            msg3 += (f"{idx}. *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n"
                                     f"   • Est: {j['estimativa']:.2f} | P(+3.5): *{j['prob_3_5']:.1f}%* | Conf: *{j['conf_3_5']:.0f}%*\n")
                        msgs.append(msg3)

                    # enviar todas as mensagens (uma por faixa)
                    send_errors = []
                    for m in msgs:
                        try:
                            enviar_para_todos(m)
                            time.sleep(0.5)  # pequeno delay entre envios
                        except Exception as e:
                            send_errors.append(str(e))
                    if send_errors:
                        st.warning("Houveram erros no envio. Veja logs.")
                    else:
                        st.success("Mensagens enviadas com sucesso (uma por faixa).")
                else:
                    # Mensagem consolidada
                    consolidated = f"🔔 *TOP 3 - Consolidados — {hoje_str}*\n\n"
                    if top_15:
                        consolidated += "• *+1.5*\n"
                        for idx, j in enumerate(top_15, start=1):
                            consolidated += f"  {idx}. *{j['home']} x {j['away']}* — {j['hora']} BRT — P:+1.5 *{j['prob_1_5']}%* | Conf *{j['conf_1_5']}%*\n"
                    if top_25:
                        consolidated += "\n• *+2.5*\n"
                        for idx, j in enumerate(top_25, start=1):
                            consolidated += f"  {idx}. *{j['home']} x {j['away']}* — {j['hora']} BRT — P:+2.5 *{j['prob_2_5']}%* | Conf *{j['conf_2_5']}%*\n"
                    if top_35:
                        consolidated += "\n• *+3.5*\n"
                        for idx, j in enumerate(top_35, start=1):
                            consolidated += f"  {idx}. *{j['home']} x {j['away']}* — {j['hora']} BRT — P:+3.5 *{j['prob_3_5']}%* | Conf *{j['conf_3_5']}%*\n"

                    try:
                        enviar_para_todos(consolidated)
                        st.success("Mensagem consolidada enviada com sucesso.")
                    except Exception as e:
                        st.error(f"Erro ao enviar mensagem consolidada: {e}")

                # salvar histórico top3
                top3_list = carregar_top3()
                novo_top = {
                    "data_envio": hoje_str,
                    "hora_envio": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "temporada": temporada_hist,
                    "top_1_5": top_15,
                    "top_2_5": top_25,
                    "top_3_5": top_35
                }
                top3_list.append(novo_top)
                salvar_top3(top3_list)

                # Exibir tabelas no Streamlit (visão profissional)
                st.markdown("### Resultado - Top Selecionados")
                def tabela_para_display(lista, faixa_label):
                    rows = []
                    for j in lista:
                        rows.append({
                            "Jogo": f"{j['home']} x {j['away']}",
                            "Competição": j['competicao'],
                            "Hora (BRT)": j['hora'],
                            f"P(+{faixa_label})": f"{j[f'prob_{faixa_label.replace('.','_')}']}%",
                            "Conf": f"{j[f'conf_{faixa_label.replace('.','_')}']}%",
                            "Estimativa": j['estimativa']
                        })
                    return rows

                if top_15:
                    st.write("#### +1.5")
                    st.table(tabela_para_display(top_15, "1.5"))
                if top_25:
                    st.write("#### +2.5")
                    st.table(tabela_para_display(top_25, "2.5"))
                if top_35:
                    st.write("#### +3.5")
                    st.table(tabela_para_display(top_35, "3.5"))

        except Exception as e:
            logger.exception("Erro ao gerar Top3")
            st.error(f"Erro ao processar: {e}")

# -----------------------------
# Aba: Jogos históricos
# -----------------------------
st.markdown("---")
st.subheader("📊 Jogos de Temporadas Passadas (OpenLigaDB) — Ligas da Alemanha")
col_a, col_b = st.columns([1, 2])
with col_a:
    temporada_hist2 = st.selectbox("Temporada histórica:", ["2022", "2023", "2024", "2025"], index=2, key="hist2")
    liga_nome_hist = st.selectbox("Escolha a Liga:", list(ligas_openliga.keys()), key="hist_liga")
    liga_id_hist = ligas_openliga[liga_nome_hist]
    if st.button("Buscar jogos da temporada", key="btn_hist"):
        with st.spinner("Buscando jogos..."):
            try:
                jogos_hist = obter_jogos_liga_temporada(liga_id_hist, temporada_hist2)
                if not jogos_hist:
                    st.info("Nenhum jogo encontrado para essa temporada/liga.")
                else:
                    st.success(f"{len(jogos_hist)} jogos encontrados na {liga_nome_hist} ({temporada_hist2})")
                    # mostrar os primeiros 100 com placar final quando houver
                    to_show = []
                    for j in jogos_hist[:200]:
                        home = j.get("team1", {}).get("teamName")
                        away = j.get("team2", {}).get("teamName")
                        placar = "-"
                        for r in j.get("matchResults", []):
                            if r.get("resultTypeID") == 2:
                                placar = f"{r.get('pointsTeam1',0)} x {r.get('pointsTeam2',0)}"
                                break
                        data = j.get("matchDateTimeUTC") or j.get("matchDateTime") or "Desconhecida"
                        to_show.append({"Jogo": f"{home} x {away}", "Data": data, "Placar": placar})
                    st.table(to_show)
            except Exception as e:
                st.error(f"Erro na busca: {e}")

# -----------------------------
# Aba: Conferência Top3 (pós-jogo)
# -----------------------------
st.markdown("---")
st.subheader("🎯 Conferência dos Top 3 enviados")
top3_salvos = carregar_top3()

if not top3_salvos:
    st.info("Nenhum Top3 registrado ainda. Gere e envie um Top3 na seção principal.")
else:
    options = [f"{idx+1} - {t['data_envio']} ({t['hora_envio']})" for idx, t in enumerate(top3_salvos)]
    seletor = st.selectbox("Selecione o lote Top3 para conferir:", options, index=len(options)-1)
    idx_selecionado = options.index(seletor)
    lote = top3_salvos[idx_selecionado]
    st.markdown(f"**Lote selecionado:** {lote['data_envio']} ({lote['hora_envio']}) — Temporada: {lote.get('temporada')}")

    if st.button("Rechecar resultados agora e enviar conferência (uma mensagem por faixa)"):
        with st.spinner("Conferindo resultados..."):
            def processar_lista_e_mandar(lista_top: List[dict], threshold_label: str):
                detalhes_local = []
                greens = reds = 0
                lines_for_msg = []
                for j in lista_top:
                    fixture_id = j.get("fixture_id")
                    liga_id = j.get("liga_id")
                    temporada = j.get("temporada")
                    info = conferir_jogo_openliga(fixture_id, liga_id, temporada, threshold_label)
                    if not info:
                        linhas = f"🏟️ {j.get('home')} x {j.get('away')} — _sem resultado disponível_"
                        lines_for_msg.append(linhas)
                        detalhes_local.append({"home": j.get("home"), "away": j.get("away"), "aposta": f"+{threshold_label}", "status": "Não encontrado"})
                        continue
                    if info.get("total_gols") is None:
                        lines_for_msg.append(f"🏟️ {info['home']} — _Em andamento / sem resultado_")
                        detalhes_local.append({"home": info["home"], "away": info["away"], "aposta": info["aposta"], "status": "Em andamento"})
                        continue
                    resultado_text = info["resultado"]
                    score = info.get("score", "")
                    lines_for_msg.append(f"🏟️ {info['home']} {score} {info['away']} — {info['aposta']} → {resultado_text}")
                    detalhes_local.append({"home": info["home"], "away": info["away"], "aposta": info["aposta"], "total_gols": info["total_gols"], "resultado": resultado_text})
                    if "GREEN" in resultado_text:
                        greens += 1
                    else:
                        reds += 1
                header = f"✅ RESULTADOS - CONFERÊNCIA +{threshold_label}\n(Lote: {lote['data_envio']})\n\n"
                body = "\n".join(lines_for_msg) if lines_for_msg else "_Nenhum jogo para conferir nesta faixa no lote selecionado._"
                resumo = f"\n\nResumo: 🟢 {greens} GREEN | 🔴 {reds} RED"
                msg = header + body + resumo
                enviar_para_todos(msg)
                return detalhes_local, {"greens": greens, "reds": reds}

            d15, r15 = processar_lista_e_mandar(lote.get("top_1_5", []), "1.5")
            d25, r25 = processar_lista_e_mandar(lote.get("top_2_5", []), "2.5")
            d35, r35 = processar_lista_e_mandar(lote.get("top_3_5", []), "3.5")

            st.success("Mensagens de conferência enviadas (uma por faixa).")
            st.write(f"+1.5 → 🟢 {r15['greens']} | 🔴 {r15['reds']}")
            st.write(f"+2.5 → 🟢 {r25['greens']} | 🔴 {r25['reds']}")
            st.write(f"+3.5 → 🟢 {r35['greens']} | 🔴 {r35['reds']}")

    if st.button("Rechecar resultados aqui (sem enviar Telegram)"):
        with st.spinner("Conferindo resultados localmente..."):
            for label, lista in [("1.5", lote.get("top_1_5", [])), ("2.5", lote.get("top_2_5", [])), ("3.5", lote.get("top_3_5", []))]:
                st.write(f"### Conferência +{label}")
                for j in lista:
                    info = conferir_jogo_openliga(j.get("fixture_id"), j.get("liga_id"), j.get("temporada"), label)
                    if not info:
                        st.warning(f"🏟️ {j.get('home')} x {j.get('away')} — Resultado não encontrado / sem atualização")
                        continue
                    if info.get("total_gols") is None:
                        st.info(f"🏟️ {info['home']} — Em andamento / sem resultado")
                        continue
                    if "GREEN" in info["resultado"]:
                        st.success(f"🏟️ {info['home']} {info.get('score','')} {info['away']} → {info['resultado']}")
                    else:
                        st.error(f"🏟️ {info['home']} {info.get('score','')} {info['away']} → {info['resultado']}")

    if st.button("Exportar lote selecionado (.json)"):
        nome_arquivo = f"relatorio_top3_{lote['data_envio'].replace('/','-')}_{lote['hora_envio'].replace(':','-').replace(' ','_')}.json"
        with open(nome_arquivo, "w", encoding="utf-8") as f:
            json.dump(lote, f, ensure_ascii=False, indent=2)
        st.success(f"Lote exportado: {nome_arquivo}")

# Fim do arquivo
