# Futebol_Alertas_Oddstop.py
import streamlit as st
from datetime import datetime, timedelta, date
import requests
import os
import json
import math

# =============================
# Configurações Odds API + Telegram
# =============================
ODDS_API_KEY = "7ad63ba7ce77a6f31c33acc766f3e9fb"
ODDS_BASE = "https://api.the-odds-api.com/v4"

# Lista ampla das ligas principais (chaves da Odds API)
# Você pode editar essa lista para incluir/excluir ligas conforme preferir.
LIGAS_PRINCIPAIS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_brazil_campeonato",
    "soccer_uefa_champs_league",
    "soccer_portugal_primeira_liga",
    "soccer_netherlands_eredivisie",
    "soccer_mexico_ligamx",
    "soccer_turkey_super_league",
    "soccer_uefa_europa_conference_league",
    "soccer_uefa_europa_league",
    # adicione mais se quiser
]

TELEGRAM_TOKEN = "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY"
TELEGRAM_CHAT_ID = "-1003073115320"
TELEGRAM_CHAT_ID_ALT2 = "-1002932611974"
BASE_URL_TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

ALERTAS_PATH = "alertas.json"
TOP3_PATH = "top3.json"

# =============================
# Persistência
# =============================
def carregar_alertas():
    if os.path.exists(ALERTAS_PATH):
        with open(ALERTAS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def salvar_alertas(alertas):
    with open(ALERTAS_PATH, "w", encoding="utf-8") as f:
        json.dump(alertas, f, ensure_ascii=False, indent=2)

def carregar_top3():
    if os.path.exists(TOP3_PATH):
        with open(TOP3_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def salvar_top3(lista):
    with open(TOP3_PATH, "w", encoding="utf-8") as f:
        json.dump(lista, f, ensure_ascii=False, indent=2)

# =============================
# Envio Telegram
# =============================
def enviar_telegram(msg, chat_id=TELEGRAM_CHAT_ID):
    try:
        requests.get(BASE_URL_TG, params={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        st.warning(f"Erro ao enviar Telegram: {e}")

# =============================
# Helpers Odds API
# =============================
def obter_odds_para_liga(liga_key, regions="eu,us,au", markets="totals", odds_format="decimal"):
    """
    Consulta a Odds API para uma liga (sport_key) retornando eventos com mercado 'totals'.
    """
    url = f"{ODDS_BASE}/sports/{liga_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            # possível 204 quando não há dados
            return []
    except Exception as e:
        st.warning(f"Erro Odds API {liga_key}: {e}")
        return []

def parse_iso_to_datetime(s):
    if not s:
        return None
    try:
        # exemplos: 2025-10-15T16:30:00Z ou com offset
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
    Retorna um dicionário com os mercados totals encontrados no evento.
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
                if key not in totals_map:
                    totals_map[key] = {"overs": [], "unders": []}
                
                if "over" in name:
                    totals_map[key]["overs"].append(price)
                elif "under" in name:
                    totals_map[key]["unders"].append(price)
    
    return totals_map

def calcular_estimativas_e_probs_por_jogo_from_odds(event):
    """
    Calcula estimativa de gols e probabilidades baseado nas odds do mercado 'totals'.
    """
    totals_map = extrair_markets_totals(event)
    
    if not totals_map:
        return {
            "estimativa": 0.0,
            "prob_1_5": 0.0, "prob_2_5": 0.0, "prob_3_5": 0.0,
            "conf_1_5": 0, "conf_2_5": 0, "conf_3_5": 0
        }
    
    # Encontra o ponto mais próximo de 2.5 (mais negociado)
    pontos = [float(k) for k in totals_map.keys()]
    ponto_ref = min(pontos, key=lambda x: abs(x - 2.5)) if pontos else 2.5
    ponto_str = str(ponto_ref)
    
    if ponto_str not in totals_map:
        return {
            "estimativa": 0.0,
            "prob_1_5": 0.0, "prob_2_5": 0.0, "prob_3_5": 0.0,
            "conf_1_5": 0, "conf_2_5": 0, "conf_3_5": 0
        }
    
    # Pega as odds para over/under no ponto de referência
    overs = totals_map[ponto_str]["overs"]
    unders = totals_map[ponto_str]["unders"]
    
    if not overs or not unders:
        return {
            "estimativa": 0.0,
            "prob_1_5": 0.0, "prob_2_5": 0.0, "prob_3_5": 0.0,
            "conf_1_5": 0, "conf_2_5": 0, "conf_3_5": 0
        }
    
    # Usa a melhor odd disponível
    best_over = max(overs)
    best_under = max(unders)
    
    # Calcula probabilidades implícitas
    prob_over = 1 / best_over
    prob_under = 1 / best_under
    margin = prob_over + prob_under
    prob_over_ajust = prob_over / margin
    prob_under_ajust = prob_under / margin
    
    # Estimativa de gols usando distribuição de Poisson
    # Para um ponto de corte 'x', P(over) = 1 - P(under) = 1 - CDF_Poisson(x, lambda)
    # Resolve para lambda que satisfaz a probabilidade
    def poisson_cdf(k, lambd):
        return sum((lambd ** i) * math.exp(-lambd) / math.factorial(i) for i in range(int(k) + 1))
    
    # Encontra lambda que melhor se ajusta às probabilidades
    best_lambda = 2.5  # default
    best_error = float('inf')
    
    for lambd in [x * 0.1 for x in range(10, 60)]:  # testa de 1.0 a 6.0
        p_over_est = 1 - poisson_cdf(ponto_ref, lambd)
        error = abs(p_over_est - prob_over_ajust)
        if error < best_error:
            best_error = error
            best_lambda = lambd
    
    # Calcula probabilidades para diferentes limites
    probs = {
        "1.5": 1 - poisson_cdf(1.5, best_lambda),
        "2.5": 1 - poisson_cdf(2.5, best_lambda), 
        "3.5": 1 - poisson_cdf(3.5, best_lambda)
    }
    
    # Confiança baseada no número de casas que oferecem
    confs = {
        "1.5": min(100, len(overs) * 20),
        "2.5": min(100, len(overs) * 20),
        "3.5": min(100, len(overs) * 20)
    }
    
    return {
        "estimativa": round(best_lambda, 2),
        "prob_1_5": round(probs["1.5"] * 100, 1),
        "prob_2_5": round(probs["2.5"] * 100, 1), 
        "prob_3_5": round(probs["3.5"] * 100, 1),
        "conf_1_5": confs["1.5"],
        "conf_2_5": confs["2.5"],
        "conf_3_5": confs["3.5"],
    }

# =============================
# Função de seleção (mesma lógica que seu código original)
# =============================
def selecionar_top3_distintos(partidas_info, max_por_faixa=3, prefer_best_fit=True):
    if not partidas_info:
        return [], [], []

    base = list(partidas_info)  # cópia

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

    # Ordem de alocação: +2.5 primeiro (mantive sua priorização original no código inicial? 
    # no seu código original você priorizava +2.5 -> +1.5 -> +3.5; aqui sigo essa ordem)
    top_25 = allocate("2_5", other_prefixes=["1_5", "3_5"])
    top_15 = allocate("1_5", other_prefixes=["2_5", "3_5"])
    top_35 = allocate("3_5", other_prefixes=["2_5", "1_5"])

    return top_15, top_25, top_35

# =============================
# Função para agregar eventos da Odds API e transformar em partidas_info
# =============================
def coletar_jogos_do_dia_por_ligas(ligas, data_obj: date, regions="eu,us,au"):
    """
    Para cada liga (sport_key) chama a Odds API e agrega eventos que caem na `data_obj`.
    """
    partidas = []
    for liga in ligas:
        eventos = obter_odds_para_liga(liga, regions=regions, markets="totals", odds_format="decimal")
        if not eventos:
            continue
        for ev in eventos:
            commence = ev.get("commence_time")
            dt = parse_iso_to_datetime(commence)
            if not dt:
                continue
            if dt.date() != data_obj:
                continue
            
            # Extrai nomes dos times de forma mais robusta
            home = ev.get("home_team")
            away = ev.get("away_team")
            if not home or not away:
                teams = ev.get("teams", [])
                if len(teams) >= 2:
                    home = teams[0]
                    away = teams[1]
                else:
                    continue  # Pula se não conseguir identificar os times
            
            hora_formatada = dt.strftime("%H:%M")
            calc = calcular_estimativas_e_probs_por_jogo_from_odds(ev)
            
            partidas.append({
                "fixture_id": ev.get("id"),
                "home": home,
                "away": away, 
                "hora": hora_formatada,
                "competicao": liga,
                "estimativa": calc.get("estimativa", 0),
                "prob_1_5": calc.get("prob_1_5", 0),
                "prob_2_5": calc.get("prob_2_5", 0),
                "prob_3_5": calc.get("prob_3_5", 0),
                "conf_1_5": calc.get("conf_1_5", 0),
                "conf_2_5": calc.get("conf_2_5", 0),
                "conf_3_5": calc.get("conf_3_5", 0),
                "liga_key": liga,
                "raw_event": ev
            })
    return partidas

# =============================
# UI Streamlit (mantendo layout igual ao original)
# =============================
st.set_page_config(page_title="Oddstop - ⚽ Alertas Top3 (Odds API)", layout="wide")
st.title("⚽ Oddstop — Alertas Top3 por Faixa (+1.5 / +2.5 / +3.5) — Odds API")

aba = st.tabs(["⚡ Gerar & Enviar Top3 (pré-jogo)", "📊 Jogos (Odds)", "🎯 Conferência Top3 (pós-jogo)"])

# ---------- ABA 1: Gerar & Enviar Top3 ----------
with aba[0]:
    st.subheader("🔎 Buscar jogos do dia nas ligas principais e enviar Top3 por faixa (via Odds API)")
    data_selecionada = st.date_input("📅 Data dos jogos:", value=datetime.today().date())
    hoje_str = data_selecionada.strftime("%Y-%m-%d")

    st.markdown("**Obs:** uso das odds para estimar P(+1.5/+2.5/+3.5). Você pode ajustar as ligas no código `LIGAS_PRINCIPAIS`.")

    if st.button("🔍 Buscar jogos do dia e enviar Top3 (cada faixa uma mensagem)"):
        with st.spinner("Buscando jogos e calculando probabilidades via Odds API..."):
            partidas_info = coletar_jogos_do_dia_por_ligas(LIGAS_PRINCIPAIS, data_selecionada, regions="eu,us,au")

            if not partidas_info:
                st.info("Nenhum jogo encontrado para essa data nas ligas selecionadas (Odds API).")
            else:
                # Seleciona Top3 distintos
                top_15, top_25, top_35 = selecionar_top3_distintos(partidas_info, max_por_faixa=3)

                # Mensagem +1.5
                if top_15:
                    msg = f"🔔 *TOP 3 +1.5 GOLS — {hoje_str}*\n\n"
                    for idx, j in enumerate(top_15, start=1):
                        msg += (f"{idx}️⃣ *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n"
                                f"   • Est: {j['estimativa']:.2f} gols | P(+1.5): *{j['prob_1_5']:.1f}%* | Conf: *{j['conf_1_5']:.0f}%*\n")
                    enviar_telegram(msg, TELEGRAM_CHAT_ID)
                    enviar_telegram(msg, TELEGRAM_CHAT_ID_ALT2)

                # Mensagem +2.5
                if top_25:
                    msg = f"🔔 *TOP 3 +2.5 GOLS — {hoje_str}*\n\n"
                    for idx, j in enumerate(top_25, start=1):
                        msg += (f"{idx}️⃣ *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n"
                                f"   • Est: {j['estimativa']:.2f} gols | P(+2.5): *{j['prob_2_5']:.1f}%* | Conf: *{j['conf_2_5']:.0f}%*\n")
                    enviar_telegram(msg, TELEGRAM_CHAT_ID)
                    enviar_telegram(msg, TELEGRAM_CHAT_ID_ALT2)

                # Mensagem +3.5
                if top_35:
                    msg = f"🔔 *TOP 3 +3.5 GOLS — {hoje_str}*\n\n"
                    for idx, j in enumerate(top_35, start=1):
                        msg += (f"{idx}️⃣ *{j['home']} x {j['away']}* — {j['competicao']} — {j['hora']} BRT\n"
                                f"   • Est: {j['estimativa']:.2f} gols | P(+3.5): *{j['prob_3_5']:.1f}%* | Conf: *{j['conf_3_5']:.0f}%*\n")
                    enviar_telegram(msg, TELEGRAM_CHAT_ID)
                    enviar_telegram(msg, TELEGRAM_CHAT_ID_ALT2)

                # salva o lote Top3 (persistente)
                top3_list = carregar_top3()
                novo_top = {
                    "data_envio": hoje_str,
                    "hora_envio": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "top_1_5": top_15,
                    "top_2_5": top_25,
                    "top_3_5": top_35
                }
                top3_list.append(novo_top)
                salvar_top3(top3_list)

                st.success("✅ Top3 gerados e enviados (uma mensagem por faixa).")
                st.write("### Top 3 +1.5")
                st.table([{ "Jogo": f"{t['home']} x {t['away']}", "P(+1.5)": f"{t['prob_1_5']}%", "Conf": f"{t['conf_1_5']}%"} for t in top_15])
                st.write("### Top 3 +2.5")
                st.table([{ "Jogo": f"{t['home']} x {t['away']}", "P(+2.5)": f"{t['prob_2_5']}%", "Conf": f"{t['conf_2_5']}%"} for t in top_25])
                st.write("### Top 3 +3.5")
                st.table([{ "Jogo": f"{t['home']} x {t['away']}", "P(+3.5)": f"{t['prob_3_5']}%", "Conf": f"{t['conf_3_5']}%"} for t in top_35])

# ---------- ABA 2: Jogos (Odds) ----------
with aba[1]:
    st.subheader("📊 Jogos do dia com Odds (Odds API)")
    data_selecionada2 = st.date_input("📅 Data dos jogos para listar:", value=datetime.today().date(), key="lista_odds")
    if st.button("🔍 Listar jogos e odds do dia", key="listar_odds"):
        with st.spinner("Consultando Odds API para listar jogos..."):
            partidas = coletar_jogos_do_dia_por_ligas(LIGAS_PRINCIPAIS, data_selecionada2, regions="eu,us,au")
            if not partidas:
                st.info("Nenhum jogo/odds encontrado para essa data nas ligas selecionadas.")
            else:
                st.success(f"{len(partidas)} jogos encontrados")
                for p in partidas:
                    st.write(f"🏟️ {p['home']} x {p['away']} — {p['competicao']} — {p['hora']} BRT")
                    st.write(f"   • Est: {p['estimativa']} | P(+1.5): {p['prob_1_5']}% ({p['conf_1_5']}%) | P(+2.5): {p['prob_2_5']}% ({p['conf_2_5']}%) | P(+3.5): {p['prob_3_5']}% ({p['conf_3_5']}%)")
                    st.markdown("---")

# ---------- ABA 3: Conferência Top 3 ----------
with aba[2]:
    st.subheader("🎯 Conferência dos Top 3 enviados — enviar conferência por faixa (cada faixa uma mensagem)")
    top3_salvos = carregar_top3()

    if not top3_salvos:
        st.info("Nenhum Top 3 registrado ainda. Gere e envie um Top 3 na aba 'Gerar & Enviar Top3'.")
    else:
        st.write(f"✅ Total de envios registrados: {len(top3_salvos)}")
        options = [f"{idx+1} - {t['data_envio']} ({t['hora_envio']})" for idx, t in enumerate(top3_salvos)]
        seletor = st.selectbox("Selecione o lote Top3 para conferir:", options, index=len(options)-1)
        idx_selecionado = options.index(seletor)
        lote = top3_salvos[idx_selecionado]
        st.markdown(f"### Lote selecionado — Envio: **{lote['data_envio']}** às **{lote['hora_envio']}**")
        st.markdown("---")

        if st.button("🔄 Rechecar resultados agora e enviar conferência (uma mensagem por faixa)"):
            with st.spinner("Conferindo resultados via Odds API e enviando mensagens..."):
                detalhes_1_5 = []
                detalhes_2_5 = []
                detalhes_3_5 = []
                # Para conferência, vamos tentar usar o id/fixture salvo e procurar no topo da lista de eventos por jogo igual
                def processar_lista_e_mandar(lista_top, threshold_label):
                    detalhes_local = []
                    greens = reds = 0
                    lines_for_msg = []
                    for j in lista_top:
                        # Tentar reconsultar pelo evento salvo: buscamos na Odds API pela liga e filtramos por home/away
                        liga = j.get("liga_key") or j.get("liga_id")
                        eventos = obter_odds_para_liga(liga, regions="eu,us,au", markets="totals")
                        found = None
                        for ev in eventos:
                            ht = ev.get("home_team") or (ev.get("teams") and ev.get("teams")[0])
                            at = ev.get("away_team") or (ev.get("teams") and ev.get("teams")[1])
                            if ht == j.get("home") and at == j.get("away"):
                                found = ev
                                break
                        if not found:
                            detalhes_local.append({
                                "home": j.get("home"),
                                "away": j.get("away"),
                                "aposta": f"+{threshold_label}",
                                "status": "Não encontrado / sem resultado"
                            })
                            lines_for_msg.append(f"🏟️ {j.get('home')} x {j.get('away')} — _sem resultado disponível_")
                            continue
                        # Odds API não guarda resultados finais; para conferência de resultados você deve usar outra fonte (OpenLiga/Football-Data/odds provider)
                        # Aqui vamos notificar que conferência automática de resultado não é possível via Odds API (somente odds).
                        lines_for_msg.append(f"🏟️ {found.get('home_team')} x {found.get('away_team')} — _Odds disponíveis — não há placar via Odds API_")
                        detalhes_local.append({
                            "home": found.get("home_team"),
                            "away": found.get("away_team"),
                            "aposta": f"+{threshold_label}",
                            "status": "Odds encontradas — sem placar (use fonte de resultados para conferência)"
                        })
                    header = f"✅ RESULTADOS - CONFERÊNCIA +{threshold_label}\n(Lote: {lote['data_envio']})\n\n"
                    body = "\n".join(lines_for_msg) if lines_for_msg else "_Nenhum jogo para conferir nesta faixa no lote selecionado._"
                    resumo = f"\n\nResumo: 🟢 {greens} GREEN | 🔴 {reds} RED"
                    msg = header + body + resumo
                    enviar_telegram(msg, TELEGRAM_CHAT_ID)
                    enviar_telegram(msg, TELEGRAM_CHAT_ID_ALT2)
                    return detalhes_local, {"greens": greens, "reds": reds}

                detalhes_1_5, resumo_1_5 = processar_lista_e_mandar(lote.get("top_1_5", []), "1.5")
                detalhes_2_5, resumo_2_5 = processar_lista_e_mandar(lote.get("top_2_5", []), "2.5")
                detalhes_3_5, resumo_3_5 = processar_lista_e_mandar(lote.get("top_3_5", []), "3.5")

                st.success("✅ Mensagens de conferência enviadas (uma por faixa).")
                st.markdown("**Resumo das conferências enviadas:**")
                st.write(f"+1.5 → 🟢 {resumo_1_5['greens']} | 🔴 {resumo_1_5['reds']}")
                st.write(f"+2.5 → 🟢 {resumo_2_5['greens']} | 🔴 {resumo_2_5['reds']}")
                st.write(f"+3.5 → 🟢 {resumo_3_5['greens']} | 🔴 {resumo_3_5['reds']}")

        if st.button("🔎 Rechecar odds aqui (sem enviar Telegram)"):
            with st.spinner("Conferindo odds localmente..."):
                for label, lista in [("1.5", lote.get("top_1_5", [])), ("2.5", lote.get("top_2_5", [])), ("3.5", lote.get("top_3_5", []))]:
                    st.write(f"### Conferência +{label}")
                    for j in lista:
                        liga = j.get("liga_key") or j.get("liga_id")
                        eventos = obter_odds_para_liga(liga, regions="eu,us,au", markets="totals")
                        found = None
                        for ev in eventos:
                            ht = ev.get("home_team") or (ev.get("teams") and ev.get("teams")[0])
                            at = ev.get("away_team") or (ev.get("teams") and ev.get("teams")[1])
                            if ht == j.get("home") and at == j.get("away"):
                                found = ev
                                break
                        if not found:
                            st.warning(f"🏟️ {j.get('home')} x {j.get('away')} — Odds/Evento não encontrado")
                            continue
                        calc = calcular_estimativas_e_probs_por_jogo_from_odds(found)
                        st.write(f"🏟️ {found.get('home_team')} {found.get('away_team')} — P(+{label}): {calc.get('prob_'+label.replace('.','_')) if False else calc.get('prob_1_5') }")
                        # Note: prints simplified info; adapt conforme preferir

        if st.button("📥 Exportar lote selecionado (.json)"):
            nome_arquivo = f"relatorio_top3_{lote['data_envio']}_{lote['hora_envio'].replace(':','-').replace(' ','_')}.json"
            with open(nome_arquivo, "w", encoding="utf-8") as f:
                json.dump(lote, f, ensure_ascii=False, indent=2)
            st.success(f"Lote exportado: {nome_arquivo}")

# Fim do arquivo
