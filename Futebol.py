# Futebol_Alertas_AllSportsAPI.py
"""
Sistema Unificado de Alertas - AllSportsAPI
- Usa exclusivamente AllSportsAPI (Fixtures, Livescore, Results, Standings, Leagues)
- Calcula tendência/estimativa de gols e confiança similar ao sistema anterior
- Envia alertas ao Telegram (automático opcional quando confiança >= threshold)
- Streamlit UI com seleção de data / ligas e exibição de Top jogos
"""

import streamlit as st
from datetime import datetime, timedelta
import requests
import json
import os
from typing import List, Dict, Any

# =============================
# CONFIGURAÇÕES (cole sua chave AllSportsAPI e Telegram)
# =============================
ALLSPORTS_API_KEY = "c7247ae8b933c5299e3d8f2eac8a7feb9d42c490baa6ccc8fac5062e5a589994"
BASE_URL_AS = "https://allsportsapi.com/api/football/"

TELEGRAM_TOKEN = "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY"
TELEGRAM_CHAT_ID = "-1002754276285"
TELEGRAM_CHAT_ID_ALT2 = "-1002754276285"
BASE_URL_TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

ALERTAS_PATH = "alertas.json"
CACHE_JOGOS = "cache_allsports_jogos.json"
CACHE_CLASSIFICACAO = "cache_allsports_classificacao.json"
CACHE_LEAGUES = "cache_allsports_leagues.json"

# =============================
# Inicialização session_state
# =============================
def inicializar_session_state():
    if 'jogos_encontrados' not in st.session_state:
        st.session_state.jogos_encontrados = []
    if 'busca_realizada' not in st.session_state:
        st.session_state.busca_realizada = False
    if 'alertas_enviados' not in st.session_state:
        st.session_state.alertas_enviados = False
    if 'top_jogos' not in st.session_state:
        st.session_state.top_jogos = []
    if 'data_ultima_busca' not in st.session_state:
        st.session_state.data_ultima_busca = None
    if 'resultados_conferidos' not in st.session_state:
        st.session_state.resultados_conferidos = []

# =============================
# Persistência / cache em disco
# =============================
def carregar_json(caminho):
    if os.path.exists(caminho):
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def salvar_json(caminho, dados):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

def carregar_cache_jogos():
    return carregar_json(CACHE_JOGOS)

def salvar_cache_jogos(dados):
    salvar_json(CACHE_JOGOS, dados)

def carregar_cache_classificacao():
    return carregar_json(CACHE_CLASSIFICACAO)

def salvar_cache_classificacao(dados):
    salvar_json(CACHE_CLASSIFICACAO, dados)

def carregar_cache_leagues():
    return carregar_json(CACHE_LEAGUES)

def salvar_cache_leagues(dados):
    salvar_json(CACHE_LEAGUES, dados)

def carregar_alertas():
    return carregar_json(ALERTAS_PATH)

def salvar_alertas(alertas):
    salvar_json(ALERTAS_PATH, alertas)

# =============================
# Envio Telegram
# =============================
def enviar_telegram(msg: str, chat_id: str = TELEGRAM_CHAT_ID) -> bool:
    try:
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
        r = requests.get(BASE_URL_TG, params=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        st.warning(f"Erro ao enviar Telegram: {e}")
        return False

def enviar_alerta_telegram_generico(home, away, data_str_brt, hora_str, liga, tendencia, estimativa, confianca, chat_id=TELEGRAM_CHAT_ID):
    msg = (
        f"⚽ *Alerta de Gols!*\n"
        f"🏟️ *{home}* vs *{away}*\n"
        f"📅 {data_str_brt} ⏰ {hora_str} (BRT)\n"
        f"🔥 Tendência: *{tendencia}*\n"
        f"📊 Estimativa: *{estimativa:.2f}* gols\n"
        f"✅ Confiança: *{confianca:.0f}%*\n"
        f"📌 Liga: {liga}"
    )
    return enviar_telegram(msg, chat_id)

# =============================
# Helper de requisição AllSportsAPI (tenta variações de nome de chave)
# =============================
def request_allsports(met: str, extra_params: dict = None) -> dict:
    if extra_params is None:
        extra_params = {}
    base_params = {"met": met}
    base_params.update(extra_params)
    # tentativas comuns de nome para a chave
    names = ["APIkey", "api_key", "key", "API_KEY"]
    for name in names:
        params = base_params.copy()
        params[name] = ALLSPORTS_API_KEY
        try:
            r = requests.get(BASE_URL_AS, params=params, timeout=12)
            r.raise_for_status()
            data = r.json()
            if data:
                return data
        except Exception:
            continue
    return {}

# =============================
# Endpoints helpers
# =============================
def listar_ligas_allsports(force_refresh=False) -> List[Dict[str, Any]]:
    cache = carregar_cache_leagues()
    if cache and not force_refresh:
        return cache.get("leagues", [])
    data = request_allsports("Leagues")
    leagues = []
    # parsing flexível
    for k in ("result", "data", "leagues", "response"):
        if isinstance(data.get(k), list):
            leagues = data.get(k)
            break
    if not leagues and isinstance(data, dict):
        # buscar primeira lista encontrada
        for v in data.values():
            if isinstance(v, list):
                leagues = v
                break
    salvar_cache_leagues({"leagues": leagues})
    return leagues

def obter_jogos_allsports_por_data(date_str: str) -> List[Dict[str, Any]]:
    """
    Retorna lista unificada de jogos (fixtures + livescore + results) para uma data.
    """
    cache = carregar_cache_jogos()
    key = f"allsports_{date_str}"
    if key in cache:
        return cache[key]

    all_events = []

    # Fixtures
    data_fixtures = request_allsports("Fixtures", {"date": date_str})
    fixtures = []
    for k in ("result", "data", "fixtures", "events", "response"):
        if isinstance(data_fixtures.get(k), list):
            fixtures = data_fixtures.get(k)
            break
    if not fixtures and isinstance(data_fixtures, dict):
        for v in data_fixtures.values():
            if isinstance(v, list):
                fixtures = v
                break

    # Livescore
    data_lives = request_allsports("Livescore", {"date": date_str})
    lives = []
    for k in ("result", "data", "livescore", "events", "response"):
        if isinstance(data_lives.get(k), list):
            lives = data_lives.get(k)
            break
    if not lives and isinstance(data_lives, dict):
        for v in data_lives.values():
            if isinstance(v, list):
                lives = v
                break

    # Results
    data_results = request_allsports("Results", {"date": date_str})
    results = []
    for k in ("result", "data", "results", "events", "response"):
        if isinstance(data_results.get(k), list):
            results = data_results.get(k)
            break
    if not results and isinstance(data_results, dict):
        for v in data_results.values():
            if isinstance(v, list):
                results = v
                break

    all_events.extend(fixtures or [])
    all_events.extend(lives or [])
    all_events.extend(results or [])

    # Normalização flexível
    normalized = []
    seen = set()
    for e in all_events:
        # campos comuns alternativos
        home = e.get("homeName") or e.get("strHomeTeam") or e.get("home_team") or e.get("home") or e.get("home_name")
        away = e.get("awayName") or e.get("strAwayTeam") or e.get("away_team") or e.get("away") or e.get("away_name")
        league = e.get("league") or e.get("competition") or e.get("competition_name") or e.get("strLeague") or e.get("league_name")
        date_field = e.get("event_date") or e.get("date") or e.get("formatted_date") or e.get("dateEvent")
        time_field = e.get("event_time") or e.get("time") or e.get("strTime") or e.get("timeEvent")
        event_id = e.get("event_key") or e.get("match_id") or e.get("idEvent") or e.get("id") or f"{league}_{home}_{away}_{date_field}_{time_field}"

        # placares
        home_score = None
        away_score = None
        for maybe_h in ("home_score", "intHomeScore", "homeGoals", "scored_home", "homegoals"):
            if maybe_h in e and e.get(maybe_h) is not None:
                try:
                    home_score = int(e.get(maybe_h))
                    break
                except:
                    pass
        for maybe_a in ("away_score", "intAwayScore", "awayGoals", "scored_away", "awaygoals"):
            if maybe_a in e and e.get(maybe_a) is not None:
                try:
                    away_score = int(e.get(maybe_a))
                    break
                except:
                    pass

        # tentar construir data ISO para parse
        data_brt, hora_brt = None, None
        utc = e.get("event_time_utc") or e.get("utcDate") or e.get("utc")
        if utc:
            # tentar parse
            try:
                dt = datetime.fromisoformat(str(utc).replace("Z", "+00:00"))
                dt_brt = dt - timedelta(hours=3)
                data_brt = dt_brt.strftime("%d/%m/%Y")
                hora_brt = dt_brt.strftime("%H:%M")
            except:
                pass
        if not data_brt:
            if date_field:
                data_brt = date_field
            else:
                data_brt = "-"
        if not hora_brt:
            hora_brt = time_field or "-"

        normalized_event = {
            "raw": e,
            "id": str(event_id),
            "home": home or "Desconhecido",
            "away": away or "Desconhecido",
            "liga": league or "Desconhecido",
            "date": date_field,
            "time": time_field,
            "data_brt": data_brt,
            "hora": hora_brt,
            "home_score": home_score,
            "away_score": away_score,
            "origem": "AllSportsAPI"
        }
        if normalized_event["id"] in seen:
            continue
        seen.add(normalized_event["id"])
        normalized.append(normalized_event)

    cache = carregar_cache_jogos()
    cache[key] = normalized
    salvar_cache_jogos(cache)
    return normalized

def obter_classificacao_allsports(league_identifier: str) -> Dict[str, Any]:
    """
    Tenta obter standings para uma liga. league_identifier pode ser id/slug/nome.
    Retorna dict: nome_time -> {scored, against, played}
    """
    cache = carregar_cache_classificacao()
    if str(league_identifier) in cache:
        return cache[str(league_identifier)]

    params = {"league": league_identifier}
    data = request_allsports("Standings", params)
    standings = {}
    teams_list = []
    for k in ("result", "data", "standings", "table", "response"):
        if isinstance(data.get(k), list):
            teams_list = data.get(k)
            break
    if not teams_list and isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                teams_list = v
                break

    for t in teams_list or []:
        name = t.get("team_name") or t.get("team") or t.get("name") or (t.get("team", {}).get("name") if isinstance(t.get("team"), dict) else None)
        gols_marcados = None
        gols_sofridos = None
        partidas = None
        for gf in ("goals_for", "goalsScored", "scored", "goalsFor"):
            if gf in t and t.get(gf) is not None:
                try:
                    gols_marcados = int(t.get(gf))
                    break
                except:
                    pass
        for ga in ("goals_against", "goalsAgainst", "against"):
            if ga in t and t.get(ga) is not None:
                try:
                    gols_sofridos = int(t.get(ga))
                    break
                except:
                    pass
        for pf in ("played", "matches", "played_games"):
            if pf in t and t.get(pf) is not None:
                try:
                    partidas = int(t.get(pf))
                    break
                except:
                    pass
        if not name:
            continue
        standings[name] = {
            "scored": gols_marcados or 0,
            "against": gols_sofridos or 0,
            "played": partidas or 1
        }

    cache[str(league_identifier)] = standings
    salvar_cache_classificacao(cache)
    return standings

# =============================
# Tendência (mesma lógica de antes)
# =============================
def calcular_tendencia_allsports(home, away, classificacao):
    dados_home = classificacao.get(home, {"scored":0, "against":0, "played":1})
    dados_away = classificacao.get(away, {"scored":0, "against":0, "played":1})

    media_home_feitos = dados_home["scored"] / max(1, dados_home["played"])
    media_home_sofridos = dados_home["against"] / max(1, dados_home["played"])
    media_away_feitos = dados_away["scored"] / max(1, dados_away["played"])
    media_away_sofridos = dados_away["against"] / max(1, dados_away["played"])

    estimativa = ((media_home_feitos + media_away_sofridos) / 2 +
                  (media_away_feitos + media_home_sofridos) / 2)

    if estimativa >= 3.0:
        tendencia = "Mais 2.5"
        confianca = min(95, 70 + (estimativa - 3.0)*10)
    elif estimativa >= 2.0:
        tendencia = "Mais 1.5"
        confianca = min(90, 60 + (estimativa - 2.0)*10)
    else:
        tendencia = "Menos 2.5"
        confianca = min(85, 55 + (2.0 - estimativa)*10)

    return round(estimativa, 2), round(confianca, 0), tendencia

# =============================
# Função principal de buscar e analisar
# =============================
def buscar_e_analisar_jogos_allsports(data_selecionada, ligas_selecionadas_names, ligas_selecionadas_ids, conf_threshold=70, auto_send=False) -> (List[dict], List[dict], List[dict]):
    data_str = data_selecionada.strftime("%Y-%m-%d")
    total_jogos = []
    total_top_jogos = []
    alertas_auto_enviados = []

    todos_jogos = obter_jogos_allsports_por_data(data_str)

    # Filtrar por ligas selecionadas (nomes)
    if ligas_selecionadas_names:
        lower_choices = [x.lower() for x in ligas_selecionadas_names]
        jogos_filtrados = [j for j in todos_jogos if any(ch in (j.get("liga") or "").lower() or (j.get("liga") or "").lower() in ch for ch in lower_choices)]
    else:
        jogos_filtrados = todos_jogos

    # Filtrar por ids se fornecido
    if ligas_selecionadas_ids:
        ids_str = [str(x) for x in ligas_selecionadas_ids]
        jogos_filtrados = [j for j in jogos_filtrados if any(i in str(j.get("raw", {}).get("league_id", "") or j.get("raw", {}).get("league_key", "") or "") for i in ids_str) or not ligas_selecionadas_names]

    for j in jogos_filtrados:
        liga = j.get("liga") or "Desconhecido"
        home = j.get("home") or "Desconhecido"
        away = j.get("away") or "Desconhecido"
        raw = j.get("raw") or {}
        league_id_guess = raw.get("league_id") or raw.get("league_key") or raw.get("league_id2") or liga

        classificacao = obter_classificacao_allsports(league_id_guess)
        estimativa, confianca, tendencia = calcular_tendencia_allsports(home, away, classificacao)

        jogo_info = {
            "id": j.get("id"),
            "home": home,
            "away": away,
            "tendencia": tendencia,
            "estimativa": estimativa,
            "confianca": confianca,
            "liga": liga,
            "hora": j.get("hora") or "-",
            "origem": "AllSportsAPI",
            "data_brt": j.get("data_brt") or "-"
        }
        total_jogos.append(jogo_info)

        # auto-send se configurado
        if auto_send and confianca >= conf_threshold:
            sucesso = enviar_alerta_telegram_generico(
                jogo_info['home'], jogo_info['away'], jogo_info['data_brt'], jogo_info['hora'],
                jogo_info['liga'], jogo_info['tendencia'], jogo_info['estimativa'], jogo_info['confianca']
            )
            if sucesso:
                alertas_auto_enviados.append(jogo_info)

    if total_jogos:
        total_top_jogos = sorted(total_jogos, key=lambda x: (x["confianca"], x["estimativa"]), reverse=True)[:5]

    return total_jogos, total_top_jogos, alertas_auto_enviados

# =============================
# Envio de alertas em massa / individual (existente)
# =============================
def enviar_alertas_individualmente(jogos):
    alertas_enviados = []
    for jogo in jogos:
        sucesso = enviar_alerta_telegram_generico(
            jogo['home'], jogo['away'], jogo['data_brt'], jogo['hora'],
            jogo['liga'], jogo['tendencia'], jogo['estimativa'], jogo['confianca']
        )
        if sucesso:
            alertas_enviados.append(jogo)
    return alertas_enviados

def enviar_top_consolidado(top_jogos):
    if not top_jogos:
        return False
    mensagem = "📢 *TOP Jogos Consolidados*\n\n"
    for t in top_jogos:
        mensagem += f"🏟️ {t['liga']}\n🏆 {t['home']} x {t['away']}\nTendência: {t['tendencia']} | Conf.: {t['confianca']}%\n\n"
    return enviar_telegram(mensagem, TELEGRAM_CHAT_ID_ALT2)

# =============================
# UI Streamlit
# =============================
def main():
    st.set_page_config(page_title="⚽ Futebol Alertas - AllSportsAPI", layout="wide")
    inicializar_session_state()

    st.title("⚽ Sistema Unificado de Alertas (AllSportsAPI)")

    # Data
    data_selecionada = st.date_input("📅 Escolha a data para os jogos:", value=datetime.today())
    data_str = data_selecionada.strftime("%Y-%m-%d")

    # Sidebar - ligas
    st.sidebar.header("Opções de Busca")
    ligas_all = []
    try:
        ligas_all = listar_ligas_allsports()
        nomes_ligas = []
        for l in ligas_all:
            if isinstance(l, dict):
                nome = l.get("league_name") or l.get("name") or l.get("strLeague") or l.get("league")
                if nome and nome not in nomes_ligas:
                    nomes_ligas.append(nome)
        nomes_ligas = sorted(list(set(nomes_ligas)))
    except Exception:
        nomes_ligas = []

    use_all = st.sidebar.checkbox("Usar todas as ligas AllSportsAPI", value=False)
    if use_all:
        ligas_selecionadas = nomes_ligas
    else:
        ligas_selecionadas = st.sidebar.multiselect("Selecione ligas (AllSportsAPI):", nomes_ligas, max_selections=12)

    ligas_ids_input = st.sidebar.text_input("Adicionar IDs de ligas (comma-separated) — opcional", "")
    ligas_selecionadas_ids = []
    if ligas_ids_input.strip():
        try:
            ligas_selecionadas_ids = [int(x.strip()) for x in ligas_ids_input.split(",") if x.strip().isdigit()]
        except:
            ligas_selecionadas_ids = []

    # Auto-send e threshold
    st.sidebar.header("Envio Automático")
    auto_send = st.sidebar.checkbox("Enviar automaticamente alertas com confiança >= threshold", value=True)
    conf_threshold = st.sidebar.number_input("Limiar de confiança (%) para envio automático:", min_value=0, max_value=100, value=70, step=5)

    # Status
    st.sidebar.header("📊 Status da Sessão")
    st.sidebar.write(f"Busca realizada: {'✅' if st.session_state.busca_realizada else '❌'}")
    st.sidebar.write(f"Alertas enviados: {'✅' if st.session_state.alertas_enviados else '❌'}")
    st.sidebar.write(f"Jogos encontrados: {len(st.session_state.jogos_encontrados)}")
    st.sidebar.write(f"Top jogos: {len(st.session_state.top_jogos)}")

    # Limpar
    if st.sidebar.button("🗑️ Limpar Dados da Sessão"):
        st.session_state.jogos_encontrados = []
        st.session_state.busca_realizada = False
        st.session_state.alertas_enviados = False
        st.session_state.top_jogos = []
        st.session_state.data_ultima_busca = None
        st.session_state.resultados_conferidos = []
        st.success("Dados da sessão limpos!")
        st.rerun()

    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        buscar_btn = st.button("🔍 Buscar partidas e analisar", type="primary")
    with c2:
        enviar_alertas_btn = st.button("🚀 Enviar Alertas Individuais", disabled=not st.session_state.busca_realizada)
    with c3:
        enviar_top_btn = st.button("📊 Enviar Top Consolidado", disabled=not st.session_state.busca_realizada)

    # BUSCA
    if buscar_btn:
        with st.spinner("Buscando partidas e analisando (AllSportsAPI)..."):
            jogos_encontrados, top_jogos, alertas_auto_enviados = buscar_e_analisar_jogos_allsports(
                data_selecionada, ligas_selecionadas, ligas_selecionadas_ids, conf_threshold, auto_send
            )
            st.session_state.jogos_encontrados = jogos_encontrados
            st.session_state.top_jogos = top_jogos
            st.session_state.busca_realizada = True
            st.session_state.data_ultima_busca = data_str
            st.session_state.alertas_enviados = len(alertas_auto_enviados) > 0

        if jogos_encontrados:
            st.success(f"✅ {len(jogos_encontrados)} jogos encontrados e analisados!")
            if alertas_auto_enviados:
                st.success(f"📨 {len(alertas_auto_enviados)} alertas automáticos enviados (conf >= {conf_threshold}%)")
            st.subheader("📋 Todos os Jogos Encontrados")
            for jogo in jogos_encontrados:
                with st.container():
                    col_a, col_b, col_c = st.columns([3,2,1])
                    with col_a:
                        st.write(f"**{jogo['home']}** vs **{jogo['away']}**")
                        st.write(f"🏆 {jogo['liga']} | 🕐 {jogo['hora']} | 📊 {jogo['origem']}")
                    with col_b:
                        st.write(f"🎯 {jogo['tendencia']}")
                        st.write(f"📈 Estimativa: {jogo['estimativa']} | ✅ Confiança: {jogo['confianca']}%")
                    with col_c:
                        if jogo in st.session_state.top_jogos:
                            st.success("🏆 TOP")
                    st.divider()
            if top_jogos:
                st.subheader("🏆 Top 5 Jogos (Maior Confiança)")
                for i, jogo in enumerate(top_jogos, 1):
                    st.info(f"{i}. **{jogo['home']}** vs **{jogo['away']}** - {jogo['tendencia']} ({jogo['confianca']}% confiança)")
        else:
            st.warning("⚠️ Nenhum jogo encontrado para os critérios selecionados.")

    # Envio manual de alertas individuais
    if enviar_alertas_btn and st.session_state.busca_realizada:
        with st.spinner("Enviando alertas individuais..."):
            alertas_enviados = enviar_alertas_individualmente(st.session_state.jogos_encontrados)
            if alertas_enviados:
                st.session_state.alertas_enviados = True
                st.success(f"✅ {len(alertas_enviados)} alertas enviados com sucesso!")
            else:
                st.error("❌ Erro ao enviar alertas (ou nenhum alerta enviado)")

    # Envio Top consolidado
    if enviar_top_btn and st.session_state.busca_realizada and st.session_state.top_jogos:
        with st.spinner("Enviando top consolidado..."):
            if enviar_top_consolidado(st.session_state.top_jogos):
                st.success("✅ Top consolidado enviado com sucesso!")
            else:
                st.error("❌ Erro ao enviar top consolidado")

    # Conferência de resultados (esqueleto)
    st.markdown("---")
    conferir_btn = st.button("📊 Conferir resultados (usar alertas salvo)")
    if conferir_btn:
        st.info("Conferindo resultados dos alertas salvos... (implemente sua lógica de conferência aqui)")
        # Exemplo: carregar alertas e verificar via Results endpoint se estiverem finalizados

if __name__ == "__main__":
    main()
