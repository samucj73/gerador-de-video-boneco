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
import numpy as np
from abc import ABC, abstractmethod

# =============================
# CLASSES PRINCIPAIS
# =============================

class CacheManager:
    """Gerencia operações de cache"""
    
    def __init__(self):
        self.CACHE_TIMEOUT = 3600
        self.ALERTAS_PATH = "alertas.json"
        self.CACHE_JOGOS = "cache_jogos.json"
        self.CACHE_CLASSIFICACAO = "cache_classificacao.json"
    
    def carregar_json(self, caminho: str) -> dict:
        """Carrega dados JSON com verificação de timeout."""
        try:
            if os.path.exists(caminho):
                with open(caminho, "r", encoding='utf-8') as f:
                    dados = json.load(f)
                
                # Verificar se o cache é muito antigo
                if caminho in [self.CACHE_JOGOS, self.CACHE_CLASSIFICACAO]:
                    agora = datetime.now().timestamp()
                    for key in list(dados.keys()):
                        if isinstance(dados[key], dict) and '_timestamp' in dados[key]:
                            if agora - dados[key]['_timestamp'] > self.CACHE_TIMEOUT:
                                del dados[key]
                        elif agora - os.path.getmtime(caminho) > self.CACHE_TIMEOUT:
                            return {}
                return dados
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Erro ao carregar {caminho}: {e}")
        return {}

    def salvar_json(self, caminho: str, dados: dict):
        """Salva dados JSON com timestamp."""
        try:
            # Adicionar timestamp para caches temporais
            if caminho in [self.CACHE_JOGOS, self.CACHE_CLASSIFICACAO]:
                dados['_timestamp'] = datetime.now().timestamp()
            
            with open(caminho, "w", encoding='utf-8') as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)
        except IOError as e:
            st.error(f"Erro ao salvar {caminho}: {e}")

    def carregar_alertas(self) -> dict:
        return self.carregar_json(self.ALERTAS_PATH)

    def salvar_alertas(self, alertas: dict):
        self.salvar_json(self.ALERTAS_PATH, alertas)

    def carregar_cache_jogos(self) -> dict:
        return self.carregar_json(self.CACHE_JOGOS)

    def salvar_cache_jogos(self, dados: dict):
        self.salvar_json(self.CACHE_JOGOS, dados)

    def carregar_cache_classificacao(self) -> dict:
        return self.carregar_json(self.CACHE_CLASSIFICACAO)

    def salvar_cache_classificacao(self, dados: dict):
        self.salvar_json(self.CACHE_CLASSIFICACAO, dados)

    def limpar_caches(self):
        """Limpa todos os caches do sistema."""
        try:
            for cache_file in [self.CACHE_JOGOS, self.CACHE_CLASSIFICACAO, self.ALERTAS_PATH]:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            st.success("✅ Caches limpos com sucesso!")
        except Exception as e:
            st.error(f"❌ Erro ao limpar caches: {e}")


class APIClient:
    """Cliente para comunicação com APIs externas"""
    
    def __init__(self):
        self.API_KEY = os.getenv("FOOTBALL_API_KEY", "9058de85e3324bdb969adc005b5d918a")
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7900056631:AAHjG6iCDqQdGTfJI6ce0AZ0E2ilV2fV9RY")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003073115320")
        self.TELEGRAM_CHAT_ID_ALT2 = os.getenv("TELEGRAM_CHAT_ID_ALT2", "-1002754276285")
        
        self.HEADERS = {"X-Auth-Token": self.API_KEY}
        self.BASE_URL_FD = "https://api.football-data.org/v4"
        self.BASE_URL_TG = f"https://api.telegram.org/bot{self.TELEGRAM_TOKEN}/sendMessage"
        
        self.cache_manager = CacheManager()
        self.liga_dict = {
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

    def enviar_telegram(self, msg: str, chat_id: str = None) -> bool:
        """Envia mensagem para o Telegram com tratamento de erro."""
        if chat_id is None:
            chat_id = self.TELEGRAM_CHAT_ID
            
        try:
            response = requests.get(
                self.BASE_URL_TG, 
                params={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                timeout=10
            )
            return response.status_code == 200
        except requests.RequestException as e:
            st.error(f"Erro ao enviar para Telegram: {e}")
            return False

    def obter_dados_api(self, url: str, timeout: int = 10) -> dict | None:
        """Faz requisição genérica à API com tratamento de erro."""
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Erro na requisição API: {e}")
            return None

    def obter_classificacao(self, liga_id: str) -> dict:
        """Obtém dados de classificação da liga."""
        cache = self.cache_manager.carregar_cache_classificacao()
        
        if liga_id in cache:
            return cache[liga_id]

        url = f"{self.BASE_URL_FD}/competitions/{liga_id}/standings"
        data = self.obter_dados_api(url)
        
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
                    "played": t.get("playedGames", 1),
                    "position": t.get("position", 0),
                    "points": t.get("points", 0)
                }

        cache[liga_id] = standings
        self.cache_manager.salvar_cache_classificacao(cache)
        return standings

    def obter_jogos(self, liga_id: str, data: str) -> list:
        """Obtém jogos da liga para uma data específica."""
        cache = self.cache_manager.carregar_cache_jogos()
        key = f"{liga_id}_{data}"
        
        if key in cache:
            return cache[key]

        url = f"{self.BASE_URL_FD}/competitions/{liga_id}/matches?dateFrom={data}&dateTo={data}"
        data = self.obter_dados_api(url)
        
        jogos = data.get("matches", []) if data else []
        cache[key] = jogos
        self.cache_manager.salvar_cache_jogos(cache)
        
        return jogos


class AnalisadorTendencias:
    """Analisa tendências de gols baseado em dados estatísticos"""
    
    @staticmethod
    def analisar_tendencia_gols(dados_casa: dict, dados_fora: dict) -> tuple[float, float, str]:
        """
        Analisa tendência de gols baseada em dados reais da API
        de forma dinâmica e não-engessada
        """
        try:
            # Coleta dados básicos
            gols_feitos_casa = dados_casa.get("scored", 0)
            gols_sofridos_casa = dados_casa.get("against", 0)
            jogos_casa = max(dados_casa.get("played", 1), 1)
            
            gols_feitos_fora = dados_fora.get("scored", 0)
            gols_sofridos_fora = dados_fora.get("against", 0)
            jogos_fora = max(dados_fora.get("played", 1), 1)
            
            # Cálculo de médias básicas
            media_gols_casa = gols_feitos_casa / jogos_casa
            media_sofridos_casa = gols_sofridos_casa / jogos_casa
            media_gols_fora = gols_feitos_fora / jogos_fora
            media_sofridos_fora = gols_sofridos_fora / jogos_fora
            
            # 1. POTENCIAL OFENSIVO DINÂMICO
            potencia_ataque_casa = media_gols_casa
            potencia_ataque_fora = media_gols_fora
            
            # 2. VULNERABILIDADE DEFENSIVA DINÂMICA  
            vulnerabilidade_casa = media_sofridos_casa
            vulnerabilidade_fora = media_sofridos_fora
            
            # 3. TENDÊNCIA REAL BASEADA EM DADOS
            expectativa_gols_casa = (potencia_ataque_casa + vulnerabilidade_fora) / 2
            expectativa_gols_fora = (potencia_ataque_fora + vulnerabilidade_casa) / 2
            
            estimativa_total = expectativa_gols_casa + expectativa_gols_fora
            
            # 4. ANÁLISE DE CONFIANÇA DINÂMICA
            fator_consistencia = AnalisadorTendencias._calcular_consistencia_dados(
                jogos_casa, jogos_fora, dados_casa, dados_fora
            )
            
            # 5. DETERMINAÇÃO DA TENDÊNCIA FLEXÍVEL
            if estimativa_total >= 3.2:
                tendencia = "Mais 2.5"
                confianca_base = 75 + min(20, (estimativa_total - 3.2) * 15)
            elif estimativa_total >= 2.5:
                tendencia = "Mais 2.5" 
                confianca_base = 65 + min(25, (estimativa_total - 2.5) * 12)
            elif estimativa_total >= 2.0:
                tendencia = "Mais 1.5"
                confianca_base = 60 + min(20, (estimativa_total - 2.0) * 15)
            elif estimativa_total >= 1.5:
                tendencia = "Mais 1.5"
                confianca_base = 55 + min(15, (estimativa_total - 1.5) * 10)
            else:
                tendencia = "Menos 2.5"
                confianca_base = 60 + min(25, (1.5 - estimativa_total) * 12)
            
            # Ajuste final da confiança
            confianca_final = min(95, confianca_base * fator_consistencia)
            
            return estimativa_total, confianca_final, tendencia
            
        except Exception as e:
            st.error(f"Erro na análise de tendência: {e}")
            return 2.5, 50, "Indefinido"

    @staticmethod
    def _calcular_consistencia_dados(jogos_casa: int, jogos_fora: int, dados_casa: dict, dados_fora: dict) -> float:
        """Calcula fator de consistência baseado na qualidade dos dados"""
        
        # Fator de amostragem (quantidade de jogos)
        fator_amostragem = min(1.0, (jogos_casa + jogos_fora) / 20)
        
        # Fator de estabilidade (variação entre ataque e defesa)
        try:
            media_gols_casa = dados_casa.get("scored", 0) / max(dados_casa.get("played", 1), 1)
            media_sofridos_casa = dados_casa.get("against", 0) / max(dados_casa.get("played", 1), 1)
            media_gols_fora = dados_fora.get("scored", 0) / max(dados_fora.get("played", 1), 1)
            media_sofridos_fora = dados_fora.get("against", 0) / max(dados_fora.get("played", 1), 1)
            
            # Equipes consistentes têm números similares entre ataque e defesa
            variacao_casa = abs(media_gols_casa - media_sofridos_casa) / max((media_gols_casa + media_sofridos_casa) / 2, 0.1)
            variacao_fora = abs(media_gols_fora - media_sofridos_fora) / max((media_gols_fora + media_sofridos_fora) / 2, 0.1)
            
            fator_estabilidade = 1.0 - (variacao_casa + variacao_fora) / 4
            fator_estabilidade = max(0.6, min(1.2, fator_estabilidade))
            
        except:
            fator_estabilidade = 1.0
        
        return fator_amostragem * fator_estabilidade

    @staticmethod
    def analisar_estilo_jogo_dinamico(dados_equipe: dict) -> str:
        """Analisa estilo de jogo baseado em dados reais"""
        try:
            gols_feitos = dados_equipe.get("scored", 0)
            gols_sofridos = dados_equipe.get("against", 0)
            jogos = max(dados_equipe.get("played", 1), 1)
            
            media_gols = gols_feitos / jogos
            media_sofridos = gols_sofridos / jogos
            
            # Análise dinâmica do estilo
            if media_gols >= 2.0 and media_sofridos >= 1.5:
                return "OFENSIVO_ABERTO"
            elif media_gols >= 1.8 and media_sofridos <= 1.0:
                return "OFENSIVO_EQUILIBRADO"
            elif media_gols <= 1.0 and media_sofridos <= 1.0:
                return "DEFENSIVO_FECHADO"
            elif media_gols <= 1.2 and media_sofridos >= 1.8:
                return "DEFENSIVO_FRAGIL"
            else:
                return "EQUILIBRADO"
                
        except:
            return "INDEFINIDO"


class GerenciadorAlertas:
    """Gerencia o sistema de alertas e notificações"""
    
    def __init__(self, api_client: APIClient, cache_manager: CacheManager):
        self.api_client = api_client
        self.cache_manager = cache_manager
        self.analisador = AnalisadorTendencias()

    def enviar_alerta_telegram(self, fixture: dict, tendencia: str, estimativa: float, confianca: float):
        """Envia alerta formatado para o Telegram."""
        home = fixture["homeTeam"]["name"]
        away = fixture["awayTeam"]["name"]
        data_formatada, hora_formatada = self._formatar_data_iso(fixture["utcDate"])
        competicao = fixture.get("competition", {}).get("name", "Desconhecido")
        
        # Abreviar nome da liga
        liga_abreviada = self._abreviar_liga(competicao)

        status = fixture.get("status", "DESCONHECIDO")
        gols_home = fixture.get("score", {}).get("fullTime", {}).get("home")
        gols_away = fixture.get("score", {}).get("fullTime", {}).get("away")
        
        placar = f"{gols_home} x {gols_away}" if gols_home is not None and gols_away is not None else None

        msg = (
            f"⚽ <b>Alerta de Gols</b>\n"
            f"🏟️ {home} vs {away}\n"
            f"🕒 {hora_formatada} BRT | 🏆 {liga_abreviada}\n"
        )
        
        if placar:
            msg += f"📊 Placar: <b>{placar}</b>\n"
            
        msg += (
            f"📈 Tendência: <b>{tendencia}</b>\n"
            f"🎯 Estimativa: <b>{estimativa:.2f} gols</b>\n"
            f"💯 Confiança: <b>{confianca:.0f}%</b>"
        )
        
        self.api_client.enviar_telegram(msg)

    def verificar_enviar_alerta(self, fixture: dict, tendencia: str, estimativa: float, confianca: float):
        """Verifica e envia alerta se necessário."""
        alertas = self.cache_manager.carregar_alertas()
        fixture_id = str(fixture["id"])
        
        if fixture_id not in alertas:
            alertas[fixture_id] = {
                "tendencia": tendencia,
                "estimativa": estimativa,
                "confianca": confianca,
                "conferido": False
            }
            self.enviar_alerta_telegram(fixture, tendencia, estimativa, confianca)
            self.cache_manager.salvar_alertas(alertas)

    def enviar_top_jogos(self, jogos: list, top_n: int):
        """Envia os top N jogos com informações simplificadas"""
        jogos_filtrados = [j for j in jogos if j["status"] not in ["FINISHED", "IN_PLAY", "POSTPONED", "SUSPENDED"]]

        if not jogos_filtrados:
            st.warning("⚠️ Nenhum jogo elegível para o Top Jogos.")
            return

        top_jogos_sorted = sorted(jogos_filtrados, key=lambda x: x["confianca"], reverse=True)[:top_n]

        # Mensagem SIMPLIFICADA com liga
        msg = f"🏆 TOP {top_n} JOGOS DO DIA\n\n"
        for j in top_jogos_sorted:
            hora_format = j["hora"].strftime("%H:%M")
            msg += (
                f"⏰ {hora_format} | 🏆 {j['liga_abreviada']}\n"
                f"🏟️ {j['home']} vs {j['away']}\n"
                f"📈 {j['tendencia']} | 🎯 {j['estimativa']:.2f} gols | 💯 {j['confianca']:.0f}%\n\n"
            )

        if self.api_client.enviar_telegram(msg, self.api_client.TELEGRAM_CHAT_ID_ALT2):
            st.success(f"🚀 Top {top_n} jogos enviados para o canal!")
        else:
            st.error("❌ Erro ao enviar top jogos")

    def _formatar_data_iso(self, data_iso: str) -> tuple[str, str]:
        """Formata data ISO para data e hora brasileira."""
        try:
            data_jogo = datetime.fromisoformat(data_iso.replace("Z", "+00:00")) - timedelta(hours=3)
            return data_jogo.strftime("%d/%m/%Y"), data_jogo.strftime("%H:%M")
        except ValueError:
            return "Data inválida", "Hora inválida"

    def _abreviar_liga(self, nome_liga: str) -> str:
        """Abrevia nomes longos de ligas."""
        abreviacoes = {
            "Premier League": "Premier League",
            "Bundesliga": "Bundesliga", 
            "Serie A": "Serie A",
            "La Liga": "La Liga",
            "Ligue 1": "Ligue 1",
            "Champions League": "UCL",
            "Europa League": "UEL",
            "Eredivisie": "Eredivisie",
            "Primeira Liga": "Liga Portugal",
            "Campeonato Brasileiro Série A": "Brasileirão"
        }
        
        for nome_completo, abreviado in abreviacoes.items():
            if nome_completo in nome_liga:
                return abreviado
        
        # Se não encontrar abreviação específica, retorna as primeiras palavras
        palavras = nome_liga.split()[:2]
        return " ".join(palavras)


class ProcessadorJogos:
    """Processa e analisa os jogos"""
    
    def __init__(self, api_client: APIClient, gerenciador_alertas: GerenciadorAlertas):
        self.api_client = api_client
        self.gerenciador_alertas = gerenciador_alertas
        self.analisador = AnalisadorTendencias()

    def processar_jogos_dia(self, data_selecionada, todas_ligas, liga_selecionada, top_n, metodo_analise):
        """Processa jogos com análise dinâmica"""
        hoje = data_selecionada.strftime("%Y-%m-%d")
        ligas_busca = self.api_client.liga_dict.values() if todas_ligas else [self.api_client.liga_dict[liga_selecionada]]
        
        st.write(f"⏳ Analisando jogos para {data_selecionada.strftime('%d/%m/%Y')}...")
        
        top_jogos = []
        progress_bar = st.progress(0)
        total_ligas = len(ligas_busca)

        for i, liga_id in enumerate(ligas_busca):
            classificacao = self.api_client.obter_classificacao(liga_id)
            jogos = self.api_client.obter_jogos(liga_id, hoje)

            for match in jogos:
                home = match["homeTeam"]["name"]
                away = match["awayTeam"]["name"]
                
                # Análise dinâmica baseada nos dados reais
                dados_home = classificacao.get(home, {})
                dados_away = classificacao.get(away, {})
                
                estimativa, confianca, tendencia = self.analisador.analisar_tendencia_gols(dados_home, dados_away)
                
                # Ajuste baseado no método selecionado
                if metodo_analise == "Conservador":
                    estimativa *= 0.9
                    confianca *= 0.95
                elif metodo_analise == "Agressivo":
                    estimativa *= 1.1
                    confianca = min(95, confianca * 1.05)

                self.gerenciador_alertas.verificar_enviar_alerta(match, tendencia, estimativa, confianca)

                top_jogos.append({
                    "id": match["id"],
                    "home": home,
                    "away": away,
                    "tendencia": tendencia,
                    "estimativa": estimativa,
                    "confianca": confianca,
                    "liga": match.get("competition", {}).get("name", "Desconhecido"),
                    "liga_abreviada": self.gerenciador_alertas._abreviar_liga(match.get("competition", {}).get("name", "Desconhecido")),
                    "hora": datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")) - timedelta(hours=3),
                    "status": match.get("status", "DESCONHECIDO"),
                    "estilo_home": self.analisador.analisar_estilo_jogo_dinamico(dados_home),
                    "estilo_away": self.analisador.analisar_estilo_jogo_dinamico(dados_away)
                })

            progress_bar.progress((i + 1) / total_ligas)

        # Resultados
        if top_jogos:
            self.gerenciador_alertas.enviar_top_jogos(top_jogos, top_n)
            self._mostrar_analise_detalhada(top_jogos)
            st.success(f"✅ Análise concluída! {len(top_jogos)} jogos processados.")
        else:
            st.warning("⚠️ Nenhum jogo encontrado para a data selecionada.")

    def _mostrar_analise_detalhada(self, jogos: list):
        """Mostra análise detalhada dos jogos"""
        st.subheader("🔍 Análise Detalhada dos Jogos")
        
        for jogo in jogos[:10]:  # Mostra apenas os 10 primeiros
            with st.expander(f"🏟️ {jogo['home']} vs {jogo['away']} - {jogo['tendencia']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Estimativa", f"{jogo['estimativa']:.2f} gols")
                with col2:
                    st.metric("💯 Confiança", f"{jogo['confianca']:.0f}%")
                with col3:
                    st.metric("🏆 Liga", jogo['liga_abreviada'])
                
                st.progress(jogo['confianca'] / 100, text=f"Confiança na análise: {jogo['confianca']:.0f}%")


class ConferenciaResultados:
    """Gerencia a conferência de resultados"""
    
    def __init__(self, api_client: APIClient, cache_manager: CacheManager):
        self.api_client = api_client
        self.cache_manager = cache_manager

    def atualizar_status_partidas(self):
        """Atualiza o status das partidas em cache."""
        cache_jogos = self.cache_manager.carregar_cache_jogos()
        mudou = False

        for key in cache_jogos.keys():
            if key == "_timestamp":
                continue
                
            liga_id, data = key.split("_")
            try:
                url = f"{self.api_client.BASE_URL_FD}/competitions/{liga_id}/matches?dateFrom={data}&dateTo={data}"
                data_api = self.api_client.obter_dados_api(url)
                
                if data_api and "matches" in data_api:
                    cache_jogos[key] = data_api["matches"]
                    mudou = True
                    
            except Exception as e:
                st.error(f"Erro ao atualizar liga {liga_id}: {e}")

        if mudou:
            self.cache_manager.salvar_cache_jogos(cache_jogos)
            st.success("✅ Status das partidas atualizado!")
        else:
            st.info("ℹ️ Nenhuma atualização disponível.")

    def conferir_resultados(self):
        """Conferência de resultados dos jogos alertados."""
        alertas = self.cache_manager.carregar_alertas()
        jogos_cache = self.cache_manager.carregar_cache_jogos()
        
        if not alertas:
            st.info("ℹ️ Nenhum alerta para conferir.")
            return

        jogos_conferidos = []
        mudou = False

        for fixture_id, info in alertas.items():
            if info.get("conferido"):
                continue

            # Encontrar jogo no cache
            jogo_dado = None
            for key, jogos in jogos_cache.items():
                if key == "_timestamp":
                    continue
                for match in jogos:
                    if str(match["id"]) == fixture_id:
                        jogo_dado = match
                        break
                if jogo_dado:
                    break

            if not jogo_dado:
                continue

            # Processar resultado
            resultado_info = self._processar_resultado_jogo(jogo_dado, info)
            if resultado_info:
                self._exibir_resultado_streamlit(resultado_info)
                
                if resultado_info["status"] == "FINISHED":
                    self._enviar_resultado_telegram(resultado_info)
                    info["conferido"] = True
                    mudou = True

            # Coletar dados para PDF
            jogos_conferidos.append(self._preparar_dados_pdf(jogo_dado, info, resultado_info))

        if mudou:
            self.cache_manager.salvar_alertas(alertas)
            st.success("✅ Resultados conferidos e atualizados!")

        # Gerar PDF se houver jogos
        if jogos_conferidos:
            self._gerar_pdf_jogos(jogos_conferidos)

    def _processar_resultado_jogo(self, jogo: dict, info: dict) -> dict | None:
        """Processa o resultado de um jogo."""
        home = jogo["homeTeam"]["name"]
        away = jogo["awayTeam"]["name"]
        status = jogo.get("status", "DESCONHECIDO")
        gols_home = jogo.get("score", {}).get("fullTime", {}).get("home")
        gols_away = jogo.get("score", {}).get("fullTime", {}).get("away")
        
        placar = f"{gols_home} x {gols_away}" if gols_home is not None and gols_away is not None else "-"
        total_gols = (gols_home or 0) + (gols_away or 0)

        # Determinar resultado da aposta
        resultado = "⏳ Aguardando"
        if status == "FINISHED":
            tendencia = info["tendencia"]
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
            "tendencia": info["tendencia"],
            "estimativa": info["estimativa"],
            "confianca": info["confianca"],
            "resultado": resultado,
            "total_gols": total_gols
        }

    def _exibir_resultado_streamlit(self, resultado: dict):
        """Exibe resultado formatado no Streamlit."""
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

    def _enviar_resultado_telegram(self, resultado: dict):
        """Envia resultado para o Telegram."""
        msg = (
            f"📊 <b>Resultado Conferido</b>\n"
            f"🏟️ {resultado['home']} vs {resultado['away']}\n"
            f"⚽ Tendência: {resultado['tendencia']} | Estim.: {resultado['estimativa']:.2f} | Conf.: {resultado['confianca']:.0f}%\n"
            f"📊 Placar Final: <b>{resultado['placar']}</b>\n"
            f"✅ Resultado: <b>{resultado['resultado']}</b>"
        )
        self.api_client.enviar_telegram(msg, self.api_client.TELEGRAM_CHAT_ID_ALT2)

    def _preparar_dados_pdf(self, jogo: dict, info: dict, resultado: dict) -> list:
        """Prepara dados para geração do PDF."""
        home = self._abreviar_nome(jogo["homeTeam"]["name"])
        away = self._abreviar_nome(jogo["awayTeam"]["name"])
        hora = datetime.fromisoformat(jogo["utcDate"].replace("Z", "+00:00")) - timedelta(hours=3)
        
        return [
            f"{home} vs {away}",
            info["tendencia"],
            f"{info['estimativa']:.2f}",
            f"{info['confianca']:.0f}%",
            resultado["placar"] if resultado else "-",
            jogo.get("status", "DESCONHECIDO"),
            resultado["resultado"] if resultado else "⏳ Aguardando",
            hora.strftime("%d/%m %H:%M")
        ]

    def _abreviar_nome(self, nome: str, max_len: int = 15) -> str:
        """Abrevia nomes longos para exibição."""
        if len(nome) <= max_len:
            return nome
        palavras = nome.split()
        abreviado = " ".join([p[0] + "." if len(p) > 2 else p for p in palavras])
        return abreviado[:max_len-3] + "..." if len(abreviado) > max_len else abreviado

    def _gerar_pdf_jogos(self, jogos_conferidos: list):
        """Gera e disponibiliza PDF dos jogos conferidos."""
        buffer = self._gerar_relatorio_pdf(jogos_conferidos)
        
        st.download_button(
            label="📄 Baixar Relatório PDF",
            data=buffer,
            file_name=f"jogos_conferidos_{datetime.today().strftime('%Y-%m-%d')}.pdf",
            mime="application/pdf"
        )

    def _gerar_relatorio_pdf(self, jogos_conferidos: list) -> io.BytesIO:
        """Gera relatório PDF dos jogos conferidos."""
        buffer = io.BytesIO()
        pdf = SimpleDocTemplate(buffer, pagesize=letter, 
                              rightMargin=20, leftMargin=20, 
                              topMargin=20, bottomMargin=20)

        data = [["Jogo", "Tendência", "Estimativa", "Confiança", 
                 "Placar", "Status", "Resultado", "Hora"]] + jogos_conferidos

        table = Table(data, repeatRows=1, 
                     colWidths=[120, 70, 60, 60, 50, 70, 60, 70])
        
        style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4B4B4B")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F5F5F5")),
            ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ])

        # Alternar cores das linhas
        for i in range(1, len(data)):
            bg_color = colors.HexColor("#E0E0E0") if i % 2 == 0 else colors.HexColor("#F5F5F5")
            style.add('BACKGROUND', (0,i), (-1,i), bg_color)

        table.setStyle(style)
        pdf.build([table])
        buffer.seek(0)
        return buffer


class SistemaAlertaGols:
    """Sistema principal de alertas de gols"""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.api_client = APIClient()
        self.gerenciador_alertas = GerenciadorAlertas(self.api_client, self.cache_manager)
        self.processador_jogos = ProcessadorJogos(self.api_client, self.gerenciador_alertas)
        self.conferencia_resultados = ConferenciaResultados(self.api_client, self.cache_manager)

    def executar(self):
        """Método principal para executar a aplicação"""
        st.set_page_config(page_title="⚽ Alerta de Gols Dinâmico", layout="wide")
        st.title("⚽ Sistema de Análise Dinâmica de Gols")
        
        # Sidebar
        with st.sidebar:
            st.header("🔧 Configurações de Análise")
            top_n = st.selectbox("📊 Jogos no Top", [3, 5, 10], index=0)
            
            st.subheader("🎯 Método de Análise")
            metodo_analise = st.radio(
                "Selecione o método:",
                ["Dinâmico (Recomendado)", "Conservador", "Agressivo"],
                index=0
            )
            
            st.info("""
            **Análise Dinâmica**: 
            - Baseada em dados real da API
            - Adaptável a cada jogo
            - Sem regras engessadas
            """)

        # Controles principais
        col1, col2 = st.columns([2, 1])
        
        with col1:
            data_selecionada = st.date_input("📅 Data para análise:", value=datetime.today())
        
        with col2:
            todas_ligas = st.checkbox("🌍 Todas as ligas", value=True)

        liga_selecionada = None
        if not todas_ligas:
            liga_selecionada = st.selectbox("📌 Liga específica:", list(self.api_client.liga_dict.keys()))

        # Botões de ação
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Analisar Partidas", type="primary"):
                self.processador_jogos.processar_jogos_dia(
                    data_selecionada, todas_ligas, liga_selecionada, top_n, metodo_analise
                )
        
        with col2:
            if st.button("🔄 Atualizar Status"):
                self.conferencia_resultados.atualizar_status_partidas()
        
        with col3:
            if st.button("📊 Conferir Resultados"):
                self.conferencia_resultados.conferir_resultados()

        # Botão adicional
        if st.button("🧹 Limpar Cache"):
            self.cache_manager.limpar_caches()

        # Estatísticas em tempo real
        st.subheader("📈 Estatísticas da Análise")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔄 Método", metodo_analise)
        with col2:
            st.metric("📅 Data", data_selecionada.strftime("%d/%m/%Y"))
        with col3:
            st.metric("🎯 Jogos no Top", top_n)


# =============================
# EXECUÇÃO PRINCIPAL
# =============================

if __name__ == "__main__":
    sistema = SistemaAlertaGols()
    sistema.executar()
