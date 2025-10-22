import requests
import time
import json
import os
from datetime import datetime, timedelta

# ===============================
# Configurações
# ===============================
CACHE_FILE = "nba_games_cache.json"
RATE_LIMIT = 1.5  # segundos entre requisições
PER_PAGE = 100    # máximo permitido pela API

# ===============================
# Funções de cache
# ===============================
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4, default=str)

# ===============================
# Função para buscar jogos por time
# ===============================
def fetch_team_games(team_id, start_date, end_date):
    """
    Busca jogos de um time usando cache local e respeitando rate limit
    """
    cache = load_cache()
    key = f"{team_id}_{start_date}_{end_date}"
    
    if key in cache:
        print(f"Usando cache para time {team_id} ({start_date} -> {end_date})")
        return cache[key]
    
    print(f"Buscando jogos para time {team_id} ({start_date} -> {end_date})")
    all_games = []
    delta_days = 30  # quebra em períodos menores para evitar timeouts
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    final_end = datetime.strptime(end_date, "%Y-%m-%d")

    while current_start <= final_end:
        current_end = min(current_start + timedelta(days=delta_days-1), final_end)
        page = 1
        while True:
            url = f"https://api.balldontlie.io/v1/games?team_ids[]={team_id}" \
                  f"&start_date={current_start.date()}&end_date={current_end.date()}" \
                  f"&per_page={PER_PAGE}&page={page}"
            
            retries = 5
            backoff = 2
            while retries > 0:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    all_games.extend(data['data'])
                    break
                elif response.status_code == 429:
                    print(f"Rate limit atingido. Esperando {backoff} segundos...")
                    time.sleep(backoff)
                    backoff *= 2
                    retries -= 1
                else:
                    print(f"Erro {response.status_code} para {url}")
                    retries = 0
            
            if len(data['data']) < PER_PAGE:
                break
            page += 1
            time.sleep(RATE_LIMIT)
        
        current_start += timedelta(days=delta_days)
        time.sleep(RATE_LIMIT)

    cache[key] = all_games
    save_cache(cache)
    return all_games

# ===============================
# Exemplo de uso para múltiplos times
# ===============================
if __name__ == "__main__":
    team_ids = [1,2,3,4,5,9,13,15,16,17,18,19,22,23,24,25,26,28,29,30]  # IDs NBA
    start_date = "2024-10-22"
    end_date = "2025-10-22"

    all_games = {}
    for team_id in team_ids:
        games = fetch_team_games(team_id, start_date, end_date)
        all_games[team_id] = games
        time.sleep(RATE_LIMIT)  # delay entre times para respeitar limite

    print(f"Todos os jogos foram salvos no cache local: {CACHE_FILE}")
