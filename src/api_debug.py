# src/api_debug.py
from dotenv import load_dotenv
import os, requests, json

load_dotenv()
API_KEY = os.environ.get("API_FOOTBALL_KEY")

url = "https://v3.football.api-sports.io/fixtures"
headers = {"x-apisports-key": API_KEY}
params = {"league": 1, "season": 2026}

resp = requests.get(url, headers=headers, params=params)
data = resp.json()

print("HTTP status:", resp.status_code)
print("Results count:", data.get("results"))
print("Errors:", data.get("errors"))
print("Response sample:", json.dumps(data.get("response", [])[:1], indent=2))