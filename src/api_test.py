# src/api_test.py
from dotenv import load_dotenv
import os, requests

load_dotenv()
API_KEY = os.environ.get("API_FOOTBALL_KEY")

url = "https://v3.football.api-sports.io/leagues"
headers = {"x-apisports-key": API_KEY}
params = {"search": "World Cup"}

resp = requests.get(url, headers=headers, params=params)
data = resp.json()

for item in data["response"]:
    league = item["league"]
    if league["type"] == "Cup":
        print(league["id"], league["name"], item["seasons"][-1]["year"] if item["seasons"] else "")