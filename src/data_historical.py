"""
Stage 2: Download and clean historical international football results.
Source: Kaggle 'International football results from 1872 to 2026' (martj42)
We fetch it directly from the maintainer's public GitHub mirror (no Kaggle auth needed).
"""

import pandas as pd
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Public CSV mirror of the dataset (results.csv)
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"

def download_historical_data():
    print("Downloading historical match results...")
    results = pd.read_csv(RESULTS_URL)
    shootouts = pd.read_csv(SHOOTOUTS_URL)

    results.to_csv(f"{RAW_DIR}/results_raw.csv", index=False)
    shootouts.to_csv(f"{RAW_DIR}/shootouts_raw.csv", index=False)

    print(f"Saved {len(results)} matches and {len(shootouts)} shootouts to {RAW_DIR}/")
    return results, shootouts


def filter_world_cups(results: pd.DataFrame) -> pd.DataFrame:
    """Keep only actual FIFA World Cup matches (for tournament-specific features)."""
    wc = results[results["tournament"] == "FIFA World Cup"].copy()
    wc["date"] = pd.to_datetime(wc["date"])
    return wc


def clean_all_matches(results: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning applied to the FULL match history (used for Elo ratings)."""
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    results, shootouts = download_historical_data()

    all_matches = clean_all_matches(results)
    all_matches.to_csv(f"{PROCESSED_DIR}/all_matches_clean.csv", index=False)
    print(f"Cleaned full match history: {len(all_matches)} rows -> {PROCESSED_DIR}/all_matches_clean.csv")

    wc_matches = filter_world_cups(all_matches)
    wc_matches.to_csv(f"{PROCESSED_DIR}/world_cup_matches.csv", index=False)
    print(f"World Cup-only matches: {len(wc_matches)} rows -> {PROCESSED_DIR}/world_cup_matches.csv")