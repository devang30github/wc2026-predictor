"""
Stage 4a: Compute Elo ratings for every national team using the full
international match history (1872-2026), processed chronologically.

Uses the standard World Football Elo Ratings approach:
https://www.eloratings.net/about
"""

import pandas as pd
import os

PROCESSED_DIR = "data/processed"

STARTING_ELO = 1500

# K-factor by competition importance (World Football Elo convention)
def get_k_factor(tournament: str) -> int:
    t = tournament.lower()
    if "world cup" in t and "qualif" not in t:
        return 60          # World Cup finals matches matter most
    elif "cup" in t or "championship" in t or "confederations" in t:
        return 50
    elif "friendly" in t:
        return 20
    else:
        return 40          # qualifiers and other competitive matches


def goal_diff_multiplier(goal_diff: int) -> float:
    """Bigger wins move Elo more, but with diminishing returns."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return (11 + gd) / 8


def compute_elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Processes matches in chronological order, updating a running Elo
    dict. Returns a DataFrame with elo_home_before/elo_away_before
    columns attached to every match (useful for building training features),
    plus we save the FINAL rating table separately.
    """
    elo = {}
    home_elo_before = []
    away_elo_before = []

    matches = matches.sort_values("date").reset_index(drop=True)

    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]
        r_home = elo.get(home, STARTING_ELO)
        r_away = elo.get(away, STARTING_ELO)

        home_elo_before.append(r_home)
        away_elo_before.append(r_away)

        # skip rating update for rows with missing scores (future/unplayed)
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue

        home_score, away_score = row["home_score"], row["away_score"]

        if home_score > away_score:
            W_home = 1.0
        elif home_score == away_score:
            W_home = 0.5
        else:
            W_home = 0.0

        We_home = 1 / (10 ** (-(r_home - r_away) / 400) + 1)

        k = get_k_factor(row["tournament"])
        gd_mult = goal_diff_multiplier(home_score - away_score)

        elo[home] = r_home + k * gd_mult * (W_home - We_home)
        elo[away] = r_away + k * gd_mult * ((1 - W_home) - (1 - We_home))

    matches["home_elo_before"] = home_elo_before
    matches["away_elo_before"] = away_elo_before

    return matches, elo


if __name__ == "__main__":
    matches = pd.read_csv(f"{PROCESSED_DIR}/all_matches_clean.csv")
    matches["date"] = pd.to_datetime(matches["date"])

    matches_with_elo, final_elo = compute_elo_ratings(matches)

    matches_with_elo.to_csv(f"{PROCESSED_DIR}/matches_with_elo.csv", index=False)

    elo_table = pd.DataFrame(
        [{"team": t, "elo": r} for t, r in final_elo.items()]
    ).sort_values("elo", ascending=False).reset_index(drop=True)

    elo_table.to_csv(f"{PROCESSED_DIR}/current_elo_ratings.csv", index=False)

    print("Top 15 teams by current Elo rating:")
    print(elo_table.head(15))