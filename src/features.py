"""
Stage 4b: Combine Elo ratings, head-to-head history, and current 2026
tournament form into a single feature-building function.

build_match_features(team_a, team_b) is the core function we'll reuse
everywhere: training, backtesting, and the final prediction.
"""

import pandas as pd
import os

PROCESSED_DIR = "data/processed"


def load_all_data():
    elo_table = pd.read_csv(f"{PROCESSED_DIR}/current_elo_ratings.csv")
    all_matches = pd.read_csv(f"{PROCESSED_DIR}/all_matches_clean.csv")
    all_matches["date"] = pd.to_datetime(all_matches["date"])
    team_summary_2026 = pd.read_csv(f"{PROCESSED_DIR}/wc2026_team_summary.csv")
    return elo_table, all_matches, team_summary_2026


def get_elo(team: str, elo_table: pd.DataFrame) -> float:
    row = elo_table[elo_table["team"] == team]
    if row.empty:
        print(f"⚠️ Warning: '{team}' not found in Elo table, using default 1500")
        return 1500.0
    return float(row["elo"].iloc[0])


def get_head_to_head(team_a: str, team_b: str, all_matches: pd.DataFrame) -> dict:
    """All-time head-to-head record between two teams, any competition."""
    mask = (
        ((all_matches["home_team"] == team_a) & (all_matches["away_team"] == team_b)) |
        ((all_matches["home_team"] == team_b) & (all_matches["away_team"] == team_a))
    )
    h2h = all_matches[mask].dropna(subset=["home_score", "away_score"])

    a_wins = b_wins = draws = 0
    a_goals = b_goals = 0

    for _, row in h2h.iterrows():
        if row["home_team"] == team_a:
            gf_a, gf_b = row["home_score"], row["away_score"]
        else:
            gf_a, gf_b = row["away_score"], row["home_score"]

        a_goals += gf_a
        b_goals += gf_b

        if gf_a > gf_b:
            a_wins += 1
        elif gf_b > gf_a:
            b_wins += 1
        else:
            draws += 1

    total = len(h2h)
    return {
        "h2h_matches": total,
        "h2h_a_wins": a_wins,
        "h2h_b_wins": b_wins,
        "h2h_draws": draws,
        "h2h_a_goals_avg": a_goals / total if total > 0 else 0,
        "h2h_b_goals_avg": b_goals / total if total > 0 else 0,
    }


def get_tournament_form(team: str, team_summary: pd.DataFrame) -> dict:
    row = team_summary[team_summary["team"] == team]
    if row.empty:
        return {
            "matches_played": 0, "wins": 0, "draws": 0, "losses": 0,
            "goals_for": 0, "goals_against": 0, "goal_diff": 0, "points": 0,
        }
    r = row.iloc[0]
    return {
        "matches_played": r["matches_played"],
        "wins": r["wins"],
        "draws": r["draws"],
        "losses": r["losses"],
        "goals_for": r["goals_for"],
        "goals_against": r["goals_against"],
        "goal_diff": r["goal_diff"],
        "points": r["points"],
    }

def get_squad_strength(team: str) -> dict:
    squad_df = pd.read_csv(f"{PROCESSED_DIR.replace('processed','raw')}/squad_strength.csv")
    row = squad_df[squad_df["team"] == team]
    if row.empty:
        print(f"⚠️ Warning: no squad strength data for '{team}', using neutral defaults")
        return {"avg_market_value_eur_m": 30.0, "squad_avg_age": 26.0, "key_players_available": 11}
    r = row.iloc[0]
    return {
        "avg_market_value_eur_m": r["avg_market_value_eur_m"],
        "squad_avg_age": r["squad_avg_age"],
        "key_players_available": r["key_players_available"],
    }

def build_match_features(team_a: str, team_b: str) -> dict:
    """
    Builds one feature row comparing team_a vs team_b.
    All features are framed as A-relative-to-B (so the label is symmetric:
    positive values favor team_a, negative favor team_b).
    """
    elo_table, all_matches, team_summary_2026 = load_all_data()

    elo_a = get_elo(team_a, elo_table)
    elo_b = get_elo(team_b, elo_table)

    h2h = get_head_to_head(team_a, team_b, all_matches)
    form_a = get_tournament_form(team_a, team_summary_2026)
    form_b = get_tournament_form(team_b, team_summary_2026)

    squad_a = get_squad_strength(team_a)
    squad_b = get_squad_strength(team_b)

    features = {
        "team_a": team_a,
        "team_b": team_b,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_diff": elo_a - elo_b,
        **h2h,
        "form_a_points": form_a["points"],
        "form_b_points": form_b["points"],
        "form_diff_points": form_a["points"] - form_b["points"],
        "form_a_goal_diff": form_a["goal_diff"],
        "form_b_goal_diff": form_b["goal_diff"],
        "form_diff_goal_diff": form_a["goal_diff"] - form_b["goal_diff"],
        "form_a_goals_for": form_a["goals_for"],
        "form_b_goals_for": form_b["goals_for"],
        "squad_value_a": squad_a["avg_market_value_eur_m"],
        "squad_value_b": squad_b["avg_market_value_eur_m"],
        "squad_value_diff": squad_a["avg_market_value_eur_m"] - squad_b["avg_market_value_eur_m"],
        "key_players_a": squad_a["key_players_available"],
        "key_players_b": squad_b["key_players_available"],
    }
    return features


if __name__ == "__main__":
    # Quick test using the two current finalists-in-progress
    test_features = build_match_features("Spain", "Argentina")
    for k, v in test_features.items():
        print(f"{k}: {v}")