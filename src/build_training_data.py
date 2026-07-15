"""
Stage 5a: Build a training dataset from historical World Cup matches,
where every feature is computed using ONLY information available
before that match was played (point-in-time — no data leakage).

For each historical WC match we generate TWO rows (A vs B and B vs A)
so the model learns a symmetric relationship, since the actual final
is played at a neutral-ish venue (no real home advantage for either
Spain or Argentina in the USA/Mexico/Canada final).
"""

import pandas as pd
import os

PROCESSED_DIR = "data/processed"


def get_h2h_before(team_a, team_b, all_matches, before_date):
    mask = (
        ((all_matches["home_team"] == team_a) & (all_matches["away_team"] == team_b)) |
        ((all_matches["home_team"] == team_b) & (all_matches["away_team"] == team_a))
    ) & (all_matches["date"] < before_date)

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
        "h2h_a_wins": a_wins,
        "h2h_b_wins": b_wins,
        "h2h_draws": draws,
        "h2h_a_goals_avg": a_goals / total if total > 0 else 0,
        "h2h_b_goals_avg": b_goals / total if total > 0 else 0,
    }


def get_form_within_edition(team, edition_matches_so_far):
    """edition_matches_so_far = matches from the SAME WC edition, strictly before this match."""
    home = edition_matches_so_far[edition_matches_so_far["home_team"] == team]
    away = edition_matches_so_far[edition_matches_so_far["away_team"] == team]

    goals_for = home["home_score"].sum() + away["away_score"].sum()
    goals_against = home["away_score"].sum() + away["home_score"].sum()
    played = len(home) + len(away)

    wins = (home["home_score"] > home["away_score"]).sum() + (away["away_score"] > away["home_score"]).sum()
    draws = (home["home_score"] == home["away_score"]).sum() + (away["away_score"] == away["home_score"]).sum()

    return {
        "points": wins * 3 + draws,
        "goal_diff": goals_for - goals_against,
        "goals_for": goals_for,
    }


def build_dataset():
    matches = pd.read_csv(f"{PROCESSED_DIR}/matches_with_elo.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    matches["year"] = matches["date"].dt.year

    wc = matches[
        (matches["tournament"] == "FIFA World Cup") &
        (matches["date"] >= "1990-01-01") &
        (matches["home_score"].notna())
    ].sort_values("date").reset_index(drop=True)

    rows = []
    for idx, m in wc.iterrows():
        edition_so_far = wc[(wc["year"] == m["year"]) & (wc["date"] < m["date"])]

        h2h = get_h2h_before(m["home_team"], m["away_team"], matches, m["date"])
        form_home = get_form_within_edition(m["home_team"], edition_so_far)
        form_away = get_form_within_edition(m["away_team"], edition_so_far)

        base = {
            "elo_diff": m["home_elo_before"] - m["away_elo_before"],
            "h2h_a_wins": h2h["h2h_a_wins"],
            "h2h_b_wins": h2h["h2h_b_wins"],
            "h2h_draws": h2h["h2h_draws"],
            "h2h_a_goals_avg": h2h["h2h_a_goals_avg"],
            "h2h_b_goals_avg": h2h["h2h_b_goals_avg"],
            "form_diff_points": form_home["points"] - form_away["points"],
            "form_diff_goal_diff": form_home["goal_diff"] - form_away["goal_diff"],
            "form_a_goals_for": form_home["goals_for"],
            "form_b_goals_for": form_away["goals_for"],
            "score_a": m["home_score"],
            "score_b": m["away_score"],
        }
        if m["home_score"] > m["away_score"]:
            base["result"] = "A_win"
        elif m["home_score"] < m["away_score"]:
            base["result"] = "B_win"
        else:
            base["result"] = "Draw"
        rows.append(base)

        # Mirrored row (B vs A) so the model has no home-side bias
        mirrored = {
            "elo_diff": -base["elo_diff"],
            "h2h_a_wins": base["h2h_b_wins"],
            "h2h_b_wins": base["h2h_a_wins"],
            "h2h_draws": base["h2h_draws"],
            "h2h_a_goals_avg": base["h2h_b_goals_avg"],
            "h2h_b_goals_avg": base["h2h_a_goals_avg"],
            "form_diff_points": -base["form_diff_points"],
            "form_diff_goal_diff": -base["form_diff_goal_diff"],
            "form_a_goals_for": base["form_b_goals_for"],
            "form_b_goals_for": base["form_a_goals_for"],
            "score_a": base["score_b"],
            "score_b": base["score_a"],
            "result": "B_win" if base["result"] == "A_win" else ("A_win" if base["result"] == "B_win" else "Draw"),
        }
        rows.append(mirrored)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv(f"{PROCESSED_DIR}/training_data.csv", index=False)
    print(f"Built training dataset: {len(df)} rows (from {len(df)//2} real matches)")
    print(df["result"].value_counts())
    print(df.head())