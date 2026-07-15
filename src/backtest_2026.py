"""
Stage 6: Backtest the trained models against real 2026 World Cup matches
played so far, to see how well they'd have predicted this actual tournament.
"""

import pandas as pd
import joblib
import numpy as np

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

FEATURE_COLS = [
    "elo_diff", "h2h_a_wins", "h2h_b_wins", "h2h_draws",
    "h2h_a_goals_avg", "h2h_b_goals_avg",
    "form_diff_points", "form_diff_goal_diff",
    "form_a_goals_for", "form_b_goals_for",
]


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
        "h2h_a_wins": a_wins, "h2h_b_wins": b_wins, "h2h_draws": draws,
        "h2h_a_goals_avg": a_goals / total if total else 0,
        "h2h_b_goals_avg": b_goals / total if total else 0,
    }


def get_form_before(team, wc2026_played, before_date):
    prior = wc2026_played[wc2026_played["date"] < before_date]
    home = prior[prior["home_team"] == team]
    away = prior[prior["away_team"] == team]

    goals_for = home["home_score"].sum() + away["away_score"].sum()
    goals_against = home["away_score"].sum() + away["home_score"].sum()
    wins = (home["home_score"] > home["away_score"]).sum() + (away["away_score"] > away["home_score"]).sum()
    draws = (home["home_score"] == home["away_score"]).sum() + (away["away_score"] == away["home_score"]).sum()

    return {"points": wins * 3 + draws, "goal_diff": goals_for - goals_against, "goals_for": goals_for}


def build_features_at_time(team_a, team_b, match_date, matches_with_elo, wc2026_played):
    prior_matches = matches_with_elo[matches_with_elo["date"] < match_date]

    def last_elo(team):
        team_matches = prior_matches[(prior_matches["home_team"] == team) | (prior_matches["away_team"] == team)]
        if team_matches.empty:
            return 1500.0
        last = team_matches.iloc[-1]
        return last["home_elo_before"] if last["home_team"] == team else last["away_elo_before"]

    elo_a, elo_b = last_elo(team_a), last_elo(team_b)
    h2h = get_h2h_before(team_a, team_b, matches_with_elo, match_date)
    form_a = get_form_before(team_a, wc2026_played, match_date)
    form_b = get_form_before(team_b, wc2026_played, match_date)

    return {
        "elo_diff": elo_a - elo_b,
        **h2h,
        "form_diff_points": form_a["points"] - form_b["points"],
        "form_diff_goal_diff": form_a["goal_diff"] - form_b["goal_diff"],
        "form_a_goals_for": form_a["goals_for"],
        "form_b_goals_for": form_b["goals_for"],
    }


if __name__ == "__main__":
    matches_with_elo = pd.read_csv(f"{PROCESSED_DIR}/matches_with_elo.csv")
    matches_with_elo["date"] = pd.to_datetime(matches_with_elo["date"])

    wc2026_played = pd.read_csv(f"{PROCESSED_DIR}/wc2026_matches_played.csv")
    wc2026_played["date"] = pd.to_datetime(wc2026_played["date"])
    wc2026_played = wc2026_played.sort_values("date").reset_index(drop=True)

    clf = joblib.load(f"{MODELS_DIR}/outcome_classifier.pkl")
    le = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")

    correct = 0
    total = 0
    results = []

    for _, m in wc2026_played.iterrows():
        team_a, team_b = m["home_team"], m["away_team"]
        feats = build_features_at_time(team_a, team_b, m["date"], matches_with_elo, wc2026_played)
        X = pd.DataFrame([feats])[FEATURE_COLS]

        pred_class = clf.predict(X)[0]
        pred_label = le.inverse_transform([pred_class])[0]
        probs = clf.predict_proba(X)[0]
        prob_dict = dict(zip(le.classes_, probs))

        if m["home_score"] > m["away_score"]:
            actual = "A_win"
        elif m["home_score"] < m["away_score"]:
            actual = "B_win"
        else:
            actual = "Draw"

        is_correct = (pred_label == actual)
        correct += int(is_correct)
        total += 1

        results.append({
            "date": m["date"].date(), "team_a": team_a, "team_b": team_b,
            "actual_score": f"{int(m['home_score'])}-{int(m['away_score'])}",
            "actual_result": actual, "predicted_result": pred_label,
            "prob_a_win": round(prob_dict.get("A_win", 0), 3),
            "prob_draw": round(prob_dict.get("Draw", 0), 3),
            "prob_b_win": round(prob_dict.get("B_win", 0), 3),
            "correct": is_correct,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{PROCESSED_DIR}/backtest_2026_results.csv", index=False)

    accuracy = correct / total
    print(f"Backtest on {total} matches from WC 2026 (through the semifinals)")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total} correct)\n")
    print(results_df.tail(15).to_string(index=False))