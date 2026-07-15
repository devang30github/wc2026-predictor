"""
Stage 7: Predict the FIFA World Cup 2026 Final.

Input: two team names.
Output: predicted winner, win probabilities, most likely scoreline.

Handles the fact that a FINAL cannot end in a draw (extra time + penalties
decide it) — draw probability is redistributed proportionally to the two
teams based on their relative strength.
"""

import pandas as pd
import joblib
import sys

sys.path.append("src")
from features import build_match_features

MODELS_DIR = "models"

FEATURE_COLS = [
    "elo_diff", "h2h_a_wins", "h2h_b_wins", "h2h_draws",
    "h2h_a_goals_avg", "h2h_b_goals_avg",
    "form_diff_points", "form_diff_goal_diff",
    "form_a_goals_for", "form_b_goals_for",
]

def load_models():
    clf = joblib.load(f"{MODELS_DIR}/outcome_classifier.pkl")
    le = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")
    reg_a = joblib.load(f"{MODELS_DIR}/score_a_regressor.pkl")
    reg_b = joblib.load(f"{MODELS_DIR}/score_b_regressor.pkl")
    reg_a_name = joblib.load(f"{MODELS_DIR}/score_a_best_name.pkl")
    reg_b_name = joblib.load(f"{MODELS_DIR}/score_b_best_name.pkl")
    scaler_a = joblib.load(f"{MODELS_DIR}/score_a_scaler.pkl")
    scaler_b = joblib.load(f"{MODELS_DIR}/score_b_scaler.pkl")
    return clf, le, reg_a, reg_b, reg_a_name, reg_b_name, scaler_a, scaler_b


def predict_with_regressor(model, model_name, scaler, X):
    """Poisson linear models were trained on scaled features; tree models weren't."""
    if model_name == "poisson_linear":
        X_input = scaler.transform(X)
    else:
        X_input = X
    return max(0, round(float(model.predict(X_input)[0])))

def prepare_feature_row(raw_features: dict) -> pd.DataFrame:
    """Map build_match_features() output onto the exact FEATURE_COLS the model expects."""
    row = {
        "elo_diff": raw_features["elo_diff"],
        "h2h_a_wins": raw_features["h2h_a_wins"],
        "h2h_b_wins": raw_features["h2h_b_wins"],
        "h2h_draws": raw_features["h2h_draws"],
        "h2h_a_goals_avg": raw_features["h2h_a_goals_avg"],
        "h2h_b_goals_avg": raw_features["h2h_b_goals_avg"],
        "form_diff_points": raw_features["form_diff_points"],
        "form_diff_goal_diff": raw_features["form_diff_goal_diff"],
        "form_a_goals_for": raw_features["form_a_goals_for"],
        "form_b_goals_for": raw_features["form_b_goals_for"],
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def redistribute_draw_probability(prob_a, prob_draw, prob_b):
    """
    A final can't end in a draw (extra time + penalties decide it).
    Split the draw probability between A and B proportionally to their
    existing win probabilities — a team already favored to win in 90 minutes
    is treated as slightly more likely to also win in ET/penalties.
    """
    if prob_a + prob_b == 0:
        split_a = split_b = 0.5
    else:
        split_a = prob_a / (prob_a + prob_b)
        split_b = prob_b / (prob_a + prob_b)

    final_prob_a = prob_a + prob_draw * split_a
    final_prob_b = prob_b + prob_draw * split_b
    return final_prob_a, final_prob_b


def predict_final(team_a: str, team_b: str):
    clf, le, reg_a, reg_b, reg_a_name, reg_b_name, scaler_a, scaler_b = load_models()

    raw_features = build_match_features(team_a, team_b)
    X = prepare_feature_row(raw_features)

    probs = clf.predict_proba(X)[0]
    prob_dict = dict(zip(le.classes_, probs))
    prob_a_win = prob_dict.get("A_win", 0)
    prob_draw = prob_dict.get("Draw", 0)
    prob_b_win = prob_dict.get("B_win", 0)

    final_prob_a, final_prob_b = redistribute_draw_probability(prob_a_win, prob_draw, prob_b_win)

    pred_score_a = predict_with_regressor(reg_a, reg_a_name, scaler_a, X)
    pred_score_b = predict_with_regressor(reg_b, reg_b_name, scaler_b, X)

    # If the regressors predict a level score, break the tie using the
    # redistributed win probability (mirrors what actually happens: ET/penalties)
    went_to_tiebreak = False
    if pred_score_a == pred_score_b:
        went_to_tiebreak = True
        if final_prob_a >= final_prob_b:
            pred_score_a += 1
        else:
            pred_score_b += 1

    winner = team_a if final_prob_a >= final_prob_b else team_b

    print("=" * 55)
    print(f"FIFA WORLD CUP 2026 FINAL PREDICTION")
    print(f"{team_a}  vs  {team_b}")
    print("=" * 55)
    print(f"\nRaw model outcome probabilities (90 minutes):")
    print(f"  {team_a} win : {prob_a_win:.1%}")
    print(f"  Draw        : {prob_draw:.1%}")
    print(f"  {team_b} win : {prob_b_win:.1%}")

    print(f"\nFinal-adjusted win probability (draw redistributed, since a final must have a winner):")
    print(f"  {team_a}: {final_prob_a:.1%}")
    print(f"  {team_b}: {final_prob_b:.1%}")

    print(f"\nPredicted winner: {winner}")

    score_line = f"{team_a} {pred_score_a} - {pred_score_b} {team_b}"
    if went_to_tiebreak:
        score_line += "  (scores level after 90 min — decided via extra time/penalties per model edge)"
    print(f"Most likely scoreline: {score_line}")
    print("=" * 55)

    return {
        "team_a": team_a, "team_b": team_b,
        "prob_a_win": final_prob_a, "prob_b_win": final_prob_b,
        "predicted_score": (pred_score_a, pred_score_b),
        "winner": winner,
    }


if __name__ == "__main__":
    if len(sys.argv) == 3:
        team_a, team_b = sys.argv[1], sys.argv[2]
    else:
        team_a, team_b = "Spain", "Argentina"   # placeholder until England vs Argentina is decided today

    predict_final(team_a, team_b)