"""
Stage 8: Flask backend serving the prediction pipeline to a simple web UI.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(__file__))
from predict_final import predict_final

app = Flask(__name__, template_folder="../templates", static_folder="../static")

PROCESSED_DIR = "data/processed"


def get_team_list():
    df = pd.read_csv(f"{PROCESSED_DIR}/wc2026_team_summary.csv")
    return sorted(df["team"].tolist())


@app.route("/")
def index():
    teams = get_team_list()
    return render_template("index.html", teams=teams)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    team_a = data.get("team_a")
    team_b = data.get("team_b")

    if not team_a or not team_b:
        return jsonify({"error": "Select both teams."}), 400
    if team_a == team_b:
        return jsonify({"error": "Pick two different teams."}), 400

    try:
        result = predict_final(team_a, team_b)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)