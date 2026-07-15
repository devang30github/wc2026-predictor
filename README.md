# 🏆 World Cup 2026 Final Predictor

Predicts the winner, win probability, and scoreline for any FIFA World Cup 2026
match — built entirely from historical Elo ratings, head-to-head records, and
live 2026 tournament form. No expert opinions, no punditry — just data.

Backtested at **67.3% accuracy** across all 101 World Cup 2026 matches played
so far (group stage through semifinals).

## How it works

1. **Historical data** (1872–2026, all internationals) → Elo ratings per team
2. **2026 tournament data** (matches before the final) → current form, goal difference, points
3. **Squad strength** → manually curated market value / key-player availability
4. All three feed a trained classifier (win/draw/loss) + regressors (scoreline)

## Setup (GitHub Codespaces)

```bash
pip install -r requirements.txt
```

Add your API key (optional, not required for core pipeline) to `.env`:

API_FOOTBALL_KEY=your_key_here

## Run the pipeline

```bash
python src/data_historical.py      # historical matches
python src/data_2026.py            # 2026 tournament matches
python src/elo.py                  # Elo ratings
python src/build_training_data.py  # leakage-safe training set
python src/train_model.py          # train + compare models (MLflow)
python src/backtest_2026.py        # validate on real 2026 matches
```

## Predict a match

```bash
python src/predict_final.py "Spain" "Argentina"
```

## Web UI

```bash
python src/app.py
```
Open the forwarded port (5001) in Codespaces, pick two teams, hit **Predict**.

## View experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · MLflow · Flask