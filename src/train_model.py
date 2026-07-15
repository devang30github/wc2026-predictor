"""
Stage 5b (revised): Train multiple candidate models for both the
outcome classifier and the score regressors, log every one to MLflow,
and print a comparison table so we can pick the best performer.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "elo_diff", "h2h_a_wins", "h2h_b_wins", "h2h_draws",
    "h2h_a_goals_avg", "h2h_b_goals_avg",
    "form_diff_points", "form_diff_goal_diff",
    "form_a_goals_for", "form_b_goals_for",
]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wc2026_predictor")


# ---------------- Outcome Classifiers ----------------
def log_model_safely(model, name):
    """XGBoost models need mlflow.xgboost, everything else uses mlflow.sklearn."""
    if "xgboost" in name.lower() or "XGB" in type(model).__name__:
        mlflow.xgboost.log_model(model, name=name)
    else:
        mlflow.sklearn.log_model(model, name=name)

def get_classifiers():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            objective="multi:softprob", num_class=3, random_state=42,
        ),
    }


def train_all_classifiers(df):
    X = df[FEATURE_COLS]
    y = df["result"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    best_model, best_name, best_acc = None, None, -1

    for name, model in get_classifiers().items():
        with mlflow.start_run(run_name=f"classifier_{name}"):
            mlflow.log_param("model_type", name)

            # Logistic regression benefits from scaled features; tree models don't need it but it doesn't hurt
            if name == "logistic_regression":
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                probs = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)

            acc = accuracy_score(y_test, preds)
            ll = log_loss(y_test, probs)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("log_loss", ll)
            log_model_safely(model, name)

            results.append({"model": name, "accuracy": acc, "log_loss": ll})

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = name

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    print("\n=== Classifier Comparison ===")
    print(results_df.to_string(index=False))
    print(f"\nBest classifier: {best_name} (accuracy={best_acc:.3f})")

    joblib.dump(best_model, f"{MODELS_DIR}/outcome_classifier.pkl")
    joblib.dump(le, f"{MODELS_DIR}/label_encoder.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/feature_scaler.pkl")
    joblib.dump(best_name, f"{MODELS_DIR}/best_classifier_name.pkl")

    return results_df, best_name


# ---------------- Score Regressors ----------------

def get_regressors():
    return {
        "poisson_linear": PoissonRegressor(max_iter=500),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
        "xgboost_poisson": XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            objective="count:poisson", random_state=42,
        ),
    }


def train_all_regressors(df, target_col, label):
    X = df[FEATURE_COLS]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    best_model, best_name, best_mae = None, None, np.inf

    for name, model in get_regressors().items():
        with mlflow.start_run(run_name=f"{label}_{name}"):
            mlflow.log_param("model_type", name)
            mlflow.log_param("target", label)

            if name == "poisson_linear":
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            mlflow.log_metric("mae", mae)
            log_model_safely(model, name)

            results.append({"model": name, "mae": mae})

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_name = name

    results_df = pd.DataFrame(results).sort_values("mae")
    print(f"\n=== {label} Regressor Comparison ===")
    print(results_df.to_string(index=False))
    print(f"\nBest {label} model: {best_name} (MAE={best_mae:.3f})")

    joblib.dump(best_model, f"{MODELS_DIR}/{label}_regressor.pkl")
    joblib.dump(best_name, f"{MODELS_DIR}/{label}_best_name.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/{label}_scaler.pkl")

    return results_df, best_name


if __name__ == "__main__":
    df = pd.read_csv(f"{PROCESSED_DIR}/training_data.csv")

    print("Training and comparing outcome classifiers...")
    clf_results, best_clf = train_all_classifiers(df)

    print("\nTraining and comparing score_a regressors...")
    reg_a_results, best_reg_a = train_all_regressors(df, "score_a", "score_a")

    print("\nTraining and comparing score_b regressors...")
    reg_b_results, best_reg_b = train_all_regressors(df, "score_b", "score_b")

    print("\n✅ Best models saved to models/ and every run logged to mlruns/")