"""
SmartWaste ML Training Pipeline
================================
Trains a Random Forest + Gradient Boosting ensemble to predict
hours_until_full from sensor readings.

Usage:
    python train_model.py               # trains on synthetic data
    python train_model.py --csv data.csv  # trains on real collected data

Output:
    waste_model.pkl  — loaded by app.py at runtime
"""

import argparse, joblib, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ── Synthetic data generator (used when no real CSV is available) ───────────
def generate_synthetic_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    fill_pct  = rng.uniform(0, 100, n)
    gas_ppm   = rng.uniform(80, 600, n)
    temp_c    = rng.uniform(20, 35, n)
    weight_kg = rng.uniform(0, 20, n)

    # Ground-truth: faster fill when gas high, temp high, already near-full
    base_rate  = 3.0 + (gas_ppm / 600) * 3.0 + (temp_c - 20) / 5
    noise      = rng.normal(0, 0.5, n)
    hours_left = np.clip((100 - fill_pct) / (base_rate + noise), 0, 72)

    return pd.DataFrame({
        "fill_pct":      fill_pct,
        "gas_ppm":       gas_ppm,
        "temp_c":        temp_c,
        "weight_kg":     weight_kg,
        "hours_to_full": hours_left,
    })


# ── Feature engineering ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fill_gas_interaction"] = df["fill_pct"] * df["gas_ppm"] / 1000
    df["remaining_pct"]        = 100 - df["fill_pct"]
    df["temp_norm"]            = (df["temp_c"] - 20) / 15
    return df

FEATURES = ["fill_pct", "gas_ppm", "temp_c", "weight_kg",
            "fill_gas_interaction", "remaining_pct", "temp_norm"]


def train(df: pd.DataFrame):
    df = engineer_features(df)
    X  = df[FEATURES].values
    y  = df["hours_to_full"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # ── Model 1: Random Forest ──────────────────────────────────────────────
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=4, random_state=42, n_jobs=-1))
    ])
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mae   = mean_absolute_error(y_test, rf_preds)
    rf_r2    = r2_score(y_test, rf_preds)
    print(f"[RF]  MAE={rf_mae:.2f}h  R²={rf_r2:.4f}")

    # ── Model 2: Gradient Boosting ──────────────────────────────────────────
    gb = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42))
    ])
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_mae   = mean_absolute_error(y_test, gb_preds)
    gb_r2    = r2_score(y_test, gb_preds)
    print(f"[GB]  MAE={gb_mae:.2f}h  R²={gb_r2:.4f}")

    # Pick best
    best = rf if rf_mae <= gb_mae else gb
    best_name = "RandomForest" if rf_mae <= gb_mae else "GradientBoosting"
    print(f"\n✔ Selected: {best_name}")

    # Cross-val on best
    cv_scores = cross_val_score(best, X, y, cv=5, scoring="neg_mean_absolute_error")
    print(f"  5-fold CV MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} h")

    # Save
    joblib.dump(best, "waste_model.pkl")
    print("  Saved → waste_model.pkl")

    # Feature importance (RF only)
    if best_name == "RandomForest":
        importances = best.named_steps["model"].feature_importances_
        for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
            print(f"  {feat:<28} {imp:.4f}")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to real sensor CSV")
    args = parser.parse_args()

    if args.csv:
        print(f"Loading real data from {args.csv}")
        df = pd.read_csv(args.csv)
        required = {"fill_pct", "gas_ppm", "temp_c", "weight_kg", "hours_to_full"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
    else:
        print("No CSV provided — generating synthetic training data (n=3000)")
        df = generate_synthetic_data()

    print(f"Dataset: {len(df)} rows\n")
    train(df)
