"""
src/train_cv.py
===============
Step-by-step training script:
  1. Load and clean training data
  2. Run time-based CV for baseline, LightGBM, XGBoost, CatBoost
  3. Print comparison table and pick best model
  4. Run Optuna tuning on best model
  5. Save CV results to artifacts/cv_results.csv
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
import optuna

from src.config import RANDOM_STATE, BASE_DIR, ARTIFACTS_DIR
from src.data_cleaning import load_and_clean_train, fit_imputer, apply_imputer
from src.models import cross_validate_model, train_catboost
from src.features import build_features
from src.validation import evaluate

optuna.logging.set_verbosity(optuna.logging.WARNING)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def mean_metrics(fold_metrics: list[dict]) -> dict:
    keys = ["MAE", "RMSE", "MAPE", "R2"]
    return {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}


def run_cv():
    print("=" * 60)
    print("FREIGHT RATE PREDICTION -- Cross-Validation")
    print("=" * 60)

    train_df = load_and_clean_train()
    print(f"Loaded train_test.csv: {train_df.shape}")

    results: dict[str, dict] = {}

    # 1: Baseline
    print("\n[1/4] Baseline (median predictor)")
    fm, _ = cross_validate_model(train_df, model_name="baseline", n_folds=3)
    results["baseline"] = mean_metrics(fm)

    # 2: LightGBM
    print("\n[2/4] LightGBM (default params)")
    fm, _ = cross_validate_model(train_df, model_name="lgbm", n_folds=3)
    results["lgbm"] = mean_metrics(fm)

    # 3: XGBoost
    print("\n[3/4] XGBoost (default params)")
    fm, _ = cross_validate_model(train_df, model_name="xgb", n_folds=3)
    results["xgb"] = mean_metrics(fm)

    # 4: CatBoost
    print("\n[4/4] CatBoost (default params)")
    fm, _ = cross_validate_model(train_df, model_name="catboost", n_folds=3)
    results["catboost"] = mean_metrics(fm)

    # Comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (mean across 3 CV folds)")
    print("=" * 60)
    df_res = pd.DataFrame(results).T.sort_values("RMSE")
    print(df_res.to_string(float_format="{:.2f}".format))

    best_model_name = df_res.index[0]
    print(f"\n[OK] Best model: {best_model_name}  (lowest CV RMSE={df_res.loc[best_model_name,'RMSE']:.2f})")

    # Optuna tuning on winner (skip if cv_meta.json already exists)
    meta_path = ARTIFACTS_DIR / "cv_meta.json"
    best_params: dict = {}
    if meta_path.exists():
        print(f"\n[5] {meta_path.name} already exists -- loading tuned params (skipping Optuna tuning)...")
        with open(meta_path) as f:
            existing_meta = json.load(f)
        best_params = existing_meta.get("best_params", {})
    elif best_model_name == "catboost":
        print("\n[5] Optuna tuning for catboost (30 trials)...")
        train_df_sorted = train_df.sort_values("date").reset_index(drop=True)
        split_idx = int(len(train_df_sorted) * 0.85)
        tune_tr = train_df_sorted.iloc[:split_idx].copy()
        tune_val = train_df_sorted.iloc[split_idx:].copy()

        imp_stats = fit_imputer(tune_tr)
        tune_tr = apply_imputer(tune_tr, imp_stats)
        tune_val = apply_imputer(tune_val, imp_stats)

        X_tr, city_enc, pair_enc, coord_lk = build_features(
            tune_tr, target=tune_tr["posted_rate"], fit=True
        )
        X_val, _, _, _ = build_features(
            tune_val, city_encoder=city_enc, pair_encoder=pair_enc,
            coord_lookup=coord_lk, fit=False
        )
        y_tr = tune_tr["posted_rate"].values
        y_val = tune_val["posted_rate"].values

        def cb_objective(trial):
            p = {
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": RANDOM_STATE,
                "verbose": 0,
                "iterations": trial.suggest_int("iterations", 500, 3000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            }
            model = cb.CatBoostRegressor(**p)
            model.fit(
                X_tr, y_tr,
                eval_set=cb.Pool(X_val, y_val),
                early_stopping_rounds=50,
            )
            preds = np.clip(model.predict(X_val), 1.0, None)
            return float(np.sqrt(np.mean((preds - y_val) ** 2)))

        study = optuna.create_study(direction="minimize")
        study.optimize(cb_objective, n_trials=30, show_progress_bar=False)
        best_params = study.best_params
        print(f"  Best RMSE (Optuna): {study.best_value:.2f}  params: {best_params}")
        tuned_model = train_catboost(X_tr, y_tr, X_val, y_val, params=best_params)
        tuned_preds = np.clip(tuned_model.predict(X_val), 1.0, None)
        evaluate(y_val, tuned_preds, label="catboost (tuned, holdout 15%)")

    # Save results
    df_res.to_csv(ARTIFACTS_DIR / "cv_results.csv")

    cv_meta = {
        "best_model": best_model_name,
        "best_params": best_params,
        "cv_metrics": results,
    }
    with open(ARTIFACTS_DIR / "cv_meta.json", "w") as f:
        json.dump(cv_meta, f, indent=2)

    print(f"\nSaved CV results to {ARTIFACTS_DIR}/cv_results.csv")
    print(f"Saved CV meta to {ARTIFACTS_DIR}/cv_meta.json")
    return best_model_name, best_params, results


if __name__ == "__main__":
    run_cv()
