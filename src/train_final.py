"""
src/train_final.py
==================
Retrain the best model on ALL of train_test.csv and save artifacts.
Reads best model name and tuned params from artifacts/cv_meta.json.
Never reruns Optuna. Uses fixed seeds.
"""

from __future__ import annotations

import json

from src.config import BASE_DIR, ARTIFACTS_DIR
from src.data_cleaning import load_and_clean_train
from src.features import FEATURE_COLS
from src.models import train_final_model, save_artifacts


def run_final_training():
    meta_path = ARTIFACTS_DIR / "cv_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run honest_tune.py first to create cv_meta.json.")

    with open(meta_path) as f:
        cv_meta = json.load(f)

    model_name = cv_meta["best_model"]
    best_params = cv_meta.get("best_params", {})
    feature_cols = cv_meta.get("feature_cols", FEATURE_COLS)
    print(f"Final training: model={model_name}  params={best_params}")
    print(f"Feature set: {len(feature_cols)} features")

    train_df = load_and_clean_train()
    model, city_enc, pair_enc, coord_lk, imp_stats = train_final_model(
        train_df, best_params=best_params, model_name=model_name,
        feature_cols=feature_cols,
    )
    save_artifacts(model, city_enc, pair_enc, coord_lk, imp_stats,
                   model_name, ARTIFACTS_DIR)

    # Also save which features were used
    cv_meta["feature_cols"] = feature_cols
    with open(meta_path, "w") as f:
        json.dump(cv_meta, f, indent=2)

    print("Final model training complete.")


if __name__ == "__main__":
    run_final_training()
