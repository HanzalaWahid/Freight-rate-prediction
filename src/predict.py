"""
src/predict.py
==============
Generate final predictions:
  1. Load the best model + encoders from artifacts/
  2. Predict validation.csv -> validation_predictions.csv
  3. Predict december_chart_inputs.csv -> fill predicted_rate column
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import (
    VALIDATION_PATH, VALIDATION_TEMPLATE_PATH,
    DECEMBER_INPUTS_PATH, VALIDATION_PREDICTIONS_PATH,
    ARTIFACTS_DIR,
)
from src.data_cleaning import load_and_clean_validation, load_december, apply_imputer
from src.features import build_features, FEATURE_COLS
from src.models import load_artifacts


def make_predictions():
    meta_path = ARTIFACTS_DIR / "cv_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run train_final.py first.")

    with open(meta_path) as f:
        cv_meta = json.load(f)
    model_name = cv_meta["best_model"]
    feature_cols = cv_meta.get("feature_cols", FEATURE_COLS)

    print(f"Loading final {model_name} model from artifacts/...")
    model, city_enc, pair_enc, coord_lk, imp_stats = load_artifacts(model_name, ARTIFACTS_DIR)

    # -- Validation predictions ------------------------------------------------
    print("Predicting validation.csv...")
    val_df = load_and_clean_validation()
    val_df = apply_imputer(val_df, imp_stats)
    # coord_lookup fills lat/lon if missing (validation has them, but use lookup for consistency)
    X_val, _, _, _ = build_features(
        val_df, city_encoder=city_enc, pair_encoder=pair_enc,
        coord_lookup=coord_lk, fit=False, feature_cols=feature_cols,
    )

    val_preds = np.clip(model.predict(X_val), 1.0, None)

    template = pd.read_csv(VALIDATION_TEMPLATE_PATH)
    assert list(template.columns) == ["load_id", "predicted_rate"]
    assert len(template) == len(val_df), "Template and validation row count mismatch!"

    out = pd.DataFrame({
        "load_id": val_df["load_id"].values,
        "predicted_rate": val_preds,
    })

    # Reorder to match template load_id order
    out = out.set_index("load_id").reindex(template["load_id"]).reset_index()
    out.columns = ["load_id", "predicted_rate"]

    assert out["predicted_rate"].isna().sum() == 0, "NaN predictions found!"
    assert (out["predicted_rate"] > 0).all(), "Non-positive predictions found!"

    out.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    print(f"Saved {len(out):,} predictions -> {VALIDATION_PREDICTIONS_PATH}")

    # -- December chart predictions --------------------------------------------
    print("Predicting december_chart_inputs.csv...")
    dec_df = load_december()

    # Use coord_lookup to derive lat/lon from city names (no hardcoding)
    dec_work = coord_lk.add_coords(dec_df.copy())
    dec_work = apply_imputer(dec_work, imp_stats)

    X_dec, _, _, _ = build_features(
        dec_work, city_encoder=city_enc, pair_encoder=pair_enc,
        coord_lookup=coord_lk, fit=False, feature_cols=feature_cols,
    )
    dec_preds = np.clip(model.predict(X_dec), 1.0, None)

    assert (dec_preds > 0).all(), "December predictions must be positive!"
    assert len(dec_preds) == 31, f"Expected 31 December rows, got {len(dec_preds)}"

    dec_out = dec_df.copy()
    dec_out["predicted_rate"] = dec_preds
    dec_out.to_csv(DECEMBER_INPUTS_PATH, index=False)
    print(f"December predictions written -> {DECEMBER_INPUTS_PATH}")
    print(f"December rate range: ${dec_preds.min():.2f} - ${dec_preds.max():.2f}")

    return out, dec_out


if __name__ == "__main__":
    make_predictions()
