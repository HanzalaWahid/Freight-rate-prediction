"""
train.py
========
Training pipeline only:
  1. Cross-validation on train_test.csv (saves cv_results.csv, cv_meta.json, oof_eval.npz)
  2. Final model retraining on all 48,000 rows (saves model, encoders, coord_lookup, imputer to artifacts/)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS_DIR
from src.train_cv import run_cv
from src.train_final import run_final_training


def _ensure_oof_eval():
    oof_path = ARTIFACTS_DIR / "oof_eval.npz"
    if not oof_path.exists():
        import json
        from src.config import ARTIFACTS_DIR
        from src.data_cleaning import load_and_clean_train, fit_imputer, apply_imputer
        from src.features import build_features
        from src.validation import get_time_folds
        from src.models import train_catboost

        meta_path = ARTIFACTS_DIR / "cv_meta.json"
        best_params = {}
        feature_cols = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            best_params = meta.get("best_params", {})
            feature_cols = meta.get("feature_cols")

        train_df = load_and_clean_train().sort_values("date").reset_index(drop=True)
        folds = get_time_folds(train_df, n_folds=3)

        y_trues, y_preds = [], []
        for fold_num, (tr_idx, val_idx) in enumerate(folds, 1):
            tr_fold = train_df.iloc[tr_idx].copy()
            val_fold = train_df.iloc[val_idx].copy()
            imp_stats = fit_imputer(tr_fold)
            tr_fold = apply_imputer(tr_fold, imp_stats)
            val_fold = apply_imputer(val_fold, imp_stats)
            X_tr, city_enc, pair_enc, coord_lk = build_features(
                tr_fold, target=tr_fold["posted_rate"], fit=True, feature_cols=feature_cols
            )
            X_val, _, _, _ = build_features(
                val_fold, city_encoder=city_enc, pair_encoder=pair_enc,
                coord_lookup=coord_lk, fit=False, feature_cols=feature_cols
            )
            y_tr = tr_fold["posted_rate"].values
            y_val = val_fold["posted_rate"].values
            model = train_catboost(X_tr, y_tr, X_val, y_val, params=best_params)
            preds = np.clip(model.predict(X_val), 1.0, None)
            y_trues.append(y_val)
            y_preds.append(preds)

        y_true = np.concatenate(y_trues)
        y_pred = np.concatenate(y_preds)
        np.savez(oof_path, y_true=y_true, y_pred=y_pred)
        print(f"Saved {oof_path}")


def run_training(skip_cv: bool = False):
    can_skip = skip_cv and (ARTIFACTS_DIR / "cv_results.csv").exists() and (ARTIFACTS_DIR / "cv_meta.json").exists()
    if can_skip:
        print("\n" + "=" * 60)
        print("STEP: Cross-Validation (SKIPPED via --skip-cv)")
        print("=" * 60)
        _ensure_oof_eval()
    else:
        print("\n" + "=" * 60)
        print("STEP: Cross-Validation + Optuna Tuning")
        print("=" * 60)
        run_cv()
        _ensure_oof_eval()

    print("\n" + "=" * 60)
    print("STEP: Final Model Training (all train data)")
    print("=" * 60)
    run_final_training()
    print(f"\n[OK] Training pipeline complete. Artifacts saved to {ARTIFACTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Freight Rate Prediction Training Pipeline")
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation if cv_results.csv and cv_meta.json exist",
    )
    args = parser.parse_args()
    run_training(skip_cv=args.skip_cv)


if __name__ == "__main__":
    main()
