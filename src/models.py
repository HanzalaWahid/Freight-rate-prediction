"""
src/models.py
=============
Model definitions, training, and artifact persistence.

Models:
  - Baseline: Median predictor
  - LightGBM
  - XGBoost
  - CatBoost (winner)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import catboost as cb

from src.config import RANDOM_STATE
from src.validation import evaluate, get_time_folds
from src.data_cleaning import fit_imputer, apply_imputer
from src.features import build_features, FEATURE_COLS

optuna.logging.set_verbosity(optuna.logging.WARNING)


# -- Baseline ------------------------------------------------------------------
class MedianBaseline:
    """Predicts the training set median for every sample."""

    def __init__(self):
        self.median_: float = 0.0

    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


# -- LightGBM -----------------------------------------------------------------
DEFAULT_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "random_state": RANDOM_STATE,
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}


def train_lgbm(X_train, y_train, X_val=None, y_val=None, params: dict | None = None):
    p = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    n_est = p.pop("n_estimators", 2000)
    callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
    eval_set = [(X_val, y_val)] if X_val is not None else None
    model = lgb.LGBMRegressor(**p, n_estimators=n_est)
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        callbacks=callbacks if eval_set else [lgb.log_evaluation(-1)],
    )
    return model


# -- XGBoost -------------------------------------------------------------------
DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
    "random_state": RANDOM_STATE,
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
}


def train_xgb(X_train, y_train, X_val=None, y_val=None, params: dict | None = None):
    p = {**DEFAULT_XGB_PARAMS, **(params or {})}
    model = xgb.XGBRegressor(**p, early_stopping_rounds=100 if X_val is not None else None)
    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


# -- CatBoost ------------------------------------------------------------------
DEFAULT_CB_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": RANDOM_STATE,
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "subsample": 0.8,
    "verbose": 0,
}


def train_catboost(X_train, y_train, X_val=None, y_val=None, params: dict | None = None):
    p = {**DEFAULT_CB_PARAMS, **(params or {})}
    early = p.pop("early_stopping_rounds", 100)
    model = cb.CatBoostRegressor(**p)
    eval_set = cb.Pool(X_val, y_val) if X_val is not None else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=early if eval_set else None,
    )
    return model


# -- Full CV loop --------------------------------------------------------------
def cross_validate_model(
    train_df: pd.DataFrame,
    model_name: str = "lgbm",
    n_folds: int = 3,
    feature_cols: list[str] | None = None,
) -> tuple[list[dict], np.ndarray]:
    """
    Run time-based CV for the given model.

    Returns
    -------
    fold_metrics : list of dicts (one per fold)
    oof_preds    : out-of-fold predictions array
    """
    train_df = train_df.sort_values("date").reset_index(drop=True)
    folds = get_time_folds(train_df, n_folds=n_folds)

    oof_preds = np.full(len(train_df), np.nan)
    fold_metrics: list[dict] = []

    for fold_num, (tr_idx, val_idx) in enumerate(folds, 1):
        print(f"\n--- Fold {fold_num} ---")
        tr_fold = train_df.iloc[tr_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()

        y_tr = tr_fold["posted_rate"].values
        y_val = val_fold["posted_rate"].values

        # Fit imputer on training fold only
        imp_stats = fit_imputer(tr_fold)
        tr_fold = apply_imputer(tr_fold, imp_stats)
        val_fold = apply_imputer(val_fold, imp_stats)

        # Build features (fit encoders on training fold only)
        X_tr, city_enc, pair_enc, coord_lk = build_features(
            tr_fold, target=tr_fold["posted_rate"], fit=True,
            feature_cols=feature_cols,
        )
        X_val, _, _, _ = build_features(
            val_fold, city_encoder=city_enc, pair_encoder=pair_enc,
            coord_lookup=coord_lk, fit=False, feature_cols=feature_cols,
        )

        # Train model
        if model_name == "baseline":
            model = MedianBaseline().fit(X_tr, y_tr)
        elif model_name == "lgbm":
            model = train_lgbm(X_tr, y_tr, X_val, y_val)
        elif model_name == "xgb":
            model = train_xgb(X_tr, y_tr, X_val, y_val)
        elif model_name == "catboost":
            model = train_catboost(X_tr, y_tr, X_val, y_val)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        preds = model.predict(X_val)
        preds = np.clip(preds, 1.0, None)
        oof_preds[val_idx] = preds
        metrics = evaluate(y_val, preds, label=f"{model_name} fold {fold_num}")
        fold_metrics.append(metrics)

    return fold_metrics, oof_preds


# -- Final model training ------------------------------------------------------
def train_final_model(
    train_df: pd.DataFrame,
    best_params: dict | None = None,
    model_name: str = "lgbm",
    feature_cols: list[str] | None = None,
) -> tuple:
    """
    Train on ALL training data.
    Returns (model, city_encoder, pair_encoder, coord_lookup, imp_stats).
    """
    train_df = train_df.sort_values("date").reset_index(drop=True)

    imp_stats = fit_imputer(train_df)
    train_df = apply_imputer(train_df, imp_stats)

    y_train = train_df["posted_rate"].values
    X_train, city_enc, pair_enc, coord_lk = build_features(
        train_df, target=train_df["posted_rate"], fit=True,
        feature_cols=feature_cols,
    )

    print(f"\nTraining final {model_name} on {len(train_df):,} rows ({X_train.shape[1]} features)...")
    if model_name == "lgbm":
        params = {**(best_params or DEFAULT_LGBM_PARAMS)}
        params["n_estimators"] = params.get("n_estimators", 2000)
        model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            verbosity=-1,
            random_state=RANDOM_STATE,
            **{k: v for k, v in params.items()
               if k not in ("objective", "metric", "verbosity", "random_state")},
        )
        model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(-1)])
    elif model_name == "xgb":
        params = {**DEFAULT_XGB_PARAMS, **(best_params or {})}
        params.pop("early_stopping_rounds", None)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
    elif model_name == "catboost":
        params = {**DEFAULT_CB_PARAMS, **(best_params or {})}
        model = cb.CatBoostRegressor(**params)
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, city_enc, pair_enc, coord_lk, imp_stats


def save_artifacts(model, city_enc, pair_enc, coord_lk, imp_stats,
                   model_name: str, artifacts_dir: Path):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / f"{model_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(artifacts_dir / "city_encoder.pkl", "wb") as f:
        pickle.dump(city_enc, f)
    with open(artifacts_dir / "pair_encoder.pkl", "wb") as f:
        pickle.dump(pair_enc, f)
    with open(artifacts_dir / "coord_lookup.pkl", "wb") as f:
        pickle.dump(coord_lk, f)
    with open(artifacts_dir / "imp_stats.pkl", "wb") as f:
        pickle.dump(imp_stats, f)
    print(f"Artifacts saved to {artifacts_dir}/")


def load_artifacts(model_name: str, artifacts_dir: Path) -> tuple:
    with open(artifacts_dir / f"{model_name}_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(artifacts_dir / "city_encoder.pkl", "rb") as f:
        city_enc = pickle.load(f)
    with open(artifacts_dir / "pair_encoder.pkl", "rb") as f:
        pair_enc = pickle.load(f)
    with open(artifacts_dir / "coord_lookup.pkl", "rb") as f:
        coord_lk = pickle.load(f)
    with open(artifacts_dir / "imp_stats.pkl", "rb") as f:
        imp_stats = pickle.load(f)
    return model, city_enc, pair_enc, coord_lk, imp_stats
