"""
src/validation.py
=================
Time-based cross-validation strategy.

Train data spans Jan–Oct 2025; validation data spans Nov–Dec 2025.
We use TimeSeriesSplit (expanding window) with 3 folds to assess
model performance without leakage.  Each fold's validation window
is a future period relative to its training window.

Fold structure (roughly 2-month increments):
  Fold 1: Train Jan–Apr | Val May–Jun
  Fold 2: Train Jan–Jun | Val Jul–Aug
  Fold 3: Train Jan–Aug | Val Sep–Oct
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute MAE, RMSE, MAPE, R² and print them."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    r2 = r2_score(y_true, y_pred)
    result = {"label": label, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
    tag = f"[{label}] " if label else ""
    print(f"{tag}MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R2={r2:.4f}")
    return result


# ── Time-based split ──────────────────────────────────────────────────────────
def get_time_folds(df: pd.DataFrame, n_folds: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate (train_idx, val_idx) pairs for an expanding-window time split.
    The training set always starts from the beginning; each fold extends
    the window by one step.
    
    df must have a 'date' column (datetime).
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    dates = df["date"].sort_values().unique()
    n_dates = len(dates)

    # Divide dates into (n_folds + 1) chunks; use first n_folds chunks as
    # expanding training windows and the next chunk as validation each time.
    chunk_size = n_dates // (n_folds + 1)
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fold in range(n_folds):
        # Training: dates 0 … (fold+1)*chunk_size - 1
        train_cutoff = dates[(fold + 1) * chunk_size - 1]
        # Validation: next chunk
        val_start = dates[(fold + 1) * chunk_size]
        val_end_idx = min((fold + 2) * chunk_size - 1, n_dates - 1)
        val_end = dates[val_end_idx]

        train_idx = df.index[df["date"] <= train_cutoff].to_numpy()
        val_idx = df.index[(df["date"] >= val_start) & (df["date"] <= val_end)].to_numpy()

        if len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
        print(f"  Fold {fold+1}: train end={train_cutoff.date()}  "
              f"val {val_start.date()} to {val_end.date()}  "
              f"(train={len(train_idx):,}, val={len(val_idx):,})")

    return folds
