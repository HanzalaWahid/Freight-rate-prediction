"""
src/data_cleaning.py
====================
Load raw CSV files, apply minimal cleaning (type casting, null imputation),
and return clean DataFrames.  All imputation statistics (medians) are fitted
on the training set ONLY and applied to any other set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TRAIN_TEST_PATH, VALIDATION_PATH, DECEMBER_INPUTS_PATH, TARGET_COL


# ── Constants ────────────────────────────────────────────────────────────────
NUMERIC_COLS = ["distance", "weight", "market_index", "quote_signal",
                "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


# ── Public API ────────────────────────────────────────────────────────────────
def load_and_clean_train() -> pd.DataFrame:
    """Load train_test.csv, parse dates, and return a cleaned DataFrame."""
    df = pd.read_csv(TRAIN_TEST_PATH)
    df = _common_clean(df)
    return df


def load_and_clean_validation() -> pd.DataFrame:
    """Load validation.csv (no target column)."""
    df = pd.read_csv(VALIDATION_PATH)
    df = _common_clean(df)
    return df


def load_december() -> pd.DataFrame:
    """Load december_chart_inputs.csv (no target column yet)."""
    df = pd.read_csv(DECEMBER_INPUTS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fit_imputer(train_df: pd.DataFrame) -> dict:
    """
    Compute imputation statistics from the training set.
    Returns a dict with medians for numeric cols that may have nulls.
    """
    stats: dict = {}
    for col in ["weight", "market_index"]:
        if col in train_df.columns:
            stats[f"{col}_median"] = float(train_df[col].median())
    return stats


def apply_imputer(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Fill nulls using pre-computed training statistics."""
    df = df.copy()
    if "weight" in df.columns:
        df["weight"] = df["weight"].fillna(stats.get("weight_median", df["weight"].median()))
    if "market_index" in df.columns:
        df["market_index"] = df["market_index"].fillna(stats.get("market_index_median", df["market_index"].median()))
    return df


# ── Internal helpers ──────────────────────────────────────────────────────────
def _common_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast and lightly clean any raw freight DataFrame."""
    df = df.copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])

    # Strip whitespace from string columns
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # Ensure numeric columns are float
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix negative weights (data entry errors: 292 in train, 145 in val)
    if "weight" in df.columns:
        df["weight"] = df["weight"].abs()

    return df
