"""
src/features.py
===============
Feature engineering for the freight-rate prediction task.

All transformations that depend on training data statistics
(target-encoded city pairs, label-encoding ordinals) are returned
as callable objects fitted on training folds only.

Feature groups:
  1. Temporal           - day-of-week, month, week-of-year, is_weekend, quarter, holidays
  2. Route              - haversine distance, bearing, log_distance
  3. Equipment          - ordinal map  (Dry Van < Flatbed < Reefer)
  4. Market signals     - market_index, quote_signal (only when available)
  5. Weight             - log1p transform
  6. City-pair target   - mean posted_rate per (pickup, delivery) pair
                          from training data, smoothed with global mean
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# -- Constants -----------------------------------------------------------------
EQUIPMENT_ORDINAL = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}

# US holidays relevant to freight (2025 dates; extend if needed)
US_HOLIDAYS_2025 = {
    pd.Timestamp("2025-01-01"): "new_year",
    pd.Timestamp("2025-01-20"): "mlk_day",
    pd.Timestamp("2025-02-17"): "presidents_day",
    pd.Timestamp("2025-05-26"): "memorial_day",
    pd.Timestamp("2025-07-04"): "july_4th",
    pd.Timestamp("2025-09-01"): "labor_day",
    pd.Timestamp("2025-10-13"): "columbus_day",
    pd.Timestamp("2025-11-27"): "thanksgiving",
    pd.Timestamp("2025-12-25"): "christmas",
    pd.Timestamp("2025-12-31"): "new_years_eve",
}

# Holiday week windows (the week containing each holiday tends to affect rates)
_HOLIDAY_WEEKS = set()
for hdate in US_HOLIDAYS_2025:
    for offset in range(-2, 3):  # 2 days before/after
        _HOLIDAY_WEEKS.add(hdate + pd.Timedelta(days=offset))


# -- Temporal features ---------------------------------------------------------
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived columns. Requires 'date' column as datetime."""
    df = df.copy()
    d = df["date"]
    df["day_of_week"] = d.dt.dayofweek          # Mon=0 ... Sun=6
    df["month"] = d.dt.month
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["day_of_month"] = d.dt.day
    df["day_of_year"] = d.dt.dayofyear
    # Cyclic encoding for day-of-week and month to capture periodicity
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    # Holiday features
    df["is_holiday"] = d.isin(US_HOLIDAYS_2025).astype(int)
    df["is_holiday_week"] = d.isin(_HOLIDAY_WEEKS).astype(int)
    # Days to nearest holiday
    holiday_dates = sorted(US_HOLIDAYS_2025.keys())
    df["days_to_holiday"] = d.apply(
        lambda x: min(abs((x - h).days) for h in holiday_dates)
    ).clip(upper=30)  # cap at 30

    return df


# -- City coordinate lookup ----------------------------------------------------
class CityCoordLookup(BaseEstimator, TransformerMixin):
    """
    Build a city -> (lat, lon) lookup from training data.
    Assigns median coords per city. Unknown cities get the global median.
    """

    def __init__(self):
        self.city_coords_: dict = {}
        self.global_lat_: float = 0.0
        self.global_lon_: float = 0.0

    def fit(self, df: pd.DataFrame, y=None) -> "CityCoordLookup":
        # Build from both pickup and delivery columns
        pickup = df.groupby("pickup")[["pickup_lat", "pickup_lon"]].median()
        delivery = df.groupby("delivery")[["delivery_lat", "delivery_lon"]].median()

        for city in pickup.index:
            self.city_coords_[city] = (
                float(pickup.loc[city, "pickup_lat"]),
                float(pickup.loc[city, "pickup_lon"]),
            )
        for city in delivery.index:
            if city not in self.city_coords_:
                self.city_coords_[city] = (
                    float(delivery.loc[city, "delivery_lat"]),
                    float(delivery.loc[city, "delivery_lon"]),
                )

        all_lats = [v[0] for v in self.city_coords_.values()]
        all_lons = [v[1] for v in self.city_coords_.values()]
        self.global_lat_ = float(np.median(all_lats))
        self.global_lon_ = float(np.median(all_lons))
        return self

    def get_coords(self, city: str) -> tuple[float, float]:
        return self.city_coords_.get(city, (self.global_lat_, self.global_lon_))

    def add_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pickup_lat/lon, delivery_lat/lon from lookup if missing."""
        df = df.copy()
        if "pickup_lat" not in df.columns or df["pickup_lat"].isna().all():
            df["pickup_lat"] = df["pickup"].map(lambda c: self.get_coords(c)[0])
            df["pickup_lon"] = df["pickup"].map(lambda c: self.get_coords(c)[1])
        if "delivery_lat" not in df.columns or df["delivery_lat"].isna().all():
            df["delivery_lat"] = df["delivery"].map(lambda c: self.get_coords(c)[0])
            df["delivery_lon"] = df["delivery"].map(lambda c: self.get_coords(c)[1])
        return df


# -- Geographic features -------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _bearing(lat1, lon1, lat2, lon2):
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(np.radians(lat2))
    y = (np.cos(np.radians(lat1)) * np.sin(np.radians(lat2))
         - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(dlon))
    return np.degrees(np.arctan2(x, y)) % 360


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add haversine distance and bearing between pickup and delivery."""
    df = df.copy()
    df["haversine_km"] = _haversine_km(
        df["pickup_lat"], df["pickup_lon"],
        df["delivery_lat"], df["delivery_lon"],
    )
    df["bearing"] = _bearing(
        df["pickup_lat"], df["pickup_lon"],
        df["delivery_lat"], df["delivery_lon"],
    )
    # Rate per mile proxy (distance already provided)
    df["log_distance"] = np.log1p(df["distance"].clip(lower=0))
    return df


# -- Equipment feature ---------------------------------------------------------
def add_equipment_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["equipment_ord"] = df["equipment"].map(EQUIPMENT_ORDINAL).fillna(-1).astype(int)
    # One-hot dummies
    for eq, code in EQUIPMENT_ORDINAL.items():
        df[f"equip_{eq.replace(' ', '_').lower()}"] = (df["equipment"] == eq).astype(int)
    return df


# -- Weight feature ------------------------------------------------------------
def add_weight_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_weight"] = np.log1p(df["weight"].clip(lower=0))
    return df


# -- City-pair target encoder --------------------------------------------------
class CityPairTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smooth mean target-encoding for (pickup, delivery) pairs.
    Fitted on training data only; unknown pairs get the global mean.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.global_mean_: float = 0.0
        self.pair_stats_: dict = {}

    def fit(self, df: pd.DataFrame, target: pd.Series) -> "CityPairTargetEncoder":
        self.global_mean_ = float(target.mean())
        pairs = df["pickup"].str.cat(df["delivery"], sep=" to ")
        grouped = target.groupby(pairs)
        counts = grouped.count()
        means = grouped.mean()
        # Smoothed estimate
        smooth = (counts * means + self.smoothing * self.global_mean_) / (counts + self.smoothing)
        self.pair_stats_ = smooth.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        pairs = df["pickup"].str.cat(df["delivery"], sep=" to ")
        df["pair_target_enc"] = pairs.map(self.pair_stats_).fillna(self.global_mean_)
        return df

    def fit_transform(self, df: pd.DataFrame, target: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        return self.fit(df, target).transform(df)


# -- City label encoder --------------------------------------------------------
class CityLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Encode pickup and delivery city names as integer codes.
    Unknown cities (in val/test but not in train) get code -1.
    """

    def __init__(self):
        self.pickup_map_: dict = {}
        self.delivery_map_: dict = {}

    def fit(self, df: pd.DataFrame, y=None) -> "CityLabelEncoder":
        pickup_cats = sorted(df["pickup"].dropna().unique())
        delivery_cats = sorted(df["delivery"].dropna().unique())
        self.pickup_map_ = {c: i for i, c in enumerate(pickup_cats)}
        self.delivery_map_ = {c: i for i, c in enumerate(delivery_cats)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["pickup_enc"] = df["pickup"].map(self.pickup_map_).fillna(-1).astype(int)
        df["delivery_enc"] = df["delivery"].map(self.delivery_map_).fillna(-1).astype(int)
        return df


# -- Feature column lists ------------------------------------------------------

# Full feature set (used when market_index and quote_signal are available)
FEATURE_COLS_FULL = [
    # Temporal
    "day_of_week", "month", "week_of_year", "quarter",
    "is_weekend", "day_of_month", "day_of_year",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_holiday", "is_holiday_week", "days_to_holiday",
    # Geo
    "haversine_km", "bearing", "log_distance", "distance",
    # Equipment
    "equipment_ord",
    "equip_dry_van", "equip_flatbed", "equip_reefer",
    # Weight
    "weight", "log_weight",
    # Market signals
    "market_index", "quote_signal",
    # City encodings
    "pickup_enc", "delivery_enc",
    # Target encoding
    "pair_target_enc",
]

# Portable feature set (no market_index / quote_signal -- works for December)
FEATURE_COLS_PORTABLE = [c for c in FEATURE_COLS_FULL
                         if c not in ("market_index", "quote_signal")]

# Default: portable (safe for all prediction targets)
FEATURE_COLS = FEATURE_COLS_PORTABLE


# -- Master feature builder ----------------------------------------------------
def build_features(
    df: pd.DataFrame,
    city_encoder: Optional[CityLabelEncoder] = None,
    pair_encoder: Optional[CityPairTargetEncoder] = None,
    coord_lookup: Optional[CityCoordLookup] = None,
    target: Optional[pd.Series] = None,
    fit: bool = False,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, CityLabelEncoder, CityPairTargetEncoder, CityCoordLookup]:
    """
    Apply all feature transformations.

    Parameters
    ----------
    df           : cleaned DataFrame
    city_encoder : pre-fitted CityLabelEncoder (or None if fit=True)
    pair_encoder : pre-fitted CityPairTargetEncoder (or None if fit=True)
    coord_lookup : pre-fitted CityCoordLookup (or None if fit=True)
    target       : target Series (required when fit=True)
    fit          : whether to fit encoders on df
    feature_cols : which feature columns to select (default: FEATURE_COLS)

    Returns
    -------
    (feature_df, city_encoder, pair_encoder, coord_lookup)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    df = df.copy()

    # Fit or apply coordinate lookup (needed for December)
    if fit:
        coord_lookup = CityCoordLookup().fit(df)
    df = coord_lookup.add_coords(df)

    df = add_temporal_features(df)
    df = add_geo_features(df)
    df = add_equipment_features(df)
    df = add_weight_features(df)

    if fit:
        city_encoder = CityLabelEncoder().fit(df)
        pair_encoder = CityPairTargetEncoder().fit(df, target)

    df = city_encoder.transform(df)
    df = pair_encoder.transform(df)

    # Select only the requested features (skip missing columns gracefully)
    available = [c for c in feature_cols if c in df.columns]
    return df[available], city_encoder, pair_encoder, coord_lookup
