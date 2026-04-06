"""
Audit Risk Analytics — Feature Engineering
Create audit-relevant features from transaction data.
"""
import logging

import numpy as np
import pandas as pd

from src.config import (
    FEATURES_PARQUET,
    PCA_FEATURES,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


def add_rolling_features(df: pd.DataFrame, windows: list[int] = [5, 10, 50]) -> pd.DataFrame:
    """Add rolling statistics on Amount to detect sudden spikes."""
    df = df.sort_values("Time").reset_index(drop=True)

    for w in windows:
        df[f"amount_rolling_mean_{w}"] = df["Amount"].rolling(window=w, min_periods=1).mean()
        df[f"amount_rolling_std_{w}"] = df["Amount"].rolling(window=w, min_periods=1).std().fillna(0)
        # Ratio of current amount to rolling mean — spikes show high ratios
        df[f"amount_ratio_{w}"] = df["Amount"] / df[f"amount_rolling_mean_{w}"].replace(0, 1)

    logger.info(f"✅ Added rolling features for windows {windows}")
    return df


def add_top_pca_interactions(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Add interaction features from the most important PCA components."""
    top_components = PCA_FEATURES[:top_n]

    for i in range(len(top_components)):
        for j in range(i + 1, len(top_components)):
            col_name = f"{top_components[i]}_x_{top_components[j]}"
            df[col_name] = df[top_components[i]] * df[top_components[j]]

    logger.info(f"✅ Added {top_n} PCA interaction features")
    return df


def add_percentile_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentile rank for Amount — useful for materiality assessment."""
    df["amount_percentile"] = df["Amount"].rank(pct=True) * 100

    # Flag top 1% as high-value transactions
    df["is_high_value"] = df["amount_percentile"] >= 99

    logger.info("✅ Added percentile rank features")
    return df


def add_time_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encoding (useful for ML models)."""
    # Cyclical encoding of hour
    df["hour_sin"] = np.sin(2 * np.pi * df["time_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time_hour"] / 24)

    logger.info("✅ Added cyclical time features")
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps and save."""
    logger.info("Starting feature engineering…")
    df = add_rolling_features(df)
    df = add_top_pca_interactions(df)
    df = add_percentile_ranks(df)
    df = add_time_derived_features(df)

    # Save enriched dataset
    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PARQUET, index=False, engine="pyarrow")
    logger.info(f"✅ Feature-enriched data saved → {FEATURES_PARQUET}")
    logger.info(f"   Total features: {len(df.columns)}")

    return df
