"""
Audit Risk Analytics — Risk Scoring Engine
Multi-factor risk scoring combining anomaly scores, amount, time, and velocity signals.
"""
import logging

import numpy as np
import pandas as pd

from src.config import RISK_LEVELS, TARGET_COL

logger = logging.getLogger(__name__)


# ── Individual Risk Components ─────────────────────────────
def _amount_risk(df: pd.DataFrame) -> np.ndarray:
    """Score based on transaction amount deviation (0-100)."""
    z = df["amount_zscore"].values
    # Sigmoid-like transformation: high z-scores → high risk
    risk = 100 / (1 + np.exp(-0.5 * (np.abs(z) - 2)))
    return np.clip(risk, 0, 100)


def _time_risk(df: pd.DataFrame) -> np.ndarray:
    """Score based on transaction timing (0-100).
    Off-hours and night transactions get higher risk."""
    hour = df["time_hour"].values
    # Night (0-6): highest risk, business (8-18): lowest
    risk = np.where(
        (hour >= 0) & (hour < 6), 80,      # Night: high risk
        np.where(
            (hour >= 6) & (hour < 8), 50,   # Early morning: medium
            np.where(
                (hour >= 18) & (hour < 22), 40,  # Evening: moderate
                np.where(
                    (hour >= 22), 70,        # Late night: high
                    20                       # Business hours: low
                )
            )
        )
    )
    return risk.astype(float)


def _velocity_risk(df: pd.DataFrame) -> np.ndarray:
    """Score based on transaction velocity/frequency patterns (0-100).
    Uses rolling amount ratios if available."""
    if "amount_ratio_5" in df.columns:
        ratio = df["amount_ratio_5"].values
        # High ratio = sudden spike relative to recent history
        risk = np.clip((ratio - 1) * 30, 0, 100)
    else:
        # Fallback: use amount percentile
        risk = df["amount_percentile"].values if "amount_percentile" in df.columns else np.zeros(len(df))
    return risk


def _pca_risk(df: pd.DataFrame) -> np.ndarray:
    """Score based on PCA magnitude deviation (0-100)."""
    mag = df["pca_magnitude"].values
    z_mag = (mag - mag.mean()) / mag.std()
    risk = 100 / (1 + np.exp(-0.3 * (np.abs(z_mag) - 2)))
    return np.clip(risk, 0, 100)


# ── Composite Risk Score ──────────────────────────────────
def calculate_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite risk score from multiple factors.

    Weights:
      - 35% Anomaly score (from ML models)
      - 25% Amount risk (z-score deviation)
      - 15% Time risk (off-hours activity)
      - 15% PCA risk (feature space deviation)
      - 10% Velocity risk (transaction patterns)
    """
    df = df.copy()

    # Component scores
    df["amount_risk"] = _amount_risk(df)
    df["time_risk"] = _time_risk(df)
    df["velocity_risk"] = _velocity_risk(df)
    df["pca_risk"] = _pca_risk(df)

    # Anomaly score from models (must exist from anomaly_model.py)
    anomaly = df["anomaly_score"].values if "anomaly_score" in df.columns else np.zeros(len(df))

    # Weighted composite
    df["risk_score"] = (
        0.35 * anomaly +
        0.25 * df["amount_risk"].values +
        0.15 * df["time_risk"].values +
        0.15 * df["pca_risk"].values +
        0.10 * df["velocity_risk"].values
    )

    # Clip to 0-100
    df["risk_score"] = df["risk_score"].clip(0, 100)

    # Assign risk category
    df["risk_category"] = pd.cut(
        df["risk_score"],
        bins=[0, 25, 50, 75, 100],
        labels=["low", "medium", "high", "critical"],
        include_lowest=True,
    )

    _log_summary(df)
    return df


def _log_summary(df: pd.DataFrame) -> None:
    """Log risk scoring summary statistics."""
    logger.info("=" * 60)
    logger.info("RISK SCORING SUMMARY")
    logger.info("=" * 60)

    # Distribution
    cat_counts = df["risk_category"].value_counts().sort_index()
    for cat in ["low", "medium", "high", "critical"]:
        count = cat_counts.get(cat, 0)
        pct = count / len(df) * 100
        emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}[cat]
        logger.info(f"  {emoji} {cat.upper():>8}: {count:>7,} ({pct:.2f}%)")

    # Fraud detection
    if TARGET_COL in df.columns:
        for cat in ["low", "medium", "high", "critical"]:
            subset = df[df["risk_category"] == cat]
            fraud = subset[TARGET_COL].sum() if len(subset) > 0 else 0
            total = len(subset)
            rate = fraud / total * 100 if total > 0 else 0
            logger.info(f"  Fraud in {cat}: {int(fraud)}/{total} ({rate:.2f}%)")

        # High + Critical capture rate
        high_risk = df[df["risk_category"].isin(["high", "critical"])]
        fraud_caught = high_risk[TARGET_COL].sum()
        total_fraud = df[TARGET_COL].sum()
        capture_rate = fraud_caught / total_fraud * 100 if total_fraud > 0 else 0
        logger.info(f"\n  🎯 Fraud captured by High+Critical: "
                    f"{int(fraud_caught)}/{int(total_fraud)} ({capture_rate:.1f}%)")

    logger.info(f"\n  📊 Risk score stats:")
    logger.info(f"     Mean:   {df['risk_score'].mean():.2f}")
    logger.info(f"     Median: {df['risk_score'].median():.2f}")
    logger.info(f"     Std:    {df['risk_score'].std():.2f}")
    logger.info(f"     Max:    {df['risk_score'].max():.2f}")
    logger.info("=" * 60)
