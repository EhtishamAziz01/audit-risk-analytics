"""
Audit Risk Analytics — ETL Pipeline
Load → Validate → Clean → Transform → Save
"""
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import (
    AMOUNT_BUCKETS,
    BUSINESS_HOURS,
    NUMERIC_FEATURES,
    PCA_FEATURES,
    PROCESSED_DIR,
    PROCESSED_PARQUET,
    RAW_CREDITCARD_CSV,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    TARGET_COL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Load ───────────────────────────────────────────────────
def load_raw_data(path: Path = RAW_CREDITCARD_CSV) -> pd.DataFrame:
    """Load the raw credit card transaction CSV."""
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ── Validate ───────────────────────────────────────────────
def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Check the dataset conforms to expected schema."""
    expected_cols = NUMERIC_FEATURES + [TARGET_COL]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check target is binary
    unique_classes = df[TARGET_COL].unique()
    if not set(unique_classes).issubset({0, 1}):
        raise ValueError(f"Target column has unexpected values: {unique_classes}")

    logger.info("✅ Schema validation passed")
    return df


# ── Clean ──────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and basic cleaning."""
    initial_rows = len(df)

    # Drop exact duplicates
    df = df.drop_duplicates()
    dupes_removed = initial_rows - len(df)
    if dupes_removed > 0:
        logger.info(f"Removed {dupes_removed:,} duplicate rows")

    # Check for nulls
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning(f"Found {total_nulls} null values — filling with median")
        df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())
    else:
        logger.info("✅ No null values found")

    # Ensure Amount is non-negative
    neg_amounts = (df["Amount"] < 0).sum()
    if neg_amounts > 0:
        logger.warning(f"Found {neg_amounts} negative amounts — taking absolute value")
        df["Amount"] = df["Amount"].abs()

    logger.info(f"✅ Cleaned dataset: {len(df):,} rows")
    return df


# ── Transform ─────────────────────────────────────────────
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived columns for analysis and modelling."""

    # Time features
    df["time_hour"] = (df["Time"] % SECONDS_PER_DAY) / SECONDS_PER_HOUR
    df["time_hour_int"] = df["time_hour"].astype(int)
    df["time_day"] = (df["Time"] / SECONDS_PER_DAY).astype(int)

    # Business hours flag
    df["is_business_hours"] = df["time_hour_int"].between(
        BUSINESS_HOURS[0], BUSINESS_HOURS[1] - 1
    )

    # Time segment
    df["time_segment"] = pd.cut(
        df["time_hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        include_lowest=True,
    )

    # Amount features
    df["amount_log"] = np.log1p(df["Amount"])
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df["is_outlier_amount"] = df["amount_zscore"].abs() > 3

    # Amount buckets (materiality)
    def _bucket(amount: float) -> str:
        for label, (lo, hi) in AMOUNT_BUCKETS.items():
            if lo <= amount < hi:
                return label
        return "very_high"

    df["amount_bucket"] = df["Amount"].apply(_bucket)

    # PCA magnitude (overall signal strength)
    df["pca_magnitude"] = np.sqrt((df[PCA_FEATURES] ** 2).sum(axis=1))

    logger.info(f"✅ Transformed dataset: {len(df.columns)} columns")
    return df


# ── Save ───────────────────────────────────────────────────
def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PARQUET) -> Path:
    """Save processed dataset to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Saved to {path} ({size_mb:.1f} MB)")
    return path


# ── DuckDB Registration ───────────────────────────────────
def register_in_duckdb(df: pd.DataFrame, table_name: str = "transactions") -> duckdb.DuckDBPyConnection:
    """Register the DataFrame as a DuckDB table for fast analytical queries."""
    con = duckdb.connect()
    con.register(table_name, df)
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info(f"✅ Registered DuckDB table '{table_name}' with {count:,} rows")
    return con


# ── Full Pipeline ──────────────────────────────────────────
def run_pipeline() -> tuple[pd.DataFrame, duckdb.DuckDBPyConnection]:
    """Execute the full ETL pipeline: Load → Validate → Clean → Transform → Save."""
    logger.info("=" * 60)
    logger.info("AUDIT RISK ANALYTICS — ETL PIPELINE")
    logger.info("=" * 60)

    df = load_raw_data()
    df = validate_schema(df)
    df = clean_data(df)
    df = transform_data(df)
    save_processed(df)

    con = register_in_duckdb(df)

    # Summary stats
    fraud_count = df[TARGET_COL].sum()
    fraud_rate = fraud_count / len(df) * 100
    logger.info(f"📊 Total transactions: {len(df):,}")
    logger.info(f"🚨 Fraud cases: {int(fraud_count):,} ({fraud_rate:.3f}%)")
    logger.info(f"💰 Amount range: €{df['Amount'].min():.2f} — €{df['Amount'].max():.2f}")
    logger.info(f"💰 Mean amount: €{df['Amount'].mean():.2f}")
    logger.info("=" * 60)

    return df, con


if __name__ == "__main__":
    run_pipeline()
