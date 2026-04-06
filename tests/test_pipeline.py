"""
Tests for the ETL pipeline.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TARGET_COL, PCA_FEATURES, NUMERIC_FEATURES
from src.pipeline import validate_schema, clean_data, transform_data


# ── Fixtures ───────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Create a minimal valid DataFrame for testing."""
    n = 100
    np.random.seed(42)
    data = {"Time": np.random.uniform(0, 172800, n), "Amount": np.random.exponential(80, n)}
    for feat in PCA_FEATURES:
        data[feat] = np.random.randn(n)
    data[TARGET_COL] = np.random.choice([0, 1], n, p=[0.98, 0.02])
    return pd.DataFrame(data)


@pytest.fixture
def dirty_df(sample_df):
    """Create a DataFrame with issues to clean."""
    df = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)  # duplicates
    df.loc[0, "Amount"] = -10  # negative amount
    return df


# ── Tests ──────────────────────────────────────────────────
class TestValidateSchema:
    def test_valid_schema_passes(self, sample_df):
        result = validate_schema(sample_df)
        assert result is not None
        assert len(result) == len(sample_df)

    def test_missing_column_raises(self, sample_df):
        df = sample_df.drop(columns=["Amount"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema(df)

    def test_invalid_target_raises(self, sample_df):
        sample_df[TARGET_COL] = 5
        with pytest.raises(ValueError, match="unexpected values"):
            validate_schema(sample_df)


class TestCleanData:
    def test_removes_duplicates(self, dirty_df):
        result = clean_data(dirty_df)
        assert len(result) < len(dirty_df)

    def test_fixes_negative_amounts(self, dirty_df):
        result = clean_data(dirty_df)
        assert (result["Amount"] >= 0).all()

    def test_no_nulls_after_cleaning(self, sample_df):
        sample_df.loc[0, "Amount"] = np.nan
        result = clean_data(sample_df)
        assert result.isnull().sum().sum() == 0


class TestTransformData:
    def test_adds_time_features(self, sample_df):
        result = transform_data(sample_df)
        assert "time_hour" in result.columns
        assert "time_hour_int" in result.columns
        assert "is_business_hours" in result.columns
        assert "time_segment" in result.columns

    def test_adds_amount_features(self, sample_df):
        result = transform_data(sample_df)
        assert "amount_log" in result.columns
        assert "amount_zscore" in result.columns
        assert "amount_bucket" in result.columns
        assert "is_outlier_amount" in result.columns

    def test_adds_pca_magnitude(self, sample_df):
        result = transform_data(sample_df)
        assert "pca_magnitude" in result.columns
        assert (result["pca_magnitude"] >= 0).all()

    def test_time_hour_range(self, sample_df):
        result = transform_data(sample_df)
        assert result["time_hour"].between(0, 24).all()

    def test_amount_buckets_valid(self, sample_df):
        result = transform_data(sample_df)
        valid_buckets = {"low", "medium", "high", "very_high"}
        assert set(result["amount_bucket"].unique()).issubset(valid_buckets)
