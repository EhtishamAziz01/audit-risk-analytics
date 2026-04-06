"""
Tests for the risk scoring engine.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TARGET_COL, PCA_FEATURES
from src.risk_scorer import calculate_risk_scores


@pytest.fixture
def scored_df():
    """Create a DataFrame with anomaly scores (simulating post-anomaly-detection)."""
    n = 200
    np.random.seed(42)
    data = {}
    for feat in PCA_FEATURES:
        data[feat] = np.random.randn(n)
    data["Amount"] = np.random.exponential(80, n)
    data["amount_zscore"] = (data["Amount"] - np.mean(data["Amount"])) / np.std(data["Amount"])
    data["time_hour"] = np.random.uniform(0, 24, n)
    data["pca_magnitude"] = np.sqrt(sum(data[f] ** 2 for f in PCA_FEATURES))
    data["anomaly_score"] = np.random.uniform(0, 100, n)
    data["amount_ratio_5"] = np.random.uniform(0.5, 3, n)
    data["amount_percentile"] = np.random.uniform(0, 100, n)
    data[TARGET_COL] = np.random.choice([0, 1], n, p=[0.98, 0.02])
    return pd.DataFrame(data)


class TestRiskScoring:
    def test_adds_risk_score(self, scored_df):
        result = calculate_risk_scores(scored_df)
        assert "risk_score" in result.columns

    def test_risk_score_range(self, scored_df):
        result = calculate_risk_scores(scored_df)
        assert result["risk_score"].min() >= 0
        assert result["risk_score"].max() <= 100

    def test_adds_risk_category(self, scored_df):
        result = calculate_risk_scores(scored_df)
        assert "risk_category" in result.columns
        valid = {"low", "medium", "high", "critical"}
        assert set(result["risk_category"].dropna().unique()).issubset(valid)

    def test_adds_component_scores(self, scored_df):
        result = calculate_risk_scores(scored_df)
        for col in ["amount_risk", "time_risk", "velocity_risk", "pca_risk"]:
            assert col in result.columns
            assert result[col].min() >= 0
            assert result[col].max() <= 100

    def test_does_not_modify_original(self, scored_df):
        original_cols = set(scored_df.columns)
        _ = calculate_risk_scores(scored_df)
        assert set(scored_df.columns) == original_cols

    def test_high_anomaly_high_risk(self, scored_df):
        scored_df["anomaly_score"] = 95  # force high anomaly
        result = calculate_risk_scores(scored_df)
        assert result["risk_score"].mean() > 30  # should be elevated
