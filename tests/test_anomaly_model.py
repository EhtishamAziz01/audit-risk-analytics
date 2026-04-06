"""
Tests for the anomaly detection models.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TARGET_COL, PCA_FEATURES
from src.anomaly_model import (
    MODEL_FEATURES,
    prepare_features,
    train_isolation_forest,
    train_lof,
    evaluate_model,
    ensemble_predictions,
)


@pytest.fixture
def feature_df():
    """Create a DataFrame with all required model features."""
    n = 500
    np.random.seed(42)
    data = {}
    for feat in PCA_FEATURES:
        data[feat] = np.random.randn(n)
    data["Amount"] = np.random.exponential(80, n)
    data["amount_log"] = np.log1p(data["Amount"])
    data["amount_zscore"] = (data["Amount"] - np.mean(data["Amount"])) / np.std(data["Amount"])
    data["time_hour"] = np.random.uniform(0, 24, n)
    data["pca_magnitude"] = np.sqrt(sum(data[f] ** 2 for f in PCA_FEATURES))
    data[TARGET_COL] = np.random.choice([0, 1], n, p=[0.98, 0.02])
    return pd.DataFrame(data)


class TestPrepareFeatures:
    def test_output_shape(self, feature_df):
        X, y, scaler = prepare_features(feature_df)
        assert X.shape == (len(feature_df), len(MODEL_FEATURES))
        assert len(y) == len(feature_df)

    def test_scaled_mean_near_zero(self, feature_df):
        X, y, scaler = prepare_features(feature_df)
        assert np.abs(X.mean(axis=0)).max() < 0.1


class TestIsolationForest:
    def test_returns_predictions(self, feature_df):
        X, y, _ = prepare_features(feature_df)
        model, preds, scores = train_isolation_forest(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_scores_in_range(self, feature_df):
        X, y, _ = prepare_features(feature_df)
        _, _, scores = train_isolation_forest(X)
        assert scores.min() >= 0
        assert scores.max() <= 100

    def test_detects_some_anomalies(self, feature_df):
        X, y, _ = prepare_features(feature_df)
        _, preds, _ = train_isolation_forest(X)
        assert preds.sum() > 0


class TestLOF:
    def test_returns_predictions(self, feature_df):
        X, y, _ = prepare_features(feature_df)
        model, preds, scores = train_lof(X)
        assert len(preds) == len(X)

    def test_scores_in_range(self, feature_df):
        X, y, _ = prepare_features(feature_df)
        _, _, scores = train_lof(X)
        assert scores.min() >= 0
        assert scores.max() <= 100


class TestEnsemble:
    def test_union_catches_more(self, feature_df):
        X, y, _ = prepare_features(feature_df)
        _, if_preds, if_scores = train_isolation_forest(X)
        _, lof_preds, lof_scores = train_lof(X)
        union_preds, _ = ensemble_predictions(if_preds, lof_preds, if_scores, lof_scores, "union")
        inter_preds, _ = ensemble_predictions(if_preds, lof_preds, if_scores, lof_scores, "intersection")
        assert union_preds.sum() >= inter_preds.sum()

    def test_combined_scores_in_range(self, feature_df):
        X, y, _ = prepare_features(feature_df)
        _, if_preds, if_scores = train_isolation_forest(X)
        _, lof_preds, lof_scores = train_lof(X)
        _, combined_scores = ensemble_predictions(if_preds, lof_preds, if_scores, lof_scores, "weighted")
        assert combined_scores.min() >= 0
        assert combined_scores.max() <= 100


class TestEvaluateModel:
    def test_returns_metrics(self, feature_df):
        X, y, _ = prepare_features(feature_df)
        _, preds, scores = train_isolation_forest(X)
        metrics = evaluate_model(y, preds, scores, "test_model")
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
