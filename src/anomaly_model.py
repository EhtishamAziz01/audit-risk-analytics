"""
Audit Risk Analytics — Anomaly Detection Models
Isolation Forest + Local Outlier Factor for transaction anomaly detection.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.config import (
    ISOLATION_FOREST_CONTAMINATION,
    LOF_CONTAMINATION,
    LOF_N_NEIGHBORS,
    PCA_FEATURES,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


# ── Feature Selection ──────────────────────────────────────
MODEL_FEATURES = PCA_FEATURES + [
    "Amount",
    "amount_log",
    "amount_zscore",
    "time_hour",
    "pca_magnitude",
]


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features for anomaly detection models."""
    X = df[MODEL_FEATURES].values
    y = df[TARGET_COL].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Prepared {X_scaled.shape[0]:,} samples × {X_scaled.shape[1]} features")
    return X_scaled, y, scaler


# ── Isolation Forest ───────────────────────────────────────
def train_isolation_forest(
    X: np.ndarray,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    random_state: int = 42,
) -> tuple[IsolationForest, np.ndarray, np.ndarray]:
    """Train an Isolation Forest model for anomaly detection."""
    logger.info(f"Training Isolation Forest (contamination={contamination})…")

    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    # Predictions: -1 = anomaly, 1 = normal → convert to 0/1 (1 = anomaly)
    raw_preds = model.predict(X)
    predictions = (raw_preds == -1).astype(int)

    # Anomaly scores: lower = more anomalous
    scores = -model.decision_function(X)  # negate so higher = more anomalous

    # Normalise scores to 0-100
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min()) * 100

    n_anomalies = predictions.sum()
    logger.info(f"✅ Isolation Forest: {n_anomalies:,} anomalies detected "
                f"({n_anomalies / len(X) * 100:.3f}%)")

    return model, predictions, scores_norm


# ── Local Outlier Factor ───────────────────────────────────
def train_lof(
    X: np.ndarray,
    contamination: float = LOF_CONTAMINATION,
    n_neighbors: int = LOF_N_NEIGHBORS,
) -> tuple[LocalOutlierFactor, np.ndarray, np.ndarray]:
    """Train a Local Outlier Factor model for anomaly detection."""
    logger.info(f"Training LOF (contamination={contamination}, n_neighbors={n_neighbors})…")

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
        n_jobs=-1,
    )

    # LOF is transductive: fit_predict in one step
    raw_preds = model.fit_predict(X)
    predictions = (raw_preds == -1).astype(int)

    # Negative outlier factor: more negative = more anomalous
    scores = -model.negative_outlier_factor_

    # Normalise to 0-100
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min()) * 100

    n_anomalies = predictions.sum()
    logger.info(f"✅ LOF: {n_anomalies:,} anomalies detected "
                f"({n_anomalies / len(X) * 100:.3f}%)")

    return model, predictions, scores_norm


# ── Evaluation ─────────────────────────────────────────────
def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    model_name: str,
) -> dict:
    """Evaluate anomaly detection model against known fraud labels."""
    metrics = {
        "model": model_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "true_positives": int(((y_pred == 1) & (y_true == 1)).sum()),
        "false_positives": int(((y_pred == 1) & (y_true == 0)).sum()),
        "false_negatives": int(((y_pred == 0) & (y_true == 1)).sum()),
        "true_negatives": int(((y_pred == 0) & (y_true == 0)).sum()),
        "total_flagged": int(y_pred.sum()),
        "total_fraud": int(y_true.sum()),
    }

    metrics["fraud_caught_pct"] = (
        metrics["true_positives"] / metrics["total_fraud"] * 100
        if metrics["total_fraud"] > 0 else 0
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} — Evaluation Results")
    logger.info(f"{'='*50}")
    logger.info(f"  Precision:  {metrics['precision']:.4f}")
    logger.info(f"  Recall:     {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:   {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    logger.info(f"  Fraud caught: {metrics['true_positives']}/{metrics['total_fraud']} "
                f"({metrics['fraud_caught_pct']:.1f}%)")
    logger.info(f"  False alarms: {metrics['false_positives']}")
    logger.info(f"{'='*50}\n")

    return metrics


# ── Ensemble ───────────────────────────────────────────────
def ensemble_predictions(
    if_preds: np.ndarray,
    lof_preds: np.ndarray,
    if_scores: np.ndarray,
    lof_scores: np.ndarray,
    strategy: str = "union",
) -> tuple[np.ndarray, np.ndarray]:
    """Combine predictions from both models.

    Strategies:
      - 'union': flag if EITHER model flags (higher recall)
      - 'intersection': flag if BOTH models flag (higher precision)
      - 'weighted': weighted average of scores
    """
    if strategy == "union":
        combined_preds = ((if_preds + lof_preds) >= 1).astype(int)
    elif strategy == "intersection":
        combined_preds = ((if_preds + lof_preds) == 2).astype(int)
    elif strategy == "weighted":
        combined_preds = ((0.6 * if_scores + 0.4 * lof_scores) > 50).astype(int)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Weighted score combination
    combined_scores = 0.6 * if_scores + 0.4 * lof_scores

    logger.info(f"Ensemble ({strategy}): {combined_preds.sum():,} anomalies flagged")
    return combined_preds, combined_scores


# ── Full Pipeline ──────────────────────────────────────────
def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full anomaly detection pipeline and add results to DataFrame."""
    logger.info("=" * 60)
    logger.info("ANOMALY DETECTION PIPELINE")
    logger.info("=" * 60)

    # Prepare
    X, y, scaler = prepare_features(df)

    # Train models
    if_model, if_preds, if_scores = train_isolation_forest(X)
    lof_model, lof_preds, lof_scores = train_lof(X)

    # Evaluate
    if_metrics = evaluate_model(y, if_preds, if_scores, "Isolation Forest")
    lof_metrics = evaluate_model(y, lof_preds, lof_scores, "Local Outlier Factor")

    # Ensemble
    ens_preds, ens_scores = ensemble_predictions(
        if_preds, lof_preds, if_scores, lof_scores, strategy="union"
    )
    ens_metrics = evaluate_model(y, ens_preds, ens_scores, "Ensemble (Union)")

    # Add results to DataFrame
    df = df.copy()
    df["if_prediction"] = if_preds
    df["if_score"] = if_scores
    df["lof_prediction"] = lof_preds
    df["lof_score"] = lof_scores
    df["ensemble_prediction"] = ens_preds
    df["anomaly_score"] = ens_scores

    logger.info("=" * 60)
    return df
