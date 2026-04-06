"""
Audit Risk Analytics — Configuration
Paths, constants, and thresholds for the pipeline.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Raw data
RAW_CREDITCARD_CSV = RAW_DIR / "creditcard.csv"

# Processed outputs
PROCESSED_PARQUET = PROCESSED_DIR / "transactions_clean.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "transactions_features.parquet"

# ── Dataset Schema ─────────────────────────────────────────
# Credit Card Fraud Detection (Kaggle / ULB)
# 31 columns: Time, V1-V28 (PCA), Amount, Class
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
NUMERIC_FEATURES = ["Time", "Amount"] + PCA_FEATURES
TARGET_COL = "Class"

# ── Risk Thresholds ────────────────────────────────────────
RISK_LEVELS = {
    "low": (0, 25),
    "medium": (26, 50),
    "high": (51, 75),
    "critical": (76, 100),
}

# Anomaly detection
ISOLATION_FOREST_CONTAMINATION = 0.002  # ~0.17% fraud rate
LOF_CONTAMINATION = 0.002
LOF_N_NEIGHBORS = 20

# ── Amount Buckets (Materiality) ───────────────────────────
AMOUNT_BUCKETS = {
    "low": (0, 50),
    "medium": (50, 200),
    "high": (200, 1000),
    "very_high": (1000, float("inf")),
}

# ── Time Segments ──────────────────────────────────────────
# Time in dataset = seconds from first transaction
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
BUSINESS_HOURS = (8, 18)  # 08:00 - 18:00
