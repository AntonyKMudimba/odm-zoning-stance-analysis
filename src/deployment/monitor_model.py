# src/deployment/monitor_model.py
# Purpose: Monitor the deployed ODM stance model for drift and data quality,
#          following Socrato Phase 9 (Sections 9.2 and 9.3).
# Author: [Your Name]
# Date Created: 2026-04-23
# Date Last Modified: 2026-04-23
# Dependencies: pandas, numpy, json, os, logging, datetime

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime

# ---------- 0. Setup logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- 1. Paths ----------
PREDICTIONS_CSV = 'data/processed/weekly_stance_predictions.csv'
BASELINE_STATS_PATH = 'models/baseline_stats.json'   # We'll create if missing
RETRAIN_THRESHOLD = 0.75   # Macro F1 threshold for retraining (Section 9.3)

# ---------- 2. Load today's predictions ----------
if not os.path.exists(PREDICTIONS_CSV):
    logger.warning("No predictions file found. Monitoring cannot run without data.")
    exit(0)

df = pd.read_csv(PREDICTIONS_CSV)
logger.info(f"Loaded {len(df)} predictions from {PREDICTIONS_CSV}")

# ---------- 3. Compute current statistics ----------
# Prediction distribution
if 'predicted_stance' in df.columns:
    current_dist = df['predicted_stance'].value_counts(normalize=True).to_dict()
else:
    logger.error("No 'predicted_stance' column found. Cannot monitor.")   
    exit(1)

# Feature statistics (for numerical columns that match selected features)
with open('models/selected_features.json', 'r') as f:
    selected_features = json.load(f)
numeric_cols = [c for c in selected_features if c in df.columns and df[c].dtype in [np.float64, np.int64, np.int32, np.float32]]
current_feature_stats = {}
for col in numeric_cols:
    series = df[col]
    current_feature_stats[col] = {
        'mean': float(series.mean()),
        'std': float(series.std()),
        'missing_rate': float(series.isnull().mean())
    }

# ---------- 4. Load or create baseline ----------
if os.path.exists(BASELINE_STATS_PATH):
    with open(BASELINE_STATS_PATH, 'r') as f:
        baseline_stats = json.load(f)
    logger.info("Baseline statistics loaded.")
else:
    # First run: use current stats as baseline (not ideal, but works for initialisation)
    # Per Section 9.2, baseline should come from training data; we'll store now and warn.
    baseline_stats = {
        'prediction_distribution': current_dist,
        'feature_stats': current_feature_stats,
        'created_on': datetime.now().isoformat()
    }
    with open(BASELINE_STATS_PATH, 'w') as f:
        json.dump(baseline_stats, f, indent=2)
    logger.warning("No baseline found. Created from current predictions. Review manually.")

# ---------- 5. Compare and alert ----------
logger.info("\n====== MONITORING HEALTH REPORT ======")
alerts = []

# 5a. Prediction distribution shift
if 'prediction_distribution' in baseline_stats:
    baseline_dist = baseline_stats['prediction_distribution']
    for label, baseline_pct in baseline_dist.items():
        current_pct = current_dist.get(label, 0.0)
        diff = abs(current_pct - baseline_pct)
        if diff > 0.10:   # 10 percentage point shift
            alert = f"Prediction distribution shift for '{label}': baseline {baseline_pct:.2%}, current {current_pct:.2%} (diff {diff:.2%})"
            alerts.append(alert)
            logger.warning(alert)

# 5b. Feature statistics drift
if 'feature_stats' in baseline_stats:
    baseline_feat = baseline_stats['feature_stats']
    for col, stats in current_feature_stats.items():
        if col in baseline_feat:
            baseline_mean = baseline_feat[col]['mean']
            baseline_std = baseline_feat[col]['std']
            # Drift detection: mean shift > 2 standard deviations
            if baseline_std > 0:
                mean_shift = abs(stats['mean'] - baseline_mean) / baseline_std
                if mean_shift > 2:
                    alert = f"Feature '{col}' mean drift: {mean_shift:.1f} std deviations"
                    alerts.append(alert)
                    logger.warning(alert)
            # Missing rate increase > 5%
            missing_diff = stats['missing_rate'] - baseline_feat[col]['missing_rate']
            if missing_diff > 0.05:
                alert = f"Feature '{col}' missing rate increased by {missing_diff:.2%}"
                alerts.append(alert)
                logger.warning(alert)

if not alerts:
    logger.info("No anomalies detected. Model appears stable.")
else:
    logger.info(f"{len(alerts)} alert(s) generated. Review immediately.")

# ---------- 6. Retraining trigger (Section 9.3) ----------
# In practice, the client would provide manually verified F1 for the previous period.
# Here we illustrate the logic with a placeholder.
last_verified_f1 = None  # Replace with actual value when available
REQUIRED_F1 = 0.75

if last_verified_f1 is not None and last_verified_f1 < REQUIRED_F1:
    logger.error(f"VERIFIED F1 {last_verified_f1:.3f} below threshold {REQUIRED_F1}. RETRAINING REQUIRED.")
else:
    logger.info(f"Retraining not triggered (current threshold: F1 >= {REQUIRED_F1}).")

# ---------- 7. Scheduled retraining reminder ----------
# Per Section 9.3, stable environments retrain quarterly; fast-changing environments monthly.
# We'll print a reminder based on the last training date.
last_train_date = datetime(2026, 4, 23)  # Date of Phase 5 training
days_since_train = (datetime.now() - last_train_date).days
if days_since_train >= 30:
    logger.info(f"Note: {days_since_train} days since last training. Consider scheduled retraining.")

logger.info("Monitoring report complete.\n")
