# src/deployment/batch_predict.py
# Purpose: Batch prediction script for ODM zoning stance classification.
#          Reads new unlabelled statements, applies the trained model,
#          and outputs a classified CSV.
# Author: [Your Name]
# Date Created: 2026-04-23
# Date Last Modified: 2026-04-23
# Dependencies: pandas, numpy, joblib, json, os, logging, datetime

import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime

# ---------- 0. Setup logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_predict.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- 1. Paths ----------
PIPELINE_PATH = 'models/final_pipeline.joblib'
SELECTED_FEATURES_PATH = 'models/selected_features.json'
ENCODING_PATH = 'models/target_label_encoding.json'
INPUT_CSV = 'data/raw/new_statements.csv'          # To be provided weekly
OUTPUT_CSV = 'data/processed/weekly_stance_predictions.csv'
ROLLBACK_MODEL = 'models/final_pipeline_previous.joblib'  # Keep previous version

logger.info("Starting batch prediction.")

# ---------- 2. Load model and metadata ----------
try:
    pipeline = joblib.load(PIPELINE_PATH)
    logger.info("Loaded current model.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Rollback: try loading previous model if current fails
    if os.path.exists(ROLLBACK_MODEL):
        pipeline = joblib.load(ROLLBACK_MODEL)
        logger.warning("Rolled back to previous model version.")
    else:
        raise RuntimeError("No model available for rollback.")

with open(SELECTED_FEATURES_PATH, 'r') as f:
    selected_features = json.load(f)
with open(ENCODING_PATH, 'r') as f:
    label_map = json.load(f)
idx_to_label = {int(v): k for k, v in label_map.items()}

# ---------- 3. Load new data ----------
if not os.path.exists(INPUT_CSV):
    logger.error(f"Input file {INPUT_CSV} not found. Creating empty template.")
    # Create a template for the client
    template = pd.DataFrame(columns=['statement_id', 'text', 'leader_name', 'county', 'rank', 'date', 'source'])
    template.to_csv(INPUT_CSV, index=False)
    raise FileNotFoundError(f"No input file. Template created at {INPUT_CSV}. Please populate with new statements.")

df = pd.read_csv(INPUT_CSV, parse_dates=['date'])
logger.info(f"Loaded {len(df)} new statements from {INPUT_CSV}")

# ---------- 4. Input validation (Section 8.2) ----------
required_cols = ['text', 'leader_name', 'county', 'rank', 'date', 'source']
for col in required_cols:
    if col not in df.columns:
        logger.warning(f"Column '{col}' missing from input. Adding with defaults.")
        if col == 'text':
            df[col] = ''
        else:
            df[col] = 'Unknown'

# Ensure text is string
df['text'] = df['text'].astype(str)

# ---------- 5. Feature engineering (replicate Phase 4 steps needed for prediction) ----------
logger.info("Building features for prediction...")
# Date features
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_august'] = (df['month'] == 8).astype(int)
df['is_december'] = (df['month'] == 12).astype(int)

# Text features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
keywords = {
    'contains_zoning': 'zoning',
    'contains_coalition': 'coalition',
    'contains_stronghold': 'stronghold',
    'contains_nationwide': 'nationwide',
    'contains_ugatuzi': 'ugatuzi',
    'contains_ukabila': 'ukabila',
    'contains_mgombea': 'mgombea',
    'contains_urais': 'urais'
}
for col_name, kw in keywords.items():
    df[col_name] = df['text'].str.lower().str.contains(kw, na=False).astype(int)

# Interaction features
df['len_x_weekend'] = df['text_length'] * df['is_weekend']
df['words_x_zoning'] = df['word_count'] * df['contains_zoning']

# Aggregation features (per leader) – use the current batch only (no historical data)
df['leader_statement_count'] = df.groupby('leader_name')['text_length'].transform('count')
df['leader_avg_text_length'] = df.groupby('leader_name')['text_length'].transform('mean')
df['leader_zoning_keyword_pct'] = df.groupby('leader_name')['contains_zoning'].transform('mean')

logger.info("Feature engineering complete.")

# ---------- 6. Prepare feature matrix ----------
feature_df = df[selected_features].copy()
feature_df = feature_df.fillna(feature_df.median())  # Impute missing with median (from training? Using batch median for simplicity)
logger.info("Feature matrix prepared.")

# ---------- 6a. Guard against empty input ----------
if feature_df.empty:
    logger.warning("No new statements provided this week. Saving empty output.")
    empty_out = pd.DataFrame(columns=['statement_id','text','leader_name','county','rank','date','source',
                                        'year','month','day_of_week','quarter','is_weekend','is_august','is_december',
                                        'text_length','word_count','contains_zoning','contains_coalition','contains_stronghold',
                                        'contains_nationwide','contains_ugatuzi','contains_ukabila','contains_mgombea','contains_urais',
                                        'len_x_weekend','words_x_zoning','leader_statement_count','leader_avg_text_length',
                                        'leader_zoning_keyword_pct','predicted_label_encoded','predicted_stance'])
    empty_out.to_csv(OUTPUT_CSV, index=False)
    logger.info("Empty predictions saved. Exiting.")
    exit(0)

# ---------- 7. Predict ----------
try:
    predictions = pipeline.predict(feature_df)
    df['predicted_label_encoded'] = predictions
    # Convert to human-readable labels
    df['predicted_stance'] = df['predicted_label_encoded'].map(idx_to_label)
    logger.info("Predictions generated.")
except Exception as e:
    logger.error(f"Prediction error: {e}")
    raise

# Try to get probabilities
try:
    proba = pipeline.predict_proba(feature_df)
    for i, cls_name in enumerate(idx_to_label.values()):
        df[f'probability_{cls_name}'] = proba[:, i]
    logger.info("Probability estimates added.")
except Exception:
    logger.warning("Model does not support probability estimates. Skipping.")

# ---------- 8. Save output ----------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
logger.info(f"Output saved to {OUTPUT_CSV} with {len(df)} rows.")

# ---------- 9. Rollback plan note ----------
# The previous model version is kept in ROLLBACK_MODEL.
# If this batch produces clearly erroneous results (e.g., all predictions are the same unexpected class),
# manually replace models/final_pipeline.joblib with models/final_pipeline_previous.joblib and re‑run.

logger.info("Batch prediction completed successfully.")
