# src/data/clean_data.py
# Purpose: Clean the raw ODM zoning statements dataset according to the Socrato DS Manual Phase 3.
# Author: [Your Name]
# Date Created: 2026-04-23
# Date Last Modified: 2026-04-23
# Dependencies: pandas, numpy, os, json, warnings, datetime, sklearn

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

RAW_DATA = 'data/raw/odm_statements_raw.csv'
PROCESSED_DIR = 'data/processed'
PROCESSED_FILE = os.path.join(PROCESSED_DIR, 'odm_statements_cleaned.csv')
ENCODERS_DIR = 'models'
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ENCODERS_DIR, exist_ok=True)

def load_raw_data(path):
    print("Loading raw data...")
    df = pd.read_csv(path, parse_dates=['date'])
    print(f"Raw data shape: {df.shape}")
    return df

df = load_raw_data(RAW_DATA)

# Fix structural issues first (Section 3.2)
df.columns = df.columns.str.strip().str.lower()
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
if 'statement_id' in df.columns:
    df['statement_id'] = df['statement_id'].astype(int)

initial_rows = len(df)
df = df.drop_duplicates()
dropped = initial_rows - len(df)
if dropped > 0:
    print(f"Removed {dropped} exact duplicate rows.")

# Handle impossible values (Section 3.2)
df['text_length'] = df['text'].str.len()
future_dates = df[df['date'] > pd.Timestamp.today()]
if len(future_dates) > 0:
    print(f"Warning: {len(future_dates)} rows with future dates. Setting to NaT.")
    df.loc[df['date'] > pd.Timestamp.today(), 'date'] = pd.NaT

# Standardise categorical values (Section 3.2)
if 'leader_name' in df.columns:
    df['leader_name'] = df['leader_name'].str.strip().str.title()
if 'county' in df.columns:
    df['county'] = df['county'].str.strip().str.title()
if 'source' in df.columns:
    df['source'] = df['source'].str.strip().str.title()

# Handle missing values (Section 3.3)
df['is_labeled'] = df['label'].notnull().astype(int)
print(f"Labeled rows: {df['is_labeled'].sum()} ({df['is_labeled'].mean()*100:.1f}%)")

for col, fill_val in [('county', 'Unknown'), ('rank', 'Other'), ('source', 'Unknown')]:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(fill_val)

# Handle outliers (Section 3.4) - cap text_length at 99th percentile
if 'text_length' in df.columns:
    q99 = df['text_length'].quantile(0.99)
    outliers_count = (df['text_length'] > q99).sum()
    if outliers_count > 0:
        print(f"Capping text_length at 99th percentile ({q99:.0f}) for {outliers_count} rows.")
        df['text_length'] = df['text_length'].clip(upper=q99)

# Encode categoricals (Section 3.5)
if 'rank' in df.columns:
    le = LabelEncoder()
    df['rank_encoded'] = le.fit_transform(df['rank'].astype(str))
    mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
    with open(os.path.join(ENCODERS_DIR, 'rank_label_encoding.json'), 'w') as f:
        json.dump(mapping, f)
    print("Rank label encoding saved.")

if 'label' in df.columns:
    target_le = LabelEncoder()
    labeled_mask = df['label'].notnull()
    target_le.fit(df.loc[labeled_mask, 'label'].astype(str))
    df.loc[labeled_mask, 'label_encoded'] = target_le.transform(df.loc[labeled_mask, 'label'].astype(str))
    mapping = {str(k): int(v) for k, v in zip(target_le.classes_, target_le.transform(target_le.classes_))}
    with open(os.path.join(ENCODERS_DIR, 'target_label_encoding.json'), 'w') as f:
        json.dump(mapping, f)
    print("Target label encoding saved.")

# Validate (Section 3.2)
print("\n=== Cleaned Data Summary ===")
print(f"Shape: {df.shape}")
print("Missing values remaining:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("Duplicate rows:", df.duplicated().sum())
print("Sample:")
print(df.head())

# Save
df.to_csv(PROCESSED_FILE, index=False)
print(f"Cleaned dataset saved to {PROCESSED_FILE}")
