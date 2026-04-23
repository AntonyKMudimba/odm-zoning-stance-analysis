# src/features/build_features.py
# Purpose: Create and select features for ODM zoning stance classification from the cleaned dataset.
# Author: [Your Name]
# Date Created: 2026-04-23
# Date Last Modified: 2026-04-23
# Dependencies: pandas, numpy, os, json, warnings, sklearn

import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

CLEANED_DATA = 'data/processed/odm_statements_cleaned.csv'
FEATURES_OUT = 'data/processed/odm_statements_features.csv'
SELECTED_FEATURES_PATH = 'models/selected_features.json'
os.makedirs('models', exist_ok=True)

def load_cleaned_data(path):
    print("Loading cleaned data...")
    df = pd.read_csv(path, parse_dates=['date'])
    print(f"Cleaned data loaded. Shape: {df.shape}")
    return df

df = load_cleaned_data(CLEANED_DATA)

# ---------- Date/time features (Section 4.2.1) ----------
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_august'] = (df['month'] == 8).astype(int)
    df['is_december'] = (df['month'] == 12).astype(int)
    print("Date features added.")

# ---------- Text features (Section 4.2.4) ----------
if 'text' in df.columns:
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
    print("Text features added.")

# ---------- Aggregation features (Section 4.2.2) ----------
if 'leader_name' in df.columns:
    df['leader_statement_count'] = df.groupby('leader_name')['statement_id'].transform('count')
    df['leader_avg_text_length'] = df.groupby('leader_name')['text_length'].transform('mean')
    if 'contains_zoning' in df.columns:
        df['leader_zoning_keyword_pct'] = df.groupby('leader_name')['contains_zoning'].transform('mean')
    print("Aggregation features added.")

# ---------- Interaction features (Section 4.2.3) ----------
if 'text_length' in df.columns and 'is_weekend' in df.columns:
    df['len_x_weekend'] = df['text_length'] * df['is_weekend']
if 'word_count' in df.columns and 'contains_zoning' in df.columns:
    df['words_x_zoning'] = df['word_count'] * df['contains_zoning']
print("Interaction features added.")

# ---------- Feature selection (Section 4.3) ----------
print("Running feature selection on labeled subset...")
labeled_df = df[df['is_labeled'] == 1].copy()
if not labeled_df.empty:
    y = labeled_df['label_encoded']
    exclude_cols = ['statement_id', 'text', 'leader_name', 'county', 'rank', 'source',
                    'label', 'label_encoded', 'is_labeled', 'date', 'rank_encoded']
    feature_cols = [c for c in labeled_df.columns if c not in exclude_cols]
    X = labeled_df[feature_cols].select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    # Filter method
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
    print("Top 10 ANOVA F-score:")
    print(scores.head(10))

    # Embedded method
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top 10 RF importance:")
    print(importances.head(10))

    selected = importances[importances > 0.01].index.tolist()
    print(f"Selected {len(selected)} features.")
    with open(SELECTED_FEATURES_PATH, 'w') as f:
        json.dump(selected, f, indent=2)
else:
    print("No labeled data for selection.")
    selected = []

# ---------- Save ----------
df.to_csv(FEATURES_OUT, index=False)
print(f"Feature-engineered dataset saved to {FEATURES_OUT}")
