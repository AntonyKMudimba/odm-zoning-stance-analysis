# src/models/train_model.py
# Purpose: Train a multi-class stance classifier for ODM zoning statements,
#          following the Socrato Model Selection Hierarchy (Section 5.1).
# Author: [Your Name]
# Date Created: 2026-04-23
# Date Last Modified: 2026-04-23
# Dependencies: pandas, numpy, sklearn, xgboost, joblib, json, os, warnings

import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------------------------------
FEATURES_PATH = 'data/processed/odm_statements_features.csv'
SELECTED_FEATURES_PATH = 'models/selected_features.json'
MODEL_OUT = 'models/final_pipeline.joblib'
os.makedirs('models', exist_ok=True)

# ------------------------------------------------------------------------------------
# 1. Load data and subset to labeled rows
# ------------------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(FEATURES_PATH, parse_dates=['date'])
labeled_df = df[df['is_labeled'] == 1].copy()
print(f"Labeled rows: {len(labeled_df)}")

# Load selected features
with open(SELECTED_FEATURES_PATH, 'r') as f:
    selected_features = json.load(f)
print(f"Using {len(selected_features)} selected features: {selected_features}")

# Separate features and target
X = labeled_df[selected_features].copy()
y = labeled_df['label_encoded'].astype(int)

# Handle any remaining missing values in features (median imputation)
X = X.fillna(X.median())

# ------------------------------------------------------------------------------------
# 2. Train/Test Split (Section 5.3 - The Test Set Is Sacred)
# ------------------------------------------------------------------------------------
# 70% train, 30% test (used exactly once at the end)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ------------------------------------------------------------------------------------
# 3. Baseline Model (Section 5.1 - always start simple)
# ------------------------------------------------------------------------------------
print("\n--- Baseline Model (DummyClassifier - majority class) ---")
baseline = DummyClassifier(strategy='most_frequent', random_state=42)
# Cross-validate on training set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    baseline.fit(X_fold_train, y_fold_train)
    y_pred = baseline.predict(X_fold_val)
    f1 = f1_score(y_fold_val, y_pred, average='macro')
    baseline_scores.append(f1)
print(f"Baseline macro F1 (mean ± std): {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}")

# ------------------------------------------------------------------------------------
# 4. Simple Model - Logistic Regression (Section 5.1, 5.5 class_weight='balanced')
# ------------------------------------------------------------------------------------
print("\n--- Simple Model: Logistic Regression with class_weight='balanced' ---")
# Pipeline with scaling (Section 3.6: StandardScaler for linear models)
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

lr_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    lr_pipeline.fit(X_fold_train, y_fold_train)
    y_pred = lr_pipeline.predict(X_fold_val)
    f1 = f1_score(y_fold_val, y_pred, average='macro')
    lr_scores.append(f1)
print(f"Logistic Regression macro F1 (mean ± std): {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}")

# ------------------------------------------------------------------------------------
# 5. Ensemble Model - Random Forest (Section 5.1, 5.4 RandomizedSearchCV)
# ------------------------------------------------------------------------------------
print("\n--- Ensemble Model: Random Forest with hyperparameter tuning ---")
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

# Hyperparameter grid (Section 5.4 - Random Search for efficiency)
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=20, cv=skf,
    scoring='f1_macro', random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)
print(f"Best RF params: {rf_search.best_params_}")
print(f"Best RF CV macro F1: {rf_search.best_score_:.4f}")

# ------------------------------------------------------------------------------------
# 6. Ensemble Model - XGBoost (Section 5.1)
# ------------------------------------------------------------------------------------
print("\n--- Ensemble Model: XGBoost with hyperparameter tuning ---")
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)

# Handle class imbalance via scale_pos_weight (compute per class)
# XGBoost's 'scale_pos_weight' is for binary; for multi-class we use sample_weight or class_weight
# We'll set the number of classes explicitly
xgb_param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    xgb, param_distributions=xgb_param_dist, n_iter=15, cv=skf,
    scoring='f1_macro', random_state=42, n_jobs=-1
)
xgb_search.fit(X_train, y_train)
print(f"Best XGB params: {xgb_search.best_params_}")
print(f"Best XGB CV macro F1: {xgb_search.best_score_:.4f}")

# ------------------------------------------------------------------------------------
# 7. Select best model, train on full training set, evaluate on TEST SET (once!)
# ------------------------------------------------------------------------------------
# Compare models based on CV F1
results = {
    'Baseline': np.mean(baseline_scores),
    'LogisticRegression': np.mean(lr_scores),
    'RandomForest': rf_search.best_score_,
    'XGBoost': xgb_search.best_score_
}
best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name} (CV F1: {results[best_model_name]:.4f})")

# Re-train best model on all training data
if best_model_name == 'LogisticRegression':
    best_model = lr_pipeline
    best_model.fit(X_train, y_train)
elif best_model_name == 'RandomForest':
    best_model = rf_search.best_estimator_
elif best_model_name == 'XGBoost':
    best_model = xgb_search.best_estimator_
else:
    best_model = baseline

# Fit on full training set (already done for the best model above)

# Evaluate on test set - exactly once (Section 5.3: The Test Set Is Sacred)
y_test_pred = best_model.predict(X_test)
test_f1 = f1_score(y_test, y_test_pred, average='macro')
test_acc = accuracy_score(y_test, y_test_pred)
print(f"\n--- Final Test Set Performance ---")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro F1: {test_f1:.4f}")

# ------------------------------------------------------------------------------------
# 8. Save full pipeline (Section 8.3)
# ------------------------------------------------------------------------------------
joblib.dump(best_model, MODEL_OUT)
print(f"Best model saved to {MODEL_OUT}")
print("Phase 5 complete.")
