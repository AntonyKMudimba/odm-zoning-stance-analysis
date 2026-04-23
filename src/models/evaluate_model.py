# src/models/evaluate_model.py
# Purpose: Evaluate the trained stance classifier on the test set,
#          following Socrato Phase 6 (Sections 6.1, 6.2, 6.3).
# Author: [Your Name]
# Date Created: 2026-04-23
# Date Last Modified: 2026-04-23
# Dependencies: pandas, numpy, sklearn, shap, matplotlib, seaborn, joblib, json, os

import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import shap

# ------ 0. Paths ------
FEATURES_PATH = 'data/processed/odm_statements_features.csv'
MODEL_PATH = 'models/final_pipeline.joblib'
SELECTED_FEATURES_PATH = 'models/selected_features.json'
ENCODING_PATH = 'models/target_label_encoding.json'
FIGURES_DIR = 'reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------ 1. Load model and data ------
print("Loading model and data...")
pipeline = joblib.load(MODEL_PATH)
df = pd.read_csv(FEATURES_PATH, parse_dates=['date'])
labeled_df = df[df['is_labeled'] == 1].copy()
with open(SELECTED_FEATURES_PATH, 'r') as f:
    selected_features = json.load(f)

X = labeled_df[selected_features].fillna(labeled_df[selected_features].median())
y = labeled_df['label_encoded'].astype(int)

# Recreate same train/test split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Test set size: {X_test.shape[0]}")

# Load label mapping
with open(ENCODING_PATH, 'r') as f:
    label_map = json.load(f)
idx_to_label = {int(v): k for k, v in label_map.items()}
classes = [idx_to_label[i] for i in range(len(idx_to_label))]

# ------ 2. Predictions ------
y_pred = pipeline.predict(X_test)
# Try probabilities for AUC; if not available, skip AUC
try:
    y_prob = pipeline.predict_proba(X_test)
    has_proba = True
except Exception:
    y_prob = None
    has_proba = False

# ------ 3. Metrics (Section 6.1) ------
print("\n=== Classification Metrics ===")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print(f"F1 (macro): {f1:.4f}")

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix – Stance Classification on Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"))
plt.show()
print("Confusion matrix (saved as confusion_matrix.png):\n", cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

# AUC-ROC (multi-class, One-vs-Rest) if probabilities available
if has_proba:
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
    print(f"AUC-ROC (macro, OvR): {auc:.4f}")
else:
    print("AUC-ROC skipped (model does not support probability estimates).")

# ------ 4. SHAP (Section 6.2) ------
print("\nGenerating SHAP summary plot...")
try:
    # Choose appropriate explainer
    if hasattr(pipeline, 'named_steps') and hasattr(pipeline.named_steps.get(list(pipeline.named_steps.keys())[-1]), 'coef_'):
        explainer = shap.LinearExplainer(pipeline[-1], X_test)
    else:
        # Use a subset of test data for speed
        background = shap.sample(X_train, 50) if len(X_train) > 50 else X_train
        explainer = shap.KernelExplainer(pipeline.predict_proba, background)
        shap_values = explainer.shap_values(X_test[:50], nsamples=100)
        shap.summary_plot(shap_values, X_test[:50], feature_names=selected_features, show=False)
except Exception as e:
    print(f"SHAP error: {e}. Using simple feature importance instead.")
    # Skip or use built-in feature importance if available
    if hasattr(pipeline[-1], 'feature_importances_'):
        importances = pipeline[-1].feature_importances_
        feat_imp = pd.Series(importances, index=selected_features).sort_values()
        feat_imp.plot(kind='barh')
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"))
        plt.show()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"))
plt.show()
print("SHAP summary saved.")

# ------ 5. Subgroup Performance (Section 6.3) ------
print("\n=== Subgroup Performance ===")
test_indices = X_test.index
subgroup_df = labeled_df.loc[test_indices].copy()
subgroup_df['y_pred'] = y_pred
subgroup_df['y_true'] = y_test

# By county
print("\n--- F1 by County ---")
for county in subgroup_df['county'].unique():
    sub = subgroup_df[subgroup_df['county'] == county]
    if len(sub) > 0:
        f1_sub = f1_score(sub['y_true'], sub['y_pred'], average='macro')
        print(f"{county}: {f1_sub:.4f} (n={len(sub)})")

# By rank
print("\n--- F1 by Rank ---")
for rank in subgroup_df['rank'].unique():
    sub = subgroup_df[subgroup_df['rank'] == rank]
    if len(sub) > 0:
        f1_sub = f1_score(sub['y_true'], sub['y_pred'], average='macro')
        print(f"{rank}: {f1_sub:.4f} (n={len(sub)})")

# By leader (top 5)
print("\n--- F1 by Leader (top 5) ---")
top_leaders = subgroup_df['leader_name'].value_counts().head(5).index
for leader in top_leaders:
    sub = subgroup_df[subgroup_df['leader_name'] == leader]
    if len(sub) > 0:
        f1_sub = f1_score(sub['y_true'], sub['y_pred'], average='macro')
        print(f"{leader}: {f1_sub:.4f} (n={len(sub)})")

# ------ 6. Sanity checks (Section 6.3) ------
print("\n=== Sanity Checks ===")
# Check for unseen labels
missing_labels = set(y_pred) - set(y)
if missing_labels:
    print(f"WARNING: Model predicted unseen labels: {missing_labels}")
else:
    print("No unseen labels predicted.")

# Business rule compliance: check if model ever predicts a stance for a leader that hasn't spoken about zoning?
# Simple check: if a leader has no statements with a certain keyword, does the model still predict a stance?
# We'll print a count of mismatches (non-neutral when text lacks zoning keywords)
if 'contains_zoning' in labeled_df.columns:
    test_zoning = labeled_df.loc[test_indices, 'contains_zoning']
    non_zoning_but_predicted = ((test_zoning == 0) & (y_pred != idx_to_label.index('Neutral') if 'Neutral' in idx_to_label.values() else False)).sum()
    print(f"Non-zoning statements predicted as non-neutral: {non_zoning_but_predicted} (may indicate context learning)")

print("Sanity checks completed.")
print("Phase 6 evaluation finished.")
