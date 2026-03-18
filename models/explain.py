"""
Phase 3: Explainable AI
=======================

Computes SHAP values exclusively for the median prediction model (P50).
Extracts the gender-specific SHAP values and base predictions required
for the Oaxaca-Blinder decomposition in Phase 4.

Saves:
- `data/shap_results.csv`
- Global explanation plots into `report/`
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from preprocessing.validator import validate_dataset
from preprocessing.title_standardiser import standardise_titles
from preprocessing.feature_engineer import engineer_features
from models.train_models import DATA_FILE, SEED, FEATURES, CATEGORICALS, TARGET

def generate_shap():
    print("Loading completely preprocessed feature matrix...")
    v_report = validate_dataset(str(DATA_FILE))
    df = standardise_titles(v_report.dataframe)
    df_engineered = engineer_features(df)
    
    for cat in CATEGORICALS:
        df_engineered[cat] = df_engineered[cat].astype('category')
        
    X = df_engineered[FEATURES]
    y_raw = df_engineered[TARGET]
    y_log = np.log(y_raw)
    
    # Strictly isolate the exact test set used during evaluation
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=SEED
    )
    
    print("Loading model_p50 (Median Baseline ONLY)...")
    # CRITICAL: SHAP is only run on the P50 model to ensure interpretability
    try:
        model_p50 = joblib.load(Path("models/saved/lgb_p50.pkl"))
    except FileNotFoundError:
        print("model_p50.pkl not found!")
        sys.exit(1)
        
    print("Computing SHAP values on preprocessed feature matrix...")
    # TreeExplainer is fast and exact for LightGBM
    explainer = shap.TreeExplainer(model_p50)
    shap_values = explainer(X_test)
    
    # ── Extract data for Oaxaca-Blinder ─────────────────────────────
    print("Extracting SHAP metadata for Oaxaca-Blinder...")
    
    # SHAP natively computes additive components in the log salary space:
    # Prediction_log = explainer.expected_value + sum(shap_values)
    
    df_results = X_test.copy()
    df_results["actual_log_salary"] = y_test_log
    
    # The P50 prediction
    df_results["predicted_log_salary"] = model_p50.predict(X_test)
    
    # Residuals = Actual - Predicted in the log space
    df_results["residual_log_salary"] = df_results["actual_log_salary"] - df_results["predicted_log_salary"]
    
    # Isolate the exact penalty/premium applied natively because of the gender feature
    gender_idx = FEATURES.index("gender")
    df_results["shap_gender_effect"] = shap_values.values[:, gender_idx]
    
    # Extract manager_rating SHAP for explicit reporting of the primary legitimate bias vector
    manager_idx = FEATURES.index("manager_rating")
    df_results["shap_manager_effect"] = shap_values.values[:, manager_idx]
    
    results_path = Path("data/shap_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_path, index=False)
    print(f"SHAP components saved to {results_path}")

    # ── Global Reporting Plots ──────────────────────────────────────
    print("Generating SHAP summary artifacts...")
    report_dir = Path("report")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(report_dir / "shap_summary.png", dpi=150)
    plt.close()
    
    print("Phase 3 XAI execution complete.")

if __name__ == "__main__":
    generate_shap()
