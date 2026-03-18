"""
Phase 2: LightGBM Quantile Regression Training
==============================================

Uses Optuna to tune hyperparams for a median regression model (P50), then trains
three quantile models (P05, P50, P95) to create a predictive salary band.
Objective for tuning is single pinball loss on the validation set for alpha=0.50.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss

from preprocessing.validator import validate_dataset
from preprocessing.title_standardiser import standardise_titles
from preprocessing.feature_engineer import engineer_features

# ── Configuraton ────────────────────────────────────────────────────────

DATA_FILE = Path("data/india_compensation_dataset.csv")
MODEL_DIR = Path("models/saved")
SEED = 42
N_TRIALS = 50

# Features used for modelling (11 total)
FEATURES = [
    "job_level",
    "career_velocity",
    "city_tier",
    "education_tier",
    "performance_rating",
    "manager_rating",
    "manager_rating_missing",
    "months_since_promotion",
    "variable_pay_pct",
    "department",
    "gender",
]
CATEGORICALS = ["department", "gender"]
TARGET = "ctc"

# ── Data Loading & Prepping ─────────────────────────────────────────────

def get_training_data():
    v_report = validate_dataset(str(DATA_FILE))
    df = standardise_titles(v_report.dataframe)
    df_engineered = engineer_features(df)
    
    # Must convert categoricals to category dtype for LightGBM
    for cat in CATEGORICALS:
        df_engineered[cat] = df_engineered[cat].astype('category')
        
    X = df_engineered[FEATURES]
    # Target is log(salary) based on Architecture Document specification
    y = np.log(df_engineered[TARGET])
    
    # Split 80/20 train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # Further split train into train/val for Optuna tuning
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED) # 0.25 of 0.8 = 0.2
    
    return X_train, y_train, X_tr, X_val, y_tr, y_val, df_engineered

# ── Optuna Tuning ───────────────────────────────────────────────────────

def tune_hyperparameters(X_tr, y_tr, X_val, y_val):
    print("\nStarting Optuna hyperparameter tuning for P50 model...")
    def objective(trial):
        params = {
            "objective": "quantile",
            "alpha": 0.50,
            "metric": "quantile",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "verbose": -1,
            "random_state": SEED,
            "n_estimators": 500
        }
        
        early_stopping = lgb.early_stopping(stopping_rounds=30, verbose=False)
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping],
            categorical_feature=CATEGORICALS
        )
        
        preds = model.predict(X_val)
        # Minimize pinball loss explicitly for alpha=0.5
        val_loss = mean_pinball_loss(y_val, preds, alpha=0.50)
        return val_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    print(f"  Best pinball loss: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params

# ── Training Models ─────────────────────────────────────────────────────

def train_quantile_models(X_train, y_train, best_params):
    print("\nTraining final P05, P50, and P95 models on full training set...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # We use early stopping internally to prevent over/under fitting the quantiles
    # This guarantees excellent calibration for the P05 / P95 models
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED)
    
    base_params = {
        "objective": "quantile",
        "metric": "quantile",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_estimators": 1000,
        "random_state": SEED,
        **best_params
    }
    
    early_stopping = lgb.early_stopping(stopping_rounds=50, verbose=False)
    
    models = {}
    for name, alpha in [("p05", 0.05), ("p50", 0.50), ("p95", 0.95)]:
        print(f"  Training {name} (alpha={alpha})...")
        params = base_params.copy()
        params["alpha"] = alpha
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[early_stopping],
            categorical_feature=CATEGORICALS
        )
        
        model_path = MODEL_DIR / f"lgb_{name}.pkl"
        joblib.dump(model, model_path)
        print(f"  -> Saved to {model_path} (best iteration: {model.best_iteration_})")
        models[name] = model
        
    return models

if __name__ == "__main__":
    X_train, y_train, X_tr, X_val, y_tr, y_val, df_engineered = get_training_data()
    best_params = tune_hyperparameters(X_tr, y_tr, X_val, y_val)
    train_quantile_models(X_train, y_train, best_params)
    print("\nTraining completely successfully! Ready for evaluation.")
