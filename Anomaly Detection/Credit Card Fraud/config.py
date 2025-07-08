"""
config.py

Centralized configuration for magic numbers and hyperparameters.
"""

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path (assumes 'creditcard.csv' in 'data/' under BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")
ZIP_PATH = os.path.join (DATA_DIR, "creditcard.zip")

# Random seed for reproducibility
RANDOM_STATE = 2025

# Train/test split
TEST_SIZE = 0.2

# Paths for saving/loading models
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_xgb_model.joblib")

# Hyperparameter grid for XGBoost (GridSearch)
HYPERPARAM_GRID = {
    "xgb__n_estimators": [100, 200],
    "xgb__max_depth": [3, 5],
    "xgb__learning_rate": [0.01, 0.1],
    "xgb__subsample": [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0],
    "xgb__scale_pos_weight": [1, 25, 50],
}

# For RandomizedSearch (if used)
RANDOMIZED_HYPERPARAM_DISTRIBUTION = {
    "xgb__n_estimators": [50, 100, 200, 500],
    "xgb__max_depth": [3, 5, 7],
    "xgb__learning_rate": [0.001, 0.01, 0.1],
    "xgb__subsample": [0.6, 0.8, 1.0],
    "xgb__colsample_bytree": [0.6, 0.8, 1.0],
    "xgb__scale_pos_weight": [1, 25, 50, 100],
}

# Cross-validation settings
CV_SPLITS = 5

# XGBoost settings
XGB_EVAL_METRIC = "auc"
XGB_USE_LABEL_ENCODER = False

# SMOTE settings
SMOTE_RANDOM_STATE = RANDOM_STATE

# Logging configuration
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "logs", "training.log")

# Schema validation (Pandera)
SCHEMA = {
    "columns": {
        "Time": {"dtype": "float64"},
        # V1-V28 are PCA components (float64)
        **{f"V{i}": {"dtype": "float64"} for i in range(1, 29)},
        "Amount": {"dtype": "float64"},
        "Class": {"dtype": "int64", "checks": [lambda s: s.isin([0, 1])]},
    },
    "index": {"dtype": ["int64"]},
}
