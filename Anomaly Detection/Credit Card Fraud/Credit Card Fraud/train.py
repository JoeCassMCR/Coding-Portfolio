"""
train.py

Main script to orchestrate:
  1. Load & validate data
  2. Split into train/validation/test
  3. Compare default pipelines (XGBoost vs RandomForest)
  4. Hyperparameter tuning for XGBoost on the training set
  5. Probability calibration on validation set
  6. Threshold tuning on validation set using a cost matrix
  7. Evaluate XGBoost on the test set at optimal threshold
  8. Compare alternative samplers (SMOTE, SMOTEN, ADASYN) on validation PR-AUC
  9. Build & manually evaluate a soft-voting ensemble (calibrated XGB + calibrated RF)
"""

import os
import logging
import argparse

from sklearn.model_selection import train_test_split
from joblib import dump, load

from config import (
    DATA_PATH,
    DEFAULT_MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    LOG_FORMAT,
    LOG_DATEFMT,
    LOG_LEVEL,
    LOG_FILE,
)
from data_loader import load_data   # Uses Pandera internally to validate
from preprocessing import preprocess_data
from model_utils import (
    compare_default_pipelines,
    build_xgb_pipeline,
    build_rf_pipeline,
    perform_grid_search_xgb,
    save_model,
    get_alternative_sampler,
)
from evaluation import (
    tune_threshold,
    evaluate_at_threshold,
)

from sklearn.calibration import CalibratedClassifierCV

# Configure root logger (console + file)
if not os.path.exists(os.path.dirname(LOG_FILE)):
    os.makedirs(os.path.dirname(LOG_FILE))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate the fraud detection pipeline.")
    parser.add_argument(
        "--cost-fp", type=float, default=1.0, help="Cost of false positive (on validation)."
    )
    parser.add_argument(
        "--cost-fn", type=float, default=10.0, help="Cost of false negative (on validation)."
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="If set, skip training and load existing calibrated model (XGB + RF)."
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------------------
    # 1) Load and validate data
    # ----------------------------------------------------------------------------
    LOGGER.info("Loading and validating data ...")
    df = load_data(DATA_PATH)

    # ----------------------------------------------------------------------------
    # 2) Preprocess (feature engineering + scaling) and split into train/val/test
    # ----------------------------------------------------------------------------
    LOGGER.info("Preprocessing and splitting data into train/val/test ...")
    X_all, y_all = preprocess_data(df)
    # First split: train+val vs test (test fraction = TEST_SIZE)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )
    # Second split: train vs val (val fraction = TEST_SIZE / (1 - TEST_SIZE))
    val_fraction = TEST_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=RANDOM_STATE
    )
    LOGGER.info(f"Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ----------------------------------------------------------------------------
    # 3) Compare default pipelines (XGB + SMOTE vs RF + SMOTE) via CV on training set
    # ----------------------------------------------------------------------------
    LOGGER.info("Comparing default pipelines on training set ...")
    default_scores = compare_default_pipelines(X_train, y_train)
    LOGGER.info(f"Default pipeline ROC-AUCs: {default_scores}")

    # ----------------------------------------------------------------------------
    # 4) Hyperparameter tuning for XGBoost on the training set
    # ----------------------------------------------------------------------------
    if args.skip_training:
        LOGGER.info("Skipping training; loading existing calibrated model ...")
        calibrated_xgb = load_model(DEFAULT_MODEL_PATH)
    else:
        LOGGER.info("Hyperparameter-tuning XGBoost (with SMOTE) on training set ...")
        smote_pipeline = build_xgb_pipeline()  # default sampler = SMOTE
        grid_search = perform_grid_search_xgb(smote_pipeline, X_train, y_train)
        best_pipeline = grid_search.best_estimator_

        # ----------------------------------------------------------------------------
        # 5) Calibrate XGBoost probabilities on the validation set
        # ----------------------------------------------------------------------------
        LOGGER.info("Calibrating XGBoost probabilities on validation set ...")
        calibrated_xgb = CalibratedClassifierCV(
            estimator=best_pipeline,
            method="isotonic",
            cv="prefit"
        )
        calibrated_xgb.fit(X_val, y_val)

        # Save calibrated model for future use
        save_model(calibrated_xgb, DEFAULT_MODEL_PATH)

    # ----------------------------------------------------------------------------
    # 6) Threshold tuning on validation set using cost matrix
    # ----------------------------------------------------------------------------
    LOGGER.info("Tuning threshold on validation set ...")
    optimal_threshold = tune_threshold(
        calibrated_xgb, X_val, y_val,
        cost_fp=args.cost_fp, cost_fn=args.cost_fn
    )

    # ----------------------------------------------------------------------------
    # 7) Evaluate calibrated XGBoost on test set at the chosen threshold
    # ----------------------------------------------------------------------------
    LOGGER.info("** EVALUATING CALIBRATED XGBoost ON TEST SET **")
    evaluate_at_threshold(calibrated_xgb, X_test, y_test, optimal_threshold)

    # ----------------------------------------------------------------------------
    # 8) Compare alternative samplers (SMOTE, SMOTEN, ADASYN) on validation set (PR-AUC)
    # ----------------------------------------------------------------------------
    LOGGER.info("Comparing alternative samplers (SMOTE, SMOTEN, ADASYN) on validation set ...")
    from sklearn.metrics import precision_recall_curve, auc as auc_score

    sampler_names = ["SMOTE", "SMOTEN", "ADASYN"]
    sampler_results = {}
    for name in sampler_names:
        sampler = get_alternative_sampler(name)
        pipeline = build_xgb_pipeline(sampler=sampler)
        pipeline.fit(X_train, y_train)
        y_proba_val = pipeline.predict_proba(X_val)[:, 1]
        precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_proba_val)
        pr_auc_val = auc_score(recall_vals, precision_vals)
        sampler_results[name] = pr_auc_val
        LOGGER.info(f"{name} on validation PR-AUC: {pr_auc_val:.4f}")
    LOGGER.info(f"Sampler comparison results: {sampler_results}")

    # ----------------------------------------------------------------------------
    # 9) Build & manually evaluate a soft-voting ensemble of calibrated XGB + calibrated RF
    # ----------------------------------------------------------------------------
    LOGGER.info("Building and evaluating soft-voting ensemble (XGB + RF) ...")

    # 9.a) Train & calibrate RandomForest on train/validation splits, just like XGBoost
    smote_for_rf = get_alternative_sampler("SMOTE")
    rf_pipeline = build_rf_pipeline(sampler=smote_for_rf)
    rf_pipeline.fit(X_train, y_train)

    calibrated_rf = CalibratedClassifierCV(
        estimator=rf_pipeline,
        method="isotonic",
        cv="prefit"
    )
    calibrated_rf.fit(X_val, y_val)

    # 9.b) Both `calibrated_xgb` and `calibrated_rf` are fitted. To “vote,” average their probabilities:
    LOGGER.info("** EVALUATING ENSEMBLE ON TEST SET (average of calibrated XGB + RF) **")
    proba_xgb = calibrated_xgb.predict_proba(X_test)[:, 1]
    proba_rf  = calibrated_rf.predict_proba(X_test)[:, 1]
    proba_ensemble = (proba_xgb + proba_rf) / 2.0

    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as auc_score, classification_report

    # Unthresholded metrics:
    roc = roc_auc_score(y_test, proba_ensemble)
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, proba_ensemble)
    pr_auc = auc_score(rec_vals, prec_vals)
    LOGGER.info(f"Ensemble ROC-AUC (unthresholded): {roc:.4f}")
    LOGGER.info(f"Ensemble PR-AUC (unthresholded): {pr_auc:.4f}")

    # Hard predictions at optimal_threshold
    y_pred_ens = (proba_ensemble >= optimal_threshold).astype(int)
    report_ens = classification_report(
        y_test, y_pred_ens, target_names=["Legitimate", "Fraud"], digits=4
    )
    LOGGER.info(f"Ensemble classification report (threshold = {optimal_threshold:.4f}):\n{report_ens}")


if __name__ == "__main__":
    main()
