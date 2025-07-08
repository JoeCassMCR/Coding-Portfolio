"""
evaluation.py

Functions to evaluate a trained model on test data, tune thresholds, and evaluate at a custom threshold.
"""

import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc as auc_score,
    classification_report,
    confusion_matrix,
)

LOGGER = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test) -> None:
    """
    Evaluate the final model on the hold-out test set at the default 0.5 threshold.

    Prints:
      - ROC AUC
      - Precision-Recall AUC
      - Classification report at 0.5 threshold
    """
    LOGGER.info("Evaluating model on test set (threshold = 0.5) ...")
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    LOGGER.info(f"Test ROC AUC: {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc_score(recall, precision)
    LOGGER.info(f"Test Precision-Recall AUC: {pr_auc:.4f}")

    y_pred = (y_proba >= 0.5).astype(int)
    report = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Fraud"], digits=4
    )
    LOGGER.info("Classification Report (threshold=0.5):\n" + report)


def tune_threshold(model, X_val, y_val, cost_fp: float = 1.0, cost_fn: float = 10.0) -> float:
    """
    Find optimal probability threshold on validation set to minimize expected cost:
      cost = cost_fp * (# false positives) + cost_fn * (# false negatives)
    """
    LOGGER.info("Tuning probability threshold based on custom cost function ...")
    y_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    best_threshold = 0.5
    best_cost = float("inf")

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    LOGGER.info(f"Optimal threshold: {best_threshold:.4f} with cost: {best_cost:.1f}")
    return best_threshold


def evaluate_at_threshold(model, X, y, threshold: float) -> None:
    """
    Evaluate model performance at a specified probability threshold.

    Prints:
      - ROC AUC (regardless of threshold)
      - Precision-Recall AUC (regardless of threshold)
      - Classification report at the given threshold
    """
    y_proba = model.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, y_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y, y_proba)
    pr_auc = auc_score(recall_vals, precision_vals)
    LOGGER.info(f"ROC AUC (unthresholded): {roc_auc:.4f}")
    LOGGER.info(f"PR AUC (unthresholded): {pr_auc:.4f}")

    y_pred = (y_proba >= threshold).astype(int)
    report = classification_report(
        y, y_pred, target_names=["Legitimate", "Fraud"], digits=4
    )
    LOGGER.info(f"Classification Report (threshold = {threshold:.4f}):\n{report}")
