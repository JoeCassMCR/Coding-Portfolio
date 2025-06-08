"""
model_utils.py

Functions to build pipelines, perform hyperparameter tuning (GridSearch or RandomizedSearch),
compare models, and serialize models. Also includes helpers for alternative samplers.
"""

import logging
import os
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, SMOTEN, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from joblib import dump, load

from config import (
    RANDOM_STATE,
    HYPERPARAM_GRID,
    RANDOMIZED_HYPERPARAM_DISTRIBUTION,
    CV_SPLITS,
    XGB_EVAL_METRIC,
    SMOTE_RANDOM_STATE,
)

LOGGER = logging.getLogger(__name__)


def build_xgb_pipeline(sampler: ImbPipeline = None) -> ImbPipeline:
    """
    Build a pipeline: (optional sampler) + XGBoost classifier.
    If `sampler` is None, defaults to SMOTE.
    """
    if sampler is None:
        sampler = SMOTE(random_state=SMOTE_RANDOM_STATE)

    pipeline = ImbPipeline(
        steps=[
            ("sampler", sampler),
            (
                "xgb",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric=XGB_EVAL_METRIC,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline


def build_rf_pipeline(sampler: ImbPipeline = None) -> ImbPipeline:
    """
    Build a pipeline: (optional sampler) + RandomForest classifier.
    If `sampler` is None, defaults to SMOTE.
    """
    if sampler is None:
        sampler = SMOTE(random_state=SMOTE_RANDOM_STATE)

    pipeline = ImbPipeline(
        steps=[
            ("sampler", sampler),
            (
                "rf",
                RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
            ),
        ]
    )
    return pipeline


def perform_grid_search_xgb(pipeline: ImbPipeline, X_train, y_train) -> GridSearchCV:
    """
    Run GridSearchCV with StratifiedKFold on training data for XGBoost pipeline.
    """
    LOGGER.info("Setting up StratifiedKFold for GridSearchCV ...")
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=HYPERPARAM_GRID,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        verbose=2,
    )
    LOGGER.info("Starting XGBoost GridSearchCV ...")
    grid_search.fit(X_train, y_train)
    LOGGER.info(f"Best XGBoost ROC AUC: {grid_search.best_score_:.4f}")
    LOGGER.info(f"Best XGBoost Params: {grid_search.best_params_}")
    return grid_search


def compare_default_pipelines(X_train, y_train) -> dict:
    """
    Cross-validate XGBoost (with default SMOTE) vs. RandomForest (with default SMOTE)
    and return their mean ROC-AUC scores.
    """
    from sklearn.model_selection import cross_val_score

    xgb_pipe = build_xgb_pipeline()
    rf_pipe = build_rf_pipeline()

    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    LOGGER.info("Cross-validating XGBoost default pipeline ...")
    xgb_score = cross_val_score(
        xgb_pipe, X_train, y_train, scoring="roc_auc", cv=skf, n_jobs=-1
    ).mean()

    LOGGER.info("Cross-validating RandomForest default pipeline ...")
    rf_score = cross_val_score(
        rf_pipe, X_train, y_train, scoring="roc_auc", cv=skf, n_jobs=-1
    ).mean()

    LOGGER.info(f"XGBoost default ROC AUC: {xgb_score:.4f}")
    LOGGER.info(f"RandomForest default ROC AUC: {rf_score:.4f}")

    return {"xgb_default": xgb_score, "rf_default": rf_score}


def save_model(model, filepath: str) -> None:
    """
    Serialize the trained model to disk using joblib.
    """
    dirname = os.path.dirname(filepath)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    LOGGER.info(f"Saving model to {filepath} ...")
    dump(model, filepath)
    LOGGER.info("Model saved successfully.")


def load_model(filepath: str):
    """
    Load a serialized model from disk.
    """
    LOGGER.info(f"Loading model from {filepath} ...")
    model = load(filepath)
    LOGGER.info("Model loaded successfully.")
    return model


def get_alternative_sampler(name: str):
    """
    Return an imblearn sampler instance by name: "SMOTE", "SMOTEN", or "ADASYN".
    """
    name = name.lower()
    if name == "smote":
        return SMOTE(random_state=SMOTE_RANDOM_STATE)
    elif name == "smoten":
        return SMOTEN(random_state=SMOTE_RANDOM_STATE)
    elif name == "adasyn":
        return ADASYN(random_state=SMOTE_RANDOM_STATE)
    else:
        raise ValueError(f"Unknown sampler: {name}")
