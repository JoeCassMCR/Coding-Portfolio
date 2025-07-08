"""
utils.py

Functions to build pipelines, perform hyperparameter tuning, and serialize models.
"""

import logging
import os
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from joblib import dump, load
from config import (
    RANDOM_STATE,
    HYPERPARAM_GRID,
    CV_SPLITS,
    XGB_EVAL_METRIC,
    XGB_USE_LABEL_ENCODER,
    SMOTE_RANDOM_STATE,
)

LOGGER = logging.getLogger(__name__)


def build_pipeline() -> ImbPipeline:
    """
    Build a pipeline: SMOTE oversampling + XGBoost classifier.
    """
    pipeline = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=SMOTE_RANDOM_STATE)),
            (
                "xgb",
                XGBClassifier(
                    objective="binary:logistic",
                    use_label_encoder=XGB_USE_LABEL_ENCODER,
                    eval_metric=XGB_EVAL_METRIC,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline


def get_hyperparameter_grid() -> dict:
    """
    Returns hyperparameter grid from config.HYPERPARAM_GRID.
    """
    return HYPERPARAM_GRID


def perform_grid_search(pipeline: ImbPipeline, X_train, y_train) -> GridSearchCV:
    """
    Run GridSearchCV with StratifiedKFold on training data.

    Returns
    -------
    grid_search : GridSearchCV
        Fitted GridSearchCV object.
    """
    LOGGER.info("Setting up StratifiedKFold ...")
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=HYPERPARAM_GRID,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        verbose=2,
    )
    LOGGER.info("Starting GridSearchCV ...")
    grid_search.fit(X_train, y_train)
    LOGGER.info(f"Best ROC AUC: {grid_search.best_score_:.4f}")
    LOGGER.info(f"Best Params: {grid_search.best_params_}")
    return grid_search


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
