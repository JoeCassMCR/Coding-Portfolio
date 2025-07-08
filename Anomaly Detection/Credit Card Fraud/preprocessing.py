"""
preprocessing.py

Preprocessing utilities: feature scaling, train/test split, feature engineering.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE

LOGGER = logging.getLogger(__name__)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from 'Time' and 'Amount'.

    - Extract 'hour_of_day' from 'Time' (seconds since first transaction).
    - Create 'log_amount' as log1p of 'Amount'.
    """
    df = df.copy()
    # Convert 'Time' (seconds) to hour of day
    df['hour_of_day'] = ((df['Time'] // 3600) % 24).astype(int)

    # Log transform for 'Amount'
    df['log_amount'] = np.log1p(df['Amount'])

    return df


def preprocess_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Split DataFrame into features and target, perform feature engineering, and scale numeric features.

    - Create 'hour_of_day' and 'log_amount'.
    - Drop the raw 'Time' and 'Amount' columns.
    - Scale numeric features (excluding PCA components which are already scaled).
    """
    df = feature_engineering(df)
    y = df['Class'].values

    # Drop original 'Time' and 'Amount'
    X = df.drop(columns=['Time', 'Amount', 'Class']).copy()

    # Identify columns to scale (including 'log_amount')
    scaler = StandardScaler()
    scale_cols = ['log_amount'] + ['V' + str(i) for i in range(1, 29)]
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    return X.values, y


def split_data(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Stratified train/test split based on config.TEST_SIZE and config.RANDOM_STATE.

    Returns train/test arrays.
    """
    LOGGER.info(f"Performing stratified train/test split (test_size={TEST_SIZE}) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    LOGGER.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
