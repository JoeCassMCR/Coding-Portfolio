"""
predictive_maintenance/data_loader.py
--------------------------------------
Functions for loading and preprocessing the CMAPSS dataset.

Corrected: sep uses raw string to avoid invalid escape
"""

import pandas as pd
import logging
from config import TRAIN_FILE

log = logging.getLogger(__name__)

def load_data(path: str = TRAIN_FILE) -> pd.DataFrame:
    cols = ['unit', 'time'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(path, sep=r'\s+', header=None, names=cols)
    return df

def remove_constant_sensors(df: pd.DataFrame) -> (pd.DataFrame, list):
    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
    const = [c for c in sensor_cols if df[c].std() == 0]
    if const:
        log.info(f"Removing constant sensors: {const}")
    df_new = df.drop(columns=const)
    sensor_cols = [c for c in df_new.columns if c.startswith('sensor_')]
    return df_new, sensor_cols