import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import VarianceThreshold
from config import ROLLING_WINDOW, RUL_THRESHOLD

log = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame, sensor_cols: list) -> (pd.DataFrame, list):
    for s in sensor_cols:
        df[f"{s}_rm"] = df.groupby("unit")[s].rolling(ROLLING_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{s}_rs"] = df.groupby("unit")[s].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
        df[f"{s}_lagdiff"] = df.groupby("unit")[s].diff().fillna(0)
        df[f"{s}_slope2"] = df.groupby("unit")[s].diff(periods=2).fillna(0)
        df[f"{s}_accel"] = df.groupby("unit")[s].diff().diff().fillna(0)
    if "sensor_2" in df.columns and "sensor_3" in df.columns:
        df["sensor_2_3_ratio"] = df["sensor_2"] / (df["sensor_3"] + 1e-5)
    df["sensors_mean_all"] = df[[c for c in df.columns if c.endswith("_rm")]].mean(axis=1)
    df["sensors_std_all"] = df[[c for c in df.columns if c.endswith("_rs")]].mean(axis=1)

    feat_cols = [c for c in df.columns if c.endswith(("_rm", "_rs", "_lagdiff", "_slope2", "_accel", "_ratio", "_all"))]
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(df[feat_cols])
    high_var = [f for f, keep in zip(feat_cols, selector.get_support()) if keep]

    corr_matrix = df[high_var].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    selected_feats = [f for f in high_var if f not in to_drop]

    model_df = df[["unit", "time"] + selected_feats].copy()
    log.info(f"Feature engineering: kept {len(selected_feats)} features out of {len(feat_cols)} after filtering.")
    return model_df, selected_feats

def label_true_anomalies(df: pd.DataFrame, threshold: int = None) -> pd.DataFrame:
    if threshold is None:
        threshold = RUL_THRESHOLD
    df["max_time"] = df.groupby("unit")["time"].transform("max")
    df["RUL"] = df["max_time"] - df["time"]
    df["true_anomaly"] = (df["RUL"] <= threshold).astype(int)
    return df