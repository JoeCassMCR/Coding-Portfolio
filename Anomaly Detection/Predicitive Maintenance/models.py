import numpy as np
import pandas as pd
import logging

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
)
from xgboost import XGBRegressor
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from config import (
    IF_N_ESTIMATORS,
    IF_CONTAM_OPTIONS,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT,
)

log = logging.getLogger(__name__)

def train_anomaly_models(X: pd.DataFrame, contamination: float):
    flags = pd.DataFrame(index=X.index)
    raw_scores = {}

    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso.fit(X.values)
    iso_scores = iso.decision_function(X.values)
    raw_scores["IsolationForest"] = iso_scores
    iso_preds = iso.predict(X.values)
    flags["IsolationForest"] = pd.Series(iso_preds, index=X.index).map({1: 0, -1: 1})

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    lof.fit(X.values)
    lof_scores = -lof.decision_function(X.values)
    raw_scores["LOF"] = lof_scores
    cutoff = np.percentile(lof_scores, 100 * (1 - contamination))
    flags["LOF"] = (lof_scores >= cutoff).astype(int)

    ocsvm = OneClassSVM(nu=contamination, kernel="rbf", gamma="auto")
    ocsvm.fit(X.values)
    ocsvm_scores = ocsvm.decision_function(X.values)
    raw_scores["OneClassSVM"] = ocsvm_scores
    ocsvm_preds = ocsvm.predict(X.values)
    flags["OneClassSVM"] = pd.Series(ocsvm_preds, index=X.index).map({1: 0, -1: 1})

    return flags, raw_scores

def tune_isolation_forest(X: pd.DataFrame):
    best_if = None
    best_mean_score = -np.inf
    for n_est in IF_N_ESTIMATORS:
        for cont in IF_CONTAM_OPTIONS:
            iso = IsolationForest(n_estimators=n_est, contamination=cont, random_state=42)
            iso.fit(X.values)
            scores = iso.decision_function(X.values)
            mean_score = np.mean(scores)
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_if = iso
    log.info(f"[Tune IF] Selected n_estimators={best_if.n_estimators}, contamination={best_if.contamination}")
    return best_if

def tune_random_forest(X: pd.DataFrame, y: pd.Series):
    best_rf = None
    best_f1 = -np.inf
    for n_est in RF_N_ESTIMATORS:
        for md in RF_MAX_DEPTH:
            for ms in RF_MIN_SAMPLES_SPLIT:
                rf = RandomForestClassifier(n_estimators=n_est, max_depth=md, min_samples_split=ms, class_weight="balanced", random_state=42)
                rf.fit(X, y)
                preds = rf.predict(X)
                f1 = f1_score(y, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_rf = rf
    log.info(f"[Tune RF] Selected n_estimators={best_rf.n_estimators}, max_depth={best_rf.max_depth}, min_samples_split={best_rf.min_samples_split}")
    return best_rf

def combine_detectors(flags_if: pd.Series, flags_rf: pd.Series) -> pd.Series:
    return ((flags_if == 1) & (flags_rf == 1)).astype(int)

def train_supervised_classifier(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    log.info("\n=== Supervised RandomForest Classification Report ===")
    log.info(f"Confusion Matrix: {confusion_matrix(y_te, y_pred)}")
    log.info(f"Precision (anomaly=1): {precision_score(y_te, y_pred):.4f}")
    log.info(f"Recall (anomaly=1): {recall_score(y_te, y_pred):.4f}")
    log.info(f"F1-Score (anomaly=1): {f1_score(y_te, y_pred):.4f}")
    log.info(f"Accuracy: {accuracy_score(y_te, y_pred):.4f}")
    return rf

def train_rul_model(X: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    log.info(f"RUL Prediction RMSE (GBR): {rmse:.2f}")
    return reg

def train_xgboost_rul(X: pd.DataFrame, y: pd.Series):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_tr, y_tr)
    y_pred = xgb.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    log.info(f"RUL Prediction RMSE (XGBoost): {rmse:.2f}")
    return xgb

def train_nn_rul(X: pd.DataFrame, y: pd.Series):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_dim = X_scaled.shape[1]
    inputs = Input(shape=(input_dim,))
    encoded = Dense(input_dim // 2, activation="relu")(inputs)
    encoded = Dense(input_dim // 4, activation="relu")(encoded)
    decoded = Dense(input_dim // 2, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    autoencoder.fit(X_scaled, X_scaled, epochs=20, batch_size=128, validation_split=0.1, verbose=0)
    encoder = Model(inputs, encoded)
    encoded_feats = encoder.predict(X_scaled)
    X_tr, X_te, y_tr, y_te = train_test_split(encoded_feats, y, test_size=0.2, random_state=42)
    nn_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    nn_reg.fit(X_tr, y_tr)
    y_pred = nn_reg.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    log.info(f"RUL Prediction RMSE (NN + GBR): {rmse:.2f}")
    return autoencoder, nn_reg
