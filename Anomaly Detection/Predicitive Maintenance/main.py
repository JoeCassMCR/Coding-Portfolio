import os
import argparse
import logging
import pandas as pd

from config import (
    BASE_DIR,
    TRAIN_FILE,
    RUL_THRESHOLD,
    GRAPH_DIR,
    IF_CONTAMINATION,
    MODEL_UNIT_TO_PLOT,
)
from data_loader import load_data, remove_constant_sensors
from features import engineer_features, label_true_anomalies
from models import (
    train_anomaly_models,
    tune_isolation_forest,
    tune_random_forest,
    combine_detectors,
    train_supervised_classifier,
    train_rul_model,
    train_xgboost_rul,
    train_nn_rul,
)
from evaluation import (
    classification_reports,
    plot_roc_curves,
    plot_pr_curves,
    detection_delay,
    find_best_pr_threshold,
)
from plotting import plot_anomaly_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "pipeline.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Predictive Maintenance Pipeline")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning steps")
    parser.add_argument("--only-evaluate", action="store_true", help="Only run evaluation & plotting; skip training")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    log.info("1) Loading and preprocessing data...")
    df = load_data(TRAIN_FILE)
    df, sensors = remove_constant_sensors(df)

    log.info("2) Feature engineering and labeling...")
    model_df, feats = engineer_features(df, sensors)
    df = label_true_anomalies(df, threshold=RUL_THRESHOLD)

    model_df = model_df.reset_index(drop=False)
    df = df.reset_index(drop=True)

    log.info("3) Training unsupervised anomaly models...")
    flags, raw_scores = train_anomaly_models(model_df[feats], contamination=IF_CONTAMINATION)
    flags["unit"] = model_df["unit"]
    flags["time"] = model_df["time"]

    log.info("4) Evaluating anomaly detectors...")
    true_labels = df.loc[model_df.index, "true_anomaly"]
    classification_reports(flags.drop(columns=["unit", "time"]), true_labels)
    plot_roc_curves(raw_scores, true_labels)
    plot_pr_curves(raw_scores, true_labels)
    detection_delay(flags.drop(columns=["unit", "time"]), df.loc[model_df.index], true_col="true_anomaly")

    tuned_flags = pd.DataFrame(index=model_df.index)
    for model_name, scores in raw_scores.items():
        best_thr, bp, br, bf1, _, _, _ = find_best_pr_threshold(scores, true_labels)
        tuned_flags[f"{model_name}_tuned"] = (-scores >= best_thr).astype(int)
    classification_reports(tuned_flags, true_labels)

    log.info(f"5) Plotting anomaly comparison for unit={MODEL_UNIT_TO_PLOT}...")
    plot_df = pd.DataFrame({"unit": df.loc[model_df.index, "unit"], "time": df.loc[model_df.index, "time"], "sensors_mean_all": df.loc[model_df.index, "sensors_mean_all"]}).reset_index(drop=True)
    tuned_flags = tuned_flags.reset_index(drop=True)
    plot_anomaly_comparison(plot_df, tuned_flags, unit_id=MODEL_UNIT_TO_PLOT, metric="sensors_mean_all", base_dir=BASE_DIR)

    if not args.skip_tuning:
        log.info("6) Hyperparameter Tuning")
        best_if = tune_isolation_forest(model_df[feats])
        best_rf = tune_random_forest(model_df[feats], df.loc[model_df.index, "true_anomaly"])
    else:
        log.info("Skipping hyperparameter tuning.")

    log.info("7) Creating ensemble of IF and RF predictions...")
    rf_preds = best_rf.predict(model_df[feats])
    ensemble_flags = combine_detectors(tuned_flags["IsolationForest_tuned"], pd.Series(rf_preds, index=tuned_flags.index))

    log.info("8) Training supervised RandomForest classifier...")
    rf_model = train_supervised_classifier(model_df[feats], df.loc[model_df.index, "true_anomaly"])

    log.info("9) Training supervised RUL regressors...")
    train_rul_model(model_df[feats], df.loc[model_df.index, "RUL"])
    train_xgboost_rul(model_df[feats], df.loc[model_df.index, "RUL"])
    train_nn_rul(model_df[feats], df.loc[model_df.index, "RUL"])

    log.info("Pipeline complete.")