import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from config import GRAPH_DIR

def classification_reports(flags_df, true_series):
    for model_name in flags_df.columns:
        print(f"\n=== {model_name} Classification Report ===")
        y_pred = flags_df[model_name]
        print("Confusion Matrix:")
        print(confusion_matrix(true_series, y_pred))
        print("\n" + classification_report(true_series, y_pred, digits=4))

def plot_roc_curves(raw_scores_dict, true_series):
    plt.figure(figsize=(8, 6))
    for name, scores in raw_scores_dict.items():
        anomaly_scores = -scores
        fpr, tpr, _ = roc_curve(true_series, anomaly_scores)
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={model_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Anomaly Models")
    plt.legend(loc="lower right")
    os.makedirs(GRAPH_DIR, exist_ok=True)
    plt.savefig(os.path.join(GRAPH_DIR, "roc_curves.png"), dpi=100)
    plt.close()

def plot_pr_curves(raw_scores_dict, true_series):
    plt.figure(figsize=(8, 6))
    for name, scores in raw_scores_dict.items():
        anomaly_scores = -scores
        precision, recall, _ = precision_recall_curve(true_series, anomaly_scores)
        avg_prec = average_precision_score(true_series, anomaly_scores)
        plt.plot(recall, precision, label=f"{name} (AP={avg_prec:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for Anomaly Models")
    plt.legend(loc="lower left")
    os.makedirs(GRAPH_DIR, exist_ok=True)
    plt.savefig(os.path.join(GRAPH_DIR, "pr_curves.png"), dpi=100)
    plt.close()

def detection_delay(flags_df, full_df, true_col="true_anomaly"):
    print("\n--- Detection Delay (in cycles) per Model ---")
    grouped = full_df.groupby("unit")
    delays_dict = {}
    for model_name in flags_df.columns:
        delays = []
        for unit_id, group in grouped:
            true_times = group.loc[group[true_col] == 1, "time"]
            if true_times.empty:
                continue
            t_true = true_times.min()
            unit_indices = group.index
            pred_indices = flags_df.loc[unit_indices, model_name][flags_df.loc[unit_indices, model_name] == 1].index
            if pred_indices.empty:
                continue
            t_pred = full_df.loc[pred_indices, "time"].min()
            delays.append(t_true - t_pred)
        if delays:
            avg_delay = np.mean(delays)
            delays_dict[model_name] = avg_delay
            print(f"{model_name:15s}: Average delay = {avg_delay:.1f} cycles")
    plt.figure(figsize=(8, 6))
    names = list(delays_dict.keys())
    values = [delays_dict[name] for name in names]
    plt.bar(names, values, color="skyblue")
    plt.ylabel("Avg Detection Delay (cycles)")
    plt.title("Detection Delay by Model")
    os.makedirs(GRAPH_DIR, exist_ok=True)
    plt.savefig(os.path.join(GRAPH_DIR, "detection_delay.png"), dpi=100)
    plt.close()

def find_best_pr_threshold(scores, true_labels):
    anomaly_scores = -scores
    precision, recall, thresholds = precision_recall_curve(true_labels, anomaly_scores)
    f1_arr = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.nanargmax(f1_arr[:-1])
    return thresholds[best_idx], precision[best_idx], recall[best_idx], f1_arr[best_idx], precision, recall, thresholds