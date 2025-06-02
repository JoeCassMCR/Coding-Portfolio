import os
import matplotlib.pyplot as plt
from config import GRAPH_DIR

def plot_anomaly_comparison(df_plot, flags, unit_id: int, metric: str, base_dir: str):
    df_unit = df_plot[df_plot["unit"] == unit_id].copy()
    flags_unit = flags.loc[df_unit.index]
    plt.figure(figsize=(12, 6))
    plt.plot(df_unit["time"], df_unit[metric], label=metric, linewidth=2, color="black")
    colors = ["red", "blue", "green", "purple"]
    for i, model_name in enumerate(flags.columns):
        times = df_unit.loc[flags_unit[model_name] == 1, "time"]
        vals = df_unit.loc[flags_unit[model_name] == 1, metric]
        plt.scatter(times, vals, s=50, edgecolor="k", label=model_name, alpha=0.7, color=colors[i % len(colors)])
    plt.title(f"Unit {unit_id}: {metric} with Anomaly Flags")
    plt.xlabel("Time (Cycles)")
    plt.ylabel(metric)
    plt.legend(loc="upper left")
    plt.grid(True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    file_path = os.path.join(GRAPH_DIR, f"unit_{unit_id}_anomaly_comparison.png")
    plt.savefig(file_path, dpi=100)
    plt.close()
    print(f"[INFO] Saved anomaly comparison plot to: {file_path}")