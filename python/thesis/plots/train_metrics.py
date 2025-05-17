import json
import os
from typing import List, Dict, Any, Tuple

def load_metrics_from_dir(metrics_dir: str) -> Dict[int, Dict[str, Any]]:
    metrics = {}
    for fname in os.listdir(metrics_dir):
        if not fname.startswith("metrics_epoch_") or not fname.endswith(".json"):
            continue
        path = os.path.join(metrics_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)
        epoch = data.get("epoch")
        if epoch is None:
            try:
                epoch = int(fname[len("metrics_epoch_"):-len(".json")])
            except ValueError:
                continue
        metrics[epoch] = data
    return dict(sorted(metrics.items(), key=lambda kv: kv[0]))

def extract_series(
    metrics: Dict[int, Dict[str, Any]],
    metric_names: List[str]
) -> Tuple[List[int], Dict[str, List[float]]]:
    epochs = list(metrics.keys())
    series = {m: [] for m in metric_names}
    for e in epochs:
        data = metrics[e]
        for m in metric_names:
            series[m].append(float(data.get(m, float("nan"))))
    return epochs, series

def plot_series(
    epochs: List[int],
    series: Dict[str, List[float]],
    output_path: str,
    title: str = None
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for metric, values in series.items():
        plt.plot(epochs, values, label=metric)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
