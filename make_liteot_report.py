"""Create the LiteOT-Fuse ablation CSV and comparison figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/liteot_experiments"
RUNS = {
    "MFNet released": "results/mfnet_reproduction/metrics.json",
    "Backbone fine-tune": "results/liteot_backbone_control/metrics.json",
    "OT only": "results/liteot_ot_only/metrics.json",
    "Gate only": "results/liteot_gate_only/metrics.json",
    "LiteOT joint": "results/liteot_full/metrics.json",
    "LiteOT frozen": "results/liteot_full_frozen/metrics.json",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {name: json.loads((ROOT / path).read_text()) for name, path in RUNS.items()}
    columns = [
        "method", "split", "class_average", "paper_miou", "eq1_class_recall",
        "standard_foreground_miou", "overall_labeled_accuracy", "parameters",
    ]
    with (OUTPUT / "ablation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, result in results.items():
            for split in ("both", "day", "night"):
                value = result[split]
                writer.writerow({
                    "method": name,
                    "split": split,
                    "class_average": value["paper_table_class_average"],
                    "paper_miou": value["paper_mean_iou"],
                    "eq1_class_recall": value["mean_class_recall_eq1"],
                    "standard_foreground_miou": value["standard_mean_iou_foreground"],
                    "overall_labeled_accuracy": value["overall_pixel_accuracy_labeled"],
                    "parameters": result["metadata"]["parameter_count"],
                })

    names = list(results)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    class_average = [results[name]["both"]["paper_table_class_average"] for name in names]
    miou = [results[name]["both"]["paper_mean_iou"] for name in names]
    axes[0].bar(x, class_average, color=["#4c78a8"] + ["#9ecae9"] * 3 + ["#f58518", "#eeca3b"])
    axes[1].bar(x, miou, color=["#4c78a8"] + ["#9ecae9"] * 3 + ["#f58518", "#eeca3b"])
    for axis, title in zip(axes, ("Paper-protocol Class average", "Paper-protocol mIoU")):
        axis.set_xticks(x, names, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.58, 0.67)
    axes[1].set_ylim(0.64, 0.655)
    fig.suptitle("LiteOT-Fuse test-set ablation (validation-selected checkpoints)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "test_ablation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, split in zip(axes, ("both", "day", "night")):
        base = results["MFNet released"][split]
        joint = results["LiteOT joint"][split]
        values = [
            base["paper_table_class_average"], joint["paper_table_class_average"],
            base["paper_mean_iou"], joint["paper_mean_iou"],
        ]
        axis.bar([0, 1], values[:2], color=["#4c78a8", "#f58518"])
        axis.bar([3, 4], values[2:], color=["#4c78a8", "#f58518"])
        axis.set_xticks([0.5, 3.5], ["Class avg.", "mIoU"])
        axis.set_title(split.capitalize())
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.45, 0.68)
    axes[-1].legend(["MFNet", "LiteOT-Fuse"], loc="lower right")
    fig.suptitle("MFNet vs LiteOT-Fuse (joint fine-tuning)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "baseline_vs_liteot.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
