"""Reproduce the MFNet test-set metrics with the released checkpoint.

The paper and the original repository disagree about the denominator used for
"class accuracy".  For reproducibility this script reports both:

* paper_table_class_score: column-normalised confusion matrix (precision), as
  implemented by the released repository and used by Table II.
* class_recall: row-normalised confusion matrix, matching Eq. (1)'s wording.

It also reports the paper's foreground-only IoU protocol and conventional IoU.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model import LiteOTFuse, MFNet
from util.MF_dataset import MF_dataset


CLASS_NAMES = [
    "unlabeled", "car", "pedestrian", "bike", "curve", "car_stop",
    "guardrail", "color_cone", "bump",
]

PAPER_TABLE = {
    "both": [0.968, 0.829, 0.852, 0.742, 0.615, 0.273, 0.000, 0.607, 0.433, 0.591, 0.649],
    "day": [0.969, 0.885, 0.830, 0.740, 0.379, 0.232, 0.000, 0.258, 0.003, 0.477, 0.574],
    "night": [0.967, 0.557, 0.910, 0.752, 0.772, 0.325, 0.000, 0.724, 0.710, 0.635, 0.621],
}


def confusion_update(confusion: np.ndarray, labels: torch.Tensor, predictions: torch.Tensor) -> None:
    valid = (labels >= 0) & (labels < len(CLASS_NAMES))
    encoded = labels[valid] * len(CLASS_NAMES) + predictions[valid]
    counts = torch.bincount(encoded, minlength=len(CLASS_NAMES) ** 2)
    confusion += counts.reshape(len(CLASS_NAMES), len(CLASS_NAMES)).cpu().numpy()


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def metrics(confusion: np.ndarray) -> dict:
    diagonal = np.diag(confusion).astype(np.float64)
    # This reproduces util.calculate_result and the values in Table II.
    paper_table_class_score = safe_divide(diagonal, confusion.sum(axis=0))
    class_recall = safe_divide(diagonal, confusion.sum(axis=1))

    fg = confusion[1:, 1:]
    fg_diagonal = np.diag(fg).astype(np.float64)
    paper_iou = safe_divide(fg_diagonal, fg.sum(axis=0) + fg.sum(axis=1) - fg_diagonal)

    standard_iou_all = safe_divide(
        diagonal, confusion.sum(axis=0) + confusion.sum(axis=1) - diagonal
    )
    labeled_total = confusion[1:, :].sum()
    overall_accuracy = float(diagonal[1:].sum() / labeled_total) if labeled_total else 0.0

    return {
        "overall_pixel_accuracy_labeled": overall_accuracy,
        "paper_table_class_score": paper_table_class_score.tolist(),
        "paper_table_class_average": float(paper_table_class_score.mean()),
        "class_recall_eq1": class_recall.tolist(),
        "mean_class_recall_eq1": float(class_recall.mean()),
        "paper_foreground_iou": paper_iou.tolist(),
        "paper_mean_iou": float(paper_iou.mean()),
        "standard_iou_all_classes": standard_iou_all.tolist(),
        "standard_mean_iou_foreground": float(standard_iou_all[1:].mean()),
        "confusion_matrix": confusion.astype(np.int64).tolist(),
    }


def find_dataset_root(path: Path) -> Path:
    candidates = [path] + [p for p in path.rglob("test.txt")]
    for candidate in candidates:
        root = candidate.parent if candidate.is_file() else candidate
        if (root / "test.txt").is_file() and (root / "images").is_dir() and (root / "labels").is_dir():
            return root
    raise FileNotFoundError(
        f"Could not find test.txt, images/, and labels/ under {path.resolve()}"
    )


def write_outputs(output_dir: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    columns = CLASS_NAMES + ["class_average", "mean_iou"]
    with (output_dir / "paper_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "source", *columns])
        for split in ("both", "day", "night"):
            measured = results[split]
            writer.writerow([
                split, "reproduced",
                *measured["paper_table_class_score"],
                measured["paper_table_class_average"], measured["paper_mean_iou"],
            ])
            writer.writerow([split, "paper", *PAPER_TABLE[split]])

    try:
        import matplotlib.pyplot as plt

        labels = ["Class avg.", "mIoU"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for axis, split in zip(axes, ("both", "day", "night")):
            measured = results[split]
            reproduced = [measured["paper_table_class_average"], measured["paper_mean_iou"]]
            paper = PAPER_TABLE[split][-2:]
            x = np.arange(len(labels))
            axis.bar(x - 0.18, paper, 0.36, label="Paper")
            axis.bar(x + 0.18, reproduced, 0.36, label="Reproduced")
            axis.set_xticks(x, labels)
            axis.set_title(split.capitalize())
            axis.set_ylim(0, 0.75)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Score")
        axes[-1].legend()
        fig.suptitle("MFNet released checkpoint: paper vs reproduced")
        fig.tight_layout()
        fig.savefig(output_dir / "paper_comparison.png", dpi=180)
        plt.close(fig)

        full_labels = [
            "Unlabeled", "Car", "Pedestrian", "Bike", "Curve", "Car stop",
            "Guardrail", "Color cone", "Bump", "Class avg.", "mIoU",
        ]
        fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True, sharey=True)
        for axis, split in zip(axes, ("both", "day", "night")):
            measured = results[split]
            reproduced = [
                *measured["paper_table_class_score"],
                measured["paper_table_class_average"], measured["paper_mean_iou"],
            ]
            paper = PAPER_TABLE[split]
            x = np.arange(len(full_labels))
            axis.bar(x - 0.2, paper, 0.4, label="Paper Table II")
            axis.bar(x + 0.2, reproduced, 0.4, label="Reproduced")
            axis.set_title(split.capitalize())
            axis.set_ylabel("Score")
            axis.set_ylim(0, 1.08)
            axis.grid(axis="y", alpha=0.25)
        axes[-1].set_xticks(np.arange(len(full_labels)), full_labels, rotation=25, ha="right")
        axes[0].legend(ncol=2)
        fig.suptitle("MFNet Table II: per-class and aggregate comparison")
        fig.tight_layout()
        fig.savefig(output_dir / "paper_table2_full_comparison.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/MF"))
    parser.add_argument("--weights", type=Path, default=Path("weights/MFNet/final.pth"))
    parser.add_argument("--model", choices=("mfnet", "liteot"), default="mfnet")
    parser.add_argument(
        "--ablation", choices=("full", "ot_only", "gate_only", "backbone"), default="full",
        help="LiteOT-Fuse component switch used for ablation studies.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/mfnet_reproduction"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root = find_dataset_root(args.data_dir)
    device = torch.device(args.device)
    if args.model == "mfnet":
        model = MFNet(n_class=len(CLASS_NAMES))
    else:
        model = LiteOTFuse(
            n_class=len(CLASS_NAMES),
            use_ot=args.ablation in ("full", "ot_only"),
            use_gate=args.ablation in ("full", "gate_only"),
        )
    model = model.to(device)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dataset = MF_dataset(str(data_root), "test", have_label=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    confusions = {key: np.zeros((9, 9), dtype=np.int64) for key in ("both", "day", "night")}
    elapsed = 0.0

    with torch.inference_mode():
        for batch_index, (images, labels, names) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            predictions = model(images).argmax(1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started

            labels = labels.to(device, non_blocking=True)
            confusion_update(confusions["both"], labels, predictions)
            for index, name in enumerate(names):
                split = "day" if str(name).upper().endswith("D") else "night"
                confusion_update(confusions[split], labels[index], predictions[index])
            print(f"evaluated {min(batch_index * args.batch_size, len(dataset))}/{len(dataset)}", end="\r")

    results = {split: metrics(value) for split, value in confusions.items()}
    results["metadata"] = {
        "dataset_root": str(data_root.resolve()),
        "weights": str(args.weights.resolve()),
        "model": args.model,
        "ablation": args.ablation if args.model == "liteot" else None,
        "device": str(device),
        "torch_version": torch.__version__,
        "test_images": len(dataset),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "timed_model_fps_batch_mode": len(dataset) / elapsed,
    }
    write_outputs(args.output_dir, results)
    print("\n" + json.dumps({
        split: {
            "class_average": results[split]["paper_table_class_average"],
            "mean_iou": results[split]["paper_mean_iou"],
        } for split in ("both", "day", "night")
    }, indent=2))


if __name__ == "__main__":
    main()
