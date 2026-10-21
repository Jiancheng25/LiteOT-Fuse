"""Create a paper-style qualitative comparison for MFNet experiments.

The default sample selection is deterministic and performance-agnostic: the
ordered day and night test sequences are each split into equal temporal bins,
then the label-richest frame in every bin is selected.  This gives broad scene
coverage without selecting examples according to either model's prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from evaluate import CLASS_NAMES, find_dataset_root
from model import LiteOTFuse, MFNet
from util.util import get_palette


DISPLAY_NAMES = [
    "Unlabeled", "Car", "Pedestrian", "Bike", "Curve", "Car stop",
    "Guardrail", "Color cone", "Bump",
]


def colourise(label: np.ndarray) -> np.ndarray:
    """Map MFNet class IDs to the official RGB palette."""
    palette = get_palette()
    safe_label = np.where((label >= 0) & (label < len(palette)), label, 0)
    return palette[safe_label].astype(np.uint8)


def label_richness(label: np.ndarray) -> tuple[float, float, float]:
    """Rank frames using annotation content only, never model performance."""
    foreground = label > 0
    foreground_ratio = float(foreground.mean())
    visible_classes = int(np.unique(label[foreground]).size) if foreground.any() else 0
    # A weak compactness term avoids choosing a frame for a handful of isolated
    # annotation pixels while still favouring rare, multi-class street scenes.
    substantial_ratio = min(foreground_ratio, 0.20) / 0.20
    return visible_classes, substantial_ratio, foreground_ratio


def select_samples(root: Path, count: int) -> list[str]:
    """Select equally many day/night frames from fixed temporal bins."""
    if count < 2 or count % 2:
        raise ValueError("--samples must be an even integer of at least 2")

    test_names = [name.strip() for name in (root / "test.txt").read_text().splitlines() if name.strip()]
    selected_by_split: dict[str, list[str]] = {}
    per_split = count // 2
    for suffix in ("D", "N"):
        names = [name for name in test_names if name.upper().endswith(suffix)]
        if len(names) < per_split:
            raise ValueError(f"Only {len(names)} test samples end in {suffix}; need {per_split}")
        bins = np.array_split(np.asarray(names, dtype=object), per_split)
        chosen: list[str] = []
        for temporal_bin in bins:
            chosen.append(max(
                temporal_bin.tolist(),
                key=lambda name: label_richness(
                    np.asarray(Image.open(root / "labels" / f"{name}.png"))
                ),
            ))
        selected_by_split[suffix] = chosen

    # Alternating day/night makes the lighting comparison easy to scan.
    return [
        name
        for pair in zip(selected_by_split["D"], selected_by_split["N"])
        for name in pair
    ]


def parse_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("provide at least one comma-separated sample name")
    return names


def load_model(model: torch.nn.Module, weights: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/MF"))
    parser.add_argument("--mfnet-weights", type=Path, default=Path("weights/MFNet/final.pth"))
    parser.add_argument(
        "--liteot-weights", type=Path,
        default=Path("weights/LiteOTFuse/full/best.pth"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/liteot_experiments/qualitative_comparison.png"),
    )
    parser.add_argument("--samples", type=int, default=8, help="Even number; half day and half night.")
    parser.add_argument(
        "--names", type=parse_names,
        help="Optional comma-separated test IDs; overrides automatic selection.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    root = find_dataset_root(args.data_dir)
    names = args.names if args.names else select_samples(root, args.samples)
    for name in names:
        for folder in ("images", "labels"):
            path = root / folder / f"{name}.png"
            if not path.is_file():
                raise FileNotFoundError(path)

    device = torch.device(args.device)
    baseline = load_model(MFNet(len(CLASS_NAMES)), args.mfnet_weights, device)
    liteot = load_model(LiteOTFuse(n_class=len(CLASS_NAMES)), args.liteot_weights, device)

    rows = ("RGB", "Thermal", "Ground Truth", "LiteOT-Fuse (Ours)", "MFNet")
    panels: list[list[np.ndarray]] = [[] for _ in rows]
    with torch.inference_mode():
        for index, name in enumerate(names, start=1):
            image = np.asarray(Image.open(root / "images" / f"{name}.png"))
            label = np.asarray(Image.open(root / "labels" / f"{name}.png"))
            tensor = torch.from_numpy(
                image.astype(np.float32).transpose(2, 0, 1) / 255.0
            )[None].to(device)
            prediction_baseline = baseline(tensor).argmax(1)[0].cpu().numpy()
            prediction_liteot = liteot(tensor).argmax(1)[0].cpu().numpy()
            panels[0].append(image[:, :, :3])
            panels[1].append(image[:, :, 3])
            panels[2].append(colourise(label))
            panels[3].append(colourise(prediction_liteot))
            panels[4].append(colourise(prediction_baseline))
            print(f"Rendered {index}/{len(names)}: {name}")

    column_width = 2.25
    fig, axes = plt.subplots(
        len(rows), len(names),
        figsize=(1.15 + column_width * len(names), 10.0),
        gridspec_kw={"wspace": 0.025, "hspace": 0.055},
        squeeze=False,
    )
    for row_index, row_name in enumerate(rows):
        for column_index, name in enumerate(names):
            axis = axes[row_index, column_index]
            if row_index == 1:
                axis.imshow(panels[row_index][column_index], cmap="gray", vmin=0, vmax=255)
            else:
                axis.imshow(panels[row_index][column_index])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column_index == 0:
                axis.set_ylabel(row_name, fontsize=13, fontweight="bold", labelpad=11)
            if row_index == 0:
                lighting = "Day" if name.upper().endswith("D") else "Night"
                axis.set_title(f"{lighting}\n{name}", fontsize=13, fontweight="semibold", pad=7)

    palette = get_palette() / 255.0
    handles = [
        mpatches.Patch(facecolor=palette[index], edgecolor="0.6" if index == 0 else "none", label=label)
        for index, label in enumerate(DISPLAY_NAMES)
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles), frameon=False,
        bbox_to_anchor=(0.53, 0.014), fontsize=12, handlelength=1.25,
        handleheight=1.25, columnspacing=1.35,
    )
    fig.suptitle(
        "Qualitative Comparison on the MFNet RGB-T Test Set",
        fontsize=16, fontweight="bold", y=0.994,
    )
    fig.text(
        0.995, 0.008,
        "Deterministic selection: balanced day/night temporal bins; ranked by ground-truth content only.",
        ha="right", va="bottom", fontsize=7.5, color="0.42",
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.925, bottom=0.085)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {args.output.resolve()}")
    print(f"Samples: {','.join(names)}")


if __name__ == "__main__":
    main()
