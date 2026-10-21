"""Reproducible MFNet/LiteOT-Fuse training and ablation entry point."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import confusion_update, find_dataset_root, metrics
from model import LiteOTFuse, MFNet
from util.MF_dataset import MF_dataset
from util.augmentation import RandomCrop


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "mfnet":
        return MFNet(n_class=9)
    return LiteOTFuse(
        n_class=9,
        use_ot=args.ablation in ("full", "ot_only"),
        use_gate=args.ablation in ("full", "gate_only"),
        window_size=args.window_size,
        sinkhorn_iterations=args.sinkhorn_iterations,
        ot_embedding_dim=args.ot_embedding_dim,
        spatial_weight=args.spatial_weight,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    max_batches: int | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    if training and getattr(model, "frozen_backbone", False):
        # Frozen BatchNorm running statistics are part of the backbone too.
        # Updating them would silently change the baseline despite frozen
        # parameters, so only the reliability block remains in train mode.
        for child_name, child in model.named_children():
            if child_name not in ("local_ot", "reliability_fusion"):
                child.eval()
    confusion = np.zeros((9, 9), dtype=np.int64)
    loss_sum = 0.0
    sample_count = 0
    rgb_weight_sum = 0.0
    rgb_weight_batches = 0
    started = time.perf_counter()

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (images, labels, _) in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            predictions = logits.detach().argmax(1)
            confusion_update(confusion, labels, predictions)
            batch_size = images.shape[0]
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size
            auxiliary = getattr(model, "last_auxiliary", None)
            if auxiliary and "mean_rgb_weight" in auxiliary:
                rgb_weight_sum += float(auxiliary["mean_rgb_weight"])
                rgb_weight_batches += 1

    result = metrics(confusion)
    result.update({
        "loss": loss_sum / max(sample_count, 1),
        "samples": sample_count,
        "seconds": time.perf_counter() - started,
        "mean_rgb_weight": rgb_weight_sum / max(rgb_weight_batches, 1),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mfnet", "liteot"), default="liteot")
    parser.add_argument("--ablation", choices=("full", "ot_only", "gate_only", "backbone"), default="full")
    parser.add_argument("--data-dir", type=Path, default=Path("data/MF"))
    parser.add_argument("--output-dir", type=Path, default=Path("weights/LiteOTFuse/full"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--lr-decay", type=float, default=0.94)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init-mfnet", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Train only local_ot and reliability_fusion as a lightweight plug-in.",
    )
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--sinkhorn-iterations", type=int, default=3)
    parser.add_argument("--ot-embedding-dim", type=int, default=32)
    parser.add_argument("--spatial-weight", type=float, default=0.20)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Only for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Only for smoke tests.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    data_root = find_dataset_root(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset = MF_dataset(str(data_root), "train", have_label=True, transform=[RandomCrop(0.1, 1.0)])
    val_dataset = MF_dataset(str(data_root), "val", have_label=True)
    common_loader_args = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, generator=generator, **common_loader_args
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common_loader_args)

    model = build_model(args)
    if args.init_mfnet is not None:
        if args.model != "liteot":
            raise ValueError("--init-mfnet is only meaningful for --model liteot")
        state = torch.load(args.init_mfnet, map_location="cpu", weights_only=True)
        model.load_mfnet_backbone(state)
    if args.freeze_backbone:
        if args.model != "liteot":
            raise ValueError("--freeze-backbone requires --model liteot")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(("local_ot.", "reliability_fusion."))
        model.frozen_backbone = True
    model = model.to(device)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    best_miou = -1.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint["best_miou"])

    history_path = args.output_dir / "history.csv"
    fieldnames = [
        "epoch", "lr", "train_loss", "train_miou", "val_loss", "val_miou",
        "val_class_average", "train_seconds", "val_seconds", "mean_rgb_weight",
    ]
    if not history_path.exists() or start_epoch == 1:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    print(
        f"model={args.model}/{args.ablation} parameters={sum(p.numel() for p in model.parameters()):,} "
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"data={data_root} train={len(train_dataset)} val={len(val_dataset)} device={device}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_result = run_epoch(
            model, train_loader, device, optimizer, scaler, args.max_train_batches
        )
        val_result = run_epoch(
            model, val_loader, device, None, None, args.max_val_batches
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_result["loss"],
            "train_miou": train_result["paper_mean_iou"],
            "val_loss": val_result["loss"],
            "val_miou": val_result["paper_mean_iou"],
            "val_class_average": val_result["paper_table_class_average"],
            "train_seconds": train_result["seconds"],
            "val_seconds": val_result["seconds"],
            "mean_rgb_weight": val_result["mean_rgb_weight"],
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)

        improved = val_result["paper_mean_iou"] > best_miou
        if improved:
            best_miou = val_result["paper_mean_iou"]
            torch.save(model.state_dict(), args.output_dir / "best.pth")
        checkpoint = {
            "epoch": epoch, "best_miou": best_miou, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        torch.save(checkpoint, args.output_dir / "last.ckpt")
        scheduler.step()
        print(
            f"epoch {epoch:03d}/{args.epochs} lr={row['lr']:.6f} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_mIoU={row['val_miou']:.4f} val_class_avg={row['val_class_average']:.4f} "
            f"rgb_gate={row['mean_rgb_weight']:.3f} best={best_miou:.4f}"
        )

    torch.save(model.state_dict(), args.output_dir / "final.pth")


if __name__ == "__main__":
    # Avoid excessive CPU oversubscription on multi-GPU servers.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
