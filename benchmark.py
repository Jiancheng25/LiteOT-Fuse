"""Measure MFNet or LiteOT-Fuse batch-1 eager inference latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model import LiteOTFuse, MFNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mfnet", "liteot"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ablation", choices=("full", "ot_only", "gate_only", "backbone"), default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.model == "mfnet":
        model = MFNet(9)
    else:
        model = LiteOTFuse(
            use_ot=args.ablation in ("full", "ot_only"),
            use_gate=args.ablation in ("full", "gate_only"),
        )
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    image = torch.zeros(1, 4, 480, 640, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(args.iterations):
            model(image)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "model": args.model,
        "ablation": args.ablation if args.model == "liteot" else None,
        "weights": str(args.weights.resolve()),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "input_shape": list(image.shape),
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "milliseconds_per_image": elapsed / args.iterations * 1000.0,
        "fps": args.iterations / elapsed,
    }
    content = json.dumps(result, indent=2)
    print(content)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
