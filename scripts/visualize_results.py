#!/usr/bin/env python3
"""
scripts/visualize_results.py — Degraded | Restored | Ground Truth grid.

    python scripts/visualize_results.py \
        --noisy_dir data/train/NoisyLR --gt_dir data/train/GT \
        --weights models/final_model.pth --num_samples 8 --output artifacts/plots/results_grid.png

If --gt_dir is omitted, only Degraded | Restored is shown (use for the
400-image test set, which has no GT).

Samples are chosen deterministically (fixed seed) and NOT cherry-picked:
this script does not rank or filter by quality, it just samples uniformly
at random from whatever directory it's given, per the "don't cherry-pick
excellent examples" requirement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference import list_input_npy_files, load_model_for_inference, restore_array  # noqa: E402
from src.utils import resolve_device  # noqa: E402
import yaml  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy_dir", type=str, required=True)
    p.add_argument("--gt_dir", type=str, default=None)
    p.add_argument("--weights", type=str, default="models/final_model.pth")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="artifacts/plots/results_grid.png")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    try:
        device = resolve_device(args.device)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.weights).is_file():
        print(f"ERROR: weights not found at {args.weights}. Train a model first.", file=sys.stderr)
        sys.exit(1)
    model = load_model_for_inference(args.weights, config, device)

    files = list_input_npy_files(args.noisy_dir)
    if not files:
        print(f"ERROR: no .npy files in {args.noisy_dir}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(files), size=min(args.num_samples, len(files)), replace=False)
    chosen = [files[i] for i in idx]

    has_gt = args.gt_dir is not None
    n_cols = 3 if has_gt else 2
    fig, axes = plt.subplots(len(chosen), n_cols, figsize=(4 * n_cols, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    for row, fpath in enumerate(chosen):
        fname = Path(fpath).name
        noisy = np.load(fpath).astype(np.float32)
        restored = restore_array(model, noisy, device)

        axes[row, 0].imshow(np.clip(noisy, 0, 1), cmap="gray")
        axes[row, 0].set_title(f"Degraded: {fname}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(restored, cmap="gray")
        axes[row, 1].set_title("Restored")
        axes[row, 1].axis("off")

        if has_gt:
            gt_path = Path(args.gt_dir) / fname
            if gt_path.is_file():
                gt = np.load(gt_path).astype(np.float32)
                axes[row, 2].imshow(np.clip(gt, 0, 1), cmap="gray")
                axes[row, 2].set_title("Ground Truth")
            else:
                axes[row, 2].text(0.5, 0.5, "GT missing", ha="center", va="center")
            axes[row, 2].axis("off")

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
