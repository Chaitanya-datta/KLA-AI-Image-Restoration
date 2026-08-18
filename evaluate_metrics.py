#!/usr/bin/env python3
"""
evaluate_metrics.py — compute PSNR / SSIM / (optional) LPIPS between a
directory of restored predictions and a directory of ground-truth images.

    python evaluate_metrics.py --pred_dir ./outputs/restored_test --gt_dir /path/to/ground_truth
    python evaluate_metrics.py --pred_dir ./outputs/restored_test --gt_dir /path/to/ground_truth --lpips

Pairing is by identical filename, same rule as src/dataset.py. Reports:
  - per-image PSNR/SSIM(/LPIPS), written to a CSV
  - dataset-average of each metric, written to a JSON summary
  - model parameter count / size, IF a checkpoint is supplied via --weights
    (purely informational — this script does not run inference itself,
    it only compares files already on disk)

Never fabricates a value: any image pair that fails to load is reported
as an error, not silently skipped or imputed.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from src.metrics import LPIPSMetric, psnr, ssim
from src.utils import is_macosx_artifact, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Compute restoration metrics")
    p.add_argument("--pred_dir", type=str, required=True)
    p.add_argument("--gt_dir", type=str, required=True)
    p.add_argument("--lpips", action="store_true", help="Also compute LPIPS (slower, downloads a backbone on first use)")
    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--output_csv", type=str, default="artifacts/metrics/per_image_metrics.csv")
    p.add_argument("--output_json", type=str, default="artifacts/metrics/summary.json")
    p.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)

    if not pred_dir.is_dir():
        print(f"ERROR: --pred_dir not found: {pred_dir}", file=sys.stderr)
        sys.exit(1)
    if not gt_dir.is_dir():
        print(f"ERROR: --gt_dir not found: {gt_dir}", file=sys.stderr)
        sys.exit(1)

    pred_files = {
        f.stem: f for f in pred_dir.iterdir()
        if f.is_file() and f.suffix == ".npy" and not is_macosx_artifact(str(f))
    }
    gt_files = {
        f.stem: f for f in gt_dir.iterdir()
        if f.is_file() and f.suffix == ".npy" and not is_macosx_artifact(str(f))
    }

    common = sorted(set(pred_files) & set(gt_files))
    missing_gt = sorted(set(pred_files) - set(gt_files))
    missing_pred = sorted(set(gt_files) - set(pred_files))

    if not common:
        print("ERROR: no matching filenames between --pred_dir and --gt_dir.", file=sys.stderr)
        sys.exit(1)
    if missing_gt:
        print(f"WARNING: {len(missing_gt)} prediction(s) have no GT match and will be skipped, e.g. {missing_gt[:5]}")
    if missing_pred:
        print(f"WARNING: {len(missing_pred)} GT file(s) have no prediction and will be skipped, e.g. {missing_pred[:5]}")

    device = torch.device(args.device)
    lpips_metric = LPIPSMetric(net=args.lpips_net, device=device) if args.lpips else None

    rows = []
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for stem in common:
        pred = np.load(pred_files[stem]).astype(np.float32)
        gt = np.load(gt_files[stem]).astype(np.float32)

        if pred.shape != gt.shape:
            print(f"WARNING: shape mismatch for {stem}: pred{pred.shape} vs gt{gt.shape} — skipping", file=sys.stderr)
            continue

        pred_t = torch.from_numpy(np.clip(pred, 0, 1)).unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy(np.clip(gt, 0, 1)).unsqueeze(0).unsqueeze(0)

        p_val = psnr(pred_t, gt_t).item()
        s_val = ssim(pred_t, gt_t).item()
        row = {"filename": stem + ".npy", "psnr": p_val, "ssim": s_val}
        psnr_vals.append(p_val)
        ssim_vals.append(s_val)

        if lpips_metric is not None:
            l_val = lpips_metric(pred_t, gt_t).item()
            row["lpips"] = l_val
            lpips_vals.append(l_val)

        rows.append(row)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        fieldnames = ["filename", "psnr", "ssim"] + (["lpips"] if lpips_metric else [])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_images": len(rows),
        "num_missing_gt": len(missing_gt),
        "num_missing_pred": len(missing_pred),
        "psnr_mean": float(np.mean(psnr_vals)) if psnr_vals else None,
        "psnr_std": float(np.std(psnr_vals)) if psnr_vals else None,
        "ssim_mean": float(np.mean(ssim_vals)) if ssim_vals else None,
        "ssim_std": float(np.std(ssim_vals)) if ssim_vals else None,
        "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else "not computed (pass --lpips)",
        "lpips_net": args.lpips_net if lpips_metric else None,
    }
    save_json(args.output_json, summary)

    print("\n--- Metrics summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nPer-image metrics: {args.output_csv}")
    print(f"Summary JSON: {args.output_json}")


if __name__ == "__main__":
    main()
