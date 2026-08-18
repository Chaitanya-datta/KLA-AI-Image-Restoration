#!/usr/bin/env python3
"""
scripts/benchmark_inference.py — isolated inference-speed benchmark,
separate from evaluate.py so evaluate.py's printed numbers reflect a real
submission run while this script gives a controlled, repeated-trial
measurement (per project brief section 19: model load time, warm-up time,
average/median inference time, throughput, parameter count, model size,
proper CUDA synchronization, disk I/O separated from model inference).

    python scripts/benchmark_inference.py --input_dir data/Test_NoisyLR/NoisyLR --weights models/final_model.pth
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference import list_input_npy_files, load_model_for_inference  # noqa: E402
from src.utils import count_parameters, model_size_mb, resolve_device, save_json, sync_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--weights", type=str, default="models/final_model.pth")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--num_warmup", type=int, default=5)
    p.add_argument("--num_trials", type=int, default=50)
    p.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--output_json", type=str, default="artifacts/metrics/inference_benchmark.json")
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
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        sys.exit(1)

    files = list_input_npy_files(args.input_dir)
    if not files:
        print(f"ERROR: no .npy files in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # --- Model load time ---
    t0 = time.perf_counter()
    model = load_model_for_inference(args.weights, config, device)
    load_time = time.perf_counter() - t0

    n_params = count_parameters(model)
    size_mb = model_size_mb(model)

    # --- Pre-load all tensors first: separates disk I/O from inference ---
    arrays = [np.load(f).astype(np.float32) for f in files[: max(args.num_warmup, args.num_trials)]]
    tensors = [torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device) for a in arrays]

    # --- Warm-up (excluded from timed stats) ---
    with torch.inference_mode():
        for t in tensors[: args.num_warmup]:
            _ = model(t)
    sync_device(device)

    # --- Timed trials ---
    trial_tensors = (tensors * (args.num_trials // len(tensors) + 1))[: args.num_trials]
    times = []
    with torch.inference_mode():
        for t in trial_tensors:
            sync_device(device)
            t_start = time.perf_counter()
            _ = model(t)
            sync_device(device)
            times.append(time.perf_counter() - t_start)

    times = np.array(times)
    result = {
        "device": str(device),
        "num_parameters": n_params,
        "model_size_mb": size_mb,
        "model_load_time_sec": load_time,
        "num_warmup": args.num_warmup,
        "num_trials": args.num_trials,
        "mean_inference_ms": float(times.mean() * 1000),
        "median_inference_ms": float(np.median(times) * 1000),
        "p95_inference_ms": float(np.percentile(times, 95) * 1000),
        "std_inference_ms": float(times.std() * 1000),
        "throughput_images_per_sec": float(1.0 / times.mean()),
    }

    save_json(args.output_json, result)
    print("\n--- Inference benchmark ---")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {args.output_json}")
    if device.type == "cpu":
        print("\nNOTE: this run was on CPU. Re-run with a CUDA GPU (e.g. the H100 "
              "benchmarking environment) for representative production numbers — "
              "CPU timings here are for pipeline verification only, not a claim "
              "about H100 performance.")


if __name__ == "__main__":
    main()
