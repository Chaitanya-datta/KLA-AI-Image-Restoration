#!/usr/bin/env python3
"""
evaluate.py — THE submission entry point.

Runs restoration inference on every .npy image in --input_dir and writes
the restored 256x256 output (same filename) to --output_dir.

    python evaluate.py --input_dir /path/to/test_images --output_dir /path/to/restored_outputs

Design constraints satisfied (per project brief section 14):
  - CLI via argparse, no notebook, no training, no internet required.
  - Loads --weights (defaults to models/final_model.pth) automatically.
  - Uses CUDA automatically if available, otherwise CPU — no manual flag
    required (though --device lets you force one).
  - torch.inference_mode() for the actual forward passes.
  - AMP autocast on CUDA.
  - Reports total inference time, average time/image, and throughput,
    with model load / warm-up time reported separately from pure inference
    time so the printed numbers aren't inflated by one-time costs.
  - Ignores __MACOSX/ and AppleDouble '._*' files automatically.
  - Creates --output_dir if missing.
  - Clear, actionable errors (missing dir, missing weights, empty dir,
    corrupt .npy) instead of stack traces.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.inference import list_input_npy_files, load_model_for_inference, restore_array
from src.utils import resolve_device, sync_device


def parse_args():
    p = argparse.ArgumentParser(description="Run KLA image-restoration inference")
    p.add_argument("--input_dir", type=str, required=True, help="Directory of degraded NoisyLR .npy images")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write restored .npy images to")
    p.add_argument("--weights", type=str, default="models/final_model.pth", help="Path to trained model weights")
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Model architecture config")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"], help="Force device; default: auto (CUDA > MPS > CPU)")
    p.add_argument("--amp", action="store_true", default=None, help="Force-enable mixed precision on CUDA")
    p.add_argument("--save_png_preview", action="store_true",
                    help="Also save a viewable .png alongside each .npy (for quick visual spot-checks)")
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: --input_dir does not exist or is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        print(
            f"ERROR: model weights not found at {weights_path}.\n"
            f"  Train a model first (`python train.py --config {args.config} --data_root <FULL_KLA_DATASET>/train`)\n"
            f"  then place/point at the resulting checkpoint, e.g.:\n"
            f"  python evaluate.py --input_dir ... --output_dir ... --weights artifacts/checkpoints/best.pth",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        device = resolve_device(args.device)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    use_amp = config.get("inference", {}).get("amp", True) if args.amp is None else args.amp

    print(f"Device: {device}")
    print(f"Weights: {weights_path}")

    t_load_start = time.perf_counter()
    model = load_model_for_inference(str(weights_path), config, device)
    load_time = time.perf_counter() - t_load_start
    print(f"Model load time: {load_time:.3f}s")

    files = list_input_npy_files(str(input_dir))
    if not files:
        print(f"ERROR: no .npy files found in {input_dir} (checked, ignoring __MACOSX/._* junk)", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} input image(s) in {input_dir}")

    # Warm-up pass (excluded from timed inference) — first CUDA call pays
    # kernel-compilation/allocator costs that would otherwise skew the
    # per-image timing average.
    sample = np.load(files[0])
    with torch.inference_mode():
        _ = restore_array(model, sample, device, use_amp=use_amp)
    sync_device(device)

    per_image_times = []
    errors = []
    t_total_start = time.perf_counter()

    for fpath in files:
        fname = Path(fpath).name
        try:
            arr = np.load(fpath)
        except Exception as e:  # noqa: BLE001 - surface a clear per-file error, keep going
            errors.append((fname, f"failed to load: {e}"))
            continue

        t0 = time.perf_counter()
        with torch.inference_mode():
            restored = restore_array(model, arr, device, use_amp=use_amp)
        sync_device(device)
        per_image_times.append(time.perf_counter() - t0)

        out_path = output_dir / fname
        np.save(out_path, restored.astype(np.float32))

        if args.save_png_preview:
            from PIL import Image
            png_arr = (np.clip(restored, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(png_arr).save(output_dir / (Path(fname).stem + ".png"))

    total_time = time.perf_counter() - t_total_start

    print("\n--- Inference summary ---")
    print(f"Images processed successfully: {len(per_image_times)}")
    if errors:
        print(f"Images with errors: {len(errors)}")
        for fname, msg in errors[:10]:
            print(f"  {fname}: {msg}")
    if per_image_times:
        arr_times = np.array(per_image_times)
        print(f"Total inference time (excl. load/warm-up): {total_time:.3f}s")
        print(f"Average time/image: {arr_times.mean() * 1000:.2f}ms")
        print(f"Median time/image:  {np.median(arr_times) * 1000:.2f}ms")
        print(f"Throughput: {len(per_image_times) / total_time:.2f} images/sec")
    print(f"Restored outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
