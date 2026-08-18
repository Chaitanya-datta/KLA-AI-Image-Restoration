#!/usr/bin/env python3
"""
scripts/validate_submission.py — PASS/FAIL checklist for the repository
before submission.

    python scripts/validate_submission.py
    python scripts/validate_submission.py --sample_input Test_NoisyLR/NoisyLR --weights models/final_model.pth

Checks (per project brief section 26):
  - README, requirements.txt, train.py, evaluate.py exist
  - model weights exist
  - src/ source code + configs/ exist
  - required output folders exist
  - model loads successfully
  - evaluate.py's internals import cleanly
  - sample inference works end-to-end (if --sample_input provided)
  - output images are valid (correct shape/dtype, finite values)
"""
from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="models/final_model.pth")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--sample_input", type=str, default=None,
                    help="Directory of real .npy test images for an end-to-end smoke test")
    return p.parse_args()


class Check:
    def __init__(self):
        self.results = []

    def run(self, name: str, fn):
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"EXCEPTION: {e}"
        self.results.append((name, ok, detail))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
        return ok

    def report(self) -> bool:
        n_pass = sum(1 for _, ok, _ in self.results if ok)
        n_total = len(self.results)
        print(f"\n{'=' * 40}\n{n_pass}/{n_total} checks passed\n{'=' * 40}")
        return n_pass == n_total


def main():
    args = parse_args()
    checker = Check()

    required_files = [
        "README.md", "requirements.txt", "train.py", "evaluate.py",
        "evaluate_metrics.py", "configs/config.yaml",
        "src/dataset.py", "src/model.py", "src/losses.py", "src/metrics.py",
    ]
    for rel in required_files:
        checker.run(f"File exists: {rel}", lambda rel=rel: ((ROOT / rel).is_file(), None))

    required_dirs = ["src", "scripts", "configs", "models", "outputs", "artifacts"]
    for rel in required_dirs:
        checker.run(f"Directory exists: {rel}", lambda rel=rel: ((ROOT / rel).is_dir(), None))

    def _weights_exist():
        p = ROOT / args.weights
        if not p.is_file():
            return False, f"not found at {p} — train a model or point --weights elsewhere"
        return True, f"{p.stat().st_size / 1e6:.2f} MB"

    weights_ok = checker.run("Model weights exist", _weights_exist)

    def _imports():
        importlib.import_module("src.dataset")
        importlib.import_module("src.model")
        importlib.import_module("src.losses")
        importlib.import_module("src.metrics")
        importlib.import_module("src.inference")
        return True, None

    checker.run("src/ modules import cleanly", _imports)

    def _model_loads():
        import torch
        import yaml
        from src.inference import load_model_for_inference
        from src.utils import get_device

        with open(ROOT / args.config) as f:
            config = yaml.safe_load(f)
        device = get_device()
        model = load_model_for_inference(str(ROOT / args.weights), config, device)
        n = sum(p.numel() for p in model.parameters())
        return True, f"{n:,} parameters on {device}"

    model_loads_ok = False
    if weights_ok:
        model_loads_ok = checker.run("Model loads from weights", _model_loads)
    else:
        print("[SKIP] Model loads from weights — no weights file")

    def _sample_inference():
        import torch
        import yaml
        from src.inference import list_input_npy_files, load_model_for_inference, restore_array
        from src.utils import get_device

        with open(ROOT / args.config) as f:
            config = yaml.safe_load(f)
        device = get_device()
        model = load_model_for_inference(str(ROOT / args.weights), config, device)

        files = list_input_npy_files(args.sample_input)
        if not files:
            return False, f"no .npy files found in {args.sample_input}"

        with tempfile.TemporaryDirectory() as tmp:
            for fpath in files[:3]:
                arr = np.load(fpath)
                out = restore_array(model, arr, device)
                if out.shape != (arr.shape[0] * 2, arr.shape[1] * 2):
                    return False, f"unexpected output shape {out.shape} for input {arr.shape}"
                if not np.isfinite(out).all():
                    return False, f"non-finite values in output for {Path(fpath).name}"
                if out.min() < -1e-6 or out.max() > 1 + 1e-6:
                    return False, f"output outside [0,1] for {Path(fpath).name}: [{out.min()}, {out.max()}]"
                np.save(Path(tmp) / Path(fpath).name, out)
        return True, f"ran on {min(3, len(files))} sample image(s), outputs valid"

    if args.sample_input and weights_ok and model_loads_ok:
        checker.run("End-to-end sample inference", _sample_inference)
    else:
        print("[SKIP] End-to-end sample inference — pass --sample_input and ensure weights load")

    all_pass = checker.report()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
