#!/usr/bin/env python3
"""
scripts/analyze_dataset.py — run against the COMPLETE local dataset
(expects --data_root/NoisyLR and --data_root/GT, matching the confirmed
layout), or a NoisyLR-only test directory via --noisy_only.

    python scripts/analyze_dataset.py --data_root /path/to/FULL_KLA_DATASET/train
    python scripts/analyze_dataset.py --data_root /path/to/Test_NoisyLR --noisy_only

Reports (never fabricated — every number here comes from actually reading
every file):
  - sample/image counts
  - resolution distribution
  - channels, dtype
  - intensity statistics (min/max/mean/std, fraction out of [0,1])
  - missing pairs (NoisyLR without GT / GT without NoisyLR)
  - duplicate detection (exact byte-content hash collisions)
  - corrupted .npy detection (file that fails to load)
  - filename/pairing validation
Automatically ignores __MACOSX/ and AppleDouble '._*' junk.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import is_macosx_artifact, save_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Analyze the KLA restoration dataset")
    p.add_argument("--data_root", type=str, required=True,
                    help="Directory containing NoisyLR/ (and GT/ unless --noisy_only)")
    p.add_argument("--noisy_only", action="store_true",
                    help="Set for a test-only directory that has no GT/ subfolder")
    p.add_argument("--output_json", type=str, default="artifacts/metrics/dataset_analysis.json")
    return p.parse_args()


def scan_folder(folder: Path) -> dict:
    result = {
        "folder": str(folder),
        "exists": folder.is_dir(),
        "num_files": 0,
        "shapes": Counter(),
        "dtypes": Counter(),
        "corrupted": [],
        "global_min": np.inf,
        "global_max": -np.inf,
        "mins": [],
        "maxs": [],
        "means": [],
        "hashes": {},
        "duplicates": [],
        "stems": [],
    }
    if not folder.is_dir():
        return result

    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if not entry.is_file() or is_macosx_artifact(str(entry)) or entry.suffix.lower() != ".npy":
            continue
        result["num_files"] += 1
        stem = entry.stem
        result["stems"].append(stem)
        try:
            arr = np.load(entry)
        except Exception as e:  # noqa: BLE001
            result["corrupted"].append({"file": entry.name, "error": str(e)})
            continue

        result["shapes"][str(arr.shape)] += 1
        result["dtypes"][str(arr.dtype)] += 1
        result["global_min"] = min(result["global_min"], float(arr.min()))
        result["global_max"] = max(result["global_max"], float(arr.max()))
        result["mins"].append(float(arr.min()))
        result["maxs"].append(float(arr.max()))
        result["means"].append(float(arr.mean()))

        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        if digest in result["hashes"]:
            result["duplicates"].append((result["hashes"][digest], entry.name))
        else:
            result["hashes"][digest] = entry.name

    return result


def summarize(scan: dict) -> dict:
    if scan["num_files"] == 0:
        return {"num_files": 0, "note": "folder missing or empty"}
    mins, maxs, means = np.array(scan["mins"]), np.array(scan["maxs"]), np.array(scan["means"])
    return {
        "folder": scan["folder"],
        "num_files": scan["num_files"],
        "num_corrupted": len(scan["corrupted"]),
        "corrupted_files": [c["file"] for c in scan["corrupted"][:20]],
        "shape_distribution": dict(scan["shapes"]),
        "dtype_distribution": dict(scan["dtypes"]),
        "global_min": scan["global_min"],
        "global_max": scan["global_max"],
        "mean_of_per_image_min": float(mins.mean()) if len(mins) else None,
        "mean_of_per_image_max": float(maxs.mean()) if len(maxs) else None,
        "mean_of_per_image_mean": float(means.mean()) if len(means) else None,
        "fraction_images_with_max_gt_1": float((maxs > 1).mean()) if len(maxs) else None,
        "fraction_images_with_min_lt_0": float((mins < 0).mean()) if len(mins) else None,
        "num_duplicate_pairs": len(scan["duplicates"]),
        "duplicate_examples": scan["duplicates"][:10],
    }


def main():
    args = parse_args()
    root = Path(args.data_root)
    noisy_dir = root / "NoisyLR" if (root / "NoisyLR").is_dir() else root
    gt_dir = root / "GT"

    print(f"Scanning NoisyLR: {noisy_dir}")
    noisy_scan = scan_folder(noisy_dir)
    noisy_summary = summarize(noisy_scan)

    report = {"noisy": noisy_summary}

    if not args.noisy_only:
        print(f"Scanning GT: {gt_dir}")
        gt_scan = scan_folder(gt_dir)
        gt_summary = summarize(gt_scan)
        report["gt"] = gt_summary

        noisy_stems, gt_stems = set(noisy_scan["stems"]), set(gt_scan["stems"])
        report["pairing"] = {
            "noisy_without_gt": sorted(noisy_stems - gt_stems)[:20],
            "gt_without_noisy": sorted(gt_stems - noisy_stems)[:20],
            "num_noisy_without_gt": len(noisy_stems - gt_stems),
            "num_gt_without_noisy": len(gt_stems - noisy_stems),
            "num_matched_pairs": len(noisy_stems & gt_stems),
        }

    save_json(args.output_json, report)

    print("\n--- Dataset analysis summary ---")
    for section, data in report.items():
        print(f"\n[{section}]")
        for k, v in data.items():
            print(f"  {k}: {v}")
    print(f"\nFull report: {args.output_json}")


if __name__ == "__main__":
    main()
