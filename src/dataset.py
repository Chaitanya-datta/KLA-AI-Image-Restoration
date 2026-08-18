"""
src/dataset.py — paired NoisyLR -> GT restoration dataset.

Confirmed structure (verified by the user against their local, complete
922MB dataset — NOT invented; see DATASET_REPORT.md referenced in README):

    train/
    |-- GT/          3200 files, XXXXXX.npy, float32, 256x256, range [0, 1]
    `-- NoisyLR/      3200 files, XXXXXX.npy, float32, 128x128, range approx
                       [-0.28, 2.16]

Pairing rule: a NoisyLR file and a GT file correspond IFF they share the
exact same filename (e.g. NoisyLR/000123.npy <-> GT/000123.npy). This is
the only pairing scheme that has been confirmed and is the only one
implemented — do not change this without re-verifying against the dataset.

This module intentionally does NOT hard-code the sample size (400) that
shipped in Test_NoisyLR.zip, does NOT hard-code any absolute path, and does
NOT depend on the uploaded sample in any way. It walks whatever directory
it is given.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import is_macosx_artifact


class PairingError(RuntimeError):
    """Raised when NoisyLR/GT files cannot be robustly paired."""


def _list_npy_files(directory: str) -> List[str]:
    """Return sorted stem names (no extension) of real .npy files in
    `directory`, silently skipping __MACOSX/ and AppleDouble '._*' junk."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    stems = []
    for entry in os.scandir(directory):
        if not entry.is_file():
            continue
        if is_macosx_artifact(entry.path):
            continue
        if entry.name.lower().endswith(".npy"):
            stems.append(Path(entry.name).stem)
    return sorted(stems)


def build_pairs(noisy_dir: str, gt_dir: Optional[str]) -> List[Tuple[str, Optional[str]]]:
    """Pair NoisyLR files with GT files by identical filename stem.

    If gt_dir is None, returns (noisy_path, None) for every NoisyLR file —
    this is the inference-only mode used for the 400-image test set, which
    has no ground truth.

    Raises PairingError with the exact mismatched filenames if any NoisyLR
    file is missing its GT counterpart, or vice versa — we never silently
    drop or guess-pair files.
    """
    noisy_stems = _list_npy_files(noisy_dir)
    if not noisy_stems:
        raise PairingError(f"No .npy files found in NoisyLR directory: {noisy_dir}")

    if gt_dir is None:
        return [(os.path.join(noisy_dir, s + ".npy"), None) for s in noisy_stems]

    gt_stems = _list_npy_files(gt_dir)
    noisy_set, gt_set = set(noisy_stems), set(gt_stems)

    missing_gt = sorted(noisy_set - gt_set)
    missing_noisy = sorted(gt_set - noisy_set)
    if missing_gt or missing_noisy:
        msg = ["Dataset pairing failed."]
        if missing_gt:
            msg.append(
                f"{len(missing_gt)} NoisyLR file(s) have no GT match, e.g. {missing_gt[:5]}"
            )
        if missing_noisy:
            msg.append(
                f"{len(missing_noisy)} GT file(s) have no NoisyLR match, e.g. {missing_noisy[:5]}"
            )
        raise PairingError(" ".join(msg))

    stems = sorted(noisy_set)
    return [
        (os.path.join(noisy_dir, s + ".npy"), os.path.join(gt_dir, s + ".npy"))
        for s in stems
    ]


def _load_npy_as_tensor(path: str) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None, ...]  # (H, W) -> (1, H, W)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))  # (H, W, C) -> (C, H, W), just in case
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr))


class PairedRestorationDataset(Dataset):
    """NoisyLR (128x128) -> GT (256x256) paired dataset.

    Augmentation (train split only): horizontal flip, vertical flip, and
    90-degree rotations, applied identically to the NoisyLR/GT pair so
    spatial correspondence is preserved. These are the only transforms used
    because semiconductor-inspection imagery has no canonical "up," but
    arbitrary-angle rotation or elastic warps would distort real structures
    and were deliberately excluded.

    Value ranges are preserved exactly as loaded — NoisyLR is NOT clipped or
    rescaled to [0, 1] (see src/utils.py module docstring, assumption A2).
    """

    def __init__(
        self,
        noisy_dir: str,
        gt_dir: Optional[str],
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        augment: bool = True,
    ):
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be 'train', 'val', or 'all', got {split!r}")

        all_pairs = build_pairs(noisy_dir, gt_dir)

        if split == "all" or gt_dir is None:
            self.pairs = all_pairs
        else:
            # Deterministic, seed-fixed split computed on the sorted file
            # list so it never depends on filesystem iteration order.
            rng = np.random.RandomState(seed)
            idx = np.arange(len(all_pairs))
            rng.shuffle(idx)
            n_val = max(1, int(round(len(all_pairs) * val_fraction)))
            val_idx = set(idx[:n_val].tolist())
            if split == "train":
                self.pairs = [p for i, p in enumerate(all_pairs) if i not in val_idx]
            else:
                self.pairs = [p for i, p in enumerate(all_pairs) if i in val_idx]

        self.split = split
        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        noisy_path, gt_path = self.pairs[idx]
        noisy = _load_npy_as_tensor(noisy_path)
        gt = _load_npy_as_tensor(gt_path) if gt_path is not None else torch.zeros(1)

        if self.augment:
            noisy, gt = self._augment_pair(noisy, gt)

        return {
            "noisy": noisy,
            "gt": gt,
            "filename": os.path.basename(noisy_path),
            "has_gt": gt_path is not None,
        }

    @staticmethod
    def _augment_pair(noisy: torch.Tensor, gt: torch.Tensor):
        if torch.rand(1).item() < 0.5:
            noisy = torch.flip(noisy, dims=[-1])
            gt = torch.flip(gt, dims=[-1])
        if torch.rand(1).item() < 0.5:
            noisy = torch.flip(noisy, dims=[-2])
            gt = torch.flip(gt, dims=[-2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            noisy = torch.rot90(noisy, k, dims=[-2, -1])
            gt = torch.rot90(gt, k, dims=[-2, -1])
        return noisy, gt


def sanity_check_dataset(noisy_dir: str, gt_dir: Optional[str]) -> dict:
    """Lightweight structural check used by scripts/validate_submission.py
    and by train.py before launching a full run. Returns a small report
    dict; raises on any hard failure."""
    pairs = build_pairs(noisy_dir, gt_dir)
    report = {"num_pairs": len(pairs)}
    noisy_sample = _load_npy_as_tensor(pairs[0][0])
    report["noisy_shape"] = tuple(noisy_sample.shape)
    report["noisy_dtype"] = str(noisy_sample.dtype)
    if gt_dir is not None:
        gt_sample = _load_npy_as_tensor(pairs[0][1])
        report["gt_shape"] = tuple(gt_sample.shape)
        report["gt_dtype"] = str(gt_sample.dtype)
    return report
