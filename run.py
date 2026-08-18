#!/usr/bin/env python3
"""
run.py — KLA Hackathon 2026 final submission entry point.

    python run.py <input-dir> <output-dir>

Self-contained by design: this file does NOT import from src/ or read
configs/config.yaml. The model architecture (LiteRestoreNet) is defined
directly below, and its hyperparameters are read from the config dict that
is embedded inside models/final_model.pth itself (the training pipeline
always saves {"model_state": ..., "config": {...}} — see
KLA-AI-Image-Restoration/src/utils.py:CheckpointState / train.py). This
means the only external file this script needs is models/final_model.pth,
matching the official submission folder:

    team_name/
    ├── run.py
    ├── requirements.txt
    ├── README.md
    └── models/
        └── final_model.pth

All paths are resolved relative to THIS SCRIPT'S location (not the
evaluator's current working directory), so it works regardless of where
it's invoked from.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Model architecture (verbatim copy of LiteRestoreNet from
# KLA-AI-Image-Restoration/src/model.py — kept in sync manually; the
# architecture itself is NOT changed here, this is a copy for submission
# self-containment, not a redesign).
# --------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(self.act(x))
        out = self.conv2(self.act(out))
        return identity + self.res_scale * out


class PixelShuffleUpsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class LiteRestoreNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 channels: int = 64, num_blocks: int = 8, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.body_fusion = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upsample = PixelShuffleUpsample(channels)
        self.tail = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.head(x)
        body_out = self.body(feat)
        body_out = self.body_fusion(body_out) + feat
        up = self.upsample(body_out)
        residual = self.tail(up)
        return base + residual


# Fallback architecture defaults — used ONLY if a checkpoint somehow lacks
# an embedded "config" dict (e.g. a hand-crafted bare state_dict). These
# match KLA-AI-Image-Restoration/configs/config.yaml's `model:` section
# exactly, i.e. the values this project's model was actually trained with.
_DEFAULT_MODEL_CFG = {
    "in_channels": 1, "out_channels": 1, "channels": 64, "num_blocks": 8, "scale": 2,
}


def build_model_from_checkpoint(ckpt: dict) -> nn.Module:
    model_cfg = ckpt.get("config", {}).get("model", {}) if isinstance(ckpt, dict) else {}
    cfg = {**_DEFAULT_MODEL_CFG, **{k: v for k, v in model_cfg.items() if k in _DEFAULT_MODEL_CFG}}
    return LiteRestoreNet(**cfg)


# --------------------------------------------------------------------------
# Minimal self-contained helpers (no src/ import)
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = SCRIPT_DIR / "models" / "final_model.pth"


def is_macosx_artifact(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return "__MACOSX" in parts or name.startswith("._")


def get_device() -> torch.device:
    """CUDA preferred (per submission requirement 'run on NVIDIA GPU'),
    CPU fallback. MPS deliberately not used here — this script targets the
    evaluator's environment, not the development Mac."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def list_npy_files(input_dir: str):
    files = []
    for entry in os.scandir(input_dir):
        if entry.is_file() and entry.name.lower().endswith(".npy") and not is_macosx_artifact(entry.path):
            files.append(entry.path)
    return sorted(files)


def load_model(weights_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model = build_model_from_checkpoint(ckpt if isinstance(ckpt, dict) else {})
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def restore_array(model: nn.Module, arr: np.ndarray, device: torch.device) -> np.ndarray:
    """Runs the model and GUARANTEES a finite, [0,1]-clamped float32 output —
    NaN/Inf are explicitly sanitized, not just left to a plain clamp (which
    would pass NaN straight through)."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]  # accept (H,W,1) input defensively, treat as (H,W)

    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    y = model(x)

    # Explicit NaN/Inf sanitation BEFORE clamping — clamp alone does not
    # remove NaN (comparisons with NaN are always false).
    y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    y = torch.clamp(y, 0.0, 1.0)

    return y.squeeze(0).squeeze(0).float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="KLA restoration inference: python run.py <input-dir> <output-dir>"
    )
    parser.add_argument("input_dir", type=str, help="Directory of degraded .npy images")
    parser.add_argument("output_dir", type=str, help="Directory to write restored .npy images to")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not DEFAULT_WEIGHTS.is_file():
        print(
            f"ERROR: model weights not found at {DEFAULT_WEIGHTS}. "
            f"Expected models/final_model.pth next to run.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    model = load_model(DEFAULT_WEIGHTS, device)

    files = list_npy_files(str(input_dir))
    if not files:
        print(f"ERROR: no .npy files found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} input file(s)")

    t_start = time.perf_counter()
    n_ok, n_err = 0, 0
    for fpath in files:
        fname = Path(fpath).name
        try:
            arr = np.load(fpath)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: failed to load {fname}: {e}", file=sys.stderr)
            n_err += 1
            continue

        restored = restore_array(model, arr, device)
        np.save(output_dir / fname, restored.astype(np.float32))
        n_ok += 1

    elapsed = time.perf_counter() - t_start
    print(f"Processed {n_ok} file(s) successfully" + (f", {n_err} error(s)" if n_err else ""))
    print(f"Total time: {elapsed:.2f}s")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
