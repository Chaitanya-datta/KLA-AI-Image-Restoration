"""
src/inference.py — shared model-loading / single-image inference helpers,
used by evaluate.py, scripts/benchmark_inference.py and
scripts/validate_submission.py so the "one true" inference path is defined
exactly once.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.model import LiteRestoreNet, build_model
from src.utils import is_macosx_artifact


def load_model_for_inference(
    weights_path: str, config: dict, device: torch.device
) -> torch.nn.Module:
    model = build_model(config)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def list_input_npy_files(input_dir: str):
    """Sorted list of real .npy files in input_dir, ignoring __MACOSX/._* junk."""
    files = []
    for entry in os.scandir(input_dir):
        if entry.is_file() and entry.name.lower().endswith(".npy") and not is_macosx_artifact(entry.path):
            files.append(entry.path)
    return sorted(files)


@torch.inference_mode()
def restore_array(model: torch.nn.Module, arr: np.ndarray, device: torch.device,
                   use_amp: bool = False) -> np.ndarray:
    """Run the model on a single (H,W) float32 array. Returns a (H*2,W*2)
    float32 array clamped to [0,1] (the GT range)."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = model(x)
    else:
        y = model(x)

    y = torch.clamp(y, 0.0, 1.0)
    return y.squeeze(0).squeeze(0).float().cpu().numpy()
