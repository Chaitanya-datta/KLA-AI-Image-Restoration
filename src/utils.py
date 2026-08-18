"""
Shared utilities: reproducibility, checkpointing, logging, running averages.

ASSUMPTIONS (see README.md "Assumptions" section for the full list):
  - A1: Data are single-channel (grayscale) float32 arrays. No channel axis is
        stored on disk; we add it (H, W) -> (1, H, W) when loading.
  - A2: NoisyLR values legitimately fall outside [0, 1] (confirmed range about
        -0.28 to 2.16). We never clip/clamp the *input* — clipping would
        destroy real information the model needs to learn to invert.
  - A3: GT values are always in [0, 1] (confirmed). We clamp model output to
        [0, 1] only at inference/metric time, never inside the loss, so
        gradients aren't killed by the clamp during training.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make training as reproducible as CPU/GPU nondeterminism allows."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms where available; benchmark mode is faster but
    # non-deterministic, so we trade a little speed for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Tracks a running mean (loss, PSNR, SSIM, timings, ...)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


class Timer:
    """Context manager returning elapsed wall-clock seconds via .elapsed."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Auto-select the best available device: CUDA > MPS (Apple Silicon) > CPU.
    `prefer_cuda=False` (or CLI --device cpu) forces CPU regardless of what's
    available, for debugging/reproducibility comparisons."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_cuda and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(requested: Optional[str]) -> torch.device:
    """Resolve a CLI --device value ('cuda' | 'mps' | 'cpu' | None) into a
    torch.device.

    - None -> auto-select: CUDA > MPS > CPU (via get_device()).
    - 'cuda' / 'mps' explicitly requested but unavailable -> raises
      RuntimeError with a clear message. We deliberately do NOT silently
      fall back to CPU here: a user who asked for an accelerator and
      unknowingly got CPU would see a confusing 10-100x slowdown with no
      explanation.
    - 'cpu' -> always honored (forces CPU even if an accelerator exists).
    """
    if requested is None:
        return get_device()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested but torch.cuda.is_available() is False. "
                "Check your CUDA drivers/PyTorch build, or drop --device to auto-select, "
                "or pass --device cpu."
            )
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "--device mps was requested but torch.backends.mps.is_available() is False. "
                "MPS requires macOS 12.3+ on Apple Silicon (M1/M2/M3/...) and a PyTorch build "
                "with MPS support (torch>=1.12). Drop --device to auto-select, or pass --device cpu."
            )
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Block until pending GPU/MPS work finishes — required before taking a
    wall-clock timestamp for accurate per-call timing, on any accelerator."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: torch.nn.Module) -> float:
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / (1024 ** 2)


@dataclass
class CheckpointState:
    epoch: int
    model_state: dict
    optimizer_state: dict
    scheduler_state: Optional[dict]
    scaler_state: Optional[dict]
    best_psnr: float
    history: list = field(default_factory=list)
    config: dict = field(default_factory=dict)


def save_checkpoint(path: str, state: CheckpointState) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": state.epoch,
            "model_state": state.model_state,
            "optimizer_state": state.optimizer_state,
            "scheduler_state": state.scheduler_state,
            "scaler_state": state.scaler_state,
            "best_psnr": state.best_psnr,
            "history": state.history,
            "config": state.config,
        },
        path,
    )


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def is_macosx_artifact(path: str) -> bool:
    """True for macOS zip-archive junk that must be ignored everywhere:
    __MACOSX/ directories and AppleDouble '._*' sidecar files."""
    parts = Path(path).parts
    name = Path(path).name
    return "__MACOSX" in parts or name.startswith("._")


class SimpleLogger:
    """Minimal dependency-free logger: prints to stdout and appends to a
    plain-text log file. Avoids pulling in a heavier logging framework for
    what is fundamentally a training-progress transcript."""

    def __init__(self, log_path: str):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
