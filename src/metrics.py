"""
src/metrics.py — PSNR / SSIM / LPIPS for single-channel restoration output.

All metrics are computed on images clamped to [0, 1] (the confirmed GT
range) since PSNR/SSIM/LPIPS are only meaningfully defined over a bounded
data range, and the model's raw output is a real number that should be
clamped for *display and metric purposes* even though it is left
unclamped during training (see src/losses.py docstring).

LPIPS expects 3-channel, roughly ImageNet-normalized input. Our images are
single-channel. We replicate the single channel to 3 channels (a standard,
documented conversion — NOT a fabricated metric) and rescale [0,1] -> [-1,1]
as required by the 'alex'/'vgg' LPIPS backbones. This is only used as an
auxiliary metric; results should be read as "LPIPS on a grayscale image
replicated to 3 channels," not as a natural-RGB LPIPS score, and are
reported as such in evaluate_metrics.py output.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """pred, target: (B,1,H,W) or (1,H,W), already clamped to data_range."""
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    mse = torch.clamp(mse, min=1e-12)
    return 20 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10 * torch.log10(mse)


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return window_2d.unsqueeze(0).unsqueeze(0)


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0,
         window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Per-image SSIM, returns (B,) tensor."""
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(pred, window, padding=pad)
    mu_y = F.conv2d(target, window, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=pad) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=pad) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map.flatten(1).mean(dim=1)


class LPIPSMetric:
    """Thin wrapper around the `lpips` package. Lazily constructed so
    scripts that don't need LPIPS (e.g. quick training sanity checks)
    don't pay the model-download/import cost."""

    def __init__(self, net: str = "alex", device: Optional[torch.device] = None):
        import lpips  # local import: optional heavy dependency

        self.device = device or torch.device("cpu")
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (B,1,H,W) in [0,1]. Returns (B,) LPIPS distances."""
        pred3 = pred.repeat(1, 3, 1, 1) * 2 - 1
        target3 = target.repeat(1, 3, 1, 1) * 2 - 1
        pred3 = pred3.to(self.device)
        target3 = target3.to(self.device)
        d = self.model(pred3, target3)
        return d.flatten()
