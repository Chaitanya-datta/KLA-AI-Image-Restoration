"""
src/losses.py — composite restoration loss.

Terms selected (and why):
  - Charbonnier (smooth L1): primary reconstruction term. More robust to the
    remaining noisy/outlier pixels than plain L2, without L1's non-smooth
    gradient at zero. Standard choice for SR/denoising (EDSR, Restormer).
  - SSIM: pushes structural/perceptual similarity beyond raw pixel error,
    directly targets one of the two evaluation metrics required by the
    problem statement's evaluation criteria.
  - Gradient (edge) loss: an L1 loss on Sobel gradients. Directly targets
    "preserve edges / recover fine detail / avoid excessive smoothing" from
    the requirements — Charbonnier alone tends to over-smooth high-frequency
    structure, which is exactly what this loss counteracts.

Deliberately NOT included:
  - Perceptual (VGG) loss: VGG was trained on 3-channel natural RGB images;
    using it on single-channel semiconductor-style imagery means feeding a
    replicated/foreign-domain input through a mismatched feature extractor.
    Risk of hallucinating natural-image-like texture ("avoid hallucinated
    semiconductor patterns" is an explicit requirement) outweighs the
    likely benefit here. LPIPS is used only as a *held-out evaluation
    metric* (evaluate_metrics.py), never inside the training loss, for the
    same reason.
  - Frequency-domain (FFT) loss: left out of the default config to keep the
    loss recipe simple and stable; the weight is wired up and defaults to
    0.0 so it can be enabled later if ablation shows it helps (see
    docs/methodology.md).

All weights are configurable via configs/config.yaml (`loss:` section).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth L1 variant: sqrt((x-y)^2 + eps^2). Differentiable everywhere,
    less sensitive to outliers than L2."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return window_2d.unsqueeze(0).unsqueeze(0)  # (1,1,K,K)


class SSIMLoss(nn.Module):
    """1 - SSIM, computed with a Gaussian window, for single-channel images.
    Operates on whatever range `pred`/`target` are in; data_range must be
    passed explicitly since GT is [0,1] but predictions during early
    training may briefly leave that range."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.register_buffer("_window", torch.zeros(1))
        self._window_ready = False

    def _get_window(self, device, dtype):
        if not self._window_ready or self._window.device != device:
            self._window = _gaussian_window(self.window_size, self.sigma, device, dtype)
            self._window_ready = True
        return self._window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = self._get_window(pred.device, pred.dtype)
        pad = self.window_size // 2

        mu_x = F.conv2d(pred, window, padding=pad)
        mu_y = F.conv2d(target, window, padding=pad)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        sigma_x2 = F.conv2d(pred * pred, window, padding=pad) - mu_x2
        sigma_y2 = F.conv2d(target * target, window, padding=pad) - mu_y2
        sigma_xy = F.conv2d(pred * target, window, padding=pad) - mu_xy

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )
        return 1.0 - ssim_map.mean()


class GradientLoss(nn.Module):
    """L1 loss between Sobel gradients of prediction and target — penalizes
    blurring of edges / loss of fine structure."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gx_pred = F.conv2d(pred, self.kx, padding=1)
        gy_pred = F.conv2d(pred, self.ky, padding=1)
        gx_tgt = F.conv2d(target, self.kx, padding=1)
        gy_tgt = F.conv2d(target, self.ky, padding=1)
        return F.l1_loss(gx_pred, gx_tgt) + F.l1_loss(gy_pred, gy_tgt)


class CompositeRestorationLoss(nn.Module):
    """Weighted sum of Charbonnier + SSIM + gradient loss. Returns
    (total_loss, dict_of_unweighted_component_values) so training can log
    each term separately."""

    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        ssim_weight: float = 0.2,
        gradient_weight: float = 0.1,
        fft_weight: float = 0.0,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss(data_range=data_range)
        self.gradient = GradientLoss()
        self.w_char = charbonnier_weight
        self.w_ssim = ssim_weight
        self.w_grad = gradient_weight
        self.w_fft = fft_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l_char = self.charbonnier(pred, target)
        l_ssim = self.ssim(pred, target)
        l_grad = self.gradient(pred, target)

        total = self.w_char * l_char + self.w_ssim * l_ssim + self.w_grad * l_grad

        if self.w_fft > 0:
            fft_pred = torch.fft.rfft2(pred)
            fft_tgt = torch.fft.rfft2(target)
            l_fft = F.l1_loss(torch.abs(fft_pred), torch.abs(fft_tgt))
            total = total + self.w_fft * l_fft
        else:
            l_fft = torch.tensor(0.0, device=pred.device)

        components = {
            "charbonnier": l_char.item(),
            "ssim": l_ssim.item(),
            "gradient": l_grad.item(),
            "fft": l_fft.item() if isinstance(l_fft, torch.Tensor) else 0.0,
        }
        return total, components


def build_loss(config: dict) -> CompositeRestorationLoss:
    lc = config.get("loss", {})
    return CompositeRestorationLoss(
        charbonnier_weight=lc.get("charbonnier_weight", 1.0),
        ssim_weight=lc.get("ssim_weight", 0.2),
        gradient_weight=lc.get("gradient_weight", 0.1),
        fft_weight=lc.get("fft_weight", 0.0),
        data_range=lc.get("data_range", 1.0),
    )
