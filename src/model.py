"""
src/model.py — LiteRestoreNet

TASK: single-channel 128x128 NoisyLR -> single-channel 256x256 GT.
Degradations (per KLA problem statement): speckle noise, additive Gaussian
noise, down-sampling, applied "in any order" — so this is a joint
denoising + 2x super-resolution problem, not SR alone.

ARCHITECTURE CHOICE
--------------------
LiteRestoreNet is a compact residual encoder-decoder CNN:

    Input (1,128,128)
      -> Head conv (1->C)
      -> N residual blocks at full LR resolution (no downsampling inside
         the body — for a 128x128 input, further downsampling throws away
         exactly the fine detail this task needs to recover)
      -> PixelShuffle x2 upsampling head (C -> 4C -> depth_to_space -> C)
      -> Tail conv (C->1)
      -> output = tail(body) + bicubic-upsampled input   (global residual)

Why this over the alternatives considered:
  - U-Net (with internal downsampling): loses fine detail at 128x128 input
    resolution before the model ever gets to use it; better suited to
    already-large inputs.
  - Restormer / SwinIR (transformer-based): stronger ceiling on quality but
    much heavier to train stably from scratch on ~2,880 training pairs
    with limited compute, and slower at inference. Overkill relative to
    the "avoid unnecessarily huge model" and "H100 inference benchmark,
    prioritize speed/size after quality" requirements.
  - RCAN (very deep, channel-attention, 400+ layers): designed for
    large-scale SR datasets (DIV2K, 800+ diverse images at full
    resolution); too large and too slow to train well here.
  - DnCNN: denoising-only, no built-in upsampling path — would need a
    bolted-on SR head, at which point it's basically a smaller version of
    this network.

LiteRestoreNet keeps NAFNet/EDSR-style design choices known to matter for
restoration quality and training stability:
  - No BatchNorm anywhere. NoisyLR values legitimately range outside
    [0, 1] (see assumption A2 in src/utils.py); BatchNorm's running
    statistics are a poor fit for inputs with per-image outliers and
    BN is known to hurt PSNR-oriented restoration (EDSR/NAFNet finding).
  - Global residual connection (predict a correction on top of a
    bicubic-upsampled input) rather than predicting the image from
    scratch — this is the single biggest stabilizer for SR training and
    lets the network focus capacity on noise removal + detail recovery
    instead of re-learning "copy the input."
  - PixelShuffle for upsampling instead of transposed convolution —
    avoids checkerboard/ringing artifacts.
  - GELU activations, pre-activation residual blocks for stable gradients.

Sized to be practical, not maximal: default config is C=64 channels,
N=8 residual blocks (~1.5M parameters, see scripts/benchmark_inference.py
for measured size/speed). configs/config.yaml exposes both so the model
can be scaled up or down without touching this file.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-activation residual block: GELU -> conv -> GELU -> conv, no norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        # Small residual scale improves training stability for deep
        # residual stacks (standard EDSR-style trick).
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(self.act(x))
        out = self.conv2(self.act(out))
        return identity + self.res_scale * out


class PixelShuffleUpsample(nn.Module):
    """2x spatial upsampling via sub-pixel convolution (no checkerboarding)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class LiteRestoreNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 64,
        num_blocks: int = 8,
        scale: int = 2,
    ):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.body_fusion = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upsample = PixelShuffleUpsample(channels)
        self.tail = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global residual base: bicubic upsample of the raw (unclipped)
        # input. The network only has to learn the *correction* on top of
        # this, not full image reconstruction from scratch.
        base = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )

        feat = self.head(x)
        body_out = self.body(feat)
        body_out = self.body_fusion(body_out) + feat  # long skip
        up = self.upsample(body_out)
        residual = self.tail(up)

        return base + residual


def build_model(config: dict) -> nn.Module:
    m = config.get("model", {})
    return LiteRestoreNet(
        in_channels=m.get("in_channels", 1),
        out_channels=m.get("out_channels", 1),
        channels=m.get("channels", 64),
        num_blocks=m.get("num_blocks", 8),
        scale=m.get("scale", 2),
    )


if __name__ == "__main__":
    model = LiteRestoreNet()
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    print(f"Params: {n_params:,}")
