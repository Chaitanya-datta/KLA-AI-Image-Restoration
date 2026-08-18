# Architecture: LiteRestoreNet

## Task framing

Input: 128×128 single-channel `NoisyLR`, real-valued, not bounded to `[0,1]`.
Output: 256×256 single-channel prediction of `GT` (`GT` is always `[0,1]`).
Degradations: additive Gaussian noise, speckle noise, and 2× down-sampling,
composed in unspecified order. This is joint denoising + super-resolution,
not either problem in isolation.

## Design

```
NoisyLR (1,128,128)
      │
      ├────────────────────────────────────────────┐
      │                                             │ bicubic ×2 upsample
      ▼                                             │ (unclipped)
 head conv (1→C)                                    │
      │                                             │
      ▼                                             │
 N × ResidualBlock(C)  (GELU, conv, GELU, conv,      │
      │                 scaled residual, no norm)    │
      ▼                                             │
 fusion conv + long skip                            │
      │                                             │
      ▼                                             │
 PixelShuffle ×2 upsample head (C→4C→depth_to_space) │
      │                                             │
      ▼                                             │
 tail conv (C→1)  ──── residual ────────────────────┤
                                                      ▼
                                              output = base + residual
                                                (256,256), clamped [0,1]
                                                only for metrics/inference
```

## Why this over the alternatives considered

| Option | Why not chosen here |
|---|---|
| U-Net (with internal downsampling) | At a 128×128 *input*, downsampling further inside the encoder discards exactly the fine detail the task needs recovered — U-Net's strength (multi-scale context) matters more when the input itself is already large. |
| Restormer / SwinIR (transformer-based) | Higher quality ceiling, but heavier to train stably from scratch on ~2,880 pairs with the compute available, and slower at inference — works against the brief's own priority order (quality, then generalization, detail, *then* speed/size/stability, "avoid unnecessarily huge model"). |
| RCAN (400+ layer channel-attention SR) | Built for large-scale, diverse SR corpora (DIV2K-style); its depth is disproportionate to a single-domain, ~3,200-pair dataset and would risk overfitting and slow inference for no measured quality benefit here. |
| DnCNN | Denoising-only architecture with no upsampling path; would need a bolted-on SR head, at which point it becomes a smaller, less structured version of LiteRestoreNet. |

## Stability choices

- **No BatchNorm anywhere.** `NoisyLR` values range outside `[0,1]` per-image
  (confirmed, not assumed); BatchNorm's running statistics are a poor fit for
  inputs with per-image outliers, and BN is a known PSNR-quality regression in
  restoration networks (this is why EDSR removed it from SRResNet, and why
  NAFNet-style designs avoid it too).
- **Global residual connection**: the network predicts a correction on top of
  a bicubic-upsampled version of the raw (unclipped) input, rather than the
  full image from scratch. This is the single biggest stabilizer for SR
  training and focuses model capacity on noise removal / detail recovery.
- **PixelShuffle, not transposed convolution**, for the 2× upsample — avoids
  the checkerboard artifacts transposed convs are prone to.
- **Small residual scale (0.2)** inside each residual block — standard
  EDSR-style trick that improves stability for stacks of residual blocks.

## Size / capacity

Default config: `channels=64`, `num_blocks=8` → **~0.77M parameters**
(measured — see `python src/model.py`). Both are exposed in
`configs/config.yaml` (`model.channels`, `model.num_blocks`) so capacity can
be scaled up if validation metrics show underfitting, or down if inference
speed on the target hardware needs to improve, without touching this file.
