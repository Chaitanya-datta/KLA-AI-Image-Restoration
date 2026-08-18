# Methodology

## Loss function

`src/losses.py` implements a weighted sum:

```
L = w_char * Charbonnier(pred, gt) + w_ssim * (1 - SSIM(pred, gt)) + w_grad * GradientL1(pred, gt)
```

Default weights (`configs/config.yaml`): `charbonnier=1.0, ssim=0.2, gradient=0.1, fft=0.0`.

- **Charbonnier** (`sqrt((x-y)^2 + eps^2)`): primary reconstruction term.
  Smooth-L1-like — more robust to residual noisy/outlier pixels than L2,
  smoother gradient than L1 at zero.
- **SSIM**: directly optimizes structural similarity, one of the two
  standard restoration metrics, computed with an 11×11 Gaussian window.
- **Gradient (Sobel) L1**: penalizes blurring of edges specifically — added
  because Charbonnier/SSIM alone tend to favor smooth solutions, and the
  brief explicitly calls out "preserve edges," "avoid excessive smoothing."
- **FFT term**: wired up (`w_fft`) but defaults to 0 — left available for a
  documented ablation rather than included by default, to keep the base
  recipe simple and stable.

Deliberately excluded: VGG/perceptual loss (trained on natural RGB images;
mismatched domain for single-channel semiconductor-style imagery, risks
hallucinating natural-image texture — the brief explicitly warns against
"hallucinated semiconductor patterns"). LPIPS is used only as a held-out
*metric* in `evaluate_metrics.py`, never inside the training loss, for the
same domain-mismatch reason.

## Augmentation

Horizontal flip, vertical flip, 90°/180°/270° rotation — applied identically
to the `NoisyLR`/`GT` pair to preserve spatial correspondence. Chosen because:
- They're the only transforms that don't require resampling (no interpolation
  artifacts introduced into either image).
- They don't distort real structure, unlike elastic warps, arbitrary-angle
  rotation, or non-uniform scaling, which the brief explicitly warns against
  ("avoid transformations that could distort the physical meaning").
- Random crops were considered but not used by default: at 128×128 input /
  256×256 target, cropping either loses global context or requires careful
  synchronized crop coordinates between the two resolutions; the flip/rotate
  set already gives an 8× effective augmentation multiplier without that
  complexity. (This can be added in `src/dataset.py` if validation curves show
  it's needed.)

## Train/validation split

No separate validation folder was provided in the confirmed dataset
structure — only `train/GT` and `train/NoisyLR`. A 10% hold-out
(`dataset.val_fraction` in the config) is carved out with a fixed seed
(`dataset.split_seed=42`) via `src/dataset.py`, computed once on the sorted,
paired file list so it's reproducible regardless of filesystem iteration
order.

**No source/category metadata was found** in the confirmed structure (just
matched filenames), so a source-aware split — recommended by the brief for
better OOD signal — could not be implemented. This is flagged, not silently
skipped: see README "Limitations."

## Training loop

`train.py`: AdamW, cosine LR annealing, gradient clipping (max norm 1.0),
mixed precision (AMP) on CUDA only (CPU AMP is skipped — no benefit),
per-epoch validation computing loss/PSNR/SSIM, best-checkpoint saving on val
PSNR improvement, JSON training history, resumable via `--resume`.

## Metrics

`src/metrics.py` / `evaluate_metrics.py`: PSNR and SSIM implemented directly
(Gaussian-window SSIM, standard formula); LPIPS via the `lpips` package,
computed on the single channel replicated to 3 (documented, not a fabricated
"grayscale LPIPS" — reported as "LPIPS on channel-replicated grayscale").
Optional (`--lpips` flag) because it downloads a backbone network on first
use and the brief doesn't confirm it's required.

## Ablation (planned, not yet run)

`configs/config.yaml`'s `loss.*_weight` fields make it straightforward to run:

```
Baseline            (charbonnier only:      ssim_weight=0, gradient_weight=0)
Baseline + SSIM     (ssim_weight=0.2,       gradient_weight=0)
Baseline + Gradient (ssim_weight=0,         gradient_weight=0.1)
Final               (ssim_weight=0.2,       gradient_weight=0.1)
```

Not run in this delivery — no real training has occurred yet (see README
"Status & Limitations"). Running it is 4 `train.py` invocations with the
weights above, followed by `evaluate_metrics.py` on each resulting
checkpoint's validation predictions.
