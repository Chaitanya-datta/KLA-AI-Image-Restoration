# KLA AI Image Restoration — SEMICON India Hackathon 2026

**[TEAM NAME — placeholder]** submission for the KLA Track: *AI-Based Restoration of
Degraded Images for Semiconductor Inspection*.

## Project Overview

This repository restores 128×128 degraded (`NoisyLR`) grayscale images to 256×256
clean (`GT`) images — a joint denoising + 2x super-resolution task. It ships a
full, runnable pipeline: data loading, training, validation, metrics, inference,
benchmarking, visualization, and submission validation. See **Status &
Limitations** below for exactly what has and hasn't been executed with real data
in this environment.

## Problem Statement

Per the official KLA kickoff deck (`docs/` references the source slides):

- Input images are degraded by **speckle noise, additive Gaussian noise, and
  down-sampling**, applied in no fixed order ("do not read into the order").
- The model must learn the inverse transform: degraded → clean.
- **`NoisyLR` values legitimately fall outside `[0, 1]`** (observed range
  roughly `-0.28` to `2.16`) — "a feature, not a bug." `GT` is always `[0, 1]`.
- Judging considers methodology (data, model, loss, compute hygiene), not only
  leaderboard score.

The source PPTX is a conceptual kickoff deck, not a numeric spec sheet — it does
not state exact evaluation metrics, submission file format, or inference-speed
targets. See **Open Questions** below for what's still needed from KLA/the
organizers before final submission.

## Dataset Structure

The complete dataset is **not included in this repository** (per instructions —
it stays local, ~922MB). Confirmed structure (verified by the team against the
full local dataset):

```
train/
├── GT/          3,200 images, XXXXXX.npy, float32, 256×256, range [0, 1]
└── NoisyLR/     3,200 images, XXXXXX.npy, float32, 128×128, range approx. [-0.28, 2.16]
```

Pairing rule: `NoisyLR/XXXXXX.npy` corresponds to `GT/XXXXXX.npy` — identical
filename. Verified: 3,200/3,200 pairs match, 0 orphaned files either direction.

A separate `Test_NoisyLR/NoisyLR/` (400 images, same format, **no GT provided**)
is the final-evaluation input set.

`__MACOSX/` and `._*` AppleDouble files (macOS zip artifacts) are present in at
least one archive and are ignored everywhere in this codebase automatically
(`src/utils.py:is_macosx_artifact`).

## Architecture

**LiteRestoreNet** — a compact residual encoder-decoder CNN with a PixelShuffle
2× upsampling head and a global residual (bicubic-upsample-plus-correction)
connection. ~0.77M parameters at the default config (C=64, 8 blocks).

Full rationale for the choice (vs. U-Net, Restormer/SwinIR, RCAN, DnCNN) is in
`src/model.py`'s module docstring and `docs/architecture.md`. Short version: at
128×128 input, further internal downsampling (as in a standard U-Net) throws
away exactly the detail this task needs to recover; heavier transformer-based
options (Restormer/SwinIR) offer a higher ceiling but are harder to train
stably on a few thousand pairs with limited compute and are unnecessarily slow
for the inference-speed requirement; DnCNN has no built-in upsampling path.

## Methodology

- **Loss**: Charbonnier (primary reconstruction) + SSIM + Sobel-gradient
  (edge-preservation) — see `src/losses.py` for the full rationale on what was
  included and deliberately excluded (VGG perceptual loss, FFT loss).
- **Augmentation**: horizontal/vertical flip + 90° rotations only — no
  distorting transforms that would misrepresent physical structure.
- **Value ranges**: `NoisyLR` is never clipped before being fed to the model —
  the out-of-range values are preserved as real signal. Model output is
  clamped to `[0,1]` only for metrics/inference, never inside the training loss.
- **Split**: no separate validation folder was provided, so 10% of `train/` is
  held out with a fixed seed (42) for a deterministic, reproducible split. No
  source/category metadata was found in the confirmed dataset structure, so
  the split is a plain random 90/10 (not source-aware) — see Limitations.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Configuration

Point the pipeline at your local, complete dataset — no paths are hard-coded.
Either edit `configs/config.yaml`:

```yaml
dataset:
  train_noisy_dir: "/path/to/FULL_KLA_DATASET/train/NoisyLR"
  train_gt_dir: "/path/to/FULL_KLA_DATASET/train/GT"
```

or pass `--data_root /path/to/FULL_KLA_DATASET/train` on the command line
(expects `NoisyLR/` and `GT/` subfolders under it), which overrides the config.

## Training

```bash
python train.py --config configs/config.yaml \
    --data_root /path/to/FULL_KLA_DATASET/train
```

Produces `artifacts/checkpoints/best.pth` (best val PSNR) and `last.pth`
(latest epoch, for resuming), plus a JSON training history and a log file.
Resume with `--resume artifacts/checkpoints/last.pth`. Quick smoke test:

```bash
python train.py --config configs/config.yaml --data_root <path> \
    --epochs 2 --limit_samples 32 --device cpu
```

**Before final submission**, copy the best checkpoint's `model_state` into
`models/final_model.pth` (see `scripts/validate_submission.py`, which checks
for it there by default):

```bash
python - <<'PY'
import torch
ckpt = torch.load("artifacts/checkpoints/best.pth", map_location="cpu", weights_only=False)
torch.save({"model_state": ckpt["model_state"]}, "models/final_model.pth")
PY
```

## Evaluation (submission entry point)

```bash
python evaluate.py \
    --input_dir /path/to/test_images \
    --output_dir /path/to/restored_outputs
```

Requires no manual code changes, no notebook, no training, no internet at
runtime. Auto-detects CUDA, uses `torch.inference_mode()`, prints total time,
average time/image, throughput. For the official 400-image test set:

```bash
python evaluate.py \
    --input_dir /path/to/Test_NoisyLR/NoisyLR \
    --output_dir outputs/restored_test
```

## Metrics

```bash
python evaluate_metrics.py \
    --pred_dir outputs/restored_test \
    --gt_dir /path/to/ground_truth \
    --lpips
```

Reports PSNR, SSIM, and (optionally) LPIPS — per-image CSV plus a JSON summary.
LPIPS on single-channel images is computed by replicating the channel to 3
(documented conversion, `src/metrics.py`); read it as such, not as natural-RGB
LPIPS.

## Results

**TBD — requires training on the complete 3,200-pair dataset**, which has not
been executed in this environment (see Status & Limitations). No PSNR/SSIM/
LPIPS numbers are reported here because none have been legitimately measured
yet — per the project's explicit no-fabrication policy.

## Inference Benchmark

```bash
python scripts/benchmark_inference.py \
    --input_dir /path/to/Test_NoisyLR/NoisyLR \
    --weights models/final_model.pth
```

Reports model load time, warm-up-excluded mean/median/p95 inference time,
throughput, parameter count, and model size. **TBD on target hardware (H100)**
— only CPU pipeline-verification numbers exist so far (see Status below).

## Repository Structure

```
KLA-AI-Image-Restoration/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── train.py                    # training entry point
├── evaluate.py                 # SUBMISSION entry point (inference)
├── evaluate_metrics.py         # PSNR/SSIM/LPIPS vs. ground truth
├── configs/
│   └── config.yaml
├── src/
│   ├── dataset.py               # pairing, splitting, augmentation
│   ├── model.py                 # LiteRestoreNet
│   ├── losses.py                # Charbonnier + SSIM + gradient
│   ├── metrics.py                # PSNR / SSIM / LPIPS implementations
│   ├── inference.py             # shared model-load / forward-pass helpers
│   └── utils.py                  # seeding, checkpoints, logging
├── scripts/
│   ├── analyze_dataset.py       # run against the FULL local dataset
│   ├── visualize_results.py     # Degraded | Restored | GT grids
│   ├── benchmark_inference.py    # isolated speed/size benchmark
│   └── validate_submission.py    # PASS/FAIL repo checklist
├── models/
│   └── final_model.pth          # TBD — generated after final training
├── outputs/restored_test/        # evaluate.py output for the 400 test images
├── artifacts/{checkpoints,metrics,logs,plots}/
└── docs/
    ├── architecture.md
    ├── methodology.md
    └── video_demo_script.md
```

## Reproducibility

- Fixed seeds throughout: `dataset.split_seed` (train/val split),
  `train.seed` (weights init, shuffling, augmentation randomness).
- `src/utils.set_seed()` sets Python/NumPy/PyTorch seeds and forces
  deterministic cuDNN algorithms.
- Every hyperparameter lives in `configs/config.yaml`, not hard-coded in
  scripts, and every checkpoint stores the exact config used to produce it.

## Limitations

- **No source/category metadata was found** in the confirmed dataset
  structure (just `GT/` and `NoisyLR/` with matching filenames), so
  source-aware OOD evaluation (recommended by the brief) could not be
  implemented — the val split is a plain random 10% hold-out from `train/`.
  If such metadata exists elsewhere in the full dataset, tell us and this can
  be upgraded.
- OOD generalization is therefore only as good as random-split validation can
  indicate; it is not a substitute for evaluation on the actual OOD test
  protocol KLA will use.
- No perceptual/VGG loss is used (see `src/losses.py` for why) — this may cap
  perceptual-quality ceiling in exchange for avoiding hallucinated texture.

## Status & Limitations of *this* build (read before relying on results)

This repository was built in an environment with **no GPU and no access to the
complete 922MB / 3,200-pair local dataset** (only a 400-image, GT-less test
sample and the conceptual PPTX were available here). Concretely:

- All code (`src/`, `train.py`, `evaluate.py`, `evaluate_metrics.py`, every
  `scripts/*.py`) has been **executed and verified**, but only against:
  1. A small **synthetic** paired dataset (random noise, not real GT) generated
     solely to exercise pairing, splitting, augmentation, loss computation,
     backprop, checkpointing, and the training loop end-to-end.
  2. The **real** 400-image `Test_NoisyLR` sample, run through `evaluate.py`
     end-to-end (all 400 processed, correct 256×256 `[0,1]` outputs, timing
     reported) — proving the submission entry point works on real data, using
     an untrained/synthetic-trained checkpoint.
- **No real training on the actual 3,200-pair dataset has occurred.** Any
  PSNR/SSIM/LPIPS number you might expect to see here is intentionally
  **absent**, not fabricated — see the no-fabrication policy this project was
  built under.
- `models/final_model.pth` is **not included** in this delivered structure —
  train on your local complete dataset and generate it (see Training above).

### Exactly what to run locally to finish this

```bash
# 1. Inspect your full local dataset (sanity-check the confirmed structure)
python scripts/analyze_dataset.py --data_root /path/to/FULL_KLA_DATASET/train

# 2. Train
python train.py --config configs/config.yaml \
    --data_root /path/to/FULL_KLA_DATASET/train

# 3. Export the final submission weights
python - <<'PY'
import torch
ckpt = torch.load("artifacts/checkpoints/best.pth", map_location="cpu", weights_only=False)
torch.save({"model_state": ckpt["model_state"]}, "models/final_model.pth")
PY

# 4. Generate restored test outputs
python evaluate.py --input_dir /path/to/Test_NoisyLR/NoisyLR --output_dir outputs/restored_test

# 5. Metrics (only possible if/when KLA provides test-set GT)
python evaluate_metrics.py --pred_dir outputs/restored_test --gt_dir /path/to/gt --lpips

# 6. Benchmark on your target hardware (ideally the H100 environment)
python scripts/benchmark_inference.py --input_dir /path/to/Test_NoisyLR/NoisyLR --weights models/final_model.pth

# 7. Validate the whole submission
python scripts/validate_submission.py --sample_input /path/to/Test_NoisyLR/NoisyLR
```

## Open Questions (need an answer from KLA / organizers, not guessed)

1. **Official evaluation metric(s) and weighting** — the PPTX doesn't specify
   whether PSNR, SSIM, LPIPS, or a combination determines the leaderboard.
2. **Exact submission file format** for restored outputs (this repo currently
   writes `.npy`, matching the input format — confirm if a different format,
   e.g. `.png`, is required).
3. **Inference-speed target/threshold**, if any, beyond "benchmarked on H100."
4. **OOD test protocol** — what makes a test image "dissimilar" per slide 7,
   and whether source/category metadata exists that we don't currently have.
5. The official **9-slide submission PPTX template** — only the conceptual
   kickoff deck was available when this repo was built; slide generation
   (`submission/`) needs the actual template to match its structure/styling.

## References

1. T. Kumar, R. Brennan, A. Mileo and M. Bendechache, "Image Data Augmentation
   Approaches: A Comprehensive Survey and Future Directions," IEEE Access,
   vol. 12, 2024.
2. Zhai, L., Wang, Y., Cui, S., Zhou, Y. "A comprehensive review of deep
   learning-based real-world image restoration." IEEE Access 11 (2023).
3. Terven, J., Cordova-Esparza, D.M., Romero-González, J.A. et al. "A
   comprehensive survey of loss functions and metrics in deep learning."
   Artif Intell Rev 58, 195 (2025).
4. V. Monga et al., "Algorithm Unrolling: Interpretable, Efficient Deep
   Learning for Signal and Image Processing," IEEE PM, vol. 38, no. 2, 2021.

(All four references are drawn directly from the official KLA kickoff deck.)
