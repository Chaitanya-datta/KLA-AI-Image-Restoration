# Video Demonstration Script

Target length: ~3-4 minutes. Record terminal + a results-grid image side by side.

1. **Problem** (20s) — One slide/screen: "Restore 128×128 degraded semiconductor
   inspection images to 256×256 clean images. Degradations: Gaussian noise,
   speckle noise, down-sampling, applied in any order."

2. **Degraded input** (20s) — Show `scripts/visualize_results.py` output, or
   `np.load` + `imshow` a single `NoisyLR/XXXXXX.npy` file live in a terminal /
   notebook cell, pointing out the out-of-[0,1] value range printed alongside.

3. **Running evaluation** (45s) — Run, on camera, the exact submission command:
   ```bash
   python evaluate.py --input_dir <Test_NoisyLR/NoisyLR> --output_dir outputs/restored_test
   ```
   Let the printed summary (images processed, total time, avg ms/image,
   throughput) appear on screen in full.

4. **Restored output** (20s) — Open one restored `.npy` from `outputs/restored_test/`
   and display it.

5. **Ground truth comparison** (20s) — If validation GT is available, show the
   Degraded | Restored | GT grid from `scripts/visualize_results.py`.

6. **Metrics** (30s) — Run and show:
   ```bash
   python evaluate_metrics.py --pred_dir outputs/restored_test --gt_dir <gt_dir>
   ```
   Read out the PSNR/SSIM (and LPIPS if computed) summary on screen.

7. **Inference speed** (20s) — Run and show:
   ```bash
   python scripts/benchmark_inference.py --input_dir <Test_NoisyLR/NoisyLR> --weights models/final_model.pth
   ```
   Call out parameter count, model size (MB), and throughput.

8. **GitHub repository** (15s) — Pan through the repo structure (README,
   `src/`, `scripts/`, `configs/config.yaml`), end on the README's exact
   evaluation command so a reviewer knows they can reproduce this without
   contacting the team.

**Status note for the presenter:** as of this repository's last update, steps
3–4 have been verified against the real 400-image test set with a
smoke-trained (not fully trained) checkpoint. Steps 5–7 require training on
the complete local dataset first — see README "Status & Limitations" and
"Exactly what to run locally to finish this" before recording.
