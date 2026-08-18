# models/final_model.pth — TBD, generated after final training

No trained weights are included in this delivery. This repository was built
and code-verified in an environment with no GPU and no access to the
complete 3,200-pair local dataset (see README "Status & Limitations").

To generate this file:

```bash
python train.py --config configs/config.yaml --data_root /path/to/FULL_KLA_DATASET/train
python - <<'PY'
import torch
ckpt = torch.load("artifacts/checkpoints/best.pth", map_location="cpu", weights_only=False)
torch.save({"model_state": ckpt["model_state"]}, "models/final_model.pth")
PY
```

The full training/inference pipeline (dataset loading, loss, model forward
pass, checkpointing, evaluate.py on real test data) HAS been executed and
verified — see README for exactly what was and wasn't run.
