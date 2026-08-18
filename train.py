#!/usr/bin/env python3
"""
train.py — train LiteRestoreNet on paired NoisyLR/GT data.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --data_root /path/to/FULL_KLA_DATASET/train
    python train.py --config configs/config.yaml --resume artifacts/checkpoints/last.pth
    python train.py --config configs/config.yaml --epochs 2 --limit_samples 32 --device cpu   # smoke test

`--data_root` should point at a directory containing NoisyLR/ and GT/
subfolders (matching the confirmed dataset layout); it overrides
dataset.train_noisy_dir / dataset.train_gt_dir from the config.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.dataset import PairedRestorationDataset, sanity_check_dataset
from src.losses import build_loss
from src.metrics import psnr as psnr_fn
from src.metrics import ssim as ssim_fn
from src.model import build_model
from src.utils import (
    AverageMeter,
    CheckpointState,
    SimpleLogger,
    count_parameters,
    load_checkpoint,
    model_size_mb,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train LiteRestoreNet")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--data_root", type=str, default=None,
                    help="Directory containing NoisyLR/ and GT/ subfolders; overrides config paths")
    p.add_argument("--epochs", type=int, default=None, help="Override config.train.epochs")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--limit_samples", type=int, default=None,
                    help="Use only the first N training pairs — for fast smoke tests, not real training")
    p.add_argument("--run_name", type=str, default="run")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict, args) -> tuple[DataLoader, DataLoader, dict]:
    noisy_dir = cfg["dataset"]["train_noisy_dir"]
    gt_dir = cfg["dataset"]["train_gt_dir"]
    if args.data_root:
        noisy_dir = os.path.join(args.data_root, "NoisyLR")
        gt_dir = os.path.join(args.data_root, "GT")

    report = sanity_check_dataset(noisy_dir, gt_dir)

    train_ds = PairedRestorationDataset(
        noisy_dir, gt_dir, split="train",
        val_fraction=cfg["dataset"]["val_fraction"],
        seed=cfg["dataset"]["split_seed"], augment=True,
    )
    val_ds = PairedRestorationDataset(
        noisy_dir, gt_dir, split="val",
        val_fraction=cfg["dataset"]["val_fraction"],
        seed=cfg["dataset"]["split_seed"], augment=False,
    )

    if args.limit_samples:
        train_ds.pairs = train_ds.pairs[: args.limit_samples]
        val_ds.pairs = val_ds.pairs[: max(1, args.limit_samples // 4)]

    bs = args.batch_size or cfg["train"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader, report


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> dict:
    model.eval()
    loss_meter, psnr_meter, ssim_meter = AverageMeter(), AverageMeter(), AverageMeter()
    for batch in loader:
        noisy = batch["noisy"].to(device)
        gt = batch["gt"].to(device)
        pred = model(noisy)
        loss, _ = loss_fn(pred, gt)
        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        b = noisy.size(0)
        loss_meter.update(loss.item(), b)
        psnr_meter.update(psnr_fn(pred_clamped, gt).mean().item(), b)
        ssim_meter.update(ssim_fn(pred_clamped, gt).mean().item(), b)
    return {"loss": loss_meter.avg, "psnr": psnr_meter.avg, "ssim": ssim_meter.avg}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    set_seed(cfg["train"]["seed"])
    device = resolve_device(args.device)  # explicit --device wins; None -> auto CUDA > MPS > CPU

    logger = SimpleLogger(cfg["train"]["log_path"])
    logger.log(f"Device: {device}")

    train_loader, val_loader, ds_report = build_dataloaders(cfg, args)
    logger.log(f"Dataset report: {ds_report}")
    logger.log(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(cfg).to(device)
    n_params = count_parameters(model)
    logger.log(f"Model: {cfg['model']['name']} | params={n_params:,} | size={model_size_mb(model):.2f}MB")

    loss_fn = build_loss(cfg).to(device)

    opt_name = cfg["train"]["optimizer"].lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                       weight_decay=cfg["train"]["weight_decay"])
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    if cfg["train"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["epochs"], eta_min=cfg["train"]["min_lr"]
        )
    else:
        scheduler = None

    # AMP is only used on CUDA — MPS autocast support is still partial/
    # unstable across PyTorch versions for arbitrary ops, so we train in
    # plain fp32 on MPS/CPU rather than risk silent numerical issues.
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_psnr = -1.0
    history = []

    resume_path = args.resume or cfg["train"].get("resume_from")
    if resume_path:
        ckpt = load_checkpoint(resume_path, map_location=str(device))
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", -1.0)
        history = ckpt.get("history", [])
        logger.log(f"Resumed from {resume_path} at epoch {start_epoch}")

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    patience = cfg["train"].get("early_stop_patience")
    epochs_since_improve = 0

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        t0 = time.time()
        train_loss_meter = AverageMeter()

        for batch in train_loader:
            noisy = batch["noisy"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp,
                                 dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                pred = model(noisy)
                loss, components = loss_fn(pred, gt)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
                optimizer.step()

            train_loss_meter.update(loss.item(), noisy.size(0))

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - t0
        log_entry = {"epoch": epoch, "train_loss": train_loss_meter.avg, "time_sec": epoch_time}

        if (epoch + 1) % cfg["train"]["val_every"] == 0:
            val_metrics = validate(model, val_loader, loss_fn, device)
            log_entry.update({f"val_{k}": v for k, v in val_metrics.items()})
            logger.log(
                f"Epoch {epoch} | train_loss={train_loss_meter.avg:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | val_psnr={val_metrics['psnr']:.2f}dB | "
                f"val_ssim={val_metrics['ssim']:.4f} | {epoch_time:.1f}s"
            )

            improved = val_metrics["psnr"] > best_psnr
            if improved:
                best_psnr = val_metrics["psnr"]
                epochs_since_improve = 0
                save_checkpoint(
                    str(ckpt_dir / "best.pth"),
                    CheckpointState(epoch, model.state_dict(), optimizer.state_dict(),
                                     scheduler.state_dict() if scheduler else None,
                                     scaler.state_dict() if use_amp else None,
                                     best_psnr, history, cfg),
                )
                logger.log(f"  -> New best val PSNR: {best_psnr:.2f}dB (saved best.pth)")
            else:
                epochs_since_improve += 1
        else:
            logger.log(f"Epoch {epoch} | train_loss={train_loss_meter.avg:.4f} | {epoch_time:.1f}s")

        history.append(log_entry)
        save_json(cfg["train"]["history_path"], history)

        save_checkpoint(
            str(ckpt_dir / "last.pth"),
            CheckpointState(epoch, model.state_dict(), optimizer.state_dict(),
                             scheduler.state_dict() if scheduler else None,
                             scaler.state_dict() if use_amp else None,
                             best_psnr, history, cfg),
        )

        if patience and epochs_since_improve >= patience:
            logger.log(f"Early stopping: no val PSNR improvement for {patience} epochs.")
            break

    logger.log(f"Training complete. Best val PSNR: {best_psnr:.2f}dB")
    logger.log(f"Best checkpoint: {ckpt_dir / 'best.pth'}")
    logger.log("To use for submission: copy/export the model_state from best.pth into models/final_model.pth "
                "(see scripts/validate_submission.py and README 'Training' section).")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
