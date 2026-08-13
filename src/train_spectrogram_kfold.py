from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.amp import GradScaler

from .config import CACHE_DIR, NUM_CLASSES, OUTPUT_DIR, RANDOM_SEED, TRAIN_ROOT
from .run_manifest import create_run_manifest, finalize_run_manifest, make_run_id, write_run_manifest
from .spectrogram import (
    DroneClassifier,
    WarmupCosineLR,
    compute_pos_weight,
    label_signature_to_multihot,
    make_loss,
)
from .train_spectrogram import make_dataset, make_loader, prediction_metrics, run_epoch, seed_everything


DEFAULT_SAVE_DIR = OUTPUT_DIR / "spectrogram_kfold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STFT spectrogram k-fold models for ensemble inference.")
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR / "stft")
    parser.add_argument("--fold", type=int, default=None, help="Train only one fold. Omit to train all folds sequentially.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--arch", choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b2", "convnext_tiny", "densenet121"], default="resnet34")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--loss", choices=["bce_pos", "asymmetric", "focal"], default="bce_pos")
    parser.add_argument("--pos-weight-max", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument("--target-time", type=int, default=768)
    parser.add_argument("--cache-time", type=int, default=1536)
    parser.add_argument("--time-mask", type=int, default=40)
    parser.add_argument("--freq-mask", type=int, default=25)
    parser.add_argument("--specaug-p", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--select-metric", choices=["strict", "micro_f1", "macro_f1"], default="strict")
    parser.add_argument("--run-id", type=str, default=None, help="Optional stable identifier for the run manifest.")
    return parser.parse_args()


def label_matrix(signatures: pd.Series) -> np.ndarray:
    return np.stack([label_signature_to_multihot(value, NUM_CLASSES) for value in signatures]).astype(np.float32)


def make_splits(df: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.arange(len(df)), df["label_signature"]))
    except ValueError:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.arange(len(df))))


def suffix(tag: str, fold: int) -> str:
    return f"_{tag}_fold{fold}" if tag else f"_fold{fold}"


def train_fold(args: argparse.Namespace, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, fold: int, device: torch.device) -> dict[str, object]:
    seed_everything(args.seed + fold)
    use_amp = device.type == "cuda" and not args.no_amp

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_dataset = make_dataset(args, train_df, augment=True)
    val_dataset = make_dataset(args, val_df, augment=False)
    train_loader = make_loader(args, train_dataset, args.batch_size, True, device, drop_last=True)
    val_loader = make_loader(args, val_dataset, args.batch_size * 2, False, device)

    train_labels = label_matrix(train_df["label_signature"])
    pos_weight = compute_pos_weight(train_labels, args.pos_weight_max).to(device)
    model = DroneClassifier(NUM_CLASSES, args.arch, not args.no_pretrained, args.dropout).to(device)
    criterion = make_loss(args.loss, pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineLR(optimizer, args.warmup_epochs, args.epochs, args.lr)
    scaler = GradScaler(device.type, enabled=use_amp)

    tag_suffix = suffix(args.tag, fold)
    model_path = args.save_dir / f"best_model{tag_suffix}.pth"
    oof_path = args.save_dir / f"oof{tag_suffix}.npz"
    history_path = args.save_dir / f"history{tag_suffix}.json"

    print(f"\nFold {fold}/{args.n_splits - 1} | train={len(train_df)} val={len(val_df)} model={args.arch} tag={args.tag or 'default'}")
    best_metric = -1.0
    best_epoch = 0
    best_probs: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        lr_now = scheduler.step()
        epoch_start = time.time()
        train_loss, train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, use_amp, args.grad_clip, f"fold{fold} epoch{epoch:03d}/train"
        )
        val_loss, val_metrics, val_probs, val_labels = run_epoch(
            model, val_loader, criterion, device, None, None, use_amp, args.grad_clip, f"fold{fold} epoch{epoch:03d}/val"
        )
        selected = float(val_metrics[args.select_metric])
        record = {
            "fold": fold,
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
            "seconds": time.time() - epoch_start,
        }
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"Fold {fold} Epoch {epoch:03d} | val_strict={val_metrics['strict']:.4f} "
            f"val_micro={val_metrics['micro_f1']:.4f} val_macro={val_metrics['macro_f1']:.4f}"
        )

        if selected > best_metric + args.min_delta:
            best_metric = selected
            best_epoch = epoch
            best_probs = val_probs
            best_labels = val_labels
            torch.save(
                {
                    "epoch": epoch,
                    "fold": fold,
                    "tag": args.tag,
                    "model_state_dict": model.state_dict(),
                    "arch": args.arch,
                    "pretrained": not args.no_pretrained,
                    "dropout": args.dropout,
                    "num_classes": NUM_CLASSES,
                    "n_fft": args.n_fft,
                    "hop": args.hop,
                    "target_freq": args.n_fft // 2 + 1,
                    "target_time": args.target_time,
                    "cache_time": args.cache_time,
                    "metrics": val_metrics,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                model_path,
            )
            print(f"Saved fold {fold} best checkpoint: {model_path}")
        elif epoch - best_epoch >= args.patience:
            print(f"Fold {fold} early stopping after {args.patience} epochs without improvement.")
            break

    if best_probs is None or best_labels is None:
        raise RuntimeError(f"Fold {fold} did not produce a checkpoint")

    np.savez(
        oof_path,
        probs=best_probs.astype(np.float16),
        labels=best_labels.astype(np.int8),
        indices=val_idx.astype(np.int32),
        fold=np.asarray(fold, dtype=np.int32),
        sample_ids=df.iloc[val_idx]["sample_id"].to_numpy(dtype=np.int64),
        metrics=np.asarray([best_metric], dtype=np.float32),
    )
    metrics = prediction_metrics(best_probs.astype(np.float32), best_labels.astype(np.int32))
    print(f"Fold {fold} done: best_epoch={best_epoch} best_{args.select_metric}={best_metric:.5f} oof={oof_path}")
    return {"fold": fold, "best_epoch": best_epoch, "best_metric": best_metric, "metrics": metrics, "model_path": str(model_path), "oof_path": str(oof_path)}


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    run_id = args.run_id or make_run_id("spectrogram-kfold", args.seed)
    manifest_path = args.save_dir / f"run-manifest_{args.tag or 'default'}.json"
    manifest = create_run_manifest(
        run_id=run_id,
        command=[sys.executable, "-m", "src.train_spectrogram_kfold", *sys.argv[1:]],
        args=vars(args),
        data={"adapter": "wairc-competition-v1", "split": "k-fold", "fold": args.fold},
        transform={
            "version": "stft-v1",
            "nFft": args.n_fft,
            "hop": args.hop,
            "targetFreq": args.n_fft // 2 + 1,
            "targetTime": args.target_time,
            "cacheTime": args.cache_time,
        },
        model={"architecture": args.arch, "numClasses": NUM_CLASSES, "pretrained": not args.no_pretrained},
        training={
            "epochs": args.epochs,
            "batchSize": args.batch_size,
            "seed": args.seed,
            "folds": [args.fold] if args.fold is not None else list(range(args.n_splits)),
            "loss": args.loss,
            "selectMetric": args.select_metric,
        },
        device=str(device),
    )
    write_run_manifest(manifest_path, manifest)

    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        df = pd.read_csv(args.train_root / "index.csv")
        if args.max_samples:
            df = df.iloc[: args.max_samples].copy().reset_index(drop=True)
        splits = make_splits(df, args.n_splits, args.seed)
        folds = [args.fold] if args.fold is not None else list(range(args.n_splits))

        config_payload = vars(args).copy()
        config_payload.update({"device": str(device), "train_rows": len(df), "folds": folds})
        (args.save_dir / f"config_{args.tag or 'default'}.json").write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        results = []
        for fold in folds:
            if fold < 0 or fold >= args.n_splits:
                raise ValueError(f"fold must be in 0..{args.n_splits - 1}")
            train_idx, val_idx = splits[fold]
            results.append(train_fold(args, df, train_idx, val_idx, fold, device))

        summary_path = args.save_dir / f"summary_{args.tag or 'default'}.json"
        summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        finalize_run_manifest(
            manifest_path,
            "completed",
            outputs={
                "summary": summary_path.name,
                "config": f"config_{args.tag or 'default'}.json",
                "folds": [result["fold"] for result in results],
                "checkpoints": [Path(str(result["model_path"])).name for result in results],
                "oof": [Path(str(result["oof_path"])).name for result in results],
            },
            metrics={
                "folds": [
                    {
                        "fold": result["fold"],
                        "bestEpoch": result["best_epoch"],
                        "bestMetric": result["best_metric"],
                        "metrics": result["metrics"],
                    }
                    for result in results
                ]
            },
        )
        print(f"\nK-fold training summary saved: {summary_path}")
    except Exception as error:
        finalize_run_manifest(manifest_path, "failed", error={"type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
