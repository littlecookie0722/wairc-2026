from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wairc_rf.reproducibility import seed_worker, set_reproducible_seed

from .config import CACHE_DIR, NUM_CLASSES, OUTPUT_DIR, RANDOM_SEED, TRAIN_ROOT, VAL_RATIO
from .checkpoint import make_checkpoint_payload
from .dataset_fingerprint import fingerprint_dataset
from .run_manifest import (
    create_run_manifest,
    finalize_run_manifest,
    finalize_run_manifest_with_artifacts,
    make_run_id,
    write_run_manifest,
)
from .rule_artifact import make_rule_payload, write_rule_artifact
from .spectrogram import (
    DroneClassifier,
    DroneSpectrogramDataset,
    WarmupCosineLR,
    apply_thresholds,
    compute_pos_weight,
    enforce_count_constraint,
    exact_match,
    label_signature_to_multihot,
    make_loss,
    search_best_inference_rule,
)


DEFAULT_SAVE_DIR = OUTPUT_DIR / "spectrogram"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a high-score STFT spectrogram + 2D backbone model.")
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR / "stft")
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
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
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


def seed_everything(seed: int) -> None:
    """Keep the legacy training helper backed by the shared seed contract."""

    set_reproducible_seed(seed)


def split_dataframe(df: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=df["label_signature"],
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    except ValueError:
        rng = random.Random(seed)
        train_parts = []
        val_parts = []
        for _, group in df.groupby("label_signature", sort=True):
            indices = list(group.index)
            rng.shuffle(indices)
            val_count = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 0
            val_parts.append(df.loc[indices[:val_count]])
            train_parts.append(df.loc[indices[val_count:]])
        train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return train_df, val_df


def make_dataset(args: argparse.Namespace, dataframe: pd.DataFrame, augment: bool) -> DroneSpectrogramDataset:
    cache_split = f"train_fft{args.n_fft}_hop{args.hop}_time{args.cache_time}"
    return DroneSpectrogramDataset(
        dataframe=dataframe,
        data_root=args.train_root,
        num_classes=NUM_CLASSES,
        n_fft=args.n_fft,
        hop=args.hop,
        target_freq=args.n_fft // 2 + 1,
        target_time=args.target_time,
        cache_time=args.cache_time,
        cache_dir=args.cache_dir,
        cache_split=cache_split,
        is_train=True,
        augment=augment,
        time_mask_param=args.time_mask,
        freq_mask_param=args.freq_mask,
        specaug_p=args.specaug_p,
    )


def make_loader(args: argparse.Namespace, dataset: DroneSpectrogramDataset, batch_size: int, shuffle: bool, device: torch.device, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
    )


def label_matrix(signatures: pd.Series) -> np.ndarray:
    return np.stack([label_signature_to_multihot(value, NUM_CLASSES) for value in signatures]).astype(np.float32)


def prediction_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = enforce_count_constraint(apply_thresholds(probs, [0.5] * labels.shape[1]), probs)
    return {
        "strict": exact_match(preds, labels),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def run_epoch(
    model: DroneClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    use_amp: bool,
    grad_clip: float,
    desc: str,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    probs_chunks = []
    labels_chunks = []

    progress = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
    for batch in progress:
        x = batch["x"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {desc}: {float(loss.detach().cpu())}")

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        probs_chunks.append(torch.sigmoid(logits.detach()).cpu().numpy())
        labels_chunks.append(labels.detach().cpu().numpy())
        progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

    probs = np.concatenate(probs_chunks, axis=0).astype(np.float32)
    labels_array = np.concatenate(labels_chunks, axis=0).astype(np.int32)
    return total_loss / max(1, total_count), prediction_metrics(probs, labels_array), probs, labels_array


def checkpoint_payload(args: argparse.Namespace, model: DroneClassifier, epoch: int, metrics: dict[str, float]) -> dict[str, object]:
    return make_checkpoint_payload(
        model_state_dict=model.state_dict(),
        arch=args.arch,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        num_classes=NUM_CLASSES,
        n_fft=args.n_fft,
        hop=args.hop,
        target_freq=args.n_fft // 2 + 1,
        target_time=args.target_time,
        cache_time=args.cache_time,
        epoch=epoch,
        metrics=metrics,
    )


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    run_id = args.run_id or make_run_id("spectrogram", args.seed)
    manifest_path = args.save_dir / "run-manifest.json"
    manifest = create_run_manifest(
        run_id=run_id,
        command=[sys.executable, "-m", "src.train_spectrogram", *sys.argv[1:]],
        args=vars(args),
        data={"adapter": "wairc-competition-v1", "split": "train-validation", "fold": None},
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
            "loss": args.loss,
            "selectMetric": args.select_metric,
        },
        device=str(device),
    )
    write_run_manifest(manifest_path, manifest)

    try:
        df = pd.read_csv(args.train_root / "index.csv")
        manifest["data"]["fingerprint"] = fingerprint_dataset(args.train_root, has_labels=True)
        write_run_manifest(manifest_path, manifest)
        if args.max_samples:
            df = df.iloc[: args.max_samples].copy()
        train_df, val_df = split_dataframe(df, args.val_ratio, args.seed)

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

        config_payload = vars(args).copy()
        config_payload.update({"device": str(device), "amp": use_amp, "train_rows": len(train_df), "val_rows": len(val_df)})
        (args.save_dir / "config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Model: {args.arch}, pretrained={not args.no_pretrained}, params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"Rows: train={len(train_df)} val={len(val_df)} | STFT={args.n_fft}/{args.hop} size={(args.n_fft // 2) + 1}x{args.target_time}")
        print(f"pos_weight: {[round(float(v), 3) for v in pos_weight.detach().cpu()]}")

        best_metric = -1.0
        best_epoch = 0
        best_probs: np.ndarray | None = None
        best_labels: np.ndarray | None = None
        history: list[dict[str, object]] = []
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            lr_now = scheduler.step()
            epoch_start = time.time()
            train_loss, train_metrics, _, _ = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler, use_amp, args.grad_clip, f"Epoch {epoch:03d}/train"
            )
            val_loss, val_metrics, val_probs, val_labels = run_epoch(
                model, val_loader, criterion, device, None, None, use_amp, args.grad_clip, f"Epoch {epoch:03d}/val"
            )

            selected = float(val_metrics[args.select_metric])
            record = {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train": train_metrics,
                "val": val_metrics,
                "seconds": time.time() - epoch_start,
            }
            history.append(record)
            (args.save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

            print(
                f"Epoch {epoch:03d} | lr={lr_now:.2e} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_strict={val_metrics['strict']:.4f} "
                f"val_micro={val_metrics['micro_f1']:.4f} val_macro={val_metrics['macro_f1']:.4f}"
            )

            if selected > best_metric + args.min_delta:
                best_metric = selected
                best_epoch = epoch
                best_probs = val_probs
                best_labels = val_labels
                torch.save(checkpoint_payload(args, model, epoch, val_metrics), args.save_dir / "best_model.pth")
                np.savez(args.save_dir / "best_val_probs.npz", probs=val_probs.astype(np.float16), labels=val_labels.astype(np.int8))
                print(f"Saved best checkpoint: epoch={epoch} {args.select_metric}={best_metric:.5f}")
            elif epoch - best_epoch >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

        if best_probs is None or best_labels is None:
            raise RuntimeError("Training finished without a best checkpoint")

        rule_payload = search_best_inference_rule(best_probs.astype(np.float32), best_labels.astype(np.int32), NUM_CLASSES)
        rule_payload = make_rule_payload(
            rule_payload["selected"],
            candidates=rule_payload.get("candidates"),
            num_classes=NUM_CLASSES,
        )
        write_rule_artifact(args.save_dir / "best_rule.json", rule_payload)
        selected_rule = rule_payload["selected"]
        print(
            f"Best inference rule: {selected_rule['method']} "
            f"accuracy={float(selected_rule['accuracy']):.5f} saved to {args.save_dir / 'best_rule.json'}"
        )
        finalize_run_manifest_with_artifacts(
            manifest_path,
            "completed",
            outputs={
                "checkpoint": "best_model.pth",
                "validationProbabilities": "best_val_probs.npz",
                "rule": "best_rule.json",
                "config": "config.json",
                "history": "history.json",
            },
            metrics={"bestEpoch": best_epoch, "bestMetric": best_metric, "rule": selected_rule},
        )
        print(f"Finished in {(time.time() - start_time) / 60:.1f} min. Best epoch={best_epoch}.")
    except Exception as error:
        finalize_run_manifest(manifest_path, "failed", error={"type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
