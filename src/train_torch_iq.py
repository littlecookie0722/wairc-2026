import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import CACHE_DIR, OUTPUT_DIR, RANDOM_SEED, TRAIN_ROOT, VAL_RATIO
from .data import load_index, stratified_split
from .torch_iq import (
    CachedIQDataset,
    IQCNN,
    IQDataset,
    build_iq_model,
    build_iq_tensor_cache,
    exact_match_accuracy_multihot,
    find_best_threshold,
    iq_cache_base_path,
    macro_f1_score,
    predictions_from_probabilities,
)


DEFAULT_MODEL_PATH = OUTPUT_DIR / "models" / "iq_cnn.pt"
DEFAULT_METRICS_PATH = OUTPUT_DIR / "metrics" / "iq_cnn_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CUDA PyTorch CNN on raw IQ samples.")
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--sequence-pairs", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-type", choices=["time", "timefreq"], default="time")
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--stft-n-fft", type=int, default=512)
    parser.add_argument("--stft-hop-length", type=int, default=256)
    parser.add_argument("--stft-max-frames", type=int, default=256)
    parser.add_argument("--freq-bins", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true", help="Read source .npz files directly for every batch.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild the IQ tensor cache before training.")
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def require_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. Install a CUDA-enabled PyTorch build "
            "in D:/Develop/Miniconda/envs/deepl, then retry."
        )
    return torch.device(requested)


def make_loader(
    root: Path,
    rows: list[dict],
    sequence_pairs: int,
    has_labels: bool,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    cache_base_path: Path | None,
) -> DataLoader:
    if cache_base_path is None:
        dataset = IQDataset(root=root, rows=rows, sequence_pairs=sequence_pairs, has_labels=has_labels)
    else:
        dataset = CachedIQDataset(cache_base_path, rows=rows, has_labels=has_labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(iq, meta)
            loss = loss_fn(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(target.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    probabilities = []
    targets = []

    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(iq, meta)
            loss = loss_fn(logits, target)

        batch_size = int(target.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())

    prob_array = np.vstack(probabilities).astype(np.float32)
    target_array = np.vstack(targets).astype(np.int64)
    threshold, threshold_metrics = find_best_threshold(target_array, prob_array)
    predictions = predictions_from_probabilities(prob_array, threshold)
    return {
        "loss": total_loss / max(1, total_samples),
        "threshold": threshold,
        "exact_match_accuracy": exact_match_accuracy_multihot(target_array, predictions),
        "macro_f1": macro_f1_score(target_array, predictions),
        "threshold_search": threshold_metrics,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model.__class__.__name__,
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "sequence_pairs": args.sequence_pairs,
            "threshold": float(metrics["threshold"]),
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = require_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    rows = load_index(args.train_root, has_labels=True)
    if args.max_samples:
        rows = rows[: args.max_samples]
        print(f"Using first {len(rows)} rows because --max-samples was provided.")

    cache_base_path = None
    if not args.no_cache:
        cache_base_path = iq_cache_base_path(
            name="train",
            sequence_pairs=args.sequence_pairs,
            max_samples=args.max_samples,
            cache_dir=args.cache_dir,
        )
        build_iq_tensor_cache(
            args.train_root,
            rows,
            sequence_pairs=args.sequence_pairs,
            has_labels=True,
            base_path=cache_base_path,
            dtype=args.cache_dtype,
            force=args.rebuild_cache,
        )

    train_rows, val_rows = stratified_split(rows, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = make_loader(
        args.train_root,
        train_rows,
        args.sequence_pairs,
        has_labels=True,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        cache_base_path=cache_base_path,
    )
    val_loader = make_loader(
        args.train_root,
        val_rows,
        args.sequence_pairs,
        has_labels=True,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        cache_base_path=cache_base_path,
    )

    model_kwargs = {}
    if args.model_type == "timefreq":
        model_kwargs = {
            "stft_n_fft": args.stft_n_fft,
            "stft_hop_length": args.stft_hop_length,
            "stft_max_frames": args.stft_max_frames,
            "freq_bins": args.freq_bins,
        }
    model = build_iq_model(
        args.model_type,
        width=args.width,
        dropout=args.dropout,
        **model_kwargs,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler(device.type, enabled=use_amp)

    best_exact = -1.0
    best_epoch = 0
    best_metrics: dict[str, object] | None = None
    history = []

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Train rows: {len(train_rows)}, validation rows: {len(val_rows)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, use_amp)
        val_metrics = evaluate(model, val_loader, loss_fn, device, use_amp)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={float(val_metrics['loss']):.4f} | "
            f"exact={float(val_metrics['exact_match_accuracy']):.4f} | "
            f"macro_f1={float(val_metrics['macro_f1']):.4f} | "
            f"threshold={float(val_metrics['threshold']):.2f}"
        )

        exact = float(val_metrics["exact_match_accuracy"])
        if exact > best_exact:
            best_exact = exact
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            save_checkpoint(args.model_path, model, optimizer, args, epoch, best_metrics)
            print(f"Saved best checkpoint: {args.model_path}")

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement.")
            break

    if best_metrics is None:
        raise RuntimeError("Training finished without producing validation metrics")

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": model.__class__.__name__,
        "model_type": args.model_type,
        "model_config": model.config,
        "train_root": str(args.train_root),
        "model_path": str(args.model_path),
        "random_seed": args.seed,
        "validation_ratio": args.val_ratio,
        "sequence_pairs": args.sequence_pairs,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "num_train_rows_total": len(rows),
        "num_train_rows_used": len(train_rows),
        "num_validation_rows": len(val_rows),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp": use_amp,
        "cache": {
            "enabled": cache_base_path is not None,
            "base_path": str(cache_base_path) if cache_base_path else None,
            "dtype": args.cache_dtype if cache_base_path else None,
            "rebuilt": bool(args.rebuild_cache) if cache_base_path else False,
        },
        "best_validation": best_metrics,
        "history": history,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
