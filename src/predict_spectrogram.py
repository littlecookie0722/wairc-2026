from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CACHE_DIR, NUM_CLASSES, OUTPUT_DIR, SUBMISSION_PATH, TEST_ROOT
from .checkpoint import load_checkpoint
from .data import load_index
from .spectrogram import (
    DroneClassifier,
    DroneSpectrogramDataset,
    apply_inference_rule,
    load_inference_rule,
)
from .submission import write_submission
from .train_spectrogram import DEFAULT_SAVE_DIR


DEFAULT_OUTPUT_PATH = SUBMISSION_PATH.parent / "submission_spectrogram.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with the STFT spectrogram model.")
    parser.add_argument("--test-root", type=Path, default=TEST_ROOT)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rule-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR / "stft")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tta-crops", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-probs", type=Path, default=OUTPUT_DIR / "spectrogram" / "test_probs.npy")
    return parser.parse_args()


@torch.no_grad()
def predict_probabilities(model: DroneClassifier, loader: DataLoader, device: torch.device) -> tuple[list[int], np.ndarray]:
    model.eval()
    sample_ids: list[int] = []
    probs_chunks = []
    for batch in tqdm(loader, desc="predict", dynamic_ncols=True):
        x = batch["x"].to(device, non_blocking=True)
        logits = model(x)
        probs_chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
        sample_ids.extend([int(value) for value in batch["sample_id"].detach().cpu().numpy().tolist()])
    return sample_ids, np.concatenate(probs_chunks, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    checkpoint_path = args.checkpoint or (args.save_dir / "best_model.pth")
    rule_path = args.rule_path or (args.save_dir / "best_rule.json")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    model = DroneClassifier(
        num_classes=int(checkpoint.get("num_classes", NUM_CLASSES)),
        arch=str(checkpoint["arch"]),
        pretrained=False,
        dropout=float(checkpoint.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    df = pd.read_csv(args.test_root / "index.csv")
    dataset = DroneSpectrogramDataset(
        dataframe=df,
        data_root=args.test_root,
        num_classes=NUM_CLASSES,
        n_fft=int(checkpoint.get("n_fft", 512)),
        hop=int(checkpoint.get("hop", 128)),
        target_freq=int(checkpoint.get("target_freq", 257)),
        target_time=int(checkpoint.get("target_time", 768)),
        cache_time=int(checkpoint.get("cache_time", 1536)),
        cache_dir=args.cache_dir,
        cache_split=(
            f"test_fft{int(checkpoint.get('n_fft', 512))}_"
            f"hop{int(checkpoint.get('hop', 128))}_"
            f"time{int(checkpoint.get('cache_time', 1536))}"
        ),
        is_train=False,
        augment=False,
        tta_crops=args.tta_crops,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    sample_ids, probs = predict_probabilities(model, loader, device)
    rule = load_inference_rule(rule_path, NUM_CLASSES)
    preds = apply_inference_rule(probs, rule).astype(int)
    rows_by_id = {int(row["sample_id"]): row for row in load_index(args.test_root, has_labels=False)}
    rows = [rows_by_id[sample_id] for sample_id in sample_ids]
    path = write_submission(rows, preds.tolist(), args.output_path)

    if args.save_probs:
        args.save_probs.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_probs, probs)

    sums = preds.sum(axis=1)
    print(f"Wrote submission: {path}")
    print(f"Used checkpoint: {checkpoint_path}")
    print(f"Used rule: {rule}")
    print(f"Prediction counts: single={(sums == 1).sum()} double={(sums == 2).sum()} other={((sums == 0) | (sums > 2)).sum()}")


if __name__ == "__main__":
    main()
