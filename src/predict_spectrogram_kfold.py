from __future__ import annotations

import argparse
import random
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CACHE_DIR, NUM_CLASSES, OUTPUT_DIR, SUBMISSION_PATH, TEST_ROOT
from .data import load_index
from .spectrogram import DroneClassifier, DroneSpectrogramDataset, apply_inference_rule, load_inference_rule
from .submission import write_submission
from .train_spectrogram_kfold import DEFAULT_SAVE_DIR


DEFAULT_OUTPUT_PATH = SUBMISSION_PATH.parent / "submission_spectrogram_kfold.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble predict with spectrogram k-fold checkpoints.")
    parser.add_argument("--test-root", type=Path, default=TEST_ROOT)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument(
        "--tag-weights",
        nargs="*",
        default=[],
        metavar="TAG=WEIGHT",
        help="Optional architecture-level weights, for example b0=0.5 r34=0.4 convnext=0.1.",
    )
    parser.add_argument("--include-default", action="store_true")
    parser.add_argument("--rule-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR / "stft")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tta-crops", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-probs", type=Path, default=OUTPUT_DIR / "spectrogram_kfold" / "test_probs_kfold.npy")
    return parser.parse_args()


def parse_tag_weights(entries: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for entry in entries:
        tag, separator, raw_weight = entry.partition("=")
        if not separator or not tag:
            raise ValueError(f"Invalid tag weight '{entry}'; expected TAG=WEIGHT")
        weight = float(raw_weight)
        if weight <= 0:
            raise ValueError(f"Weight for tag '{tag}' must be positive")
        weights[tag] = weight
    total = sum(weights.values())
    return {tag: weight / total for tag, weight in weights.items()} if total else {}


def seed_inference(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def discover_model_files(save_dir: Path, tags: list[str], include_default: bool) -> list[Path]:
    patterns = []
    if include_default or not tags:
        patterns.append("best_model_fold*.pth")
    for tag in tags:
        patterns.append(f"best_model_{tag}_fold*.pth")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(save_dir.glob(pattern)))
    return sorted(set(paths))


def stft_key(checkpoint: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(checkpoint.get("n_fft", 512)),
        int(checkpoint.get("hop", 128)),
        int(checkpoint.get("target_freq", 257)),
        int(checkpoint.get("target_time", 768)),
        int(checkpoint.get("cache_time", 1536)),
    )


def cache_split_from_key(key: tuple[int, int, int, int, int]) -> str:
    n_fft, hop, target_freq, target_time, cache_time = key
    return f"test_fft{n_fft}_hop{hop}_freq{target_freq}_target{target_time}_time{cache_time}"


def inspect_model_paths(
    paths: list[Path],
) -> tuple[dict[tuple[int, int, int, int, int], list[Path]], dict[Path, str]]:
    groups: dict[tuple[int, int, int, int, int], list[Path]] = defaultdict(list)
    tags: dict[Path, str] = {}
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu")
        groups[stft_key(checkpoint)].append(path)
        tags[path] = str(checkpoint.get("tag", ""))
        del checkpoint
    return dict(groups), tags


def group_model_paths(paths: list[Path]) -> dict[tuple[int, int, int, int, int], list[Path]]:
    groups, _ = inspect_model_paths(paths)
    return groups


@torch.no_grad()
def predict_one_model(
    path: Path,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], np.ndarray]:
    checkpoint = torch.load(path, map_location=device)
    model = DroneClassifier(
        num_classes=int(checkpoint.get("num_classes", NUM_CLASSES)),
        arch=str(checkpoint["arch"]),
        pretrained=False,
        dropout=float(checkpoint.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_ids: list[int] = []
    chunks = []
    for batch in tqdm(loader, desc=path.name, dynamic_ncols=True, leave=False):
        x = batch["x"].to(device, non_blocking=True)
        logits = model(x)
        chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
        sample_ids.extend([int(value) for value in batch["sample_id"].detach().cpu().numpy().tolist()])

    del model
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sample_ids, np.concatenate(chunks, axis=0).astype(np.float32)


def make_loader_for_key(args: argparse.Namespace, df: pd.DataFrame, key: tuple[int, int, int, int, int]) -> DataLoader:
    n_fft, hop, target_freq, target_time, cache_time = key
    dataset = DroneSpectrogramDataset(
        dataframe=df,
        data_root=args.test_root,
        num_classes=NUM_CLASSES,
        n_fft=n_fft,
        hop=hop,
        target_freq=target_freq,
        target_time=target_time,
        cache_time=cache_time,
        cache_dir=args.cache_dir,
        cache_split=cache_split_from_key(key),
        is_train=False,
        augment=False,
        tta_crops=args.tta_crops,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    seed_inference(args.seed)
    tag_weights = parse_tag_weights(args.tag_weights)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    model_paths = discover_model_files(args.save_dir, args.tags, args.include_default)
    if not model_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {args.save_dir}")

    print("Model files:")
    for path in model_paths:
        print(f"  {path}")

    df = pd.read_csv(args.test_root / "index.csv")
    groups, model_tags = inspect_model_paths(model_paths)
    model_counts_by_tag = Counter(model_tags.values())
    if tag_weights:
        selected_tags = set(model_counts_by_tag)
        if selected_tags != set(tag_weights):
            raise ValueError(
                f"Selected model tags {sorted(selected_tags)} do not match weighted tags {sorted(tag_weights)}"
            )
        model_weights = {
            path: tag_weights[tag] / model_counts_by_tag[tag]
            for path, tag in model_tags.items()
        }
    else:
        model_weights = {path: 1.0 / len(model_paths) for path in model_paths}
    total_probs: np.ndarray | None = None
    reference_ids: list[int] | None = None
    model_count = 0
    total_weight = 0.0

    for key, paths in groups.items():
        print(f"\nSTFT group {key}: {len(paths)} model(s)")
        loader = make_loader_for_key(args, df, key)
        for path in paths:
            sample_ids, probs = predict_one_model(path, loader, device)
            if total_probs is None:
                total_probs = np.zeros_like(probs, dtype=np.float32)
                reference_ids = sample_ids
            elif sample_ids != reference_ids:
                raise RuntimeError(f"Sample order mismatch while predicting {path}")
            model_weight = model_weights[path]
            total_probs += probs * model_weight
            total_weight += model_weight
            model_count += 1

    if total_probs is None or reference_ids is None or model_count == 0:
        raise RuntimeError("No probabilities were produced")

    probs = total_probs / total_weight
    rule_path = args.rule_path or (args.save_dir / "best_rule_kfold.json")
    rule = load_inference_rule(rule_path, NUM_CLASSES)
    preds = apply_inference_rule(probs, rule).astype(int)

    rows_by_id = {int(row["sample_id"]): row for row in load_index(args.test_root, has_labels=False)}
    rows = [rows_by_id[sample_id] for sample_id in reference_ids]
    path = write_submission(rows, preds.tolist(), args.output_path)

    if args.save_probs:
        args.save_probs.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_probs, probs)

    sums = preds.sum(axis=1)
    print(f"\nWrote submission: {path}")
    print(f"Averaged models: {model_count}")
    print(f"Tag weights: {tag_weights or 'equal per model'}")
    print(f"Used rule: {rule}")
    print(f"Prediction counts: single={(sums == 1).sum()} double={(sums == 2).sum()} other={((sums == 0) | (sums > 2)).sum()}")


if __name__ == "__main__":
    main()
