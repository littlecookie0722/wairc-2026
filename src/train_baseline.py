import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import (
    MAX_IQ_PAIRS,
    METRICS_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
    TRAIN_ROOT,
    VAL_RATIO,
)
from .data import label_to_multihot, load_index, stratified_split
from .features import cache_path, extract_features_for_rows
from .model import NearestCentroidClassifier, save_model


def exact_match_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline AI-radio classifier.")
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-pairs", type=int, default=MAX_IQ_PAIRS)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_index(args.train_root, has_labels=True)
    if args.max_samples:
        rows = rows[: args.max_samples]
        print(f"Using first {len(rows)} rows because --max-samples was provided.")

    # Fail fast if label parsing ever drifts from the competition format.
    for row in rows[:10]:
        label_to_multihot(row["label_signature"])

    train_rows, val_rows = stratified_split(rows, val_ratio=args.val_ratio, seed=args.seed)
    train_cache = cache_path("train_split", args.max_pairs, args.max_samples)
    val_cache = cache_path("val_split", args.max_pairs, args.max_samples)

    x_train = extract_features_for_rows(
        args.train_root,
        train_rows,
        cache_path=train_cache,
        max_pairs=args.max_pairs,
        force=args.force_cache,
    )
    x_val = extract_features_for_rows(
        args.train_root,
        val_rows,
        cache_path=val_cache,
        max_pairs=args.max_pairs,
        force=args.force_cache,
    )
    y_train = [row["label_signature"] for row in train_rows]
    y_val = [row["label_signature"] for row in val_rows]

    model = NearestCentroidClassifier().fit(x_train, y_train)
    val_pred = model.predict(x_val)
    accuracy = exact_match_accuracy(y_val, val_pred)

    save_model(model, args.model_path)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "NearestCentroidClassifier",
        "train_root": str(args.train_root),
        "random_seed": args.seed,
        "validation_ratio": args.val_ratio,
        "max_pairs": args.max_pairs,
        "num_train_rows_total": len(rows),
        "num_train_rows_used": len(train_rows),
        "num_validation_rows": len(val_rows),
        "num_features": int(x_train.shape[1]),
        "num_label_combinations": len(set(y_train)),
        "local_exact_match_accuracy": accuracy,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model: {args.model_path}")
    print(f"Saved metrics: {args.metrics_path}")
    print(f"Local exact-match accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

