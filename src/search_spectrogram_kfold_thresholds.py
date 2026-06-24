from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .config import NUM_CLASSES, OUTPUT_DIR
from .spectrogram import apply_inference_rule, search_best_inference_rule
from .train_spectrogram_kfold import DEFAULT_SAVE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search inference rules from spectrogram k-fold OOF predictions.")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--include-default", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def discover_oof_files(save_dir: Path, tags: list[str], include_default: bool) -> list[Path]:
    patterns = []
    if include_default or not tags:
        patterns.append("oof_fold*.npz")
    for tag in tags:
        patterns.append(f"oof_{tag}_fold*.npz")

    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(save_dir.glob(pattern)))
    return sorted(set(files))


def load_averaged_oof(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob_by_index: dict[int, list[np.ndarray]] = defaultdict(list)
    label_by_index: dict[int, np.ndarray] = {}
    sample_id_by_index: dict[int, int] = {}

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            probs = data["probs"].astype(np.float32)
            labels = data["labels"].astype(np.int32)
            indices = data["indices"].astype(np.int64)
            sample_ids = data["sample_ids"].astype(np.int64) if "sample_ids" in data else indices
        for local_idx, original_idx in enumerate(indices.tolist()):
            prob_by_index[int(original_idx)].append(probs[local_idx])
            label_by_index[int(original_idx)] = labels[local_idx]
            sample_id_by_index[int(original_idx)] = int(sample_ids[local_idx])

    ordered = sorted(prob_by_index)
    probs_out = np.stack([np.mean(prob_by_index[idx], axis=0) for idx in ordered]).astype(np.float32)
    labels_out = np.stack([label_by_index[idx] for idx in ordered]).astype(np.int32)
    sample_ids_out = np.asarray([sample_id_by_index[idx] for idx in ordered], dtype=np.int64)
    return probs_out, labels_out, sample_ids_out


def main() -> None:
    args = parse_args()
    output = args.output or (args.save_dir / "best_rule_kfold.json")
    paths = discover_oof_files(args.save_dir, args.tags, args.include_default)
    if not paths:
        raise FileNotFoundError(f"No OOF files found in {args.save_dir}")

    print("OOF files:")
    for path in paths:
        print(f"  {path}")

    probs, labels, sample_ids = load_averaged_oof(paths)
    rule_payload = search_best_inference_rule(probs, labels, NUM_CLASSES)
    selected = rule_payload["selected"]
    preds = apply_inference_rule(probs, selected)
    rule_payload["oof_micro_f1"] = float(f1_score(labels, preds, average="micro", zero_division=0))
    rule_payload["oof_macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    rule_payload["num_oof_samples"] = int(len(labels))
    rule_payload["source_files"] = [str(path) for path in paths]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rule_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(output.with_suffix(".oof_probs.npz"), probs=probs.astype(np.float16), labels=labels.astype(np.int8), sample_ids=sample_ids)

    sums = preds.sum(axis=1)
    print(f"Saved rule: {output}")
    print(f"Selected: {selected}")
    print(f"OOF micro_f1={rule_payload['oof_micro_f1']:.5f} macro_f1={rule_payload['oof_macro_f1']:.5f}")
    print(f"OOF prediction counts: single={(sums == 1).sum()} double={(sums == 2).sum()} other={((sums == 0) | (sums > 2)).sum()}")


if __name__ == "__main__":
    main()
