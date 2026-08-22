from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .config import NUM_CLASSES
from .oof_aggregate_artifact import write_oof_aggregate_artifact
from .oof_artifact import load_oof_artifact
from .rule_artifact import make_rule_payload, write_rule_artifact
from .spectrogram import apply_inference_rule, search_best_inference_rule
from .train_spectrogram_kfold import DEFAULT_SAVE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search inference rules from spectrogram k-fold OOF predictions.")
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
    parser.add_argument("--output", type=Path, default=None)
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


def oof_tag(path: Path) -> str:
    match = re.fullmatch(r"oof_(.+)_fold\d+", path.stem)
    if match:
        return match.group(1)
    if re.fullmatch(r"oof_fold\d+", path.stem):
        return ""
    raise ValueError(f"Cannot determine tag from OOF filename: {path.name}")


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
        artifact = load_oof_artifact(path)
        probs = artifact["probs"]
        labels = artifact["labels"]
        indices = artifact["indices"]
        sample_ids = artifact["sample_ids"]
        for local_idx, original_idx in enumerate(indices.tolist()):
            index = int(original_idx)
            sample_id = int(sample_ids[local_idx])
            existing_label = label_by_index.get(index)
            if existing_label is not None and not np.array_equal(existing_label, labels[local_idx]):
                raise ValueError(f"OOF label mismatch for original index {index}")
            existing_sample_id = sample_id_by_index.get(index)
            if existing_sample_id is not None and existing_sample_id != sample_id:
                raise ValueError(f"OOF sample ID mismatch for original index {index}")
            prob_by_index[index].append(probs[local_idx])
            label_by_index[index] = labels[local_idx]
            sample_id_by_index[index] = sample_id

    ordered = sorted(prob_by_index)
    probs_out = np.stack([np.mean(prob_by_index[idx], axis=0) for idx in ordered]).astype(np.float32)
    labels_out = np.stack([label_by_index[idx] for idx in ordered]).astype(np.int32)
    sample_ids_out = np.asarray([sample_id_by_index[idx] for idx in ordered], dtype=np.int64)
    return probs_out, labels_out, sample_ids_out


def load_weighted_oof(paths: list[Path], tag_weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob_by_index: dict[int, list[tuple[float, np.ndarray]]] = defaultdict(list)
    label_by_index: dict[int, np.ndarray] = {}
    sample_id_by_index: dict[int, int] = {}
    seen_tags: set[str] = set()

    for path in paths:
        tag = oof_tag(path)
        if tag not in tag_weights:
            raise ValueError(f"No weight supplied for OOF tag '{tag}'")
        seen_tags.add(tag)
        artifact = load_oof_artifact(path)
        probs = artifact["probs"]
        labels = artifact["labels"]
        indices = artifact["indices"]
        sample_ids = artifact["sample_ids"]
        for local_idx, original_idx in enumerate(indices.tolist()):
            index = int(original_idx)
            sample_id = int(sample_ids[local_idx])
            existing_label = label_by_index.get(index)
            if existing_label is not None and not np.array_equal(existing_label, labels[local_idx]):
                raise ValueError(f"OOF label mismatch for original index {index}")
            existing_sample_id = sample_id_by_index.get(index)
            if existing_sample_id is not None and existing_sample_id != sample_id:
                raise ValueError(f"OOF sample ID mismatch for original index {index}")
            prob_by_index[index].append((tag_weights[tag], probs[local_idx]))
            label_by_index[index] = labels[local_idx]
            sample_id_by_index[index] = sample_id

    missing_tags = set(tag_weights) - seen_tags
    if missing_tags:
        raise FileNotFoundError(f"No OOF files found for weighted tags: {sorted(missing_tags)}")

    ordered = sorted(prob_by_index)
    weighted_probs = []
    for idx in ordered:
        entries = prob_by_index[idx]
        total_weight = sum(weight for weight, _ in entries)
        weighted_probs.append(sum(weight * probs for weight, probs in entries) / total_weight)
    probs_out = np.stack(weighted_probs).astype(np.float32)
    labels_out = np.stack([label_by_index[idx] for idx in ordered]).astype(np.int32)
    sample_ids_out = np.asarray([sample_id_by_index[idx] for idx in ordered], dtype=np.int64)
    return probs_out, labels_out, sample_ids_out


def main() -> None:
    args = parse_args()
    tag_weights = parse_tag_weights(args.tag_weights)
    output = args.output or (args.save_dir / "best_rule_kfold.json")
    paths = discover_oof_files(args.save_dir, args.tags, args.include_default)
    if not paths:
        raise FileNotFoundError(f"No OOF files found in {args.save_dir}")

    print("OOF files:")
    for path in paths:
        print(f"  {path}")

    probs, labels, sample_ids = (
        load_weighted_oof(paths, tag_weights) if tag_weights else load_averaged_oof(paths)
    )
    rule_payload = search_best_inference_rule(probs, labels, NUM_CLASSES)
    selected = rule_payload["selected"]
    preds = apply_inference_rule(probs, selected)
    rule_payload["oof_micro_f1"] = float(f1_score(labels, preds, average="micro", zero_division=0))
    rule_payload["oof_macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    rule_payload["num_oof_samples"] = int(len(labels))
    rule_payload["source_files"] = [path.name for path in paths]
    rule_payload["tag_weights"] = tag_weights
    rule_payload = make_rule_payload(
        rule_payload["selected"],
        candidates=rule_payload.get("candidates"),
        num_classes=NUM_CLASSES,
        oof_micro_f1=rule_payload["oof_micro_f1"],
        oof_macro_f1=rule_payload["oof_macro_f1"],
        num_oof_samples=rule_payload["num_oof_samples"],
        source_files=rule_payload["source_files"],
        tag_weights=tag_weights,
    )

    write_rule_artifact(output, rule_payload)
    write_oof_aggregate_artifact(
        output.with_suffix(".oof_probs.npz"),
        probs=probs,
        labels=labels,
        sample_ids=sample_ids,
        source_files=[path.name for path in paths],
        tag_weights=tag_weights,
    )

    sums = preds.sum(axis=1)
    print(f"Saved rule: {output}")
    print(f"Selected: {selected}")
    print(f"OOF micro_f1={rule_payload['oof_micro_f1']:.5f} macro_f1={rule_payload['oof_macro_f1']:.5f}")
    print(f"OOF prediction counts: single={(sums == 1).sum()} double={(sums == 2).sum()} other={((sums == 0) | (sums > 2)).sum()}")


if __name__ == "__main__":
    main()
