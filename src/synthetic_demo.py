from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import NUM_CLASSES, RANDOM_SEED
from .data import label_to_multihot, load_index, resolve_iq_path
from .spectrogram import iq_to_spectrogram
from .submission import write_submission
from .validate_submission import validate_submission


SAMPLE_RATE = 4096.0
CLASS_FREQUENCIES = np.linspace(160.0, 1440.0, NUM_CLASSES)
INDEX_FIELDS = [
    "sample_id",
    "iq_npz_relpath",
    "has_node0",
    "has_node1",
    "has_node2",
    "sample_rate_node0",
    "sample_rate_node1",
    "sample_rate_node2",
]


@dataclass(frozen=True)
class DemoResult:
    output_dir: str
    train_samples: int
    test_samples: int
    exact_match_accuracy: float
    threshold: float
    submission_path: str
    model_path: str
    metrics_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic synthetic IQ training and inference demo on CPU."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--train-samples-per-class", type=int, default=4)
    return parser.parse_args()


def _interleave_iq(signal: np.ndarray) -> np.ndarray:
    max_value = max(float(np.max(np.abs(signal))), 1.0)
    scaled = signal / max_value * 12_000.0
    output = np.empty(signal.size * 2, dtype=np.int16)
    output[0::2] = np.clip(scaled.real, -32768, 32767).astype(np.int16)
    output[1::2] = np.clip(scaled.imag, -32768, 32767).astype(np.int16)
    return output


def _make_signal(labels: tuple[int, ...], node: int, rng: np.random.Generator) -> np.ndarray:
    sample_count = 2048
    times = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    signal = np.zeros(sample_count, dtype=np.complex128)
    for label in labels:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        node_offset = (node - 1) * 3.0
        signal += np.exp(2j * np.pi * (CLASS_FREQUENCIES[label] + node_offset) * times + 1j * phase)
    noise = rng.normal(0.0, 0.08, sample_count) + 1j * rng.normal(0.0, 0.08, sample_count)
    return _interleave_iq(signal + noise)


def _write_dataset(
    root: Path,
    label_sets: list[tuple[int, ...]],
    seed: int,
    include_labels: bool,
) -> None:
    iq_dir = root / "iq_sample"
    iq_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for sample_id, labels in enumerate(label_sets):
        missing_node = sample_id % 4 if sample_id % 4 < 3 else None
        arrays: dict[str, np.ndarray] = {}
        row: dict[str, object] = {
            "sample_id": sample_id,
            "iq_npz_relpath": f"iq_sample/{sample_id:04d}.npz",
        }
        for node in range(3):
            present = node != missing_node
            arrays[f"iq_node{node}"] = (
                _make_signal(labels, node, rng) if present else np.asarray([], dtype=np.int16)
            )
            arrays[f"sample_rate_node{node}"] = np.float32(SAMPLE_RATE if present else 0.0)
            row[f"has_node{node}"] = int(present)
            row[f"sample_rate_node{node}"] = SAMPLE_RATE if present else 0.0
        if include_labels:
            row["label_signature"] = "|".join(str(label) for label in labels)
        np.savez(iq_dir / f"{sample_id:04d}.npz", **arrays)
        rows.append(row)

    fields = [*INDEX_FIELDS, "label_signature"] if include_labels else INDEX_FIELDS
    with (root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _feature_vector(npz_path: Path) -> np.ndarray:
    features: list[np.ndarray] = []
    with np.load(npz_path) as data:
        for node in range(3):
            raw = data[f"iq_node{node}"]
            sample_rate = float(data[f"sample_rate_node{node}"])
            spec = iq_to_spectrogram(
                raw,
                sample_rate,
                n_fft=128,
                hop=32,
                target_freq=65,
                target_time=64,
            )
            if spec is None:
                features.append(np.zeros(65, dtype=np.float32))
            else:
                features.append(spec.mean(axis=1).astype(np.float32))
    return np.concatenate(features)


def _features(root: Path, rows: list[dict]) -> np.ndarray:
    return np.stack([_feature_vector(resolve_iq_path(root, row)) for row in rows])


def _constrain_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    predictions = (probabilities >= threshold).astype(np.int64)
    for row_index, row in enumerate(predictions):
        active = np.flatnonzero(row)
        if active.size == 0:
            row[int(np.argmax(probabilities[row_index]))] = 1
        elif active.size > 2:
            keep = np.argsort(probabilities[row_index])[-2:]
            row[:] = 0
            row[keep] = 1
    return predictions


def _select_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.linspace(0.25, 0.75, 21)
    return float(
        max(
            candidates,
            key=lambda threshold: np.mean(
                np.all(_constrain_predictions(probabilities, float(threshold)) == labels, axis=1)
            ),
        )
    )


def run_demo(output_dir: Path, seed: int = RANDOM_SEED, train_samples_per_class: int = 4) -> DemoResult:
    if train_samples_per_class < 2:
        raise ValueError("train_samples_per_class must be at least 2")

    output_dir = Path(output_dir)
    train_root = output_dir / "data" / "train"
    test_root = output_dir / "data" / "test"
    train_labels = [
        (label,)
        for _ in range(train_samples_per_class)
        for label in range(NUM_CLASSES)
    ]
    train_labels.extend((label, (label + 1) % NUM_CLASSES) for label in range(NUM_CLASSES))
    test_labels = [(label,) for label in range(NUM_CLASSES)]
    test_labels.extend((label, (label + 1) % NUM_CLASSES) for label in range(0, NUM_CLASSES, 2))

    _write_dataset(train_root, train_labels, seed=seed, include_labels=True)
    _write_dataset(test_root, test_labels, seed=seed + 1, include_labels=False)
    train_rows = load_index(train_root, has_labels=True)
    test_rows = load_index(test_root, has_labels=False)
    train_x = _features(train_root, train_rows)
    test_x = _features(test_root, test_rows)
    train_y = np.asarray([label_to_multihot(row["label_signature"]) for row in train_rows])
    test_y = np.asarray([label_to_multihot("|".join(map(str, labels))) for labels in test_labels])

    model = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(C=10.0, max_iter=500, random_state=seed, solver="liblinear")
        ),
    )
    model.fit(train_x, train_y)
    train_probabilities = model.predict_proba(train_x)
    threshold = _select_threshold(train_probabilities, train_y)
    predictions = _constrain_predictions(model.predict_proba(test_x), threshold)
    accuracy = float(np.mean(np.all(predictions == test_y, axis=1)))

    model_path = output_dir / "model.joblib"
    submission_path = output_dir / "submission.txt"
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    write_submission(test_rows, predictions.tolist(), submission_path)
    errors = validate_submission(submission_path, test_root)
    if errors:
        raise RuntimeError("Synthetic submission validation failed: " + "; ".join(errors))

    result = DemoResult(
        output_dir=str(output_dir.resolve()),
        train_samples=len(train_rows),
        test_samples=len(test_rows),
        exact_match_accuracy=accuracy,
        threshold=threshold,
        submission_path=str(submission_path.resolve()),
        model_path=str(model_path.resolve()),
        metrics_path=str(metrics_path.resolve()),
    )
    metrics_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = run_demo(args.output_dir, seed=args.seed, train_samples_per_class=args.train_samples_per_class)
    print("Synthetic CPU demo passed")
    print(f"Train samples: {result.train_samples}")
    print(f"Test samples: {result.test_samples}")
    print(f"Synthetic exact-match accuracy: {result.exact_match_accuracy:.3f}")
    print(f"Submission: {result.submission_path}")
    print(f"Metrics: {result.metrics_path}")


if __name__ == "__main__":
    main()
