from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from wairc_rf import CompetitionDatasetAdapter, RFDatasetAdapter, RFSample, SyntheticDatasetAdapter

from .config import NUM_CLASSES, RANDOM_SEED
from .data import label_to_multihot
from .spectrogram import iq_to_spectrogram
from .submission import write_submission
from .validate_submission import validate_submission


SAMPLE_RATE = 4096.0
CLASS_FREQUENCIES = np.linspace(160.0, 1440.0, NUM_CLASSES)
SYNTHETIC_GENERATOR_VERSION = "synthetic-iq-v1"
SIGNAL_SAMPLE_COUNT = 2048
NOISE_STD = 0.08
NODE_FREQUENCY_OFFSET = 3.0
MISSING_NODE_PATTERN: tuple[int | None, ...] = (0, 1, 2, None)
DEFAULT_FREQUENCY_OFFSET_HZ = 0.0
DEFAULT_TIMING_OFFSET_SAMPLES = 0
DEFAULT_SIGNAL_GAIN = 1.0
SYNTHETIC_N_FFT = 128
SYNTHETIC_HOP = 32
SYNTHETIC_TARGET_FREQ = 65
SYNTHETIC_TARGET_TIME = 64
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
    macro_f1: float
    per_class_recall: tuple[float, ...]
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


def _make_signal(
    labels: tuple[int, ...],
    node: int,
    rng: np.random.Generator,
    noise_std: float = NOISE_STD,
    frequency_offset_hz: float = DEFAULT_FREQUENCY_OFFSET_HZ,
    timing_offset_samples: int = DEFAULT_TIMING_OFFSET_SAMPLES,
    signal_gain: float = DEFAULT_SIGNAL_GAIN,
) -> np.ndarray:
    times = (np.arange(SIGNAL_SAMPLE_COUNT, dtype=np.float64) + timing_offset_samples) / SAMPLE_RATE
    signal = np.zeros(SIGNAL_SAMPLE_COUNT, dtype=np.complex128)
    for label in labels:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        node_offset = (node - 1) * NODE_FREQUENCY_OFFSET + frequency_offset_hz
        signal += np.exp(2j * np.pi * (CLASS_FREQUENCIES[label] + node_offset) * times + 1j * phase)
    noise = rng.normal(0.0, noise_std, SIGNAL_SAMPLE_COUNT) + 1j * rng.normal(
        0.0, noise_std, SIGNAL_SAMPLE_COUNT
    )
    return _interleave_iq(signal * signal_gain + noise)


def _write_dataset(
    root: Path,
    label_sets: list[tuple[int, ...]],
    seed: int,
    include_labels: bool,
    noise_std: float = NOISE_STD,
    missing_node_pattern: tuple[int | None, ...] = MISSING_NODE_PATTERN,
    frequency_offset_hz: float = DEFAULT_FREQUENCY_OFFSET_HZ,
    timing_offset_samples: int = DEFAULT_TIMING_OFFSET_SAMPLES,
    signal_gain: float = DEFAULT_SIGNAL_GAIN,
) -> None:
    iq_dir = root / "iq_sample"
    iq_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for sample_id, labels in enumerate(label_sets):
        missing_node = missing_node_pattern[sample_id % len(missing_node_pattern)]
        arrays: dict[str, np.ndarray] = {}
        row: dict[str, object] = {
            "sample_id": sample_id,
            "iq_npz_relpath": f"iq_sample/{sample_id:04d}.npz",
        }
        for node in range(3):
            present = node != missing_node
            arrays[f"iq_node{node}"] = (
                _make_signal(
                    labels,
                    node,
                    rng,
                    noise_std=noise_std,
                    frequency_offset_hz=frequency_offset_hz,
                    timing_offset_samples=timing_offset_samples,
                    signal_gain=signal_gain,
                )
                if present
                else np.asarray([], dtype=np.int16)
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


def _feature_vector(sample: RFSample) -> np.ndarray:
    features: list[np.ndarray] = []
    for node in sample.nodes:
        spec = iq_to_spectrogram(
            node.iq,
            node.sample_rate,
            n_fft=SYNTHETIC_N_FFT,
            hop=SYNTHETIC_HOP,
            target_freq=SYNTHETIC_TARGET_FREQ,
            target_time=SYNTHETIC_TARGET_TIME,
        )
        if spec is None:
            features.append(np.zeros(SYNTHETIC_TARGET_FREQ, dtype=np.float32))
        else:
            features.append(spec.mean(axis=1).astype(np.float32))
    return np.concatenate(features)


def _features(samples: RFDatasetAdapter) -> np.ndarray:
    return np.stack([_feature_vector(samples[index]) for index in range(len(samples))])


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


def _normalize_test_condition(
    noise_std: float | None,
    missing_node_pattern: tuple[int | None, ...] | None,
    frequency_offset_hz: float | None,
    timing_offset_samples: int | None,
    signal_gain: float | None,
) -> tuple[float, tuple[int | None, ...], float, int, float]:
    normalized_noise = NOISE_STD if noise_std is None else float(noise_std)
    normalized_pattern = MISSING_NODE_PATTERN if missing_node_pattern is None else tuple(missing_node_pattern)
    normalized_frequency = (
        DEFAULT_FREQUENCY_OFFSET_HZ if frequency_offset_hz is None else float(frequency_offset_hz)
    )
    normalized_timing = (
        DEFAULT_TIMING_OFFSET_SAMPLES if timing_offset_samples is None else timing_offset_samples
    )
    normalized_gain = DEFAULT_SIGNAL_GAIN if signal_gain is None else float(signal_gain)
    if not np.isfinite(normalized_noise) or normalized_noise < 0.0:
        raise ValueError("noise_std must be a finite non-negative number")
    if not normalized_pattern:
        raise ValueError("missing_node_pattern must not be empty")
    if any(node is not None and node not in {0, 1, 2} for node in normalized_pattern):
        raise ValueError("missing_node_pattern entries must be 0, 1, 2, or None")
    if not np.isfinite(normalized_frequency):
        raise ValueError("frequency_offset_hz must be finite")
    if isinstance(normalized_timing, bool) or not isinstance(normalized_timing, int):
        raise ValueError("timing_offset_samples must be an integer")
    if not np.isfinite(normalized_gain) or normalized_gain <= 0.0:
        raise ValueError("signal_gain must be a finite positive number")
    return normalized_noise, normalized_pattern, normalized_frequency, normalized_timing, normalized_gain


def run_demo(
    output_dir: Path,
    seed: int = RANDOM_SEED,
    train_samples_per_class: int = 4,
    test_noise_std: float | None = None,
    test_missing_node_pattern: tuple[int | None, ...] | None = None,
    test_frequency_offset_hz: float | None = None,
    test_timing_offset_samples: int | None = None,
    test_signal_gain: float | None = None,
) -> DemoResult:
    if train_samples_per_class < 2:
        raise ValueError("train_samples_per_class must be at least 2")
    (
        test_noise_std,
        test_missing_node_pattern,
        test_frequency_offset_hz,
        test_timing_offset_samples,
        test_signal_gain,
    ) = _normalize_test_condition(
        test_noise_std,
        test_missing_node_pattern,
        test_frequency_offset_hz,
        test_timing_offset_samples,
        test_signal_gain,
    )

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

    _write_dataset(
        train_root,
        train_labels,
        seed=seed,
        include_labels=True,
        noise_std=NOISE_STD,
        missing_node_pattern=MISSING_NODE_PATTERN,
        frequency_offset_hz=DEFAULT_FREQUENCY_OFFSET_HZ,
        timing_offset_samples=DEFAULT_TIMING_OFFSET_SAMPLES,
        signal_gain=DEFAULT_SIGNAL_GAIN,
    )
    _write_dataset(
        test_root,
        test_labels,
        seed=seed + 1,
        include_labels=False,
        noise_std=test_noise_std,
        missing_node_pattern=test_missing_node_pattern,
        frequency_offset_hz=test_frequency_offset_hz,
        timing_offset_samples=test_timing_offset_samples,
        signal_gain=test_signal_gain,
    )
    train_samples = SyntheticDatasetAdapter(CompetitionDatasetAdapter(train_root, has_labels=True))
    test_samples = SyntheticDatasetAdapter(CompetitionDatasetAdapter(test_root, has_labels=False))
    train_x = _features(train_samples)
    test_x = _features(test_samples)
    train_y = np.asarray(
        [
            label_to_multihot("|".join(str(label) for label in train_samples[index].labels or ()))
            for index in range(len(train_samples))
        ]
    )
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
    macro_f1 = float(f1_score(test_y, predictions, average="macro", zero_division=0))
    per_class_recall = tuple(
        float(value) for value in recall_score(test_y, predictions, average=None, zero_division=0)
    )

    model_path = output_dir / "model.joblib"
    submission_path = output_dir / "submission.txt"
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    submission_rows = [
        {"sample_id": test_samples[index].sample_id} for index in range(len(test_samples))
    ]
    write_submission(submission_rows, predictions.tolist(), submission_path)
    errors = validate_submission(submission_path, test_root)
    if errors:
        raise RuntimeError("Synthetic submission validation failed: " + "; ".join(errors))

    result = DemoResult(
        output_dir=str(output_dir.resolve()),
        train_samples=len(train_samples),
        test_samples=len(test_samples),
        exact_match_accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_recall=per_class_recall,
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
