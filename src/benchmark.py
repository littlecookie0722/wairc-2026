from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .config import NUM_CLASSES, RANDOM_SEED
from .synthetic_demo import (
    CLASS_FREQUENCIES,
    DemoResult,
    MISSING_NODE_PATTERN,
    NODE_FREQUENCY_OFFSET,
    NOISE_STD,
    SAMPLE_RATE,
    SIGNAL_SAMPLE_COUNT,
    SYNTHETIC_GENERATOR_VERSION,
    SYNTHETIC_HOP,
    SYNTHETIC_N_FFT,
    SYNTHETIC_TARGET_FREQ,
    SYNTHETIC_TARGET_TIME,
    run_demo,
)


BENCHMARK_MANIFEST_SCHEMA = "benchmark-manifest-v1"
BENCHMARK_REPORT_SCHEMA = "benchmark-report-v1"
BENCHMARK_GENERATOR_NAME = "wairc.synthetic_iq"


@dataclass(frozen=True)
class BenchmarkCondition:
    name: str
    noise_std: float
    missing_node_pattern: tuple[int | None, ...]
    artifact_dir: str


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    train_samples_per_class: int
    conditions: tuple[BenchmarkCondition, ...] = ()


BENCHMARK_PROFILES = {
    "cpu-smoke": BenchmarkProfile(name="cpu-smoke", train_samples_per_class=2),
    "robustness-small": BenchmarkProfile(
        name="robustness-small",
        train_samples_per_class=2,
        conditions=(
            BenchmarkCondition("baseline", NOISE_STD, MISSING_NODE_PATTERN, "conditions/baseline"),
            BenchmarkCondition("high-noise", 0.20, MISSING_NODE_PATTERN, "conditions/high-noise"),
            BenchmarkCondition("node0-missing", NOISE_STD, (0,), "conditions/node0-missing"),
        ),
    ),
}


@dataclass(frozen=True)
class BenchmarkResult:
    output_dir: str
    profile: str
    seed: int
    manifest_path: str
    report_path: str
    exact_match_accuracy: float
    deterministic_signature: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _condition_manifest(condition: BenchmarkCondition) -> dict[str, object]:
    return {
        "name": condition.name,
        "noise_std": condition.noise_std,
        "missing_node_pattern": list(condition.missing_node_pattern),
    }


def _manifest(profile: BenchmarkProfile, seed: int) -> dict[str, object]:
    return {
        "schemaVersion": BENCHMARK_MANIFEST_SCHEMA,
        "profile": profile.name,
        "seed": seed,
        "generator": {
            "name": BENCHMARK_GENERATOR_NAME,
            "version": SYNTHETIC_GENERATOR_VERSION,
        },
        "data": {
            "sample_rate_hz": SAMPLE_RATE,
            "node_count": 3,
            "num_classes": NUM_CLASSES,
            "class_mapping": list(range(NUM_CLASSES)),
            "class_frequencies_hz": [float(value) for value in CLASS_FREQUENCIES],
            "signal_sample_count": SIGNAL_SAMPLE_COUNT,
            "noise_std": NOISE_STD,
            "node_frequency_offset_hz": NODE_FREQUENCY_OFFSET,
            "missing_node_pattern": list(MISSING_NODE_PATTERN),
        },
        "transform": {
            "profile": "stft-v1",
            "n_fft": SYNTHETIC_N_FFT,
            "hop": SYNTHETIC_HOP,
            "target_freq": SYNTHETIC_TARGET_FREQ,
            "target_time": SYNTHETIC_TARGET_TIME,
        },
        "training": {
            "model": "sklearn.OneVsRestClassifier(LogisticRegression)",
            "train_samples_per_class": profile.train_samples_per_class,
            "solver": "liblinear",
            "max_iter": 500,
            "C": 10.0,
        },
        "evaluation": {
            "metric": "exact_match_accuracy",
            "additional_metrics": ["macro_f1", "per_class_recall"],
            "synthetic_only": True,
            "conditions": [
                _condition_manifest(condition)
                for condition in (
                    profile.conditions
                    or (BenchmarkCondition("cpu-smoke", NOISE_STD, MISSING_NODE_PATTERN, "demo"),)
                )
            ],
        },
    }


def _metrics(result: DemoResult) -> dict[str, object]:
    return {
        "exact_match_accuracy": float(result.exact_match_accuracy),
        "macro_f1": float(result.macro_f1),
        "per_class_recall": [float(value) for value in result.per_class_recall],
        "threshold": float(result.threshold),
        "train_samples": int(result.train_samples),
        "test_samples": int(result.test_samples),
    }


def _signature(manifest: dict[str, object], metrics: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json({"manifest": manifest, "metrics": metrics})).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_benchmark(
    output_dir: Path,
    profile: str = "cpu-smoke",
    seed: int = RANDOM_SEED,
) -> BenchmarkResult:
    selected_profile = BENCHMARK_PROFILES.get(profile)
    if selected_profile is None:
        available = ", ".join(sorted(BENCHMARK_PROFILES))
        raise ValueError(f"Unknown benchmark profile {profile!r}. Available profiles: {available}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(selected_profile, seed)
    manifest_path = output_dir / "benchmark-manifest.json"
    report_path = output_dir / "benchmark-report.json"
    _write_json(manifest_path, manifest)

    started = time.perf_counter()
    if selected_profile.conditions:
        condition_metrics: list[dict[str, object]] = []
        for condition in selected_profile.conditions:
            demo_result = run_demo(
                output_dir / condition.artifact_dir,
                seed=seed,
                train_samples_per_class=selected_profile.train_samples_per_class,
                test_noise_std=condition.noise_std,
                test_missing_node_pattern=condition.missing_node_pattern,
            )
            condition_metrics.append(
                {
                    "name": condition.name,
                    "metrics": _metrics(demo_result),
                    "artifacts": [
                        f"{condition.artifact_dir}/metrics.json",
                        f"{condition.artifact_dir}/submission.txt",
                    ],
                }
            )
        metrics: dict[str, object] = {"conditions": condition_metrics}
        exact_match_accuracy = float(condition_metrics[0]["metrics"]["exact_match_accuracy"])
        artifacts = [
            artifact
            for condition in condition_metrics
            for artifact in condition["artifacts"]
        ]
    else:
        demo_result = run_demo(
            output_dir / "demo",
            seed=seed,
            train_samples_per_class=selected_profile.train_samples_per_class,
        )
        metrics = _metrics(demo_result)
        exact_match_accuracy = float(demo_result.exact_match_accuracy)
        artifacts = ["demo/metrics.json", "demo/submission.txt"]
    deterministic_signature = _signature(manifest, metrics)
    report = {
        "schemaVersion": BENCHMARK_REPORT_SCHEMA,
        "profile": selected_profile.name,
        "seed": seed,
        "status": "passed",
        "metrics": metrics,
        "deterministic_signature": deterministic_signature,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "manifest": manifest_path.name,
        "artifacts": artifacts,
    }
    _write_json(report_path, report)
    return BenchmarkResult(
        output_dir=str(output_dir.resolve()),
        profile=selected_profile.name,
        seed=seed,
        manifest_path=str(manifest_path.resolve()),
        report_path=str(report_path.resolve()),
        exact_match_accuracy=exact_match_accuracy,
        deterministic_signature=deterministic_signature,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible synthetic benchmarks.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="Run a synthetic benchmark profile")
    run_parser.add_argument("--profile", choices=sorted(BENCHMARK_PROFILES), default="cpu-smoke")
    run_parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    run_parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_benchmark(args.output_dir, profile=args.profile, seed=args.seed)
    print("Synthetic benchmark passed")
    print(f"Profile: {result.profile}")
    metric_label = (
        "Baseline synthetic exact-match accuracy"
        if result.profile == "robustness-small"
        else "Synthetic exact-match accuracy"
    )
    print(f"{metric_label}: {result.exact_match_accuracy:.3f}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Report: {result.report_path}")


if __name__ == "__main__":
    main()
