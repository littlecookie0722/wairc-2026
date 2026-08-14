from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

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
BENCHMARK_FIXTURE_SCHEMA = "benchmark-fixture-v1"
BENCHMARK_GENERATOR_NAME = "wairc.synthetic_iq"

_EXPECTED_FIXTURE_REDISTRIBUTION = {
    "status": "repository-authored-parameters-only",
    "license": "MIT",
    "raw_iq_included": False,
    "model_weights_included": False,
    "private_labels_included": False,
    "external_recordings_included": False,
}


@dataclass(frozen=True)
class BenchmarkCondition:
    name: str
    noise_std: float
    missing_node_pattern: tuple[int | None, ...]
    artifact_dir: str
    frequency_offset_hz: float = 0.0
    timing_offset_samples: int = 0
    signal_gain: float = 1.0


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
            BenchmarkCondition("all-nodes-present", NOISE_STD, (None,), "conditions/all-nodes-present"),
            BenchmarkCondition("high-noise", 0.20, MISSING_NODE_PATTERN, "conditions/high-noise"),
            BenchmarkCondition("node0-missing", NOISE_STD, (0,), "conditions/node0-missing"),
            BenchmarkCondition("node1-missing", NOISE_STD, (1,), "conditions/node1-missing"),
            BenchmarkCondition("node2-missing", NOISE_STD, (2,), "conditions/node2-missing"),
            BenchmarkCondition(
                "frequency-offset",
                NOISE_STD,
                MISSING_NODE_PATTERN,
                "conditions/frequency-offset",
                frequency_offset_hz=180.0,
            ),
            BenchmarkCondition(
                "timing-offset",
                NOISE_STD,
                MISSING_NODE_PATTERN,
                "conditions/timing-offset",
                timing_offset_samples=32,
            ),
            BenchmarkCondition(
                "low-gain",
                NOISE_STD,
                MISSING_NODE_PATTERN,
                "conditions/low-gain",
                signal_gain=0.5,
            ),
            BenchmarkCondition(
                "combined-stress",
                0.20,
                MISSING_NODE_PATTERN,
                "conditions/combined-stress",
                frequency_offset_hz=180.0,
                timing_offset_samples=32,
                signal_gain=0.5,
            ),
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
        "frequency_offset_hz": condition.frequency_offset_hz,
        "timing_offset_samples": condition.timing_offset_samples,
        "signal_gain": condition.signal_gain,
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


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Report field {field!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Report field {field!r} must be finite")
    return number


def _metric_summary(metrics: object) -> dict[str, object]:
    if not isinstance(metrics, dict):
        raise ValueError("Report metrics must be an object")
    exact_match = _number(metrics.get("exact_match_accuracy"), "exact_match_accuracy")
    macro_f1 = _number(metrics.get("macro_f1"), "macro_f1")
    threshold = _number(metrics.get("threshold"), "threshold")
    recall = metrics.get("per_class_recall")
    if not isinstance(recall, list) or len(recall) != NUM_CLASSES:
        raise ValueError(f"Report per_class_recall must contain {NUM_CLASSES} values")
    recall_values = [_number(value, "per_class_recall") for value in recall]
    if any(not 0.0 <= value <= 1.0 for value in [exact_match, macro_f1, threshold, *recall_values]):
        raise ValueError("Report metrics must be between 0 and 1")
    test_samples = metrics.get("test_samples")
    if isinstance(test_samples, bool) or not isinstance(test_samples, int) or test_samples <= 0:
        raise ValueError("Report test_samples must be a positive integer")
    return {
        "exact_match_accuracy": exact_match,
        "macro_f1": macro_f1,
        "mean_recall": sum(recall_values) / len(recall_values),
        "min_recall": min(recall_values),
        "threshold": threshold,
        "test_samples": test_samples,
    }


def _relative_name(value: str, field: str) -> str:
    path = PureWindowsPath(value)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ValueError(f"Report field {field!r} must be a relative path")
    return value.replace("\\", "/")


def render_benchmark_summary(report: object) -> str:
    if not isinstance(report, dict) or report.get("schemaVersion") != BENCHMARK_REPORT_SCHEMA:
        raise ValueError(f"Expected {BENCHMARK_REPORT_SCHEMA} report")
    profile = report.get("profile")
    status = report.get("status")
    signature = report.get("deterministic_signature")
    seed = report.get("seed")
    if not isinstance(profile, str) or not profile:
        raise ValueError("Report profile must be a non-empty string")
    if not isinstance(status, str) or not status:
        raise ValueError("Report status must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Report seed must be an integer")
    if not isinstance(signature, str) or not signature:
        raise ValueError("Report deterministic_signature must be a non-empty string")

    raw_metrics = report.get("metrics")
    rows: list[tuple[str, dict[str, object]]] = []
    if isinstance(raw_metrics, dict) and "conditions" in raw_metrics:
        conditions = raw_metrics["conditions"]
        if not isinstance(conditions, list) or not conditions:
            raise ValueError("Report conditions must be a non-empty list")
        for condition in conditions:
            if not isinstance(condition, dict) or not isinstance(condition.get("name"), str):
                raise ValueError("Each report condition must have a name")
            rows.append((condition["name"], _metric_summary(condition.get("metrics"))))
    else:
        rows.append((profile, _metric_summary(raw_metrics)))

    lines = [
        "# Synthetic benchmark summary",
        "",
        f"- Profile: `{profile}`",
        f"- Seed: `{seed}`",
        f"- Status: **{status.upper()}**",
        "- Synthetic-only: `true`",
        f"- Deterministic signature: `{signature}`",
    ]
    manifest = report.get("manifest")
    if isinstance(manifest, str) and manifest:
        lines.append(f"- Manifest: `{_relative_name(manifest, 'manifest')}`")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Condition | Exact-match | Macro F1 | Mean recall | Min recall | Threshold | Test samples |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in rows:
        lines.append(
            f"| `{name}` | {metrics['exact_match_accuracy']:.3f} | "
            f"{metrics['macro_f1']:.3f} | {metrics['mean_recall']:.3f} | "
            f"{metrics['min_recall']:.3f} | {metrics['threshold']:.3f} | {metrics['test_samples']} |"
        )

    artifacts = report.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        lines.extend(["", "## Relative artifacts", ""])
        lines.extend(
            f"- `{_relative_name(artifact, 'artifacts')}`"
            for artifact in artifacts
            if isinstance(artifact, str)
        )
    lines.extend(
        [
            "",
            "> This summary reports a functional check on generated synthetic data; it is not a real-data benchmark or competition score.",
            "",
        ]
    )
    return "\n".join(lines)


def write_benchmark_summary(report_path: Path, output_path: Path | None = None) -> Path:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = render_benchmark_summary(report)
    output_path = report_path.with_name("benchmark-summary.md") if output_path is None else Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    return output_path


def _load_benchmark_fixture(fixture_path: Path) -> dict[str, object]:
    fixture_path = Path(fixture_path)
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Benchmark fixture {fixture_path.name!r} was not found") from None
    except OSError:
        raise ValueError(f"Benchmark fixture {fixture_path.name!r} could not be read") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Benchmark fixture {fixture_path.name!r} is not valid JSON") from error

    if not isinstance(fixture, dict):
        raise ValueError("Benchmark fixture must be a JSON object")
    if fixture.get("schemaVersion") != BENCHMARK_FIXTURE_SCHEMA:
        raise ValueError(f"Expected {BENCHMARK_FIXTURE_SCHEMA} fixture")
    if fixture.get("redistribution") != _EXPECTED_FIXTURE_REDISTRIBUTION:
        raise ValueError("Benchmark fixture redistribution metadata is not supported")

    profile = fixture.get("profile")
    if not isinstance(profile, str) or profile not in BENCHMARK_PROFILES:
        raise ValueError("Benchmark fixture profile is not supported")
    seed = fixture.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Benchmark fixture seed must be an integer")

    expected = fixture.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("Benchmark fixture expected metadata must be an object")
    if expected.get("report_schema") != BENCHMARK_REPORT_SCHEMA:
        raise ValueError(f"Benchmark fixture expected report must use {BENCHMARK_REPORT_SCHEMA}")
    signature = expected.get("deterministic_signature")
    if (
        not isinstance(signature, str)
        or len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        raise ValueError("Benchmark fixture deterministic_signature must be a SHA-256 hex digest")

    for field in ("generator", "data", "transform", "training"):
        if not isinstance(fixture.get(field), dict):
            raise ValueError(f"Benchmark fixture field {field!r} must be an object")
    if "evaluation" in fixture and not isinstance(fixture["evaluation"], dict):
        raise ValueError("Benchmark fixture field 'evaluation' must be an object")
    return fixture


def verify_benchmark_fixture(fixture_path: Path, output_dir: Path) -> BenchmarkResult:
    fixture = _load_benchmark_fixture(fixture_path)
    result = run_benchmark(output_dir, profile=fixture["profile"], seed=fixture["seed"])
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))

    for field in ("generator", "profile", "seed", "data", "transform", "training"):
        if fixture.get(field) != manifest.get(field):
            raise ValueError(f"Benchmark fixture field {field!r} does not match generated manifest")
    if "evaluation" in fixture and fixture["evaluation"] != manifest.get("evaluation"):
        raise ValueError("Benchmark fixture field 'evaluation' does not match generated manifest")

    expected = fixture["expected"]
    if report.get("schemaVersion") != expected["report_schema"]:
        raise ValueError("Benchmark fixture expected report schema does not match generated report")
    if report.get("deterministic_signature") != expected["deterministic_signature"]:
        raise ValueError("Benchmark fixture deterministic_signature does not match generated report")
    return result


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
                test_frequency_offset_hz=condition.frequency_offset_hz,
                test_timing_offset_samples=condition.timing_offset_samples,
                test_signal_gain=condition.signal_gain,
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
    summarize_parser = subparsers.add_parser("summarize", help="Render a Markdown summary from a benchmark report")
    summarize_parser.add_argument("report_path", type=Path)
    summarize_parser.add_argument("--output", type=Path)
    verify_parser = subparsers.add_parser(
        "verify-fixture", help="Replay and verify a redistributable benchmark fixture"
    )
    verify_parser.add_argument("fixture_path", type=Path)
    verify_parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark-fixture"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.action == "summarize":
        output_path = write_benchmark_summary(args.report_path, args.output)
        print(f"Benchmark summary: {output_path.resolve()}")
        return
    if args.action == "verify-fixture":
        result = verify_benchmark_fixture(args.fixture_path, args.output_dir)
        print("Benchmark fixture verified")
        print(f"Profile: {result.profile}")
        print(f"Deterministic signature: {result.deterministic_signature}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Report: {result.report_path}")
        return
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
