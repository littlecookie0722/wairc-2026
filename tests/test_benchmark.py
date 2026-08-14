import csv
import json
from pathlib import Path

import pytest

from src.benchmark import run_benchmark, verify_benchmark_fixture, write_benchmark_summary


FIXTURE = Path(__file__).parent / "fixtures" / "benchmark" / "synthetic_iq_v1.json"
ROBUSTNESS_FIXTURE = Path(__file__).parent / "fixtures" / "benchmark" / "synthetic_iq_robustness_v1.json"


def test_synthetic_benchmark_writes_path_free_deterministic_report(tmp_path):
    first = run_benchmark(tmp_path / "first", seed=2026)
    second = run_benchmark(tmp_path / "second", seed=2026)

    first_manifest = json.loads((tmp_path / "first" / "benchmark-manifest.json").read_text(encoding="utf-8"))
    first_report = json.loads((tmp_path / "first" / "benchmark-report.json").read_text(encoding="utf-8"))
    second_report = json.loads((tmp_path / "second" / "benchmark-report.json").read_text(encoding="utf-8"))

    assert first_manifest["schemaVersion"] == "benchmark-manifest-v1"
    assert first_manifest["generator"]["version"] == "synthetic-iq-v1"
    assert first_manifest["data"]["class_mapping"] == list(range(9))
    assert first_manifest["data"]["missing_node_pattern"] == [0, 1, 2, None]
    assert first_report["schemaVersion"] == "benchmark-report-v1"
    assert first_report["status"] == "passed"
    assert first_report["profile"] == "cpu-smoke"
    assert first_report["metrics"] == second_report["metrics"]
    assert first_report["deterministic_signature"] == second_report["deterministic_signature"]
    assert first.deterministic_signature == second.deterministic_signature
    assert first_report["artifacts"] == ["demo/metrics.json", "demo/submission.txt"]
    assert str(tmp_path) not in json.dumps(first_manifest)
    assert str(tmp_path) not in json.dumps(first_report)
    assert (tmp_path / "first" / "demo" / "metrics.json").exists()
    assert (tmp_path / "first" / "demo" / "submission.txt").exists()


def test_redistributable_fixture_matches_cpu_smoke_manifest(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = run_benchmark(tmp_path / "fixture", profile="cpu-smoke", seed=fixture["seed"])
    manifest = json.loads((tmp_path / "fixture" / "benchmark-manifest.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "fixture" / "benchmark-report.json").read_text(encoding="utf-8"))

    assert fixture["schemaVersion"] == "benchmark-fixture-v1"
    assert fixture["redistribution"] == {
        "status": "repository-authored-parameters-only",
        "license": "MIT",
        "raw_iq_included": False,
        "model_weights_included": False,
        "private_labels_included": False,
        "external_recordings_included": False,
    }
    assert fixture["generator"] == manifest["generator"]
    assert fixture["profile"] == manifest["profile"]
    for section in ("data", "transform", "training"):
        assert fixture[section] == manifest[section]
    assert fixture["expected"]["report_schema"] == report["schemaVersion"]
    assert fixture["expected"]["deterministic_signature"] == report["deterministic_signature"]
    assert result.deterministic_signature == fixture["expected"]["deterministic_signature"]


def test_redistributable_fixture_matches_robustness_manifest(tmp_path):
    fixture = json.loads(ROBUSTNESS_FIXTURE.read_text(encoding="utf-8"))
    result = run_benchmark(tmp_path / "fixture", profile=fixture["profile"], seed=fixture["seed"])
    manifest = json.loads((tmp_path / "fixture" / "benchmark-manifest.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "fixture" / "benchmark-report.json").read_text(encoding="utf-8"))

    assert fixture["schemaVersion"] == "benchmark-fixture-v1"
    assert fixture["fixtureId"] == "synthetic-iq-v1-robustness-small"
    assert fixture["redistribution"] == {
        "status": "repository-authored-parameters-only",
        "license": "MIT",
        "raw_iq_included": False,
        "model_weights_included": False,
        "private_labels_included": False,
        "external_recordings_included": False,
    }
    assert fixture["generator"] == manifest["generator"]
    assert fixture["profile"] == manifest["profile"]
    for section in ("data", "transform", "training", "evaluation"):
        assert fixture[section] == manifest[section]
    assert fixture["expected"]["report_schema"] == report["schemaVersion"]
    assert fixture["expected"]["deterministic_signature"] == report["deterministic_signature"]
    assert result.deterministic_signature == fixture["expected"]["deterministic_signature"]


@pytest.mark.parametrize("fixture_path", [FIXTURE, ROBUSTNESS_FIXTURE])
def test_verify_benchmark_fixture_replays_manifest_and_signature(tmp_path, fixture_path):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    result = verify_benchmark_fixture(fixture_path, tmp_path / fixture["profile"])

    assert result.profile == fixture["profile"]
    assert result.deterministic_signature == fixture["expected"]["deterministic_signature"]


def test_verify_benchmark_fixture_rejects_signature_mismatch_without_path_leak(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture["expected"]["deterministic_signature"] = "0" * 64
    fixture_path = tmp_path / "invalid-fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic_signature") as error:
        verify_benchmark_fixture(fixture_path, tmp_path / "output")

    assert str(tmp_path) not in str(error.value)


def test_synthetic_benchmark_cli_accepts_profile_and_output_dir(tmp_path, capsys):
    from src.benchmark import main

    main(["run", "--profile", "cpu-smoke", "--output-dir", str(tmp_path / "cli")])

    output = capsys.readouterr().out
    assert "Synthetic benchmark passed" in output
    assert (tmp_path / "cli" / "benchmark-report.json").exists()

    from src.cli import main as cli_main

    cli_main(
        [
            "benchmark",
            "summarize",
            str(tmp_path / "cli" / "benchmark-report.json"),
            "--output",
            str(tmp_path / "cli" / "summary.md"),
        ]
    )
    assert "Benchmark summary:" in capsys.readouterr().out
    assert (tmp_path / "cli" / "summary.md").exists()


def test_synthetic_benchmark_cli_verifies_redistributable_fixture(tmp_path, capsys):
    from src.benchmark import main

    main(["verify-fixture", str(FIXTURE), "--output-dir", str(tmp_path / "fixture")])

    output = capsys.readouterr().out
    assert "Benchmark fixture verified" in output
    assert "Deterministic signature:" in output
    assert (tmp_path / "fixture" / "benchmark-report.json").exists()


def test_robustness_small_reports_each_controlled_condition(tmp_path):
    result = run_benchmark(tmp_path / "robustness", profile="robustness-small", seed=2026)

    manifest = json.loads(
        (tmp_path / "robustness" / "benchmark-manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (tmp_path / "robustness" / "benchmark-report.json").read_text(encoding="utf-8")
    )
    conditions = report["metrics"]["conditions"]

    assert result.profile == "robustness-small"
    expected_names = [
        "baseline",
        "all-nodes-present",
        "high-noise",
        "node0-missing",
        "node1-missing",
        "node2-missing",
        "frequency-offset",
        "timing-offset",
        "low-gain",
        "combined-stress",
    ]
    assert [condition["name"] for condition in manifest["evaluation"]["conditions"]] == expected_names
    manifest_by_name = {condition["name"]: condition for condition in manifest["evaluation"]["conditions"]}
    assert manifest_by_name["all-nodes-present"]["missing_node_pattern"] == [None]
    assert manifest_by_name["node0-missing"]["missing_node_pattern"] == [0]
    assert manifest_by_name["node1-missing"]["missing_node_pattern"] == [1]
    assert manifest_by_name["node2-missing"]["missing_node_pattern"] == [2]
    assert manifest_by_name["frequency-offset"]["frequency_offset_hz"] == 180.0
    assert manifest_by_name["timing-offset"]["timing_offset_samples"] == 32
    assert manifest_by_name["low-gain"]["signal_gain"] == 0.5
    assert manifest_by_name["combined-stress"]["noise_std"] == 0.20
    assert manifest_by_name["combined-stress"]["frequency_offset_hz"] == 180.0
    assert manifest_by_name["combined-stress"]["timing_offset_samples"] == 32
    assert manifest_by_name["combined-stress"]["signal_gain"] == 0.5
    assert [condition["name"] for condition in conditions] == expected_names
    assert all(0.0 <= condition["metrics"]["exact_match_accuracy"] <= 1.0 for condition in conditions)
    assert all(len(condition["metrics"]["per_class_recall"]) == 9 for condition in conditions)
    assert all(
        (tmp_path / "robustness" / condition["artifacts"][0]).exists()
        and (tmp_path / "robustness" / condition["artifacts"][1]).exists()
        for condition in conditions
    )
    with (tmp_path / "robustness" / "conditions" / "all-nodes-present" / "data" / "test" / "index.csv").open(
        newline="", encoding="utf-8"
    ) as index_file:
        all_nodes_rows = list(csv.DictReader(index_file))
    assert all(
        int(row[f"has_node{node}"]) == 1 and float(row[f"sample_rate_node{node}"]) == 4096.0
        for row in all_nodes_rows
        for node in range(3)
    )
    assert str(tmp_path) not in json.dumps(manifest)
    assert str(tmp_path) not in json.dumps(report)


def test_benchmark_summary_renders_cpu_and_robustness_reports(tmp_path):
    cpu_output = tmp_path / "cpu"
    robust_output = tmp_path / "robust"
    run_benchmark(cpu_output, profile="cpu-smoke", seed=2026)
    run_benchmark(robust_output, profile="robustness-small", seed=2026)

    cpu_summary = write_benchmark_summary(cpu_output / "benchmark-report.json")
    robust_summary = write_benchmark_summary(
        robust_output / "benchmark-report.json", tmp_path / "robust-summary.md"
    )
    cpu_text = cpu_summary.read_text(encoding="utf-8")
    robust_text = robust_summary.read_text(encoding="utf-8")

    assert "# Synthetic benchmark summary" in cpu_text
    assert "| `cpu-smoke` |" in cpu_text
    assert "Synthetic-only: `true`" in cpu_text
    assert "| `baseline` |" in robust_text
    assert "| `high-noise` |" in robust_text
    assert "| `node0-missing` |" in robust_text
    assert "| `frequency-offset` |" in robust_text
    assert "| `timing-offset` |" in robust_text
    assert "| `low-gain` |" in robust_text
    assert "| `combined-stress` |" in robust_text
    assert str(tmp_path) not in cpu_text
    assert str(tmp_path) not in robust_text
