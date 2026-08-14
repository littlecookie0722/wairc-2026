import json

from src.benchmark import run_benchmark


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


def test_synthetic_benchmark_cli_accepts_profile_and_output_dir(tmp_path, capsys):
    from src.benchmark import main

    main(["run", "--profile", "cpu-smoke", "--output-dir", str(tmp_path / "cli")])

    output = capsys.readouterr().out
    assert "Synthetic benchmark passed" in output
    assert (tmp_path / "cli" / "benchmark-report.json").exists()


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
    assert [condition["name"] for condition in manifest["evaluation"]["conditions"]] == [
        "baseline",
        "high-noise",
        "node0-missing",
    ]
    assert [condition["name"] for condition in conditions] == [
        "baseline",
        "high-noise",
        "node0-missing",
    ]
    assert all(0.0 <= condition["metrics"]["exact_match_accuracy"] <= 1.0 for condition in conditions)
    assert all(len(condition["metrics"]["per_class_recall"]) == 9 for condition in conditions)
    assert all(
        (tmp_path / "robustness" / condition["artifacts"][0]).exists()
        and (tmp_path / "robustness" / condition["artifacts"][1]).exists()
        for condition in conditions
    )
    assert str(tmp_path) not in json.dumps(manifest)
    assert str(tmp_path) not in json.dumps(report)
