import json

import numpy as np
import pytest
import torch

from src.artifact_cli import main as artifact_main
from src.artifact_index import ARTIFACT_INDEX_V2_SCHEMA, build_artifact_index
from src.artifact_inspect import inspect_artifact, validate_run_manifest
from src.cache_artifact import write_cache_artifact
from src.checkpoint import make_checkpoint_payload
from src.oof_artifact import write_oof_artifact
from src.oof_aggregate_artifact import write_oof_aggregate_artifact
from src.rule_artifact import make_rule_payload, write_rule_artifact
from src.validation_artifact import write_validation_artifact


def _checkpoint_payload():
    return make_checkpoint_payload(
        model_state_dict={"weight": torch.ones(1)},
        arch="resnet18",
        dropout=0.3,
        num_classes=9,
        n_fft=8,
        hop=2,
        target_freq=5,
        target_time=4,
        cache_time=7,
        epoch=2,
        metrics={"strict": 0.5},
        pretrained=False,
        fold=1,
        tag="demo",
    )


def _oof_values():
    return {
        "probs": np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]], dtype=np.float32),
        "labels": np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.int32),
        "indices": np.asarray([4], dtype=np.int32),
        "sample_ids": np.asarray([104], dtype=np.int64),
    }


def _write_versioned_artifacts(tmp_path):
    checkpoint = tmp_path / "model.pth"
    torch.save(_checkpoint_payload(), checkpoint)

    oof = tmp_path / "oof.npz"
    write_oof_artifact(oof, **_oof_values(), fold=0, metric=0.75)

    rule = tmp_path / "rule.json"
    selected = {"method": "per_class_thresholds", "thresholds": [0.5] * 9, "accuracy": 0.75}
    write_rule_artifact(rule, make_rule_payload(selected, num_classes=9))

    cache = tmp_path / "cache.npz"
    write_cache_artifact(
        cache,
        x=np.zeros((3, 5, 7), dtype=np.float32),
        node_mask=np.asarray([1, 0, 1], dtype=np.float32),
        n_fft=8,
        hop=2,
        target_freq=5,
        cache_time=7,
    )
    aggregate = tmp_path / "rule.oof_probs.npz"
    values = _oof_values()
    write_oof_aggregate_artifact(
        aggregate,
        probs=values["probs"],
        labels=values["labels"],
        sample_ids=values["sample_ids"],
        source_files=[oof.name],
    )
    validation = tmp_path / "best_val_probs.npz"
    write_validation_artifact(
        validation,
        probs=values["probs"],
        labels=values["labels"],
        sample_ids=values["sample_ids"],
        epoch=2,
        metric_name="strict",
        metric_value=0.75,
    )
    return {
        "checkpoint": checkpoint,
        "oof": oof,
        "rule": rule,
        "cache": cache,
        "aggregate": aggregate,
        "validation": validation,
    }


def _write_manifest(tmp_path, *, mismatch=False, with_index=False, index_schema="artifact-index-v1"):
    paths = _write_versioned_artifacts(tmp_path)
    checkpoint_payload = torch.load(paths["checkpoint"], map_location="cpu")
    checkpoint_payload["fold"] = 0
    torch.save(checkpoint_payload, paths["checkpoint"])
    manifest = {
        "schemaVersion": "run-manifest-v1",
        "runId": "demo-run",
        "status": "completed",
        "model": {"numClasses": 9},
        "transform": {"version": "stft-v1", "nFft": 8, "hop": 2, "targetFreq": 5, "targetTime": 4, "cacheTime": 7},
        "training": {"folds": [0]},
        "outputs": {
            "checkpoint": "model.pth",
            "oof": ["oof.npz"],
            "rule": "rule.json",
            "validationProbabilities": "best_val_probs.npz",
            "config": "config.json",
            "history": "history.json",
        },
    }
    rule_payload = json.loads(paths["rule"].read_text(encoding="utf-8"))
    rule_payload["source_files"] = ["missing.npz" if mismatch else "oof.npz"]
    paths["rule"].write_text(json.dumps(rule_payload), encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "history.json").write_text("[]", encoding="utf-8")
    manifest_path = tmp_path / "run-manifest.json"
    if with_index:
        manifest["artifactIndex"] = build_artifact_index(
            manifest["runId"],
            tmp_path,
            manifest["outputs"],
            schema_version=index_schema,
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


@pytest.mark.parametrize("kind", ["checkpoint", "oof", "rule", "cache", "aggregate", "validation"])
def test_inspect_artifact_summarizes_versioned_artifacts_without_absolute_paths(tmp_path, kind):
    path = _write_versioned_artifacts(tmp_path)[kind]

    result = inspect_artifact(path)

    assert result["valid"] is True
    assert result["fileName"] == path.name
    assert str(path) not in json.dumps(result)
    assert result["schemaVersion"] in {
        "checkpoint-v1",
        "oof-v1",
        "rule-v1",
        "cache-v1",
        "oof-aggregate-v1",
        "validation-predictions-v1",
    }


@pytest.mark.parametrize("kind", ["checkpoint", "oof", "rule", "cache", "aggregate", "validation"])
def test_inspect_artifact_accepts_legacy_formats(tmp_path, kind):
    paths = {
        "checkpoint": tmp_path / "legacy.pth",
        "oof": tmp_path / "legacy.npz",
        "rule": tmp_path / "legacy.json",
        "cache": tmp_path / "legacy-cache.npz",
        "aggregate": tmp_path / "legacy-aggregate.npz",
        "validation": tmp_path / "legacy-validation.npz",
    }
    if kind == "checkpoint":
        torch.save({"model_state_dict": {"weight": torch.zeros(1)}, "arch": "resnet18"}, paths[kind])
    elif kind == "oof":
        values = _oof_values()
        np.savez(paths[kind], probs=values["probs"], labels=values["labels"], indices=values["indices"])
    elif kind == "rule":
        paths[kind].write_text(json.dumps({"thresholds": [0.5] * 9}), encoding="utf-8")
    elif kind == "aggregate":
        values = _oof_values()
        np.savez(
            paths[kind],
            probs=values["probs"],
            labels=values["labels"],
            sample_ids=values["sample_ids"],
        )
    elif kind == "validation":
        values = _oof_values()
        np.savez(paths[kind], probs=values["probs"], labels=values["labels"])
    else:
        np.savez_compressed(
            paths[kind],
            x=np.zeros((3, 5, 7), dtype=np.float16),
            node_mask=np.asarray([1, 0, 1], dtype=np.float32),
        )

    result = inspect_artifact(paths[kind])

    assert result["valid"] is True
    assert result["schemaVersion"] == "legacy-unversioned"


def test_inspect_artifact_routes_malformed_aggregate_to_aggregate_validation(tmp_path):
    path = tmp_path / "malformed-aggregate.npz"
    values = _oof_values()
    np.savez(
        path,
        schemaVersion=np.asarray(["oof-aggregate-v1"]),
        aggregationMethod=np.asarray("mean"),
        probs=values["probs"],
        labels=values["labels"],
        sample_ids=values["sample_ids"],
    )

    result = inspect_artifact(path)

    assert result["valid"] is False
    assert "OOF aggregate metadata schemaVersion must be a scalar string" in result["error"]
    assert "Unable to read NPZ artifact" not in result["error"]


def test_validate_run_manifest_checks_linked_artifacts_without_paths(tmp_path):
    path = _write_manifest(tmp_path)

    result = validate_run_manifest(path)

    assert result["valid"] is True
    assert result["artifactType"] == "run-manifest"
    assert result["details"]["outputCount"] == 6
    assert result["details"]["artifactIndexPresent"] is False
    assert result["details"]["indexedArtifactCount"] == 0
    assert result["details"]["validatedArtifactCount"] == 4
    assert str(tmp_path) not in json.dumps(result)


def test_validate_run_manifest_accepts_kfold_fold_metadata(tmp_path):
    path = _write_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["outputs"]["folds"] = [0]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run_manifest(path)

    assert result["valid"] is True


def test_validate_run_manifest_rejects_mismatched_rule_sources(tmp_path):
    path = _write_manifest(tmp_path, mismatch=True)

    result = validate_run_manifest(path)

    assert result["valid"] is False
    assert "rule source_files do not match manifest oof outputs" in result["errors"]


def test_validate_run_manifest_rejects_mismatched_validation_metadata(tmp_path):
    path = _write_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["training"]["selectMetric"] = "macro_f1"
    manifest["metrics"] = {"bestEpoch": 3, "bestMetric": 0.5}
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run_manifest(path)

    assert result["valid"] is False
    assert "validationProbabilities metric does not match manifest" in result["errors"]
    assert "validationProbabilities epoch does not match manifest" in result["errors"]
    assert "validationProbabilities metric value does not match manifest" in result["errors"]


def test_validate_run_manifest_rejects_unsafe_output_reference(tmp_path):
    path = _write_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["outputs"]["checkpoint"] = str(tmp_path / "model.pth")
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run_manifest(path)

    assert result["valid"] is False
    assert "checkpoint references an unsafe path" in result["errors"]


def test_validate_run_manifest_checks_artifact_index_integrity(tmp_path):
    path = _write_manifest(tmp_path, with_index=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    result = validate_run_manifest(path)

    assert result["valid"] is True
    assert result["details"]["artifactIndexPresent"] is True
    assert result["details"]["indexedArtifactCount"] == 3
    assert manifest["artifactIndex"]["schemaVersion"] == "artifact-index-v1"
    assert {entry["role"] for entry in manifest["artifactIndex"]["artifacts"]} == {
        "checkpoint",
        "oof",
        "rule",
    }

    rule_path = tmp_path / "rule.json"
    rule_payload = json.loads(rule_path.read_text(encoding="utf-8"))
    rule_payload["note"] = "changed after indexing"
    rule_path.write_text(json.dumps(rule_payload), encoding="utf-8")

    changed = validate_run_manifest(path)
    assert changed["valid"] is False
    assert "artifactIndex size mismatch for rule.json" in changed["errors"]
    assert "artifactIndex digest mismatch for rule.json" in changed["errors"]


def test_validate_run_manifest_checks_v2_validation_probability_integrity(tmp_path):
    path = _write_manifest(tmp_path, with_index=True, index_schema=ARTIFACT_INDEX_V2_SCHEMA)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    result = validate_run_manifest(path)

    assert result["valid"] is True
    assert result["details"]["indexedArtifactCount"] == 4
    validation_entry = next(
        entry
        for entry in manifest["artifactIndex"]["artifacts"]
        if entry["role"] == "validationProbabilities"
    )
    assert validation_entry["artifactType"] == "validation-predictions"
    assert validation_entry["schemaVersion"] == "validation-predictions-v1"

    validation_path = tmp_path / "best_val_probs.npz"
    with validation_path.open("ab") as file:
        file.write(b"\n")
    assert inspect_artifact(validation_path)["valid"] is True

    changed = validate_run_manifest(path)
    assert changed["valid"] is False
    assert "artifactIndex size mismatch for best_val_probs.npz" in changed["errors"]
    assert "artifactIndex digest mismatch for best_val_probs.npz" in changed["errors"]


def test_validate_run_manifest_rejects_unknown_artifact_index_schema(tmp_path):
    path = _write_manifest(tmp_path, with_index=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifactIndex"]["schemaVersion"] = "artifact-index-v3"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run_manifest(path)

    assert result["valid"] is False
    assert (
        "artifactIndex must use artifact-index-v1 or artifact-index-v2 metadata"
        in result["errors"]
    )


def test_validate_run_manifest_rejects_incomplete_artifact_index(tmp_path):
    path = _write_manifest(tmp_path, with_index=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifactIndex"]["artifacts"].pop()
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run_manifest(path)

    assert result["valid"] is False
    assert "artifactIndex entries do not match manifest artifact outputs" in result["errors"]


def test_validate_run_manifest_rejects_malformed_artifact_index_without_paths(tmp_path):
    path = _write_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifactIndex"] = []
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run_manifest(path)

    assert result["valid"] is False
    assert "artifactIndex must be an object" in result["errors"]
    assert str(tmp_path) not in json.dumps(result)


def test_validate_run_manifest_rejects_unsafe_artifact_index_filename_without_paths(tmp_path):
    path = _write_manifest(tmp_path, with_index=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifactIndex"]["artifacts"][0]["fileName"] = str(tmp_path / "model.pth")
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run_manifest(path)

    assert result["valid"] is False
    assert "artifactIndex entry has an unsafe filename" in result["errors"]
    assert str(tmp_path) not in json.dumps(result)


def test_validate_run_manifest_strictly_checks_indexed_fold_and_tag(tmp_path):
    path = _write_manifest(tmp_path, with_index=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_entry = next(
        entry for entry in manifest["artifactIndex"]["artifacts"] if entry["role"] == "checkpoint"
    )
    checkpoint_entry["fold"] = False
    checkpoint_entry["tag"] = "other"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run_manifest(path)

    assert result["valid"] is False
    assert "artifactIndex fold mismatch for model.pth" in result["errors"]
    assert "artifactIndex tag mismatch for model.pth" in result["errors"]


def test_build_artifact_index_rejects_unsafe_output_without_leaking_path(tmp_path):
    unsafe_name = str(tmp_path / "model.pth")

    with pytest.raises(ValueError) as error:
        build_artifact_index("demo-run", tmp_path, {"checkpoint": unsafe_name})

    assert "filename in the run directory" in str(error.value)
    assert str(tmp_path) not in str(error.value)


def test_build_artifact_index_rejects_artifact_role_mismatch(tmp_path):
    rule_path = _write_versioned_artifacts(tmp_path)["rule"]

    with pytest.raises(ValueError, match="checkpoint has an unexpected artifact type"):
        build_artifact_index("demo-run", tmp_path, {"checkpoint": rule_path.name})


def test_artifact_cli_json_reports_invalid_artifact_and_returns_nonzero(tmp_path, capsys):
    path = tmp_path / "invalid.pth"
    torch.save({"schemaVersion": "checkpoint-v1", "arch": "resnet18"}, path)

    with pytest.raises(SystemExit) as error:
        artifact_main(["validate", str(path), "--json"])

    assert error.value.code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is False
    assert output["fileName"] == path.name
    assert str(path) not in json.dumps(output)


def test_artifact_cli_inspect_json_reports_valid_artifact(tmp_path, capsys):
    path = _write_versioned_artifacts(tmp_path)["rule"]

    artifact_main(["inspect", str(path), "--json"])

    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["artifactType"] == "inference-rule"


def test_artifact_cli_validate_run_json_reports_linkage(tmp_path, capsys):
    path = _write_manifest(tmp_path)

    artifact_main(["validate-run", str(path), "--json"])

    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["artifactType"] == "run-manifest"
