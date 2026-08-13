import json

import numpy as np
import pytest
import torch

from src.artifact_cli import main as artifact_main
from src.artifact_inspect import inspect_artifact
from src.cache_artifact import write_cache_artifact
from src.checkpoint import make_checkpoint_payload
from src.oof_artifact import write_oof_artifact
from src.rule_artifact import make_rule_payload, write_rule_artifact


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
    return {"checkpoint": checkpoint, "oof": oof, "rule": rule, "cache": cache}


@pytest.mark.parametrize("kind", ["checkpoint", "oof", "rule", "cache"])
def test_inspect_artifact_summarizes_versioned_artifacts_without_absolute_paths(tmp_path, kind):
    path = _write_versioned_artifacts(tmp_path)[kind]

    result = inspect_artifact(path)

    assert result["valid"] is True
    assert result["fileName"] == path.name
    assert str(path) not in json.dumps(result)
    assert result["schemaVersion"] in {"checkpoint-v1", "oof-v1", "rule-v1", "cache-v1"}


@pytest.mark.parametrize("kind", ["checkpoint", "oof", "rule", "cache"])
def test_inspect_artifact_accepts_legacy_formats(tmp_path, kind):
    paths = {
        "checkpoint": tmp_path / "legacy.pth",
        "oof": tmp_path / "legacy.npz",
        "rule": tmp_path / "legacy.json",
        "cache": tmp_path / "legacy-cache.npz",
    }
    if kind == "checkpoint":
        torch.save({"model_state_dict": {"weight": torch.zeros(1)}, "arch": "resnet18"}, paths[kind])
    elif kind == "oof":
        values = _oof_values()
        np.savez(paths[kind], probs=values["probs"], labels=values["labels"], indices=values["indices"])
    elif kind == "rule":
        paths[kind].write_text(json.dumps({"thresholds": [0.5] * 9}), encoding="utf-8")
    else:
        np.savez_compressed(
            paths[kind],
            x=np.zeros((3, 5, 7), dtype=np.float16),
            node_mask=np.asarray([1, 0, 1], dtype=np.float32),
        )

    result = inspect_artifact(paths[kind])

    assert result["valid"] is True
    assert result["schemaVersion"] == "legacy-unversioned"


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
