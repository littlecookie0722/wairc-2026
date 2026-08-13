import torch
import pytest

from src.checkpoint import CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA, load_checkpoint, make_checkpoint_payload


def _payload(**kwargs):
    values = {
        "model_state_dict": {"weight": torch.ones(1)},
        "arch": "resnet18",
        "dropout": 0.3,
        "num_classes": 9,
        "n_fft": 512,
        "hop": 128,
        "target_freq": 257,
        "target_time": 768,
        "cache_time": 1536,
        "epoch": 2,
        "metrics": {"strict": 0.5},
        "pretrained": False,
    }
    values.update(kwargs)
    return make_checkpoint_payload(**values)


def test_checkpoint_v1_keeps_legacy_fields_and_adds_schema_metadata():
    checkpoint = _payload(fold=1, tag="r34")

    assert checkpoint["schemaVersion"] == CHECKPOINT_SCHEMA
    assert checkpoint["artifactType"] == "model-checkpoint"
    assert checkpoint["stftProfile"] == "stft-v1"
    assert checkpoint["created_at"].endswith("+00:00")
    assert checkpoint["model_state_dict"]["weight"].equal(torch.ones(1))
    assert checkpoint["fold"] == 1
    assert checkpoint["tag"] == "r34"


def test_checkpoint_v1_preserves_empty_kfold_tag():
    assert _payload(fold=0, tag="")["tag"] == ""


def test_load_checkpoint_accepts_v1_and_normalizes_legacy_checkpoints(tmp_path):
    v1_path = tmp_path / "v1.pth"
    legacy_path = tmp_path / "legacy.pth"
    torch.save(_payload(), v1_path)
    torch.save({"model_state_dict": {"weight": torch.zeros(1)}, "arch": "resnet18"}, legacy_path)

    v1 = load_checkpoint(v1_path)
    legacy = load_checkpoint(legacy_path)

    assert v1["schemaVersion"] == CHECKPOINT_SCHEMA
    assert legacy["schemaVersion"] == LEGACY_CHECKPOINT_SCHEMA
    assert legacy["artifactType"] == "model-checkpoint"


@pytest.mark.parametrize(
    "checkpoint, message",
    [
        ({"schemaVersion": "checkpoint-v9", "model_state_dict": {}, "arch": "resnet18"}, "Unsupported checkpoint schema"),
        ({"schemaVersion": CHECKPOINT_SCHEMA, "artifactType": "oof", "model_state_dict": {}, "arch": "resnet18"}, "artifact type"),
        ({"schemaVersion": CHECKPOINT_SCHEMA, "arch": "resnet18"}, "missing"),
        ({"schemaVersion": CHECKPOINT_SCHEMA, "model_state_dict": {}}, "arch"),
        ([], "must contain a dictionary"),
    ],
)
def test_load_checkpoint_rejects_unsupported_or_incomplete_payloads(tmp_path, checkpoint, message):
    path = tmp_path / "invalid.pth"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("stftProfile", "future-stft", "STFT profile"),
        ("n_fft", 0, "n_fft must be a positive integer"),
        ("num_classes", True, "num_classes must be a positive integer"),
    ],
)
def test_load_checkpoint_rejects_incompatible_v1_metadata(tmp_path, field, value, message):
    path = tmp_path / "invalid-v1.pth"
    checkpoint = _payload()
    checkpoint[field] = value
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path)
