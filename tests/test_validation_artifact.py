import numpy as np
import pytest

from src.validation_artifact import (
    LEGACY_VALIDATION_SCHEMA,
    VALIDATION_SCHEMA,
    load_validation_artifact,
    write_validation_artifact,
)


def arrays():
    return {
        "probs": np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]], dtype=np.float32),
        "labels": np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.int32),
        "sample_ids": np.asarray([104], dtype=np.int64),
    }


def test_writer_adds_metadata_without_changing_historical_arrays(tmp_path):
    path = tmp_path / "best_val_probs.npz"
    values = arrays()

    write_validation_artifact(
        path,
        **values,
        epoch=3,
        metric_name="strict",
        metric_value=0.75,
    )
    with np.load(path, allow_pickle=False) as raw:
        assert raw["probs"].dtype == np.float16
        assert raw["labels"].dtype == np.int8
    artifact = load_validation_artifact(path)

    assert artifact["schemaVersion"] == VALIDATION_SCHEMA
    assert artifact["artifactType"] == "validation-predictions"
    assert artifact["numClasses"] == 9
    assert artifact["epoch"] == 3
    assert artifact["metricName"] == "strict"
    assert artifact["metricValue"] == pytest.approx(0.75)
    np.testing.assert_allclose(artifact["probs"], values["probs"], atol=1e-3)
    np.testing.assert_array_equal(artifact["labels"], values["labels"])
    np.testing.assert_array_equal(artifact["sample_ids"], values["sample_ids"])


def test_loader_accepts_historical_probs_and_labels(tmp_path):
    path = tmp_path / "legacy.npz"
    values = arrays()
    np.savez(path, probs=values["probs"], labels=values["labels"])

    artifact = load_validation_artifact(path)

    assert artifact["schemaVersion"] == LEGACY_VALIDATION_SCHEMA
    assert artifact["sample_ids"] is None
    assert artifact["epoch"] is None
    assert artifact["metricName"] is None


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"probs": np.zeros((1, 8), dtype=np.float32)}, "probabilities"),
        ({"labels": np.full((1, 9), 2, dtype=np.int8)}, "labels"),
        ({"sample_ids": np.asarray([104, 104], dtype=np.int64)}, "sample_ids"),
        ({"probs": np.full((1, 9), np.nan, dtype=np.float32)}, "probabilities"),
    ],
)
def test_writer_rejects_invalid_arrays(tmp_path, changes, message):
    values = arrays()
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        write_validation_artifact(
            tmp_path / "invalid.npz",
            **values,
            epoch=1,
            metric_name="strict",
            metric_value=0.5,
        )


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"epoch": 0, "metric_name": "strict", "metric_value": 0.5}, "epoch"),
        ({"epoch": 1, "metric_name": "", "metric_value": 0.5}, "metricName"),
        ({"epoch": 1, "metric_name": "strict", "metric_value": float("nan")}, "metricValue"),
    ],
)
def test_writer_rejects_invalid_metadata(tmp_path, metadata, message):
    with pytest.raises(ValueError, match=message):
        write_validation_artifact(tmp_path / "invalid.npz", **arrays(), **metadata)


def test_loader_rejects_unknown_schema(tmp_path):
    path = tmp_path / "unknown.npz"
    np.savez(
        path,
        probs=arrays()["probs"],
        labels=arrays()["labels"],
        schemaVersion=np.asarray("validation-predictions-v9"),
    )
    with pytest.raises(ValueError, match="Unsupported validation schema"):
        load_validation_artifact(path)
