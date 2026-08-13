import numpy as np
import pytest

from src.oof_artifact import LEGACY_OOF_SCHEMA, OOF_SCHEMA, load_oof_artifact, write_oof_artifact
from src.search_spectrogram_kfold_thresholds import load_averaged_oof, load_weighted_oof


def arrays():
    return {
        "probs": np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]], dtype=np.float32),
        "labels": np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.int32),
        "indices": np.asarray([4], dtype=np.int32),
        "sample_ids": np.asarray([104], dtype=np.int64),
    }


def test_oof_writer_adds_metadata_and_loader_preserves_arrays(tmp_path):
    path = tmp_path / "oof_fold0.npz"
    values = arrays()
    write_oof_artifact(path, **values, fold=0, metric=0.75)

    artifact = load_oof_artifact(path)

    assert artifact["schemaVersion"] == OOF_SCHEMA
    assert artifact["artifactType"] == "oof-predictions"
    np.testing.assert_allclose(artifact["probs"], values["probs"], atol=1e-3)
    np.testing.assert_array_equal(artifact["labels"], values["labels"])
    np.testing.assert_array_equal(artifact["indices"], values["indices"])
    np.testing.assert_array_equal(artifact["sample_ids"], values["sample_ids"])
    assert artifact["fold"] == 0


def test_loader_accepts_legacy_unversioned_oof_files(tmp_path):
    path = tmp_path / "legacy.npz"
    values = arrays()
    np.savez(
        path,
        probs=values["probs"],
        labels=values["labels"],
        indices=values["indices"],
    )

    artifact = load_oof_artifact(path)

    assert artifact["schemaVersion"] == LEGACY_OOF_SCHEMA
    np.testing.assert_array_equal(artifact["sample_ids"], values["indices"])
    assert artifact["fold"] is None
    assert artifact["metrics"].size == 0


def test_loader_preserves_optional_legacy_metadata_when_present(tmp_path):
    path = tmp_path / "legacy-with-metadata.npz"
    values = arrays()
    np.savez(path, **values, fold=np.asarray(1, dtype=np.int32), metrics=np.asarray([0.5], dtype=np.float32))

    artifact = load_oof_artifact(path)

    assert artifact["schemaVersion"] == LEGACY_OOF_SCHEMA
    assert artifact["fold"] == 1
    np.testing.assert_array_equal(artifact["metrics"], [0.5])


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"probs": np.zeros((1, 8), dtype=np.float32)}, "probabilities"),
        ({"labels": np.full((1, 9), 2, dtype=np.int8)}, "labels"),
        ({"indices": np.asarray([1, 1], dtype=np.int32)}, "indices"),
        ({"sample_ids": np.asarray([104, 104], dtype=np.int64)}, "sample_ids"),
        ({"probs": np.full((1, 9), 2.0, dtype=np.float32)}, "probabilities"),
    ],
)
def test_writer_rejects_invalid_oof_arrays(tmp_path, changes, message):
    values = arrays()
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        write_oof_artifact(tmp_path / "invalid.npz", **values, fold=0, metric=0.5)


def test_loader_rejects_unknown_schema_and_missing_fields(tmp_path):
    unknown = tmp_path / "unknown.npz"
    values = arrays()
    np.savez(
        unknown,
        schemaVersion=np.asarray("oof-v9"),
        artifactType=np.asarray("oof-predictions"),
        **values,
        fold=np.asarray(0, dtype=np.int32),
        metrics=np.asarray([0.5], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Unsupported OOF schema"):
        load_oof_artifact(unknown)

    missing = tmp_path / "missing.npz"
    np.savez(missing, probs=values["probs"], labels=values["labels"])
    with pytest.raises(ValueError, match="missing"):
        load_oof_artifact(missing)


def test_writer_rejects_non_finite_metric(tmp_path):
    with pytest.raises(ValueError, match="metric"):
        write_oof_artifact(tmp_path / "invalid.npz", **arrays(), fold=0, metric=float("nan"))


def test_oof_aggregation_preserves_average_and_weighted_values(tmp_path):
    first = arrays()
    second = arrays()
    second["probs"] = first["probs"] * 0.5
    first_path = tmp_path / "oof_r34_fold0.npz"
    second_path = tmp_path / "oof_b0_fold1.npz"
    write_oof_artifact(first_path, **first, fold=0, metric=0.5)
    write_oof_artifact(second_path, **second, fold=1, metric=0.25)

    averaged_probs, averaged_labels, averaged_ids = load_averaged_oof([first_path, second_path])
    weighted_probs, weighted_labels, weighted_ids = load_weighted_oof(
        [first_path, second_path], {"r34": 0.75, "b0": 0.25}
    )

    np.testing.assert_allclose(averaged_probs[0], first["probs"][0] * 0.75, atol=1e-3)
    np.testing.assert_array_equal(averaged_labels[0], first["labels"][0])
    np.testing.assert_array_equal(averaged_ids, [104])
    np.testing.assert_allclose(weighted_probs[0], first["probs"][0] * 0.875, atol=1e-3)
    np.testing.assert_array_equal(weighted_labels, averaged_labels)
    np.testing.assert_array_equal(weighted_ids, averaged_ids)


def test_oof_aggregation_rejects_cross_file_label_or_sample_id_conflicts(tmp_path):
    first = arrays()
    second = arrays()
    second["labels"] = first["labels"].copy()
    second["labels"][0, 8] = 0
    first_path = tmp_path / "oof_r34_fold0.npz"
    second_path = tmp_path / "oof_b0_fold1.npz"
    write_oof_artifact(first_path, **first, fold=0, metric=0.5)
    write_oof_artifact(second_path, **second, fold=1, metric=0.25)

    with pytest.raises(ValueError, match="label mismatch"):
        load_averaged_oof([first_path, second_path])

    second["labels"] = first["labels"].copy()
    second["sample_ids"] = np.asarray([999], dtype=np.int64)
    write_oof_artifact(second_path, **second, fold=1, metric=0.25)
    with pytest.raises(ValueError, match="sample ID mismatch"):
        load_weighted_oof([first_path, second_path], {"r34": 0.5, "b0": 0.5})
