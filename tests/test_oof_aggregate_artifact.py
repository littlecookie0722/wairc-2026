from types import SimpleNamespace

import numpy as np
import pytest

import src.search_spectrogram_kfold_thresholds as rule_search
from src.oof_aggregate_artifact import (
    LEGACY_OOF_AGGREGATE_SCHEMA,
    OOF_AGGREGATE_SCHEMA,
    load_oof_aggregate_artifact,
    write_oof_aggregate_artifact,
)
from src.oof_artifact import write_oof_artifact


def arrays():
    return {
        "probs": np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]], dtype=np.float32),
        "labels": np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.int32),
        "sample_ids": np.asarray([104], dtype=np.int64),
    }


def test_writer_adds_mean_metadata_without_changing_historical_arrays(tmp_path):
    path = tmp_path / "best_rule_kfold.oof_probs.npz"
    values = arrays()

    write_oof_aggregate_artifact(path, **values, source_files=["oof_r34_fold0.npz"])
    artifact = load_oof_aggregate_artifact(path)

    assert artifact["schemaVersion"] == OOF_AGGREGATE_SCHEMA
    assert artifact["artifactType"] == "aggregated-oof-probabilities"
    assert artifact["aggregationMethod"] == "mean"
    assert artifact["source_files"] == ["oof_r34_fold0.npz"]
    assert artifact["tag_weights"] == {}
    np.testing.assert_allclose(artifact["probs"], values["probs"], atol=1e-3)
    np.testing.assert_array_equal(artifact["labels"], values["labels"])
    np.testing.assert_array_equal(artifact["sample_ids"], values["sample_ids"])


def test_writer_records_normalized_tag_weights_and_redacts_source_paths(tmp_path):
    path = tmp_path / "weighted.oof_probs.npz"
    write_oof_aggregate_artifact(
        path,
        **arrays(),
        source_files=[tmp_path / "oof_r34_fold0.npz", "nested/oof_b0_fold0.npz"],
        tag_weights={"r34": 0.75, "b0": 0.25},
    )

    artifact = load_oof_aggregate_artifact(path)

    assert artifact["aggregationMethod"] == "tag-weighted"
    assert artifact["source_files"] == ["oof_r34_fold0.npz", "oof_b0_fold0.npz"]
    assert artifact["tag_weights"] == pytest.approx({"b0": 0.25, "r34": 0.75})
    assert str(tmp_path) not in str(artifact)


def test_loader_accepts_historical_unversioned_aggregate(tmp_path):
    path = tmp_path / "legacy.oof_probs.npz"
    np.savez(path, **arrays())

    artifact = load_oof_aggregate_artifact(path)

    assert artifact["schemaVersion"] == LEGACY_OOF_AGGREGATE_SCHEMA
    assert artifact["aggregationMethod"] is None
    assert artifact["source_files"] == []
    np.testing.assert_array_equal(artifact["sample_ids"], [104])


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
        write_oof_aggregate_artifact(
            tmp_path / "invalid.npz", **values, source_files=["oof_r34_fold0.npz"]
        )


def test_writer_rejects_invalid_weight_metadata(tmp_path):
    with pytest.raises(ValueError, match="sum to 1"):
        write_oof_aggregate_artifact(
            tmp_path / "invalid.npz",
            **arrays(),
            source_files=["oof_r34_fold0.npz"],
            tag_weights={"r34": 0.8},
        )


def test_loader_rejects_unknown_schema(tmp_path):
    path = tmp_path / "unknown.npz"
    np.savez(
        path,
        **arrays(),
        schemaVersion=np.asarray("oof-aggregate-v9"),
        artifactType=np.asarray("aggregated-oof-probabilities"),
    )
    with pytest.raises(ValueError, match="Unsupported OOF aggregate schema"):
        load_oof_aggregate_artifact(path)


def test_rule_search_writes_versioned_weighted_aggregate_without_changing_values(tmp_path, monkeypatch):
    values = arrays()
    first_path = tmp_path / "oof_r34_fold0.npz"
    second_path = tmp_path / "oof_b0_fold0.npz"
    write_oof_artifact(
        first_path,
        **values,
        indices=np.asarray([4], dtype=np.int32),
        fold=0,
        metric=0.5,
    )
    second_probs = values["probs"] * 0.5
    write_oof_artifact(
        second_path,
        probs=second_probs,
        labels=values["labels"],
        indices=np.asarray([4], dtype=np.int32),
        sample_ids=values["sample_ids"],
        fold=0,
        metric=0.25,
    )
    output = tmp_path / "best_rule.json"
    monkeypatch.setattr(
        rule_search,
        "parse_args",
        lambda: SimpleNamespace(
            save_dir=tmp_path,
            tags=["r34", "b0"],
            tag_weights=["r34=3", "b0=1"],
            include_default=False,
            output=output,
        ),
    )

    rule_search.main()

    artifact = load_oof_aggregate_artifact(output.with_suffix(".oof_probs.npz"))
    expected = values["probs"] * 0.75 + second_probs * 0.25
    assert artifact["schemaVersion"] == OOF_AGGREGATE_SCHEMA
    assert artifact["aggregationMethod"] == "tag-weighted"
    assert artifact["tag_weights"] == pytest.approx({"b0": 0.25, "r34": 0.75})
    np.testing.assert_allclose(artifact["probs"], expected, atol=1e-3)
