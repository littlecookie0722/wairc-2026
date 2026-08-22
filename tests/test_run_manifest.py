import json

import pytest
import torch

from src.artifact_index import ARTIFACT_INDEX_V2_SCHEMA
from src.checkpoint import make_checkpoint_payload
from src.run_manifest import (
    RUN_MANIFEST_SCHEMA,
    create_run_manifest,
    finalize_run_manifest,
    finalize_run_manifest_with_artifacts,
    make_run_id,
    write_run_manifest,
)


def test_run_manifest_is_serializable_and_excludes_local_paths(tmp_path):
    manifest = create_run_manifest(
        run_id=make_run_id("spectrogram", 2026),
        command=["python", "-m", "src.train_spectrogram", "--save-dir", str(tmp_path)],
        args={"save_dir": tmp_path, "epochs": 2, "seed": 2026},
        data={"adapter": "wairc-competition-v1", "split": "train-validation", "fold": None},
        transform={"version": "stft-v1", "nFft": 512, "hop": 128},
        model={"architecture": "resnet34", "numClasses": 9},
        training={"epochs": 2, "batchSize": 4, "seed": 2026},
        device="cpu",
    )

    assert manifest["schemaVersion"] == RUN_MANIFEST_SCHEMA
    assert manifest["status"] == "running"
    assert manifest["arguments"] == {"epochs": 2, "seed": 2026}
    encoded = json.dumps(manifest)
    assert str(tmp_path) not in encoded
    assert manifest["command"][-2:] == ["--save-dir", "<save_dir>"]


def test_run_manifest_sanitizes_inline_paths():
    manifest = create_run_manifest(
        run_id="test-run",
        command=["python", "-m", "src.train_spectrogram", "--cache-dir=C:\\private cache"],
        args={},
        data={},
        transform={},
        model={},
        training={},
        device="cpu",
    )

    assert manifest["command"][-1] == "--cache-dir=<cache_dir>"


def test_run_manifest_can_be_finalized(tmp_path):
    path = tmp_path / "run-manifest.json"
    manifest = create_run_manifest(
        run_id="test-run",
        command=["python", "-m", "src.train_spectrogram"],
        args={},
        data={},
        transform={},
        model={},
        training={},
        device="cpu",
    )
    write_run_manifest(path, manifest)

    finalize_run_manifest(path, "completed", outputs={"checkpoint": "best_model.pth"})

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "completed"
    assert saved["outputs"] == {"checkpoint": "best_model.pth"}
    assert "finishedAt" in saved


def test_run_manifest_can_index_linked_artifacts_without_paths(tmp_path):
    path = tmp_path / "run-manifest.json"
    manifest = create_run_manifest(
        run_id="test-run",
        command=["python", "-m", "src.train_spectrogram"],
        args={},
        data={},
        transform={},
        model={},
        training={},
        device="cpu",
    )
    write_run_manifest(path, manifest)
    checkpoint = tmp_path / "best_model.pth"
    torch.save(
        make_checkpoint_payload(
            model_state_dict={"weight": torch.ones(1)},
            arch="resnet18",
            dropout=0.3,
            num_classes=9,
            n_fft=8,
            hop=2,
            target_freq=5,
            target_time=4,
            cache_time=7,
            epoch=1,
            metrics={"strict": 0.5},
            pretrained=False,
        ),
        checkpoint,
    )

    finalize_run_manifest_with_artifacts(path, "completed", outputs={"checkpoint": checkpoint.name})

    saved = json.loads(path.read_text(encoding="utf-8"))
    index = saved["artifactIndex"]
    assert index["schemaVersion"] == "artifact-index-v1"
    assert index["runId"] == "test-run"
    assert index["artifacts"][0]["fileName"] == checkpoint.name
    assert len(index["artifacts"][0]["sha256"]) == 64
    assert str(tmp_path) not in json.dumps(index)


def test_run_manifest_can_explicitly_write_v2_artifact_index(tmp_path):
    path = tmp_path / "run-manifest.json"
    manifest = create_run_manifest(
        run_id="test-run",
        command=["python", "-m", "src.train_spectrogram"],
        args={},
        data={},
        transform={},
        model={},
        training={},
        device="cpu",
    )
    write_run_manifest(path, manifest)

    finalize_run_manifest_with_artifacts(
        path,
        "completed",
        outputs={},
        artifact_index_schema=ARTIFACT_INDEX_V2_SCHEMA,
    )

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["artifactIndex"]["schemaVersion"] == ARTIFACT_INDEX_V2_SCHEMA
    assert saved["artifactIndex"]["artifacts"] == []


def test_run_manifest_rejects_explicit_unknown_artifact_index_schema(tmp_path):
    path = tmp_path / "run-manifest.json"
    manifest = create_run_manifest(
        run_id="test-run",
        command=["python", "-m", "src.train_spectrogram"],
        args={},
        data={},
        transform={},
        model={},
        training={},
        device="cpu",
    )
    write_run_manifest(path, manifest)

    with pytest.raises(ValueError, match="Unsupported artifact index schema"):
        finalize_run_manifest_with_artifacts(
            path,
            "completed",
            outputs={},
            artifact_index_schema="",
        )

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "running"
    assert "artifactIndex" not in saved


def test_artifact_index_finalizer_requires_completed_status(tmp_path):
    path = tmp_path / "run-manifest.json"
    manifest = create_run_manifest(
        run_id="test-run",
        command=["python", "-m", "src.train_spectrogram"],
        args={},
        data={},
        transform={},
        model={},
        training={},
        device="cpu",
    )
    write_run_manifest(path, manifest)

    with pytest.raises(ValueError, match="requires status 'completed'"):
        finalize_run_manifest_with_artifacts(path, "failed", outputs={})

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "running"
    assert "artifactIndex" not in saved
