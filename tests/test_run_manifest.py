import json

from src.run_manifest import (
    RUN_MANIFEST_SCHEMA,
    create_run_manifest,
    finalize_run_manifest,
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
