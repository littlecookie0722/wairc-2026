import csv
import json
import shutil

import numpy as np

from src.dataset_fingerprint import DATASET_FINGERPRINT_SCHEMA, fingerprint_dataset


def write_dataset(root, *, label_signature="0|2", value=1):
    root.mkdir(parents=True, exist_ok=True)
    iq_dir = root / "iq_sample"
    iq_dir.mkdir(exist_ok=True)
    np.savez(
        iq_dir / "0007.npz",
        iq_node0=np.asarray([value, value + 1], dtype=np.int16),
        iq_node1=np.asarray([], dtype=np.int16),
        iq_node2=np.asarray([value + 2], dtype=np.int16),
        sample_rate_node0=np.asarray(125_000_000.0, dtype=np.float32),
        sample_rate_node1=np.asarray(0.0, dtype=np.float32),
        sample_rate_node2=np.asarray(125_000_000.0, dtype=np.float32),
    )
    fields = [
        "sample_id",
        "iq_npz_relpath",
        "has_node0",
        "has_node1",
        "has_node2",
        "sample_rate_node0",
        "sample_rate_node1",
        "sample_rate_node2",
        "label_signature",
    ]
    with (root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": 7,
                "iq_npz_relpath": "iq_sample/0007.npz",
                "has_node0": 1,
                "has_node1": 0,
                "has_node2": 1,
                "sample_rate_node0": 125_000_000,
                "sample_rate_node1": 0,
                "sample_rate_node2": 125_000_000,
                "label_signature": label_signature,
            }
        )


def test_fingerprint_is_root_independent_and_sanitized(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_dataset(first)
    shutil.copytree(first, second)

    fingerprint = fingerprint_dataset(first, has_labels=True)
    copied = fingerprint_dataset(second, has_labels=True)

    assert fingerprint == copied
    assert fingerprint["schemaVersion"] == DATASET_FINGERPRINT_SCHEMA
    assert fingerprint["algorithm"] == "sha256"
    assert fingerprint["rowCount"] == 1
    assert fingerprint["referencedFileCount"] == 1
    assert fingerprint["nodePresence"] == {"node0": 1, "node1": 0, "node2": 1}
    encoded = json.dumps(fingerprint)
    assert str(first) not in encoded
    assert "iq_sample" not in encoded
    assert "0007" not in encoded


def test_fingerprint_changes_when_index_or_iq_content_changes(tmp_path):
    root = tmp_path / "dataset"
    write_dataset(root)
    original = fingerprint_dataset(root, has_labels=True)

    (root / "index.csv").write_text(
        (root / "index.csv").read_text(encoding="utf-8").replace("0|2", "2|0"),
        encoding="utf-8",
    )
    reordered_label = fingerprint_dataset(root, has_labels=True)
    assert reordered_label == original

    (root / "index.csv").write_text(
        (root / "index.csv").read_text(encoding="utf-8").replace("2|0", "0"),
        encoding="utf-8",
    )
    changed_index = fingerprint_dataset(root, has_labels=True)
    assert changed_index["indexDigest"] != original["indexDigest"]
    assert changed_index["contentDigest"] != original["contentDigest"]

    write_dataset(root, value=9)
    changed_file = fingerprint_dataset(root, has_labels=True)
    assert changed_file["indexDigest"] == original["indexDigest"]
    assert changed_file["contentDigest"] != original["contentDigest"]
