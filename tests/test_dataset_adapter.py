import csv
from pathlib import Path

import numpy as np
import pytest

from wairc_rf import (
    CompetitionDatasetAdapter,
    RFDatasetAdapter,
    RFNode,
    RFSample,
    SyntheticDatasetAdapter,
)


INDEX_FIELDS = [
    "sample_id",
    "iq_npz_relpath",
    "has_node0",
    "has_node1",
    "has_node2",
    "sample_rate_node0",
    "sample_rate_node1",
    "sample_rate_node2",
]


def _valid_arrays() -> dict[str, np.ndarray]:
    return {
        "iq_node0": np.asarray([1, -1, 2, -2], dtype=np.int16),
        "iq_node1": np.asarray([], dtype=np.int16),
        "iq_node2": np.asarray([3, -3, 4, -4], dtype=np.int16),
        "sample_rate_node0": np.asarray(4_096.0, dtype=np.float32),
        "sample_rate_node1": np.asarray(0.0, dtype=np.float32),
        "sample_rate_node2": np.asarray(8_192.0, dtype=np.float32),
    }


def _valid_row(*, sample_id: int = 7, relpath: str = "iq_sample/0007.npz") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "iq_npz_relpath": relpath,
        "has_node0": 1,
        "has_node1": 0,
        "has_node2": 1,
        "sample_rate_node0": 4_096.0,
        "sample_rate_node1": 0.0,
        "sample_rate_node2": 8_192.0,
        "label_signature": "2|0",
    }


def _write_index(root: Path, rows: list[dict[str, object]], *, has_labels: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fieldnames = [*INDEX_FIELDS, "label_signature"] if has_labels else INDEX_FIELDS
    with (root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_valid_dataset(root: Path, *, has_labels: bool = True) -> None:
    iq_dir = root / "iq_sample"
    iq_dir.mkdir(parents=True)
    np.savez(iq_dir / "0007.npz", **_valid_arrays())
    row = _valid_row()
    if not has_labels:
        row.pop("label_signature")
    _write_index(root, [row], has_labels=has_labels)


def test_competition_adapter_preserves_ids_node_order_missing_nodes_and_labels(tmp_path):
    _write_valid_dataset(tmp_path)

    adapter = CompetitionDatasetAdapter(tmp_path, has_labels=True)
    sample = adapter[0]

    assert isinstance(adapter, RFDatasetAdapter)
    assert adapter.root == tmp_path.resolve()
    assert adapter.has_labels is True
    assert adapter.sample_ids == (7,)
    assert [item.sample_id for item in adapter[:]] == [7]
    assert sample.sample_id == 7
    assert sample.labels == (0, 2)
    assert sample.node_mask == (True, False, True)
    assert [node.sample_rate for node in sample.nodes] == [4_096.0, 0.0, 8_192.0]
    assert [node.iq_format for node in sample.nodes] == ["interleaved", "interleaved", "interleaved"]
    np.testing.assert_array_equal(sample.nodes[0].iq, [1, -1, 2, -2])
    assert sample.nodes[1].iq.dtype == np.int16
    assert sample.nodes[1].iq.size == 0
    assert "[ 1, -1,  2, -2]" not in repr(sample)


def test_competition_adapter_returns_none_labels_for_public_test_rows(tmp_path):
    _write_valid_dataset(tmp_path, has_labels=False)

    adapter = CompetitionDatasetAdapter(tmp_path, has_labels=False)

    assert adapter[0].labels is None


def test_rf_sample_contract_supports_native_complex_nodes_and_canonical_labels():
    node = RFNode(np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64), np.float32(2_000_000))
    sample = RFSample(np.int64(11), (node,), labels=(2, 0))

    assert node.iq_format == "complex"
    assert sample.sample_id == 11
    assert sample.labels == (0, 2)
    assert sample.node_mask == (True,)


def test_synthetic_adapter_preserves_sequence_order_and_supports_slices():
    node = RFNode(np.asarray([1, 2], dtype=np.int16), 1.0)
    first = RFSample("first", (node,), labels=(0,))
    second = RFSample("second", (node,))
    adapter = SyntheticDatasetAdapter([first, second])

    assert isinstance(adapter, RFDatasetAdapter)
    assert adapter.sample_ids == ("first", "second")
    assert adapter[0] is first
    assert [sample.sample_id for sample in adapter[1:]] == ["second"]
    assert [sample.sample_id for sample in adapter] == ["first", "second"]


def test_synthetic_adapter_rejects_invalid_or_duplicate_samples():
    node = RFNode(np.asarray([1, 2], dtype=np.int16), 1.0)
    sample = RFSample(1, (node,))

    with pytest.raises(ValueError, match="unique sample_id"):
        SyntheticDatasetAdapter([sample, sample])
    with pytest.raises(TypeError, match="RFSample"):
        SyntheticDatasetAdapter(["not a sample"])


@pytest.mark.parametrize(
    ("node", "error_type", "message"),
    [
        ((np.asarray([1, 2, 3], dtype=np.int16), 1.0, True), ValueError, "complete I/Q pairs"),
        ((np.asarray([], dtype=np.int16), 1.0, True), ValueError, "present nodes"),
        ((np.asarray([1, 2], dtype=np.int16), 0.0, False), ValueError, "missing nodes"),
        ((np.asarray([], dtype=np.int16), 0.0, "no"), TypeError, "present must be"),
    ],
)
def test_rf_node_rejects_inconsistent_contracts(node, error_type, message):
    with pytest.raises(error_type, match=message):
        RFNode(*node)


def test_competition_adapter_rejects_duplicate_sample_ids(tmp_path):
    rows = [_valid_row(), _valid_row(relpath="iq_sample/other.npz")]
    _write_index(tmp_path, rows)

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        CompetitionDatasetAdapter(tmp_path, has_labels=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("has_node1", 2, "must be 0 or 1"),
        ("sample_rate_node1", 1.0, "requires sample rate 0"),
        ("sample_rate_node0", float("nan"), "must be finite"),
    ],
)
def test_competition_adapter_rejects_invalid_index_node_metadata(tmp_path, field, value, message):
    row = _valid_row()
    row[field] = value
    _write_index(tmp_path, [row])

    with pytest.raises(ValueError, match=message):
        CompetitionDatasetAdapter(tmp_path, has_labels=True)


@pytest.mark.parametrize(
    ("case", "error_type", "message"),
    [
        ("missing-field", ValueError, "missing fields"),
        ("wrong-iq-dtype", TypeError, "int16 interleaved IQ"),
        ("odd-iq", ValueError, "complete I/Q pairs"),
        ("rate-vector", ValueError, "must be a scalar"),
        ("wrong-rate-dtype", TypeError, "must use float32"),
        ("rate-mismatch", ValueError, "does not match index.csv"),
        ("present-empty", ValueError, "present node 0"),
        ("missing-has-data", ValueError, "missing node 1"),
    ],
)
def test_competition_adapter_rejects_malformed_npz(tmp_path, case, error_type, message):
    arrays = _valid_arrays()
    if case == "missing-field":
        arrays.pop("iq_node2")
    elif case == "wrong-iq-dtype":
        arrays["iq_node0"] = arrays["iq_node0"].astype(np.float32)
    elif case == "odd-iq":
        arrays["iq_node0"] = np.asarray([1, 2, 3], dtype=np.int16)
    elif case == "rate-vector":
        arrays["sample_rate_node0"] = np.asarray([4_096.0], dtype=np.float32)
    elif case == "wrong-rate-dtype":
        arrays["sample_rate_node0"] = np.asarray(4_096.0, dtype=np.float64)
    elif case == "rate-mismatch":
        arrays["sample_rate_node0"] = np.asarray(2_048.0, dtype=np.float32)
    elif case == "present-empty":
        arrays["iq_node0"] = np.asarray([], dtype=np.int16)
    elif case == "missing-has-data":
        arrays["iq_node1"] = np.asarray([1, 2], dtype=np.int16)

    iq_dir = tmp_path / "iq_sample"
    iq_dir.mkdir(parents=True)
    np.savez(iq_dir / "0007.npz", **arrays)
    _write_index(tmp_path, [_valid_row()])
    adapter = CompetitionDatasetAdapter(tmp_path, has_labels=True)

    with pytest.raises(error_type, match=message):
        adapter[0]


def test_competition_adapter_rejects_paths_outside_dataset_root(tmp_path):
    dataset_root = tmp_path / "dataset"
    np.savez(tmp_path / "outside.npz", **_valid_arrays())
    _write_index(dataset_root, [_valid_row(relpath="../outside.npz")])
    adapter = CompetitionDatasetAdapter(dataset_root, has_labels=True)

    with pytest.raises(ValueError, match="escapes the dataset root"):
        adapter[0]


def test_competition_adapter_reports_missing_relative_iq_file_without_absolute_path(tmp_path):
    _write_index(tmp_path, [_valid_row(relpath="iq_sample/missing.npz")])
    adapter = CompetitionDatasetAdapter(tmp_path, has_labels=True)

    with pytest.raises(FileNotFoundError, match=r"sample 7: iq_sample/missing\.npz") as error:
        adapter[0]
    assert str(tmp_path) not in str(error.value)
