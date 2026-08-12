import csv

import pytest

from src.data import (
    label_to_multihot,
    load_index,
    multihot_to_signature,
    normalize_label_signature,
    parse_label_signature,
)


def write_index(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_label_round_trip_is_canonical():
    assert parse_label_signature("2|0") == [0, 2]
    assert normalize_label_signature("2|0") == "0|2"
    assert label_to_multihot("2|0") == [1, 0, 1, 0, 0, 0, 0, 0, 0]
    assert multihot_to_signature([1, 0, 1, 0, 0, 0, 0, 0, 0]) == "0|2"


@pytest.mark.parametrize("signature", ["", "9", "1|1", "hello"])
def test_invalid_label_signature_is_rejected(signature):
    with pytest.raises(ValueError):
        parse_label_signature(signature)


def test_load_index_normalizes_numeric_fields(tmp_path):
    write_index(
        tmp_path,
        [
            {
                "sample_id": "7",
                "iq_npz_relpath": "iq_sample/0007.npz",
                "has_node0": "1",
                "has_node1": "0",
                "has_node2": "1",
                "sample_rate_node0": "125000000",
                "sample_rate_node1": "0",
                "sample_rate_node2": "122880000",
                "label_signature": "2|0",
            }
        ],
    )

    rows = load_index(tmp_path, has_labels=True)

    assert rows == [
        {
            "sample_id": 7,
            "iq_npz_relpath": "iq_sample/0007.npz",
            "has_node0": 1,
            "has_node1": 0,
            "has_node2": 1,
            "sample_rate_node0": 125000000.0,
            "sample_rate_node1": 0.0,
            "sample_rate_node2": 122880000.0,
            "label_signature": "0|2",
        }
    ]
