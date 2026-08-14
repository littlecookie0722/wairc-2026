import json
from pathlib import Path

import numpy as np
import pytest

from wairc_rf.sigmf import (
    SigMFCapture,
    SigMFDatasetAdapter,
    SigMFMetadata,
    parse_sigmf_metadata,
)


FIXTURE = Path(__file__).parent / "fixtures" / "sigmf" / "minimal.sigmf-meta"


def _write_metadata(tmp_path, document):
    path = tmp_path / "sample.sigmf-meta"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _valid_document():
    return {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": 2_000_000.0,
            "core:version": "1.2.0",
        },
        "captures": [{"core:sample_start": 0}],
        "annotations": [],
    }


def test_sigmf_fixture_parses_supported_metadata_subset():
    metadata = parse_sigmf_metadata(FIXTURE)

    assert isinstance(metadata, SigMFMetadata)
    assert metadata.datatype == "ci16_le"
    assert metadata.iq_format == "interleaved"
    assert metadata.sample_rate == 4096.0
    assert metadata.dataset == "minimal.sigmf-data"
    assert metadata.captures == (SigMFCapture(sample_start=0, frequency=915_000_000.0),)
    assert metadata.annotations[0].sample_count == 8
    assert metadata.annotations[0].label == "synthetic-fixture"


def test_sigmf_empty_captures_use_the_specified_implicit_zero_capture(tmp_path):
    document = _valid_document()
    document["captures"] = []

    metadata = parse_sigmf_metadata(_write_metadata(tmp_path, document))

    assert metadata.captures == (SigMFCapture(sample_start=0),)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda doc: doc.pop("global"), "global object"),
        (lambda doc: doc["global"].pop("core:datatype"), "core:datatype"),
        (lambda doc: doc["global"].update({"core:datatype": "ri16_le"}), "complex datatype"),
        (lambda doc: doc["global"].update({"core:version": "2.0.0"}), "specification version"),
        (lambda doc: doc["global"].update({"core:sample_rate": 0}), "finite and positive"),
        (lambda doc: doc.update({"captures": [{"core:sample_start": 2}, {"core:sample_start": 1}]}), "captures must be sorted"),
        (lambda doc: doc.update({"annotations": [{"core:sample_start": 0, "core:sample_count": -1}]}), "sample_count"),
        (lambda doc: doc["global"].update({"core:extensions": [{"name": "future", "version": "1.0.0", "optional": True}]}), "extensions"),
    ],
)
def test_sigmf_parser_rejects_unsupported_or_malformed_metadata(tmp_path, change, message):
    document = _valid_document()
    change(document)

    with pytest.raises(ValueError, match=message):
        parse_sigmf_metadata(_write_metadata(tmp_path, document))


@pytest.mark.parametrize(
    "dataset_name",
    ["../outside.sigmf-data", r"..\outside.sigmf-data", r"C:\\outside.sigmf-data", ".."],
)
def test_sigmf_parser_rejects_unsafe_dataset_filename(tmp_path, dataset_name):
    document = _valid_document()
    document["global"]["core:dataset"] = dataset_name

    with pytest.raises(ValueError, match="filename"):
        parse_sigmf_metadata(_write_metadata(tmp_path, document))


def test_sigmf_parser_does_not_expose_absolute_metadata_path_in_missing_error(tmp_path):
    missing = tmp_path / "missing.sigmf-meta"

    with pytest.raises(FileNotFoundError, match=r"missing\.sigmf-meta") as error:
        parse_sigmf_metadata(missing)
    assert str(tmp_path) not in str(error.value)


def test_sigmf_dataset_adapter_loads_interleaved_integer_iq_lazily(tmp_path):
    document = _valid_document()
    document["global"]["core:datatype"] = "ci16_le"
    metadata_path = _write_metadata(tmp_path, document)
    np.asarray([1, -1, 2, -2], dtype="<i2").tofile(tmp_path / "sample.sigmf-data")

    adapter = SigMFDatasetAdapter(metadata_path)
    sample = adapter[0]

    assert adapter.sample_ids == ("sample",)
    assert sample.labels is None
    assert sample.node_mask == (True,)
    assert sample.nodes[0].iq_format == "interleaved"
    assert sample.nodes[0].sample_rate == 2_000_000.0
    np.testing.assert_array_equal(sample.nodes[0].iq, [1, -1, 2, -2])


def test_sigmf_dataset_adapter_loads_native_complex_float_iq(tmp_path):
    document = _valid_document()
    document["global"]["core:datatype"] = "cf32_le"
    metadata_path = _write_metadata(tmp_path, document)
    np.asarray([1 + 2j, 3 + 4j], dtype="<c8").tofile(tmp_path / "sample.sigmf-data")

    sample = SigMFDatasetAdapter(metadata_path)[-1]

    assert sample.nodes[0].iq_format == "complex"
    np.testing.assert_array_equal(sample.nodes[0].iq, [1 + 2j, 3 + 4j])


@pytest.mark.parametrize(
    ("setup", "error_type", "message"),
    [
        (lambda path: None, FileNotFoundError, "dataset file not found"),
        (lambda path: path.write_bytes(b"\x00"), ValueError, "byte length"),
        (lambda path: np.asarray([1], dtype="<i2").tofile(path), ValueError, "complete I/Q pairs"),
    ],
)
def test_sigmf_dataset_adapter_rejects_missing_or_malformed_raw_data(
    tmp_path, setup, error_type, message
):
    document = _valid_document()
    document["global"]["core:datatype"] = "ci16_le"
    metadata_path = _write_metadata(tmp_path, document)
    data_path = tmp_path / "sample.sigmf-data"
    if error_type is FileNotFoundError:
        with pytest.raises(error_type, match=message):
            SigMFDatasetAdapter(metadata_path)
        return
    setup(data_path)
    adapter = SigMFDatasetAdapter(metadata_path)

    with pytest.raises(error_type, match=message):
        adapter[0]


def test_sigmf_dataset_adapter_rejects_metadata_only_recordings(tmp_path):
    document = _valid_document()
    document["global"]["core:metadata_only"] = True

    with pytest.raises(ValueError, match="metadata_only"):
        SigMFDatasetAdapter(_write_metadata(tmp_path, document))
