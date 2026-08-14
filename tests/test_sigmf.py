import json
from pathlib import Path

import pytest

from wairc_rf.sigmf import SigMFCapture, SigMFMetadata, parse_sigmf_metadata


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
