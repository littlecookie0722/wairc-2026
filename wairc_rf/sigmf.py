"""Conservative parsing for the supported SigMF metadata subset."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path, PureWindowsPath
from typing import Literal, overload

import numpy as np

from .datasets import RFNode, RFSample


_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_DATATYPE_RE = re.compile(r"^c(?P<kind>f32|f64|i32|i16|u32|u16|i8|u8)(?:_(?P<byteorder>le|be))?$")
SigMFIQFormat = Literal["interleaved", "complex"]


@dataclass(frozen=True, slots=True)
class SigMFCapture:
    """The capture fields consumed by the current SigMF integration."""

    sample_start: int
    frequency: float | None = None


@dataclass(frozen=True, slots=True)
class SigMFAnnotation:
    """The annotation fields retained without assigning project labels."""

    sample_start: int
    sample_count: int | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class SigMFMetadata:
    """Validated metadata for one single-channel SigMF recording."""

    datatype: str
    version: str
    sample_rate: float
    captures: tuple[SigMFCapture, ...]
    annotations: tuple[SigMFAnnotation, ...]
    dataset: str | None = None
    metadata_only: bool = False
    num_channels: int = 1

    @property
    def iq_format(self) -> SigMFIQFormat:
        """Return the RFNode representation implied by ``core:datatype``."""

        if self.datatype[1] == "f":
            return "complex"
        return "interleaved"


class SigMFDatasetAdapter(Sequence[RFSample]):
    """Read one supported SigMF recording as a single unlabeled RF sample."""

    def __init__(self, metadata_path: str | Path, data_path: str | Path | None = None) -> None:
        self._metadata_path = Path(metadata_path).resolve()
        self.metadata = parse_sigmf_metadata(self._metadata_path)
        if self.metadata.metadata_only:
            raise ValueError("SigMF metadata_only recordings do not contain a dataset to load")
        self._data_path = self._resolve_data_path(data_path)
        self._sample_id = self._metadata_path.name.removesuffix(".sigmf-meta")

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return (self._sample_id,)

    def __len__(self) -> int:
        return 1

    @overload
    def __getitem__(self, index: int) -> RFSample: ...

    @overload
    def __getitem__(self, index: slice) -> list[RFSample]: ...

    def __getitem__(self, index: int | slice) -> RFSample | list[RFSample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if isinstance(index, bool) or index not in (0, -1):
            raise IndexError("SigMF recording adapter contains one sample")

        values = self._read_data()
        node = RFNode(values, self.metadata.sample_rate)
        return RFSample(sample_id=self._sample_id, nodes=(node,), labels=None)

    def _resolve_data_path(self, data_path: str | Path | None) -> Path:
        parent = self._metadata_path.parent
        if data_path is None:
            if self.metadata.dataset is not None:
                candidate = parent / self.metadata.dataset
            else:
                candidate = self._metadata_path.with_suffix(".sigmf-data")
        else:
            candidate = Path(data_path)
            if not candidate.is_absolute():
                candidate = parent / candidate

        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"SigMF dataset file not found: {candidate.name}") from exc
        if resolved.parent != parent:
            raise ValueError("SigMF dataset file must remain in the metadata directory")
        if not resolved.is_file():
            raise ValueError(f"SigMF dataset path is not a file: {resolved.name}")
        return resolved

    def _read_data(self) -> np.ndarray:
        dtype = _sigmf_numpy_dtype(self.metadata.datatype)
        try:
            byte_size = self._data_path.stat().st_size
        except OSError as exc:
            raise FileNotFoundError(f"SigMF dataset file is unavailable: {self._data_path.name}") from exc
        if byte_size % dtype.itemsize != 0:
            raise ValueError("SigMF dataset byte length is not aligned to its datatype")

        values = np.fromfile(self._data_path, dtype=dtype)
        if values.size == 0:
            raise ValueError("SigMF dataset contains no samples")
        if not np.issubdtype(values.dtype, np.complexfloating) and values.size % 2 != 0:
            raise ValueError("SigMF interleaved IQ dataset must contain complete I/Q pairs")
        return values


def parse_sigmf_metadata(metadata_path: str | Path) -> SigMFMetadata:
    """Parse the supported, single-channel SigMF metadata subset.

    The parser intentionally does not load a dataset file or interpret labels as
    the repository's nine-class competition mapping. It accepts complex SigMF
    datatypes, global sample rate, capture starts/frequencies, and annotation
    starts/counts/labels. Unsupported file-layout semantics fail explicitly.
    """

    path = Path(metadata_path)
    display_name = path.name or "metadata"
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"SigMF metadata file not found: {display_name}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid SigMF metadata JSON: {display_name}") from exc

    if not isinstance(document, dict):
        raise ValueError("SigMF metadata root must be an object")
    global_data = document.get("global")
    if not isinstance(global_data, dict):
        raise ValueError("SigMF metadata requires a global object")

    datatype = _require_string(global_data, "core:datatype")
    datatype_match = _DATATYPE_RE.fullmatch(datatype)
    if datatype_match is None:
        raise ValueError(f"Unsupported SigMF complex datatype: {datatype!r}")

    version = _require_string(global_data, "core:version")
    version_match = _VERSION_RE.fullmatch(version)
    if version_match is None or int(version_match.group(1)) != 1:
        raise ValueError(f"Unsupported SigMF specification version: {version!r}")

    sample_rate_value = global_data.get("core:sample_rate")
    if isinstance(sample_rate_value, (bool,)) or not isinstance(sample_rate_value, Real):
        raise ValueError("SigMF global.core:sample_rate must be a real number")
    sample_rate = float(sample_rate_value)
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("SigMF global.core:sample_rate must be finite and positive")

    num_channels = _optional_nonnegative_int(global_data, "core:num_channels", default=1)
    if num_channels != 1:
        raise ValueError("SigMF adapter currently supports only core:num_channels=1")
    offset = _optional_nonnegative_int(global_data, "core:offset", default=0)
    if offset != 0:
        raise ValueError("SigMF adapter currently supports only core:offset=0")
    trailing_bytes = _optional_nonnegative_int(global_data, "core:trailing_bytes", default=0)
    if trailing_bytes != 0:
        raise ValueError("SigMF adapter does not support core:trailing_bytes")

    extensions = global_data.get("core:extensions", [])
    if not isinstance(extensions, list):
        raise ValueError("SigMF global.core:extensions must be an array")
    if extensions:
        raise ValueError("SigMF extensions are outside the supported metadata subset")

    dataset = global_data.get("core:dataset")
    if dataset is not None:
        if (
            not isinstance(dataset, str)
            or not dataset
            or dataset in {".", ".."}
            or "/" in dataset
            or "\\" in dataset
            or PureWindowsPath(dataset).drive
            or Path(dataset).name != dataset
        ):
            raise ValueError("SigMF core:dataset must be a filename in the metadata directory")

    captures = _parse_captures(document.get("captures"))
    annotations = _parse_annotations(document.get("annotations"))
    metadata_only = global_data.get("core:metadata_only", False)
    if not isinstance(metadata_only, bool):
        raise ValueError("SigMF global.core:metadata_only must be boolean")
    return SigMFMetadata(
        datatype=datatype,
        version=version,
        sample_rate=sample_rate,
        captures=captures,
        annotations=annotations,
        dataset=dataset,
        metadata_only=metadata_only,
        num_channels=num_channels,
    )


def _sigmf_numpy_dtype(datatype: str) -> np.dtype:
    match = _DATATYPE_RE.fullmatch(datatype)
    if match is None:
        raise ValueError(f"Unsupported SigMF complex datatype: {datatype!r}")
    kind = match.group("kind")
    code = {
        "f32": "c8",
        "f64": "c16",
        "i32": "i4",
        "i16": "i2",
        "u32": "u4",
        "u16": "u2",
        "i8": "i1",
        "u8": "u1",
    }[kind]
    byteorder = match.group("byteorder")
    if byteorder is None or kind in {"i8", "u8"}:
        return np.dtype(code)
    return np.dtype(("<" if byteorder == "le" else ">") + code)


def _require_string(values: dict, key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"SigMF global.{key} must be a non-empty string")
    return value


def _optional_nonnegative_int(values: dict, key: str, *, default: int) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"SigMF global.{key} must be a non-negative integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"SigMF global.{key} must be a non-negative integer")
    return resolved


def _parse_captures(value) -> tuple[SigMFCapture, ...]:
    if value is None:
        raise ValueError("SigMF metadata requires a captures array")
    if not isinstance(value, list):
        raise ValueError("SigMF captures must be an array")
    if not value:
        return (SigMFCapture(sample_start=0),)

    captures: list[SigMFCapture] = []
    previous_start = -1
    for capture in value:
        if not isinstance(capture, dict):
            raise ValueError("SigMF capture entries must be objects")
        sample_start = _segment_start(capture, "capture")
        if sample_start < previous_start:
            raise ValueError("SigMF captures must be sorted by core:sample_start")
        previous_start = sample_start
        frequency = capture.get("core:frequency")
        if frequency is not None:
            if isinstance(frequency, bool) or not isinstance(frequency, Real):
                raise ValueError("SigMF capture core:frequency must be a real number")
            frequency = float(frequency)
            if not math.isfinite(frequency):
                raise ValueError("SigMF capture core:frequency must be finite")
        header_bytes = capture.get("core:header_bytes", 0)
        if isinstance(header_bytes, bool) or not isinstance(header_bytes, Integral) or header_bytes < 0:
            raise ValueError("SigMF capture core:header_bytes must be a non-negative integer")
        if header_bytes:
            raise ValueError("SigMF adapter does not support capture core:header_bytes")
        captures.append(SigMFCapture(sample_start=sample_start, frequency=frequency))
    return tuple(captures)


def _parse_annotations(value) -> tuple[SigMFAnnotation, ...]:
    if value is None:
        raise ValueError("SigMF metadata requires an annotations array")
    if not isinstance(value, list):
        raise ValueError("SigMF annotations must be an array")

    annotations: list[SigMFAnnotation] = []
    previous_start = -1
    for annotation in value:
        if not isinstance(annotation, dict):
            raise ValueError("SigMF annotation entries must be objects")
        sample_start = _segment_start(annotation, "annotation")
        if sample_start < previous_start:
            raise ValueError("SigMF annotations must be sorted by core:sample_start")
        previous_start = sample_start
        sample_count = annotation.get("core:sample_count")
        if sample_count is not None:
            if isinstance(sample_count, bool) or not isinstance(sample_count, Integral) or sample_count < 0:
                raise ValueError("SigMF annotation core:sample_count must be a non-negative integer")
            sample_count = int(sample_count)
        label = annotation.get("core:label")
        if label is not None and not isinstance(label, str):
            raise ValueError("SigMF annotation core:label must be a string")
        annotations.append(
            SigMFAnnotation(sample_start=sample_start, sample_count=sample_count, label=label)
        )
    return tuple(annotations)


def _segment_start(segment: dict, kind: str) -> int:
    value = segment.get("core:sample_start")
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"SigMF {kind} core:sample_start must be a non-negative integer")
    return int(value)
