"""Public RF sample contracts and dataset adapters."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Protocol, overload, runtime_checkable

import numpy as np

from src.data import load_index

from .labels import parse_label_signature


IQFormat = Literal["interleaved", "complex"]


@dataclass(frozen=True, slots=True, eq=False)
class RFNode:
    """One receiver node's IQ samples and sampling rate."""

    iq: np.ndarray = field(repr=False)
    sample_rate: float
    present: bool = True

    def __post_init__(self) -> None:
        values = np.asarray(self.iq)
        if values.ndim != 1:
            raise ValueError("iq must be a one-dimensional array")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("iq must contain numeric values")
        if not np.issubdtype(values.dtype, np.complexfloating) and values.size % 2 != 0:
            raise ValueError("interleaved iq must contain complete I/Q pairs")

        if isinstance(self.sample_rate, (bool, np.bool_)) or not isinstance(self.sample_rate, Real):
            raise TypeError("sample_rate must be a real number")
        sample_rate = float(self.sample_rate)
        if not np.isfinite(sample_rate):
            raise ValueError("sample_rate must be finite")

        if not isinstance(self.present, (bool, np.bool_)):
            raise TypeError("present must be a boolean")
        present = bool(self.present)
        if present and (values.size == 0 or sample_rate <= 0):
            raise ValueError("present nodes require IQ samples and a positive sample_rate")
        if not present and (values.size != 0 or sample_rate != 0):
            raise ValueError("missing nodes require empty IQ samples and sample_rate 0")

        object.__setattr__(self, "iq", values)
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "present", present)

    @property
    def iq_format(self) -> IQFormat:
        """Return the transform input format implied by the IQ dtype."""

        if np.issubdtype(self.iq.dtype, np.complexfloating):
            return "complex"
        return "interleaved"


@dataclass(frozen=True, slots=True, eq=False)
class RFSample:
    """A labeled or unlabeled RF observation from one or more nodes."""

    sample_id: int | str
    nodes: tuple[RFNode, ...]
    labels: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        sample_id = self.sample_id
        if isinstance(sample_id, (bool, np.bool_)):
            raise TypeError("sample_id must be an integer or non-empty string")
        if isinstance(sample_id, Integral):
            sample_id = int(sample_id)
        elif not isinstance(sample_id, str) or not sample_id.strip():
            raise TypeError("sample_id must be an integer or non-empty string")

        nodes = tuple(self.nodes)
        if not nodes:
            raise ValueError("nodes must contain at least one RFNode")
        if any(not isinstance(node, RFNode) for node in nodes):
            raise TypeError("nodes must contain only RFNode instances")

        labels: tuple[int, ...] | None = None
        if self.labels is not None:
            resolved: list[int] = []
            for label in self.labels:
                if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral):
                    raise TypeError("labels must contain non-negative integers")
                value = int(label)
                if value < 0:
                    raise ValueError("labels must contain non-negative integers")
                resolved.append(value)
            if len(set(resolved)) != len(resolved):
                raise ValueError("labels must not contain duplicates")
            labels = tuple(sorted(resolved))

        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "labels", labels)

    @property
    def node_mask(self) -> tuple[bool, ...]:
        """Return node availability in the same order as ``nodes``."""

        return tuple(node.present for node in self.nodes)


@runtime_checkable
class RFDatasetAdapter(Protocol):
    """Minimal sequence-like contract for RF dataset interoperability."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> RFSample: ...


class SyntheticDatasetAdapter(Sequence[RFSample]):
    """Expose generated or fixture samples through the public sequence contract."""

    def __init__(self, samples: Iterable[RFSample]) -> None:
        resolved = tuple(samples)
        if any(not isinstance(sample, RFSample) for sample in resolved):
            raise TypeError("samples must contain only RFSample instances")

        sample_ids = [sample.sample_id for sample in resolved]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Synthetic samples must have unique sample_id values")

        self._samples = resolved

    @property
    def sample_ids(self) -> tuple[int | str, ...]:
        return tuple(sample.sample_id for sample in self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @overload
    def __getitem__(self, index: int) -> RFSample: ...

    @overload
    def __getitem__(self, index: slice) -> list[RFSample]: ...

    def __getitem__(self, index: int | slice) -> RFSample | list[RFSample]:
        if isinstance(index, slice):
            return list(self._samples[index])
        return self._samples[index]


class CompetitionDatasetAdapter(Sequence[RFSample]):
    """Read the three-node competition CSV/NPZ schema as public RF samples."""

    node_count = 3

    def __init__(self, root: str | Path, *, has_labels: bool) -> None:
        if not isinstance(has_labels, (bool, np.bool_)):
            raise TypeError("has_labels must be a boolean")

        self._root = Path(root).resolve()
        self._has_labels = bool(has_labels)
        self._rows = tuple(load_index(self._root, has_labels=self._has_labels))
        self._validate_index_rows()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def has_labels(self) -> bool:
        return self._has_labels

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return tuple(row["sample_id"] for row in self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    @overload
    def __getitem__(self, index: int) -> RFSample: ...

    @overload
    def __getitem__(self, index: slice) -> list[RFSample]: ...

    def __getitem__(self, index: int | slice) -> RFSample | list[RFSample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]

        row = self._rows[index]
        path = self._resolve_iq_path(row)
        nodes = self._load_nodes(path, row)
        labels = None
        if self._has_labels:
            labels = tuple(parse_label_signature(row["label_signature"]))
        return RFSample(sample_id=row["sample_id"], nodes=nodes, labels=labels)

    def _validate_index_rows(self) -> None:
        seen_ids: set[int] = set()
        for row in self._rows:
            sample_id = row["sample_id"]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id in index.csv: {sample_id}")
            seen_ids.add(sample_id)

            for node_index in range(self.node_count):
                present = row[f"has_node{node_index}"]
                if present not in (0, 1):
                    raise ValueError(f"has_node{node_index} must be 0 or 1 for sample {sample_id}")
                sample_rate = row[f"sample_rate_node{node_index}"]
                if not np.isfinite(sample_rate):
                    raise ValueError(f"sample_rate_node{node_index} must be finite for sample {sample_id}")
                if present == 1 and sample_rate <= 0:
                    raise ValueError(f"present node {node_index} requires a positive sample rate for sample {sample_id}")
                if present == 0 and sample_rate != 0:
                    raise ValueError(f"missing node {node_index} requires sample rate 0 for sample {sample_id}")

    def _resolve_iq_path(self, row: dict) -> Path:
        sample_id = row["sample_id"]
        raw_relpath = str(row["iq_npz_relpath"])
        normalized = raw_relpath.replace("\\", "/")
        if (
            not normalized
            or PurePosixPath(normalized).is_absolute()
            or PureWindowsPath(raw_relpath).is_absolute()
        ):
            raise ValueError(f"IQ path must be relative to the dataset root for sample {sample_id}")

        try:
            path = (self._root / normalized).resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing IQ file for sample {sample_id}: {normalized}") from exc
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(f"IQ path escapes the dataset root for sample {sample_id}") from exc
        if not path.is_file():
            raise ValueError(f"IQ path is not a file for sample {sample_id}: {normalized}")
        return path

    def _load_nodes(self, path: Path, row: dict) -> tuple[RFNode, ...]:
        required = {
            *(f"iq_node{node_index}" for node_index in range(self.node_count)),
            *(f"sample_rate_node{node_index}" for node_index in range(self.node_count)),
        }
        with np.load(path, allow_pickle=False) as data:
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"IQ NPZ for sample {row['sample_id']} is missing fields: {sorted(missing)}")

            nodes = tuple(self._load_node(data, row, node_index) for node_index in range(self.node_count))
        return nodes

    @staticmethod
    def _load_node(data: np.lib.npyio.NpzFile, row: dict, node_index: int) -> RFNode:
        sample_id = row["sample_id"]
        iq_key = f"iq_node{node_index}"
        rate_key = f"sample_rate_node{node_index}"
        iq = np.asarray(data[iq_key])
        if iq.ndim != 1:
            raise ValueError(f"{iq_key} must be one-dimensional for sample {sample_id}")
        if iq.dtype != np.dtype(np.int16):
            raise TypeError(f"{iq_key} must use int16 interleaved IQ for sample {sample_id}")
        if iq.size % 2 != 0:
            raise ValueError(f"{iq_key} must contain complete I/Q pairs for sample {sample_id}")

        stored_rate = np.asarray(data[rate_key])
        if stored_rate.shape != ():
            raise ValueError(f"{rate_key} must be a scalar for sample {sample_id}")
        if stored_rate.dtype != np.dtype(np.float32):
            raise TypeError(f"{rate_key} must use float32 for sample {sample_id}")
        sample_rate = float(stored_rate)
        if not np.isfinite(sample_rate):
            raise ValueError(f"{rate_key} must be finite for sample {sample_id}")
        if sample_rate != row[rate_key]:
            raise ValueError(f"{rate_key} does not match index.csv for sample {sample_id}")

        present = bool(row[f"has_node{node_index}"])
        if present and (iq.size == 0 or sample_rate <= 0):
            raise ValueError(f"present node {node_index} has missing IQ data for sample {sample_id}")
        if not present and (iq.size != 0 or sample_rate != 0):
            raise ValueError(f"missing node {node_index} has IQ data for sample {sample_id}")
        return RFNode(iq=iq, sample_rate=sample_rate, present=present)
