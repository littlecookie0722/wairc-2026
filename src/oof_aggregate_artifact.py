"""Versioned aggregate OOF probabilities with legacy-compatible loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import NUM_CLASSES


OOF_AGGREGATE_SCHEMA = "oof-aggregate-v1"
OOF_AGGREGATE_ARTIFACT_TYPE = "aggregated-oof-probabilities"
LEGACY_OOF_AGGREGATE_SCHEMA = "legacy-unversioned"
SUPPORTED_AGGREGATION_METHODS = {"mean", "tag-weighted"}


def write_oof_aggregate_artifact(
    path: Path,
    *,
    probs: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    source_files: list[str],
    tag_weights: dict[str, float] | None = None,
) -> None:
    """Write aggregate probabilities without changing the historical arrays."""
    weights = dict(tag_weights or {})
    method = "tag-weighted" if weights else "mean"
    sanitized_sources = [Path(str(source)).name for source in source_files]
    _validate_arrays(probs, labels, sample_ids)
    _validate_metadata(method, sanitized_sources, weights)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tag_names = sorted(weights)
    np.savez(
        path,
        schemaVersion=np.asarray(OOF_AGGREGATE_SCHEMA),
        artifactType=np.asarray(OOF_AGGREGATE_ARTIFACT_TYPE),
        numClasses=np.asarray(NUM_CLASSES, dtype=np.int32),
        aggregationMethod=np.asarray(method),
        probs=probs.astype(np.float16),
        labels=labels.astype(np.int8),
        sample_ids=sample_ids.astype(np.int64),
        source_files=np.asarray(sanitized_sources, dtype=str),
        tag_names=np.asarray(tag_names, dtype=str),
        tag_weights=np.asarray([weights[tag] for tag in tag_names], dtype=np.float32),
    )


def load_oof_aggregate_artifact(path: Path) -> dict[str, Any]:
    """Load and validate a versioned or historical aggregate OOF file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        schema = _read_scalar_string(data, "schemaVersion", LEGACY_OOF_AGGREGATE_SCHEMA)
        artifact_type = _read_scalar_string(data, "artifactType", OOF_AGGREGATE_ARTIFACT_TYPE)
        if schema not in {OOF_AGGREGATE_SCHEMA, LEGACY_OOF_AGGREGATE_SCHEMA}:
            raise ValueError(f"Unsupported OOF aggregate schema {schema!r} in {path}")
        if artifact_type != OOF_AGGREGATE_ARTIFACT_TYPE:
            raise ValueError(f"Unsupported OOF aggregate artifact type {artifact_type!r} in {path}")

        required = {"probs", "labels", "sample_ids"}
        if schema == OOF_AGGREGATE_SCHEMA:
            required.update(
                {
                    "artifactType",
                    "numClasses",
                    "aggregationMethod",
                    "source_files",
                    "tag_names",
                    "tag_weights",
                }
            )
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"OOF aggregate artifact {path} is missing {', '.join(missing)}")

        probs = data["probs"].astype(np.float32)
        labels = data["labels"].astype(np.int32)
        sample_ids = data["sample_ids"].astype(np.int64)
        if schema == OOF_AGGREGATE_SCHEMA:
            num_classes = _read_scalar_int(data, "numClasses")
            method = _read_scalar_string(data, "aggregationMethod", "")
            source_files = _read_string_array(data, "source_files")
            tag_names = _read_string_array(data, "tag_names")
            weight_values = _read_float_array(data, "tag_weights")
            if num_classes != NUM_CLASSES:
                raise ValueError(f"OOF aggregate artifact {path} has incompatible numClasses")
            if len(tag_names) != len(weight_values):
                raise ValueError(f"OOF aggregate artifact {path} has mismatched tag weights")
            tag_weights = dict(zip(tag_names, weight_values, strict=True))
        else:
            num_classes = int(probs.shape[1]) if probs.ndim == 2 else None
            method = None
            source_files = []
            tag_weights = {}

    _validate_arrays(probs, labels, sample_ids)
    if schema == OOF_AGGREGATE_SCHEMA:
        _validate_metadata(method, source_files, tag_weights)
    return {
        "schemaVersion": schema,
        "artifactType": artifact_type,
        "numClasses": num_classes,
        "aggregationMethod": method,
        "probs": probs,
        "labels": labels,
        "sample_ids": sample_ids,
        "source_files": source_files,
        "tag_weights": tag_weights,
    }


def _read_scalar_string(data: Any, key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    if value.shape != () or not isinstance(value.item(), str):
        raise ValueError(f"OOF aggregate metadata {key} must be a scalar string")
    return value.item()


def _read_scalar_int(data: Any, key: str) -> int:
    value = data[key]
    if value.shape != () or isinstance(value.item(), (bool, np.bool_)):
        raise ValueError(f"OOF aggregate metadata {key} must be a scalar integer")
    scalar = value.item()
    if not isinstance(scalar, (int, np.integer)):
        raise ValueError(f"OOF aggregate metadata {key} must be a scalar integer")
    return int(scalar)


def _read_string_array(data: Any, key: str) -> list[str]:
    value = data[key]
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"OOF aggregate metadata {key} must be a string array")
    return [str(item) for item in value.tolist()]


def _read_float_array(data: Any, key: str) -> list[float]:
    value = data[key]
    if value.ndim != 1 or value.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"OOF aggregate metadata {key} must be a numeric array")
    return [float(item) for item in value.tolist()]


def _validate_arrays(probs: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray) -> None:
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise ValueError(f"OOF aggregate probabilities must have shape (rows, {NUM_CLASSES})")
    rows = probs.shape[0]
    if labels.shape != probs.shape:
        raise ValueError("OOF aggregate labels must have the same shape as probabilities")
    if sample_ids.shape != (rows,):
        raise ValueError("OOF aggregate sample_ids must match the probability row count")
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("OOF aggregate probabilities must be finite values in [0, 1]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("OOF aggregate labels must contain only 0 or 1")
    if np.unique(sample_ids).size != rows or np.any(sample_ids < 0):
        raise ValueError("OOF aggregate sample_ids must be unique non-negative values")


def _validate_metadata(method: str, source_files: list[str], tag_weights: dict[str, float]) -> None:
    if method not in SUPPORTED_AGGREGATION_METHODS:
        raise ValueError(f"Unsupported OOF aggregate method {method!r}")
    if not source_files or len(set(source_files)) != len(source_files):
        raise ValueError("OOF aggregate source_files must contain unique filenames")
    if any(not name or Path(name).name != name or name in {".", ".."} for name in source_files):
        raise ValueError("OOF aggregate source_files must contain filenames only")
    if method == "mean" and tag_weights:
        raise ValueError("Mean OOF aggregates must not include tag weights")
    if method == "tag-weighted":
        if not tag_weights or any(not isinstance(tag, str) or not tag for tag in tag_weights):
            raise ValueError("Tag-weighted OOF aggregates require named tag weights")
        weights = np.asarray(list(tag_weights.values()), dtype=np.float64)
        if not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError("OOF aggregate tag weights must be positive and sum to 1")
