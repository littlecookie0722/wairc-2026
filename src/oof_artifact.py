"""Versioned out-of-fold prediction artifacts and compatibility loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import NUM_CLASSES


OOF_SCHEMA = "oof-v1"
OOF_ARTIFACT_TYPE = "oof-predictions"
LEGACY_OOF_SCHEMA = "legacy-unversioned"
NUM_OOF_CLASSES = NUM_CLASSES


def write_oof_artifact(
    path: Path,
    *,
    probs: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    fold: int,
    sample_ids: np.ndarray,
    metric: float,
) -> None:
    """Write an OOF file with explicit metadata while preserving array names."""
    _validate_arrays(probs, labels, indices, sample_ids, fold)
    if not np.isfinite(metric):
        raise ValueError("OOF metric must be finite")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schemaVersion=np.asarray(OOF_SCHEMA),
        artifactType=np.asarray(OOF_ARTIFACT_TYPE),
        probs=probs.astype(np.float16),
        labels=labels.astype(np.int8),
        indices=indices.astype(np.int32),
        fold=np.asarray(fold, dtype=np.int32),
        sample_ids=sample_ids.astype(np.int64),
        metrics=np.asarray([metric], dtype=np.float32),
    )


def load_oof_artifact(path: Path) -> dict[str, Any]:
    """Load a v1 or legacy OOF file and validate its row-level contract."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        schema = _read_scalar_string(data, "schemaVersion", LEGACY_OOF_SCHEMA)
        artifact_type = _read_scalar_string(data, "artifactType", OOF_ARTIFACT_TYPE)
        if schema not in {OOF_SCHEMA, LEGACY_OOF_SCHEMA}:
            raise ValueError(f"Unsupported OOF schema {schema!r} in {path}")
        if schema == OOF_SCHEMA and "artifactType" not in keys:
            raise ValueError(f"OOF artifact {path} is missing artifactType")
        if artifact_type != OOF_ARTIFACT_TYPE:
            raise ValueError(f"Unsupported OOF artifact type {artifact_type!r} in {path}")
        required = {"probs", "labels", "indices"}
        if schema == OOF_SCHEMA:
            required.update({"fold", "metrics"})
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"OOF artifact {path} is missing {', '.join(missing)}")
        probs = data["probs"].astype(np.float32)
        labels = data["labels"].astype(np.int32)
        indices = data["indices"].astype(np.int64)
        fold = _read_scalar_int(data, "fold") if "fold" in data else None
        metrics = data["metrics"].astype(np.float32) if "metrics" in data else np.asarray([], dtype=np.float32)
        sample_ids = data["sample_ids"].astype(np.int64) if "sample_ids" in data else indices.copy()

    _validate_arrays(probs, labels, indices, sample_ids, fold)
    if metrics.size not in {0, 1} or (metrics.size == 1 and not np.isfinite(metrics[0])):
        raise ValueError(f"OOF artifact {path} metrics must contain one finite value")
    return {
        "schemaVersion": schema,
        "artifactType": artifact_type,
        "probs": probs,
        "labels": labels,
        "indices": indices,
        "fold": fold,
        "sample_ids": sample_ids,
        "metrics": metrics,
    }


def _read_scalar_string(data: Any, key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    if value.shape != ():
        raise ValueError(f"OOF metadata {key} must be scalar")
    scalar = value.item()
    if not isinstance(scalar, str):
        raise ValueError(f"OOF metadata {key} must be a string")
    return scalar


def _read_scalar_int(data: Any, key: str) -> int:
    value = data[key]
    if value.shape != () or isinstance(value.item(), (bool, np.bool_)):
        raise ValueError(f"OOF metadata {key} must be a scalar integer")
    scalar = value.item()
    if not isinstance(scalar, (int, np.integer)):
        raise ValueError(f"OOF metadata {key} must be a scalar integer")
    return int(scalar)


def _validate_arrays(
    probs: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    sample_ids: np.ndarray,
    fold: int | None,
) -> None:
    if probs.ndim != 2 or probs.shape[1] != NUM_OOF_CLASSES:
        raise ValueError(f"OOF probabilities must have shape (rows, {NUM_OOF_CLASSES})")
    rows = probs.shape[0]
    if labels.shape != probs.shape:
        raise ValueError("OOF labels must have the same shape as probabilities")
    if indices.shape != (rows,) or sample_ids.shape != (rows,):
        raise ValueError("OOF indices and sample_ids must match the probability row count")
    if fold is not None and (isinstance(fold, (bool, np.bool_)) or not isinstance(fold, (int, np.integer))):
        raise ValueError("OOF fold must be an integer")
    if fold is not None and fold < 0:
        raise ValueError("OOF fold must be non-negative")
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("OOF probabilities must be finite values in [0, 1]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("OOF labels must contain only 0 or 1")
    if np.unique(indices).size != rows:
        raise ValueError("OOF indices must be unique within one artifact")
    if np.unique(sample_ids).size != rows:
        raise ValueError("OOF sample_ids must be unique within one artifact")
    if np.any(indices < 0) or np.any(sample_ids < 0):
        raise ValueError("OOF indices and sample_ids must be non-negative")
