"""Versioned single-model validation predictions with legacy loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import NUM_CLASSES


VALIDATION_SCHEMA = "validation-predictions-v1"
VALIDATION_ARTIFACT_TYPE = "validation-predictions"
LEGACY_VALIDATION_SCHEMA = "legacy-unversioned"


def write_validation_artifact(
    path: Path,
    *,
    probs: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    epoch: int,
    metric_name: str,
    metric_value: float,
) -> None:
    """Write validation predictions while preserving historical array names."""
    _validate_arrays(probs, labels, sample_ids)
    _validate_metadata(epoch, metric_name, metric_value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schemaVersion=np.asarray(VALIDATION_SCHEMA),
        artifactType=np.asarray(VALIDATION_ARTIFACT_TYPE),
        numClasses=np.asarray(NUM_CLASSES, dtype=np.int32),
        probs=probs.astype(np.float16),
        labels=labels.astype(np.int8),
        sample_ids=sample_ids.astype(np.int64),
        epoch=np.asarray(epoch, dtype=np.int32),
        metricName=np.asarray(metric_name),
        metricValue=np.asarray(metric_value, dtype=np.float32),
    )


def load_validation_artifact(path: Path) -> dict[str, Any]:
    """Load and validate a versioned or historical validation artifact."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        schema = _read_scalar_string(data, "schemaVersion", LEGACY_VALIDATION_SCHEMA)
        artifact_type = _read_scalar_string(data, "artifactType", VALIDATION_ARTIFACT_TYPE)
        if schema not in {VALIDATION_SCHEMA, LEGACY_VALIDATION_SCHEMA}:
            raise ValueError(f"Unsupported validation schema {schema!r} in {path}")
        if artifact_type != VALIDATION_ARTIFACT_TYPE:
            raise ValueError(f"Unsupported validation artifact type {artifact_type!r} in {path}")

        required = {"probs", "labels"}
        if schema == VALIDATION_SCHEMA:
            required.update(
                {"artifactType", "numClasses", "sample_ids", "epoch", "metricName", "metricValue"}
            )
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"Validation artifact {path} is missing {', '.join(missing)}")

        probs = data["probs"].astype(np.float32)
        labels = data["labels"].astype(np.int32)
        sample_ids = data["sample_ids"].astype(np.int64) if "sample_ids" in keys else None
        if schema == VALIDATION_SCHEMA:
            num_classes = _read_scalar_int(data, "numClasses")
            epoch = _read_scalar_int(data, "epoch")
            metric_name = _read_scalar_string(data, "metricName", "")
            metric_value = _read_scalar_float(data, "metricValue")
            if num_classes != NUM_CLASSES:
                raise ValueError(f"Validation artifact {path} has incompatible numClasses")
        else:
            num_classes = int(probs.shape[1]) if probs.ndim == 2 else None
            epoch = None
            metric_name = None
            metric_value = None

    _validate_arrays(probs, labels, sample_ids)
    if schema == VALIDATION_SCHEMA:
        _validate_metadata(epoch, metric_name, metric_value)
    return {
        "schemaVersion": schema,
        "artifactType": artifact_type,
        "numClasses": num_classes,
        "probs": probs,
        "labels": labels,
        "sample_ids": sample_ids,
        "epoch": epoch,
        "metricName": metric_name,
        "metricValue": metric_value,
    }


def _read_scalar_string(data: Any, key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    if value.shape != () or not isinstance(value.item(), str):
        raise ValueError(f"Validation metadata {key} must be a scalar string")
    return value.item()


def _read_scalar_int(data: Any, key: str) -> int:
    value = data[key]
    if value.shape != () or isinstance(value.item(), (bool, np.bool_)):
        raise ValueError(f"Validation metadata {key} must be a scalar integer")
    scalar = value.item()
    if not isinstance(scalar, (int, np.integer)):
        raise ValueError(f"Validation metadata {key} must be a scalar integer")
    return int(scalar)


def _read_scalar_float(data: Any, key: str) -> float:
    value = data[key]
    if value.shape != () or isinstance(value.item(), (bool, np.bool_)):
        raise ValueError(f"Validation metadata {key} must be a scalar number")
    scalar = value.item()
    if not isinstance(scalar, (int, float, np.integer, np.floating)):
        raise ValueError(f"Validation metadata {key} must be a scalar number")
    return float(scalar)


def _validate_arrays(
    probs: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray | None,
) -> None:
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise ValueError(f"Validation probabilities must have shape (rows, {NUM_CLASSES})")
    rows = probs.shape[0]
    if labels.shape != probs.shape:
        raise ValueError("Validation labels must have the same shape as probabilities")
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Validation probabilities must be finite values in [0, 1]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Validation labels must contain only 0 or 1")
    if sample_ids is not None:
        if sample_ids.shape != (rows,):
            raise ValueError("Validation sample_ids must match the probability row count")
        if np.unique(sample_ids).size != rows or np.any(sample_ids < 0):
            raise ValueError("Validation sample_ids must be unique non-negative values")


def _validate_metadata(epoch: int, metric_name: str, metric_value: float) -> None:
    if isinstance(epoch, (bool, np.bool_)) or not isinstance(epoch, (int, np.integer)) or epoch <= 0:
        raise ValueError("Validation epoch must be a positive integer")
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise ValueError("Validation metricName must be a non-empty string")
    if isinstance(metric_value, (bool, np.bool_)) or not isinstance(
        metric_value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("Validation metricValue must be numeric")
    if not np.isfinite(metric_value):
        raise ValueError("Validation metricValue must be finite")
