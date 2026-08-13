"""Versioned checkpoint metadata and backward-compatible loading."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import torch


CHECKPOINT_SCHEMA = "checkpoint-v1"
LEGACY_CHECKPOINT_SCHEMA = "legacy-unversioned"
SUPPORTED_CHECKPOINT_SCHEMAS = {CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA}
CHECKPOINT_V1_REQUIRED_FIELDS = {
    "model_state_dict",
    "arch",
    "num_classes",
    "n_fft",
    "hop",
    "target_freq",
    "target_time",
    "cache_time",
    "stftProfile",
}


def make_checkpoint_payload(
    *,
    model_state_dict: Mapping[str, Any],
    arch: str,
    dropout: float,
    num_classes: int,
    n_fft: int,
    hop: int,
    target_freq: int,
    target_time: int,
    cache_time: int,
    epoch: int,
    metrics: Mapping[str, float],
    pretrained: bool,
    fold: int | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Build the stable checkpoint envelope while retaining legacy fields."""
    payload: dict[str, Any] = {
        "schemaVersion": CHECKPOINT_SCHEMA,
        "artifactType": "model-checkpoint",
        "epoch": epoch,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": model_state_dict,
        "arch": arch,
        "pretrained": pretrained,
        "dropout": dropout,
        "num_classes": num_classes,
        "n_fft": n_fft,
        "hop": hop,
        "target_freq": target_freq,
        "target_time": target_time,
        "cache_time": cache_time,
        "stftProfile": "stft-v1",
        "metrics": dict(metrics),
    }
    if fold is not None:
        payload["fold"] = fold
    if tag is not None:
        payload["tag"] = tag
    return payload


def load_checkpoint(path: Any, *, map_location: Any = "cpu") -> dict[str, Any]:
    """Load a v1 or legacy checkpoint and validate its model metadata.

    Checkpoints without ``schemaVersion`` are intentionally accepted because
    all existing training outputs use the original flat dictionary format.
    """
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} must contain a dictionary")

    schema = checkpoint.get("schemaVersion", LEGACY_CHECKPOINT_SCHEMA)
    if not isinstance(schema, str) or schema not in SUPPORTED_CHECKPOINT_SCHEMAS:
        raise ValueError(f"Unsupported checkpoint schema {schema!r} in {path}")
    artifact_type = checkpoint.get("artifactType", "model-checkpoint")
    if artifact_type != "model-checkpoint":
        raise ValueError(f"Unsupported checkpoint artifact type {artifact_type!r} in {path}")
    required_fields = {"model_state_dict", "arch"}
    if schema == CHECKPOINT_SCHEMA:
        required_fields = CHECKPOINT_V1_REQUIRED_FIELDS
    missing = sorted(required_fields.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint {path} is missing {', '.join(missing)}")
    if schema == CHECKPOINT_SCHEMA:
        if checkpoint["stftProfile"] != "stft-v1":
            raise ValueError(f"Unsupported checkpoint STFT profile {checkpoint['stftProfile']!r} in {path}")
        for field in ("num_classes", "n_fft", "hop", "target_freq", "target_time", "cache_time"):
            value = checkpoint[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Checkpoint {path} field {field} must be a positive integer")

    normalized = dict(checkpoint)
    normalized["schemaVersion"] = schema
    normalized["artifactType"] = artifact_type
    return normalized
