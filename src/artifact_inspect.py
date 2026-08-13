"""Inspect and validate supported research artifacts without exposing paths."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .cache_artifact import CACHE_ARTIFACT_TYPE, CACHE_SCHEMA, LEGACY_CACHE_SCHEMA, load_cache_artifact
from .checkpoint import load_checkpoint
from .oof_artifact import load_oof_artifact
from .rule_artifact import LEGACY_RULE_SCHEMA, RULE_ARTIFACT_TYPE, RULE_SCHEMA, load_rule_artifact


CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth"}


def inspect_artifact(path: Path) -> dict[str, Any]:
    """Return a JSON-serializable, path-redacted validation summary."""
    path = Path(path)
    result: dict[str, Any] = {
        "valid": False,
        "artifactType": "unknown",
        "schemaVersion": None,
        "fileName": path.name,
    }
    if not path.exists():
        result["error"] = "File does not exist"
        return result
    if not path.is_file():
        result["error"] = "Artifact path is not a file"
        return result

    try:
        kind = _detect_kind(path)
        if kind == "checkpoint":
            result.update(_inspect_checkpoint(path))
        elif kind == "oof":
            result.update(_inspect_oof(path))
        elif kind == "cache":
            result.update(_inspect_cache(path))
        elif kind == "rule":
            result.update(_inspect_rule(path))
        else:
            result["error"] = "Unsupported artifact format"
    except Exception as error:  # Loaders expose several format-specific exception types.
        result["error"] = _safe_error(error, path)
    return result


def _detect_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in CHECKPOINT_SUFFIXES:
        return "checkpoint"
    if suffix == ".json":
        return "rule"
    if suffix != ".npz":
        return "unknown"

    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
    except Exception as error:
        raise ValueError("Unable to read NPZ artifact") from error
    if {"probs", "labels", "indices"}.issubset(keys):
        return "oof"
    if {"x", "node_mask"}.issubset(keys):
        return "cache"
    return "unknown"


def _inspect_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    transform = {
        name: int(checkpoint[name])
        for name in ("n_fft", "hop", "target_freq", "target_time", "cache_time")
        if name in checkpoint
    }
    details: dict[str, Any] = {
        "architecture": str(checkpoint["arch"]),
        "numClasses": _optional_int(checkpoint.get("num_classes")),
        "stftProfile": checkpoint.get("stftProfile"),
        "transform": transform,
        "hasStateDict": "model_state_dict" in checkpoint,
    }
    for source, target in (("fold", "fold"), ("tag", "tag")):
        if source in checkpoint:
            details[target] = checkpoint[source]
    return _valid_summary(
        artifact_type=str(checkpoint["artifactType"]),
        schema=str(checkpoint["schemaVersion"]),
        details=details,
    )


def _inspect_oof(path: Path) -> dict[str, Any]:
    artifact = load_oof_artifact(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
    metrics = artifact["metrics"]
    details: dict[str, Any] = {
        "rows": int(artifact["probs"].shape[0]),
        "classes": int(artifact["probs"].shape[1]),
        "fold": artifact["fold"],
        "sampleIdsPresent": "sample_ids" in keys,
        "metrics": [float(value) for value in metrics.tolist()],
    }
    return _valid_summary(
        artifact_type=str(artifact["artifactType"]),
        schema=str(artifact["schemaVersion"]),
        details=details,
    )


def _inspect_cache(path: Path) -> dict[str, Any]:
    metadata = _read_cache_metadata(path)
    if metadata["artifactType"] != CACHE_ARTIFACT_TYPE:
        raise ValueError("Unsupported cache artifact type")
    loaded = load_cache_artifact(
        path,
        n_fft=metadata["n_fft"],
        hop=metadata["hop"],
        target_freq=metadata["target_freq"],
        cache_time=metadata["cache_time"],
    )
    if loaded is None:
        raise ValueError("Cache artifact failed validation")
    x, node_mask = loaded
    details = {
        "shape": [int(value) for value in x.shape],
        "nodeCount": int(x.shape[0]),
        "nodeMask": [int(value) for value in node_mask.tolist()],
        "stftProfile": metadata["stftProfile"],
        "transform": {
            "nFft": metadata["n_fft"] if metadata["schemaVersion"] == CACHE_SCHEMA else None,
            "hop": metadata["hop"] if metadata["schemaVersion"] == CACHE_SCHEMA else None,
            "targetFreq": metadata["target_freq"],
            "cacheTime": metadata["cache_time"],
        },
    }
    return _valid_summary(
        artifact_type=metadata["artifactType"],
        schema=metadata["schemaVersion"],
        details=details,
    )


def _inspect_rule(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = load_rule_artifact(path)
    schema = payload.get("schemaVersion", LEGACY_RULE_SCHEMA)
    artifact_type = payload.get("artifactType", RULE_ARTIFACT_TYPE)
    thresholds = selected.get("thresholds")
    details: dict[str, Any] = {
        "selectedMethod": selected.get("method"),
        "thresholdCount": len(thresholds) if isinstance(thresholds, list) else 0,
        "candidateCount": len(payload.get("candidates", [])) if isinstance(payload.get("candidates", []), list) else 0,
    }
    if schema == RULE_SCHEMA:
        details["numClasses"] = payload.get("numClasses")
    elif isinstance(thresholds, list):
        details["numClasses"] = len(thresholds)
    return _valid_summary(artifact_type=str(artifact_type), schema=str(schema), details=details)


def _read_cache_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        schema = _scalar_text(data, "schemaVersion", LEGACY_CACHE_SCHEMA)
        artifact_type = _scalar_text(data, "artifactType", CACHE_ARTIFACT_TYPE)
        stft_profile = _scalar_text(data, "stftProfile", "") or None
        if schema == CACHE_SCHEMA:
            n_fft = _positive_scalar_int(data, "n_fft")
            hop = _positive_scalar_int(data, "hop")
            target_freq = _positive_scalar_int(data, "target_freq")
            cache_time = _positive_scalar_int(data, "cache_time")
        else:
            x = np.asarray(data["x"])
            if x.ndim != 3:
                raise ValueError("Legacy cache x must be a three-dimensional array")
            n_fft = 1
            hop = 1
            target_freq = int(x.shape[1])
            cache_time = int(x.shape[2])
    return {
        "schemaVersion": schema,
        "artifactType": artifact_type,
        "stftProfile": stft_profile,
        "n_fft": n_fft,
        "hop": hop,
        "target_freq": target_freq,
        "cache_time": cache_time,
    }


def _scalar_text(data: Any, key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    if value.shape != () or not isinstance(value.item(), str):
        raise ValueError(f"Cache metadata {key} must be a scalar string")
    return value.item()


def _positive_scalar_int(data: Any, key: str) -> int:
    if key not in data:
        raise ValueError(f"Cache metadata is missing {key}")
    value = data[key]
    if value.shape != () or isinstance(value.item(), (bool, np.bool_)):
        raise ValueError(f"Cache metadata {key} must be a positive integer")
    scalar = value.item()
    if not isinstance(scalar, (int, np.integer)) or scalar <= 0:
        raise ValueError(f"Cache metadata {key} must be a positive integer")
    return int(scalar)


def _valid_summary(*, artifact_type: str, schema: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": True,
        "artifactType": artifact_type,
        "schemaVersion": schema,
        "details": details,
    }


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _safe_error(error: Exception, path: Path) -> str:
    message = " ".join(str(error).split())
    for candidate in (str(path.resolve()), str(path), path.as_posix()):
        message = message.replace(candidate, path.name)
    if re.search(r"(?i)(?:[a-z]:[\\/]|\\\\|/)", message):
        message = "artifact failed validation"
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
