"""Versioned STFT cache artifacts with legacy-compatible loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA = "cache-v1"
CACHE_ARTIFACT_TYPE = "stft-cache"
LEGACY_CACHE_SCHEMA = "legacy-unversioned"
NODE_COUNT = 3


def write_cache_artifact(
    path: Path,
    *,
    x: np.ndarray,
    node_mask: np.ndarray,
    n_fft: int,
    hop: int,
    target_freq: int,
    cache_time: int,
) -> None:
    """Write cached tensors with explicit transform and shape metadata."""
    for name, value in {"n_fft": n_fft, "hop": hop, "target_freq": target_freq, "cache_time": cache_time}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Cache metadata {name} must be a positive integer")
    _validate_arrays(x, node_mask, target_freq=target_freq, cache_time=cache_time)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schemaVersion=np.asarray(CACHE_SCHEMA),
        artifactType=np.asarray(CACHE_ARTIFACT_TYPE),
        stftProfile=np.asarray("stft-v1"),
        n_fft=np.asarray(n_fft, dtype=np.int32),
        hop=np.asarray(hop, dtype=np.int32),
        target_freq=np.asarray(target_freq, dtype=np.int32),
        cache_time=np.asarray(cache_time, dtype=np.int32),
        node_count=np.asarray(NODE_COUNT, dtype=np.int32),
        x=x.astype(np.float16),
        node_mask=node_mask.astype(np.float32),
    )


def load_cache_artifact(
    path: Path,
    *,
    n_fft: int,
    hop: int,
    target_freq: int,
    cache_time: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a matching cache, returning ``None`` for stale/corrupt caches."""
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (n_fft, hop, target_freq, cache_time)):
        return None
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            schema = _read_scalar_string(data, "schemaVersion", LEGACY_CACHE_SCHEMA)
            artifact_type = _read_scalar_string(data, "artifactType", CACHE_ARTIFACT_TYPE)
            if schema not in {CACHE_SCHEMA, LEGACY_CACHE_SCHEMA}:
                return None
            if schema == CACHE_SCHEMA and "artifactType" not in data.files:
                return None
            if artifact_type != CACHE_ARTIFACT_TYPE:
                return None
            if schema == CACHE_SCHEMA:
                if _read_scalar_string(data, "stftProfile", "") != "stft-v1":
                    return None
                expected = {"n_fft": n_fft, "hop": hop, "target_freq": target_freq, "cache_time": cache_time, "node_count": NODE_COUNT}
                for key, value in expected.items():
                    if _read_scalar_int(data, key) != value:
                        return None
            x = data["x"].astype(np.float32)
            node_mask = data["node_mask"].astype(np.float32)
        _validate_arrays(x, node_mask, target_freq=target_freq, cache_time=cache_time)
        return x, node_mask
    except (EOFError, OSError, ValueError, KeyError, TypeError):
        return None


def _read_scalar_string(data: Any, key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    if value.shape != () or not isinstance(value.item(), str):
        raise ValueError(f"Cache metadata {key} must be a scalar string")
    return value.item()


def _read_scalar_int(data: Any, key: str) -> int:
    value = data[key]
    if value.shape != () or isinstance(value.item(), (bool, np.bool_)):
        raise ValueError(f"Cache metadata {key} must be a scalar integer")
    scalar = value.item()
    if not isinstance(scalar, (int, np.integer)):
        raise ValueError(f"Cache metadata {key} must be a scalar integer")
    return int(scalar)


def _validate_arrays(x: np.ndarray, node_mask: np.ndarray, *, target_freq: int, cache_time: int) -> None:
    if x.shape != (NODE_COUNT, target_freq, cache_time):
        raise ValueError(f"Cache x must have shape ({NODE_COUNT}, {target_freq}, {cache_time})")
    if node_mask.shape != (NODE_COUNT,):
        raise ValueError(f"Cache node_mask must have shape ({NODE_COUNT},)")
    if not np.isfinite(x).all() or not np.isfinite(node_mask).all():
        raise ValueError("Cache arrays must contain finite values")
    if not np.isin(node_mask, [0, 1]).all():
        raise ValueError("Cache node_mask must contain only 0 or 1")
