"""Privacy-safe references for artifacts linked from a run manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

ARTIFACT_INDEX_V1_SCHEMA = "artifact-index-v1"
ARTIFACT_INDEX_V2_SCHEMA = "artifact-index-v2"
ARTIFACT_INDEX_SCHEMA = ARTIFACT_INDEX_V1_SCHEMA
ARTIFACT_INDEX_ALGORITHM = "sha256"
INDEXED_OUTPUT_ROLES = frozenset({"checkpoint", "checkpoints", "oof", "rule"})
INDEXED_OUTPUT_ROLES_V2 = INDEXED_OUTPUT_ROLES | {"validationProbabilities"}
_EXPECTED_ARTIFACT_TYPES = {
    "checkpoint": "model-checkpoint",
    "checkpoints": "model-checkpoint",
    "oof": "oof-predictions",
    "rule": "inference-rule",
    "validationProbabilities": "validation-predictions",
}


def indexed_output_roles(schema_version: str) -> frozenset[str]:
    """Return the exact manifest roles covered by one index schema."""
    if schema_version == ARTIFACT_INDEX_V1_SCHEMA:
        return INDEXED_OUTPUT_ROLES
    if schema_version == ARTIFACT_INDEX_V2_SCHEMA:
        return INDEXED_OUTPUT_ROLES_V2
    raise ValueError(f"Unsupported artifact index schema {schema_version!r}")


def build_artifact_index(
    run_id: str,
    output_dir: Path,
    outputs: dict[str, Any],
    *,
    schema_version: str = ARTIFACT_INDEX_SCHEMA,
) -> dict[str, Any]:
    """Build content-addressed references for supported manifest outputs."""
    from .artifact_inspect import inspect_artifact

    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Artifact index runId must be a non-empty string")
    indexed_roles = indexed_output_roles(schema_version)
    output_dir = Path(output_dir).resolve()
    artifacts: list[dict[str, Any]] = []
    for role in sorted(indexed_roles):
        values = outputs.get(role)
        if values is None:
            continue
        names = [values] if isinstance(values, str) else values
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"Artifact index output {role} must be a filename or list of filenames")
        for name in names:
            path = _resolve_output(output_dir, name)
            if not path.is_file():
                raise FileNotFoundError(f"Artifact index output is missing: {path.name}")
            summary = inspect_artifact(path)
            if not summary["valid"]:
                raise ValueError(f"Artifact index output is invalid: {path.name}")
            if summary["artifactType"] != _EXPECTED_ARTIFACT_TYPES[role]:
                raise ValueError(f"Artifact index output {role} has an unexpected artifact type")
            entry: dict[str, Any] = {
                "role": role,
                "fileName": path.name,
                "artifactType": summary["artifactType"],
                "schemaVersion": summary["schemaVersion"],
                "sizeBytes": path.stat().st_size,
                "sha256": _file_digest(path),
            }
            details = summary.get("details", {})
            fold = details.get("fold")
            if fold is not None:
                if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
                    raise ValueError("Artifact index output has invalid fold metadata")
                entry["fold"] = fold
            tag = details.get("tag")
            if tag is not None:
                if not isinstance(tag, str):
                    raise ValueError("Artifact index output has invalid tag metadata")
                entry["tag"] = tag
            artifacts.append(entry)
    return {
        "schemaVersion": schema_version,
        "runId": run_id,
        "algorithm": ARTIFACT_INDEX_ALGORITHM,
        "artifacts": artifacts,
    }


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file without exposing its path."""
    return _file_digest(Path(path))


def _resolve_output(output_dir: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() or candidate.name != name or name in {".", ".."}:
        raise ValueError("Artifact index output must be a filename in the run directory")
    resolved = (output_dir / candidate).resolve()
    try:
        resolved.relative_to(output_dir)
    except ValueError as error:
        raise ValueError("Artifact index output must stay inside the run directory") from error
    return resolved


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
