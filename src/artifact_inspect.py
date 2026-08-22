"""Inspect and validate supported research artifacts without exposing paths."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .cache_artifact import CACHE_ARTIFACT_TYPE, CACHE_SCHEMA, LEGACY_CACHE_SCHEMA, load_cache_artifact
from .checkpoint import load_checkpoint
from .oof_aggregate_artifact import (
    OOF_AGGREGATE_ARTIFACT_TYPE,
    OOF_AGGREGATE_SCHEMA,
    load_oof_aggregate_artifact,
)
from .oof_artifact import load_oof_artifact
from .rule_artifact import LEGACY_RULE_SCHEMA, RULE_ARTIFACT_TYPE, RULE_SCHEMA, load_rule_artifact
from .run_manifest import RUN_MANIFEST_SCHEMA
from .validation_artifact import VALIDATION_ARTIFACT_TYPE, VALIDATION_SCHEMA, load_validation_artifact


CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth"}
MANIFEST_NAMES = {"run-manifest.json"}
LINKED_OUTPUT_ROLES = {"checkpoint", "checkpoints", "oof", "rule", "validationProbabilities"}


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
        if kind == "manifest":
            result.update(_inspect_manifest(path))
        elif kind == "checkpoint":
            result.update(_inspect_checkpoint(path))
        elif kind == "oof":
            result.update(_inspect_oof(path))
        elif kind == "oof-aggregate":
            result.update(_inspect_oof_aggregate(path))
        elif kind == "validation":
            result.update(_inspect_validation(path))
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
    if path.name in MANIFEST_NAMES or path.name.startswith("run-manifest_"):
        return "manifest"
    if suffix in CHECKPOINT_SUFFIXES:
        return "checkpoint"
    if suffix == ".json":
        return "rule"
    if suffix != ".npz":
        return "unknown"

    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            schema = _optional_npz_scalar_string(data, "schemaVersion")
            artifact_type = _optional_npz_scalar_string(data, "artifactType")
    except Exception as error:
        raise ValueError("Unable to read NPZ artifact") from error
    if {"probs", "labels", "indices"}.issubset(keys):
        return "oof"
    if (
        schema == OOF_AGGREGATE_SCHEMA
        or artifact_type == OOF_AGGREGATE_ARTIFACT_TYPE
        or "aggregationMethod" in keys
    ):
        return "oof-aggregate"
    if (
        schema == VALIDATION_SCHEMA
        or artifact_type == VALIDATION_ARTIFACT_TYPE
        or {"epoch", "metricName", "metricValue"}.intersection(keys)
    ):
        return "validation"
    if {"probs", "labels", "sample_ids"}.issubset(keys):
        return "oof-aggregate"
    if {"probs", "labels"}.issubset(keys):
        return "validation"
    if {"x", "node_mask"}.issubset(keys):
        return "cache"
    return "unknown"


def _optional_npz_scalar_string(data: Any, key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if value.shape != ():
        return None
    scalar = value.item()
    return scalar if isinstance(scalar, str) else None


def validate_run_manifest(path: Path) -> dict[str, Any]:
    """Validate a run manifest and the public artifacts it links to."""
    path = Path(path)
    result: dict[str, Any] = {
        "valid": False,
        "artifactType": "run-manifest",
        "schemaVersion": None,
        "fileName": path.name,
    }
    if not path.exists():
        result["error"] = "File does not exist"
        return result
    if not path.is_file():
        result["error"] = "Manifest path is not a file"
        return result
    try:
        result.update(_inspect_manifest(path))
    except Exception as error:
        result["error"] = _safe_error(error, path)
    return result


def _inspect_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Run manifest must be a JSON object")
    if payload.get("schemaVersion") != RUN_MANIFEST_SCHEMA:
        raise ValueError("Run manifest must use run-manifest-v1 metadata")
    run_id = payload.get("runId")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Run manifest is missing runId")
    outputs = payload.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("Run manifest outputs must be an object")
    output_paths, output_errors = _manifest_output_paths(outputs)
    loaded: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    errors: list[str] = list(output_errors)
    auxiliary_outputs: dict[str, list[str]] = {}
    for role, names in output_paths.items():
        for name in names:
            artifact_path = _resolve_manifest_output(path, name)
            if artifact_path is None:
                errors.append(f"{role} references an unsafe path")
                continue
            if not artifact_path.exists():
                errors.append(f"Missing {role} artifact {artifact_path.name}")
                continue
            if role not in LINKED_OUTPUT_ROLES:
                auxiliary_outputs.setdefault(role, []).append(artifact_path.name)
                continue
            summary = inspect_artifact(artifact_path)
            if not summary["valid"]:
                errors.append(f"Invalid {role} artifact {artifact_path.name}")
                continue
            expected_type = _expected_artifact_type(role)
            if summary.get("artifactType") != expected_type:
                errors.append(f"Unexpected {role} artifact type for {artifact_path.name}")
                continue
            loaded.setdefault(role, []).append((artifact_path, summary))

    _validate_manifest_linkage(payload, output_paths, loaded, errors)
    indexed_count = _validate_artifact_index(payload, path, output_paths, loaded, errors)
    details = {
        "runId": run_id,
        "status": payload.get("status"),
        "outputCount": sum(len(names) for names in output_paths.values()),
        "validatedArtifactCount": sum(len(entries) for entries in loaded.values()),
        "artifactIndexPresent": "artifactIndex" in payload,
        "indexedArtifactCount": indexed_count,
        "outputs": {role: [path.name for path, _ in entries] for role, entries in loaded.items()},
        "auxiliaryOutputs": auxiliary_outputs,
        "errorCount": len(errors),
    }
    if errors:
        return {
            "valid": False,
            "artifactType": "run-manifest",
            "schemaVersion": RUN_MANIFEST_SCHEMA,
            "details": details,
            "errors": errors,
        }
    return _valid_summary(
        artifact_type="run-manifest",
        schema=RUN_MANIFEST_SCHEMA,
        details=details,
    )


def _manifest_output_paths(outputs: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    output_paths: dict[str, list[str]] = {}
    errors: list[str] = []
    for role, value in outputs.items():
        if role == "folds" and isinstance(value, list) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            continue
        if isinstance(value, str):
            output_paths[role] = [value]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            output_paths[role] = list(value)
        else:
            errors.append(f"{role} outputs must be a string or list of strings")
    return output_paths, errors


def _expected_artifact_type(role: str) -> str:
    if role in {"checkpoint", "checkpoints"}:
        return "model-checkpoint"
    if role == "oof":
        return "oof-predictions"
    if role == "rule":
        return RULE_ARTIFACT_TYPE
    if role == "validationProbabilities":
        return VALIDATION_ARTIFACT_TYPE
    raise ValueError(f"Unsupported linked output role {role}")


def _resolve_manifest_output(manifest_path: Path, name: str) -> Path | None:
    candidate = Path(name)
    if candidate.is_absolute() or candidate.name != name or name in {".", ".."}:
        return None
    resolved = (manifest_path.parent / candidate).resolve()
    try:
        resolved.relative_to(manifest_path.parent.resolve())
    except ValueError:
        return None
    return resolved


def _validate_manifest_linkage(
    payload: dict[str, Any],
    output_paths: dict[str, list[str]],
    loaded: dict[str, list[tuple[Path, dict[str, Any]]]],
    errors: list[str],
) -> None:
    expected_classes = _manifest_num_classes(payload)
    expected_transform = _manifest_transform(payload)
    for role in ("checkpoint", "checkpoints"):
        for _, summary in loaded.get(role, []):
            details = summary.get("details", {})
            if expected_classes is not None and details.get("numClasses") not in {None, expected_classes}:
                errors.append(f"{role} class count does not match manifest")
            _compare_transform(details.get("transform", {}), expected_transform, role, errors)

    for _, summary in loaded.get("oof", []):
        details = summary.get("details", {})
        if expected_classes is not None and details.get("classes") != expected_classes:
            errors.append("oof class count does not match manifest")

    for _, summary in loaded.get("rule", []):
        details = summary.get("details", {})
        if expected_classes is not None and details.get("numClasses") not in {None, expected_classes}:
            errors.append("rule class count does not match manifest")

    for _, summary in loaded.get("validationProbabilities", []):
        details = summary.get("details", {})
        if expected_classes is not None and details.get("classes") != expected_classes:
            errors.append("validationProbabilities class count does not match manifest")
        _validate_validation_linkage(payload, details, errors)

    if output_paths.get("rule") and output_paths.get("oof"):
        for rule_path, _ in loaded.get("rule", []):
            try:
                payload_data = json.loads(rule_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            sources = payload_data.get("source_files")
            if isinstance(sources, list):
                expected_oof = {Path(name).name for name in output_paths["oof"]}
                actual_oof = {Path(str(name)).name for name in sources}
                if actual_oof != expected_oof:
                    errors.append("rule source_files do not match manifest oof outputs")

    manifest_folds = _manifest_folds(payload)
    if manifest_folds is not None:
        artifact_folds = {
            int(summary["details"]["fold"])
            for role in ("oof", "checkpoint", "checkpoints")
            for _, summary in loaded.get(role, [])
            if summary.get("details", {}).get("fold") is not None
        }
        if artifact_folds and artifact_folds != manifest_folds:
            errors.append("artifact folds do not match manifest")


def _manifest_num_classes(payload: dict[str, Any]) -> int | None:
    model = payload.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("numClasses")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _validate_validation_linkage(payload: dict[str, Any], details: dict[str, Any], errors: list[str]) -> None:
    training = payload.get("training")
    expected_metric = training.get("selectMetric") if isinstance(training, dict) else None
    actual_metric = details.get("metricName")
    if isinstance(expected_metric, str) and actual_metric not in {None, expected_metric}:
        errors.append("validationProbabilities metric does not match manifest")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return
    expected_epoch = metrics.get("bestEpoch")
    actual_epoch = details.get("epoch")
    if isinstance(expected_epoch, int) and not isinstance(expected_epoch, bool) and actual_epoch not in {
        None,
        expected_epoch,
    }:
        errors.append("validationProbabilities epoch does not match manifest")
    expected_value = metrics.get("bestMetric")
    actual_value = details.get("metricValue")
    if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool) and actual_value is not None:
        if not math.isclose(float(actual_value), float(expected_value), rel_tol=1e-6, abs_tol=1e-7):
            errors.append("validationProbabilities metric value does not match manifest")


def _manifest_transform(payload: dict[str, Any]) -> dict[str, int | str]:
    transform = payload.get("transform")
    if not isinstance(transform, dict):
        return {}
    aliases = {"nFft": "n_fft", "targetFreq": "target_freq", "targetTime": "target_time", "cacheTime": "cache_time"}
    expected: dict[str, int | str] = {}
    for source, target in aliases.items():
        value = transform.get(source)
        if isinstance(value, int) and not isinstance(value, bool):
            expected[target] = value
    if isinstance(transform.get("hop"), int) and not isinstance(transform["hop"], bool):
        expected["hop"] = transform["hop"]
    if isinstance(transform.get("version"), str):
        expected["stftProfile"] = transform["version"]
    return expected


def _compare_transform(details: dict[str, Any], expected: dict[str, int | str], role: str, errors: list[str]) -> None:
    for key, value in expected.items():
        if details.get(key) not in {None, value}:
            errors.append(f"{role} {key} does not match manifest")


def _manifest_folds(payload: dict[str, Any]) -> set[int] | None:
    training = payload.get("training")
    if not isinstance(training, dict):
        return None
    folds = training.get("folds")
    if not isinstance(folds, list) or not all(isinstance(value, int) and not isinstance(value, bool) for value in folds):
        return None
    return set(folds)


def _validate_artifact_index(
    payload: dict[str, Any],
    manifest_path: Path,
    output_paths: dict[str, list[str]],
    loaded: dict[str, list[tuple[Path, dict[str, Any]]]],
    errors: list[str],
) -> int:
    from .artifact_index import (
        ARTIFACT_INDEX_ALGORITHM,
        ARTIFACT_INDEX_V1_SCHEMA,
        ARTIFACT_INDEX_V2_SCHEMA,
        INDEXED_OUTPUT_ROLES_V2,
        file_sha256,
        indexed_output_roles,
    )

    index = payload.get("artifactIndex")
    if index is None:
        return 0
    if not isinstance(index, dict):
        errors.append("artifactIndex must be an object")
        return 0
    schema_version = index.get("schemaVersion")
    try:
        indexed_roles = indexed_output_roles(schema_version)
    except ValueError:
        errors.append(
            f"artifactIndex must use {ARTIFACT_INDEX_V1_SCHEMA} or {ARTIFACT_INDEX_V2_SCHEMA} metadata"
        )
        indexed_roles = INDEXED_OUTPUT_ROLES_V2
    if index.get("runId") != payload.get("runId"):
        errors.append("artifactIndex runId does not match manifest")
    if index.get("algorithm") != ARTIFACT_INDEX_ALGORITHM:
        errors.append("artifactIndex algorithm must be sha256")
    entries = index.get("artifacts")
    if not isinstance(entries, list):
        errors.append("artifactIndex artifacts must be a list")
        return 0

    expected = {
        (role, name)
        for role, names in output_paths.items()
        if role in indexed_roles
        for name in names
    }
    loaded_by_key = {
        (role, path.name): (path, summary)
        for role, artifacts in loaded.items()
        for path, summary in artifacts
    }
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("artifactIndex entries must be objects")
            continue
        role = entry.get("role")
        file_name = entry.get("fileName")
        if not isinstance(role, str) or role not in indexed_roles:
            errors.append("artifactIndex entry has an unsupported role")
            continue
        if not isinstance(file_name, str) or _resolve_manifest_output(manifest_path, file_name) is None:
            errors.append("artifactIndex entry has an unsafe filename")
            continue
        key = (role, file_name)
        if key in seen:
            errors.append(f"artifactIndex contains duplicate entry {file_name}")
            continue
        seen.add(key)
        loaded_entry = loaded_by_key.get(key)
        if loaded_entry is None:
            continue
        path, summary = loaded_entry
        if entry.get("artifactType") != summary.get("artifactType"):
            errors.append(f"artifactIndex artifact type mismatch for {file_name}")
        if entry.get("schemaVersion") != summary.get("schemaVersion"):
            errors.append(f"artifactIndex schema mismatch for {file_name}")
        size_bytes = entry.get("sizeBytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes != path.stat().st_size:
            errors.append(f"artifactIndex size mismatch for {file_name}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append(f"artifactIndex digest is invalid for {file_name}")
        elif digest != file_sha256(path):
            errors.append(f"artifactIndex digest mismatch for {file_name}")
        details = summary.get("details", {})
        expected_fold = details.get("fold")
        indexed_fold = entry.get("fold")
        if expected_fold is None:
            fold_matches = "fold" not in entry
        else:
            fold_matches = (
                isinstance(indexed_fold, int)
                and not isinstance(indexed_fold, bool)
                and indexed_fold == expected_fold
            )
        if not fold_matches:
            errors.append(f"artifactIndex fold mismatch for {file_name}")

        expected_tag = details.get("tag")
        indexed_tag = entry.get("tag")
        if expected_tag is None:
            tag_matches = "tag" not in entry
        else:
            tag_matches = isinstance(indexed_tag, str) and indexed_tag == expected_tag
        if not tag_matches:
            errors.append(f"artifactIndex tag mismatch for {file_name}")

    if seen != expected:
        errors.append("artifactIndex entries do not match manifest artifact outputs")
    return len(seen)


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


def _inspect_oof_aggregate(path: Path) -> dict[str, Any]:
    artifact = load_oof_aggregate_artifact(path)
    details: dict[str, Any] = {
        "rows": int(artifact["probs"].shape[0]),
        "classes": int(artifact["probs"].shape[1]),
        "aggregationMethod": artifact["aggregationMethod"],
        "sourceFiles": artifact["source_files"],
        "tagWeights": artifact["tag_weights"],
    }
    return _valid_summary(
        artifact_type=str(artifact["artifactType"]),
        schema=str(artifact["schemaVersion"]),
        details=details,
    )


def _inspect_validation(path: Path) -> dict[str, Any]:
    artifact = load_validation_artifact(path)
    details: dict[str, Any] = {
        "rows": int(artifact["probs"].shape[0]),
        "classes": int(artifact["probs"].shape[1]),
        "sampleIdsPresent": artifact["sample_ids"] is not None,
        "epoch": artifact["epoch"],
        "metricName": artifact["metricName"],
        "metricValue": artifact["metricValue"],
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
