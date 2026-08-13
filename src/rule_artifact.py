"""Versioned inference-rule artifacts and legacy-compatible loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import NUM_CLASSES


RULE_SCHEMA = "rule-v1"
RULE_ARTIFACT_TYPE = "inference-rule"
LEGACY_RULE_SCHEMA = "legacy-unversioned"
SUPPORTED_RULE_METHODS = {"per_class_thresholds", "top2_second_threshold"}


def make_rule_payload(
    selected: dict[str, Any],
    *,
    candidates: list[dict[str, Any]] | None = None,
    num_classes: int = NUM_CLASSES,
    **metadata: Any,
) -> dict[str, Any]:
    """Wrap an existing search result without changing its selected rule."""
    validate_rule(selected, num_classes=num_classes)
    for candidate in candidates or []:
        validate_rule(candidate, num_classes=num_classes)
    payload: dict[str, Any] = {
        "schemaVersion": RULE_SCHEMA,
        "artifactType": RULE_ARTIFACT_TYPE,
        "numClasses": num_classes,
        "selected": dict(selected),
        "candidates": [dict(candidate) for candidate in (candidates or [])],
    }
    payload.update(metadata)
    return payload


def load_rule_artifact(path: Path, *, num_classes: int = NUM_CLASSES) -> dict[str, Any]:
    """Load a versioned or legacy rule and return the selected rule only."""
    path = Path(path)
    if not path.exists():
        return default_rule(num_classes)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Rule artifact {path} must contain a JSON object")

    schema = payload.get("schemaVersion", LEGACY_RULE_SCHEMA)
    if not isinstance(schema, str) or schema not in {RULE_SCHEMA, LEGACY_RULE_SCHEMA}:
        raise ValueError(f"Unsupported rule schema {schema!r} in {path}")
    if schema == RULE_SCHEMA:
        if payload.get("artifactType") != RULE_ARTIFACT_TYPE:
            raise ValueError(f"Unsupported rule artifact type in {path}")
        if payload.get("numClasses") != num_classes:
            raise ValueError(f"Rule artifact {path} has incompatible numClasses")
        selected = payload.get("selected")
        if not isinstance(selected, dict):
            raise ValueError(f"Rule artifact {path} is missing selected")
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError(f"Rule artifact {path} candidates must be a list")
        for candidate in candidates:
            validate_rule(candidate, num_classes=num_classes)
    elif "selected" in payload:
        selected = payload["selected"]
    elif "thresholds" in payload:
        selected = {
            "method": "per_class_thresholds",
            "thresholds": payload["thresholds"],
            "accuracy": payload.get("per_class_acc"),
        }
    else:
        selected = payload
    validate_rule(selected, num_classes=num_classes)
    return selected


def write_rule_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Write a validated rule payload without embedding local source paths."""
    if not isinstance(payload, dict):
        raise ValueError("Rule artifact payload must be a JSON object")
    if payload.get("schemaVersion") != RULE_SCHEMA or payload.get("artifactType") != RULE_ARTIFACT_TYPE:
        raise ValueError("Rule artifact payload must use rule-v1 inference-rule metadata")
    num_classes = payload.get("numClasses")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("Rule artifact numClasses must be a positive integer")
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("Rule artifact payload is missing selected")
    validate_rule(selected, num_classes=num_classes)
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("Rule artifact candidates must be a list")
    for candidate in candidates:
        validate_rule(candidate, num_classes=num_classes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = dict(payload)
    if "source_files" in sanitized:
        if not isinstance(sanitized["source_files"], list):
            raise ValueError("Rule artifact source_files must be a list")
        sanitized["source_files"] = [Path(str(source)).name for source in sanitized["source_files"]]
    path.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def default_rule(num_classes: int) -> dict[str, Any]:
    return {"method": "per_class_thresholds", "thresholds": [0.5] * num_classes, "accuracy": None}


def validate_rule(rule: Any, *, num_classes: int = NUM_CLASSES) -> None:
    if not isinstance(rule, dict):
        raise ValueError("Inference rule must be a JSON object")
    method = rule.get("method", "per_class_thresholds")
    if method not in SUPPORTED_RULE_METHODS:
        raise ValueError(f"Unknown inference rule method: {method}")
    if method == "per_class_thresholds":
        thresholds = rule.get("thresholds")
        if not isinstance(thresholds, list) or len(thresholds) != num_classes:
            raise ValueError(f"Inference rule thresholds must contain {num_classes} values")
        try:
            values = np.asarray(thresholds, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Inference rule thresholds must be numeric") from error
    else:
        if "second_threshold" not in rule:
            raise ValueError("Inference rule is missing second_threshold")
        try:
            values = np.asarray([rule["second_threshold"]], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Inference rule second_threshold must be numeric") from error
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Inference rule thresholds must be finite values in [0, 1]")
