"""Versioned provenance records for data-backed training runs."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any


RUN_MANIFEST_SCHEMA = "run-manifest-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_provenance() -> dict[str, Any]:
    commit = _run_git("rev-parse", "HEAD")
    dirty_output = _run_git("status", "--porcelain")
    return {
        "commit": commit or "unknown",
        "dirty": dirty_output is None or bool(dirty_output),
    }


def _installed_package_version(distribution_name: str) -> str:
    try:
        return distribution_version(distribution_name)
    except PackageNotFoundError:
        return "unavailable"


def _runtime_provenance(device: str) -> dict[str, str]:
    torch_version = _installed_package_version("torch")
    cuda_version = "unavailable"
    device_name = "unavailable"
    try:
        import torch

        cuda_version = getattr(torch.version, "cuda", None) or "unavailable"
        if device.startswith("cuda") and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.system().lower(),
        "numpy": _installed_package_version("numpy"),
        "scipy": _installed_package_version("scipy"),
        "scikitLearn": _installed_package_version("scikit-learn"),
        "torch": torch_version,
        "torchvision": _installed_package_version("torchvision"),
        "cuda": str(cuda_version),
        "device": device,
        "deviceName": device_name,
    }


def make_run_id(prefix: str, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{seed}-{uuid.uuid4().hex[:8]}"


def create_run_manifest(
    *,
    run_id: str,
    command: list[str],
    args: dict[str, Any],
    data: dict[str, Any],
    transform: dict[str, Any],
    model: dict[str, Any],
    training: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Create a serializable manifest without exposing local absolute paths."""
    return {
        "schemaVersion": RUN_MANIFEST_SCHEMA,
        "runId": run_id,
        "status": "running",
        "createdAt": _utc_now(),
        "git": _git_provenance(),
        "runtime": _runtime_provenance(device),
        "data": data,
        "transform": transform,
        "model": model,
        "training": training,
        "command": _sanitize_command(command, args),
        "arguments": _public_arguments(args),
    }


def _sanitize_command(command: list[str], args: dict[str, Any]) -> list[str]:
    path_flags = {"--train-root": "<train_root>", "--save-dir": "<save_dir>", "--cache-dir": "<cache_dir>"}
    sanitized: list[str] = []
    replacement: str | None = None
    for token in command:
        value = "python" if token == sys.executable else token
        if replacement is not None:
            value = replacement
            replacement = None
        elif value in path_flags:
            replacement = path_flags[value]
        elif any(value.startswith(f"{flag}=") for flag in path_flags):
            flag = next(flag for flag in path_flags if value.startswith(f"{flag}="))
            value = f"{flag}={path_flags[flag]}"
        else:
            value = re.sub(r"(?:[A-Za-z]:[\\/]|/(?:Users|home|tmp)/)[^\s]*", "<local-path>", value)
        sanitized.append(value)
    return sanitized


def _public_arguments(args: dict[str, Any]) -> dict[str, Any]:
    """Keep useful CLI context while removing machine-specific paths."""
    output: dict[str, Any] = {}
    for key, value in args.items():
        if key in {"train_root", "save_dir", "cache_dir"}:
            continue
        if isinstance(value, Path):
            output[key] = value.name
        elif isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value
        else:
            output[key] = str(value)
    return output


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finalize_run_manifest(path: Path, status: str, **updates: Any) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError("status must be 'completed' or 'failed'")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["finishedAt"] = _utc_now()
    manifest.update(updates)
    write_run_manifest(path, manifest)
