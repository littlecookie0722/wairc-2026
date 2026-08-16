"""Lightweight environment diagnostics for the installed WAIRC command."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from importlib import import_module
from typing import Any

from . import __version__


DOCTOR_SCHEMA = "doctor-v1"
_CHECK_NAMES = ("python", "torch", "torchvision", "cuda", "package")


def _ok(**details: Any) -> dict[str, Any]:
    return {"status": "ok", **details}


def _error(error: BaseException, **details: Any) -> dict[str, Any]:
    return {"status": "error", "errorType": type(error).__name__, **details}


def _load_optional_module(name: str) -> tuple[Any | None, dict[str, Any]]:
    try:
        module = import_module(name)
    except Exception as error:  # Import failures are the diagnostic result.
        return None, _error(error)
    version = getattr(module, "__version__", "unknown")
    return module, _ok(version=str(version))


def collect_doctor_report() -> dict[str, Any]:
    """Collect package and runtime facts without touching project data."""

    checks: dict[str, dict[str, Any]] = {
        "python": _ok(version=platform.python_version()),
    }
    torch, checks["torch"] = _load_optional_module("torch")
    _torchvision, checks["torchvision"] = _load_optional_module("torchvision")

    if torch is None:
        checks["cuda"] = _error(RuntimeError("torch is unavailable"), available=None)
    else:
        try:
            checks["cuda"] = _ok(available=bool(torch.cuda.is_available()))
        except Exception as error:  # A broken CUDA runtime is diagnostic output.
            checks["cuda"] = _error(error, available=None)

    checks["package"] = _ok(version=__version__)
    status = "ok" if all(checks[name]["status"] == "ok" for name in _CHECK_NAMES) else "error"
    return {
        "schemaVersion": DOCTOR_SCHEMA,
        "status": status,
        "checks": checks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local WAIRC runtime environment.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report.")
    return parser.parse_args(argv)


def _human_value(name: str, check: dict[str, Any]) -> str:
    if name == "cuda":
        return str(check.get("available"))
    if "version" in check:
        return str(check["version"])
    return str(check.get("errorType", "unknown error"))


def _print_human(report: dict[str, Any]) -> None:
    labels = {
        "python": "Python",
        "torch": "Torch",
        "torchvision": "Torchvision",
        "cuda": "CUDA available",
        "package": "Package version",
    }
    print("WAIRC-2026 environment doctor")
    for name in _CHECK_NAMES:
        check = report["checks"][name]
        state = "PASS" if check["status"] == "ok" else "ERROR"
        print(f"[{state}] {labels[name]}: {_human_value(name, check)}")
    print(f"Result: {report['status'].upper()}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = collect_doctor_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    else:
        _print_human(report)
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
