"""Command-line interface for inspecting and validating research artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .artifact_inspect import inspect_artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or validate a WAIRC-2026 artifact.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("inspect", "validate"):
        subparser = subparsers.add_parser(action, help=f"{action.title()} one artifact.")
        subparser.add_argument("path", type=Path)
        subparser.add_argument("--json", action="store_true", help="Print a machine-readable JSON summary.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = inspect_artifact(args.path)
    if args.json:
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    else:
        _print_human(result)
    if not result["valid"]:
        raise SystemExit(1)


def _print_human(result: dict[str, Any]) -> None:
    state = "VALID" if result["valid"] else "INVALID"
    print(f"{state}: {result['artifactType']} / {result['schemaVersion']} / {result['fileName']}")
    if result.get("details"):
        for key, value in result["details"].items():
            print(f"- {key}: {json.dumps(value, ensure_ascii=True, sort_keys=True)}")
    if result.get("error"):
        print(f"- error: {result['error']}")


if __name__ == "__main__":
    main()
