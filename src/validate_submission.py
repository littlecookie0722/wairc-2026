import argparse
import ast
from pathlib import Path

from .config import NUM_CLASSES, SUBMISSION_PATH, TEST_ROOT
from .data import load_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a competition submission txt file.")
    parser.add_argument("submission_path", type=Path, nargs="?", default=SUBMISSION_PATH)
    parser.add_argument("--test-root", type=Path, default=TEST_ROOT)
    parser.add_argument("--allow-partial", action="store_true", help="Allow fewer rows for smoke tests.")
    return parser.parse_args()


def validate_submission(path: Path, test_root: Path, allow_partial: bool = False) -> list[str]:
    rows = load_index(test_root, has_labels=False)
    expected_ids = {int(row["sample_id"]) for row in rows}
    expected_count = len(expected_ids)
    errors: list[str] = []

    if not path.exists():
        return [f"Submission file does not exist: {path}"]

    seen_ids: set[int] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    if not allow_partial and len(lines) != expected_count:
        errors.append(f"Expected {expected_count} lines, got {len(lines)}")
    if allow_partial and len(lines) > expected_count:
        errors.append(f"Expected at most {expected_count} lines, got {len(lines)}")

    for line_no, line in enumerate(lines, start=1):
        if ":" not in line:
            errors.append(f"Line {line_no}: missing ':'")
            continue
        left, right = line.split(":", 1)
        try:
            sample_id = int(left.strip())
        except ValueError:
            errors.append(f"Line {line_no}: invalid sample_id {left!r}")
            continue
        if sample_id in seen_ids:
            errors.append(f"Line {line_no}: duplicate sample_id {sample_id}")
        seen_ids.add(sample_id)
        if sample_id not in expected_ids:
            errors.append(f"Line {line_no}: sample_id {sample_id} not found in public test index")

        try:
            values = ast.literal_eval(right.strip())
        except (SyntaxError, ValueError):
            errors.append(f"Line {line_no}: prediction is not a Python-style list")
            continue
        if not isinstance(values, list):
            errors.append(f"Line {line_no}: prediction must be a list")
            continue
        if len(values) != NUM_CLASSES:
            errors.append(f"Line {line_no}: expected {NUM_CLASSES} values, got {len(values)}")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
                errors.append(f"Line {line_no}: prediction values must be integer 0 or 1")
                break

    if not allow_partial:
        missing_ids = expected_ids - seen_ids
        if missing_ids:
            preview = sorted(missing_ids)[:10]
            errors.append(f"Missing {len(missing_ids)} sample IDs, first few: {preview}")
    return errors


def main() -> None:
    args = parse_args()
    errors = validate_submission(args.submission_path, args.test_root, allow_partial=args.allow_partial)
    if errors:
        print("Submission validation failed:")
        for error in errors[:50]:
            print(f"- {error}")
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more errors")
        raise SystemExit(1)
    print(f"Submission validation passed: {args.submission_path}")


if __name__ == "__main__":
    main()
