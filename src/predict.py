import argparse
from pathlib import Path

from .config import MAX_IQ_PAIRS, MODEL_PATH, SUBMISSION_PATH, TEST_ROOT
from .data import load_index
from .features import cache_path, extract_features_for_rows
from .model import load_model
from .submission import signature_predictions_to_multihot, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict public test labels and write submission.")
    parser.add_argument("--test-root", type=Path, default=TEST_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=SUBMISSION_PATH)
    parser.add_argument("--max-pairs", type=int, default=MAX_IQ_PAIRS)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_index(args.test_root, has_labels=False)
    if args.max_samples:
        rows = rows[: args.max_samples]
        print(f"Using first {len(rows)} rows because --max-samples was provided.")

    model = load_model(args.model_path)
    test_cache = cache_path("test_public", args.max_pairs, args.max_samples)
    x_test = extract_features_for_rows(
        args.test_root,
        rows,
        cache_path=test_cache,
        max_pairs=args.max_pairs,
        force=args.force_cache,
    )
    signature_predictions = model.predict(x_test)
    predictions = signature_predictions_to_multihot(signature_predictions)
    path = write_submission(rows, predictions, args.output_path)
    print(f"Wrote submission: {path}")


if __name__ == "__main__":
    main()

