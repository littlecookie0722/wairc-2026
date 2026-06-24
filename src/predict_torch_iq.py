import argparse
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from .config import CACHE_DIR, SUBMISSION_PATH, TEST_ROOT
from .data import load_index
from .submission import write_submission
from .torch_iq import (
    CachedIQDataset,
    IQCNN,
    IQDataset,
    build_iq_tensor_cache,
    iq_cache_base_path,
    predictions_from_probabilities,
)
from .train_torch_iq import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_PATH = SUBMISSION_PATH.parent / "submission_iq_cnn.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict public test labels with a trained IQCNN model.")
    parser.add_argument("--test-root", type=Path, default=TEST_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision inference.")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true", help="Read source .npz files directly for every batch.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild the IQ tensor cache before prediction.")
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def require_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. Install a CUDA-enabled PyTorch build "
            "in D:/Develop/Miniconda/envs/deepl, then retry."
        )
    return torch.device(requested)


@torch.no_grad()
def predict_probabilities(
    model: IQCNN,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    probabilities = []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(iq, meta)
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.vstack(probabilities).astype(np.float32)


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model = IQCNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = float(args.threshold if args.threshold is not None else checkpoint["threshold"])
    sequence_pairs = int(checkpoint["sequence_pairs"])
    use_amp = device.type == "cuda" and not args.no_amp

    rows = load_index(args.test_root, has_labels=False)
    if args.max_samples:
        rows = rows[: args.max_samples]
        print(f"Using first {len(rows)} rows because --max-samples was provided.")

    if args.no_cache:
        dataset = IQDataset(args.test_root, rows, sequence_pairs=sequence_pairs, has_labels=False)
    else:
        cache_base_path = iq_cache_base_path(
            name="test_public",
            sequence_pairs=sequence_pairs,
            max_samples=args.max_samples,
            cache_dir=args.cache_dir,
        )
        build_iq_tensor_cache(
            args.test_root,
            rows,
            sequence_pairs=sequence_pairs,
            has_labels=False,
            base_path=cache_base_path,
            dtype=args.cache_dtype,
            force=args.rebuild_cache,
        )
        dataset = CachedIQDataset(cache_base_path, rows=rows, has_labels=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    probabilities = predict_probabilities(model, loader, device, use_amp)
    predictions = predictions_from_probabilities(probabilities, threshold).astype(int).tolist()
    path = write_submission(rows, predictions, args.output_path)
    print(f"Wrote submission: {path}")
    print(f"Used model: {args.model_path}")
    print(f"Used threshold: {threshold:.2f}")


if __name__ == "__main__":
    main()
