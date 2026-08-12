import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spectrogram import DroneClassifier, apply_inference_rule, iq_to_spectrogram  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(2026)
    raw = rng.integers(-100, 100, size=4096, dtype=np.int16)
    spectrogram = iq_to_spectrogram(
        raw,
        sample_rate=125_000_000,
        n_fft=128,
        hop=32,
        target_freq=65,
        target_time=64,
    )
    if spectrogram is None:
        raise RuntimeError("Synthetic IQ did not produce a spectrogram")

    inputs = torch.from_numpy(np.stack([spectrogram] * 3, axis=0)[None])
    model = DroneClassifier(num_classes=9, arch="resnet18", pretrained=False, dropout=0.0)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(inputs)).cpu().numpy()

    predictions = apply_inference_rule(
        probabilities,
        {"method": "per_class_thresholds", "thresholds": [0.5] * 9},
    )
    if probabilities.shape != (1, 9) or predictions.shape != (1, 9):
        raise AssertionError("Unexpected smoke-test output shape")
    if not np.isfinite(probabilities).all():
        raise AssertionError("Smoke-test probabilities contain NaN or Inf")
    if not np.isin(predictions, [0, 1]).all():
        raise AssertionError("Smoke-test predictions are not binary")

    print("smoke test passed: synthetic IQ -> STFT -> CPU model -> multi-hot rule")


if __name__ == "__main__":
    main()
