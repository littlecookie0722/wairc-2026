"""Run a small data-free CPU compatibility probe across supported runtimes."""

from __future__ import annotations

import json
import platform

import numpy as np
import torch

from wairc_rf import STFTConfig, complex_iq_to_spectrogram, iq_to_spectrogram

from .spectrogram import DroneClassifier


def run_cpu_compatibility() -> dict[str, object]:
    """Check public transforms and a randomly initialized model on CPU."""

    rng = np.random.default_rng(2026)
    raw = rng.integers(-100, 100, size=4096, dtype=np.int16)
    complex_iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    config = STFTConfig(n_fft=128, hop=32, target_freq=65, target_time=64)

    interleaved = iq_to_spectrogram(raw, sample_rate=125_000_000.0, config=config)
    native_complex = complex_iq_to_spectrogram(
        complex_iq,
        sample_rate=125_000_000.0,
        config=config,
    )
    if interleaved is None or native_complex is None:
        raise AssertionError("CPU compatibility probe did not produce spectrograms")
    if not np.array_equal(interleaved, native_complex):
        raise AssertionError("Interleaved and native-complex stft-v1 outputs differ")

    torch.manual_seed(2026)
    device = torch.device("cpu")
    model = DroneClassifier(num_classes=9, arch="resnet18", pretrained=False, dropout=0.0)
    model.to(device)
    model.eval()
    inputs = torch.from_numpy(np.stack([interleaved] * 3, axis=0)[None]).to(device)
    with torch.inference_mode():
        logits = model(inputs)
    if logits.shape != (1, 9) or not torch.isfinite(logits).all():
        raise AssertionError("CPU model output is invalid")

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "stft_shape": list(interleaved.shape),
        "logits_shape": list(logits.shape),
    }


def main() -> None:
    result = run_cpu_compatibility()
    print("CPU compatibility passed: " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
