"""Run a small data-free CPU compatibility probe across supported runtimes."""

from __future__ import annotations

import json
import platform

import numpy as np
import torch

from wairc_rf import STFTConfig, complex_iq_to_spectrogram, iq_to_spectrogram, set_reproducible_seed

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

    set_reproducible_seed(2026, deterministic=True)
    device = torch.device("cpu")
    model = DroneClassifier(num_classes=9, arch="resnet18", pretrained=False, dropout=0.0)
    model.to(device)
    model.eval()
    inputs = torch.from_numpy(np.stack([interleaved] * 3, axis=0)[None]).to(device)
    with torch.inference_mode():
        logits = model(inputs)
        scripted_model = torch.jit.trace(model, inputs)
        scripted_logits = scripted_model(inputs)
    if logits.shape != (1, 9) or not torch.isfinite(logits).all():
        raise AssertionError("CPU model output is invalid")
    if scripted_logits.shape != logits.shape or not torch.isfinite(scripted_logits).all():
        raise AssertionError("TorchScript CPU model output is invalid")
    max_abs_difference = float(torch.max(torch.abs(logits - scripted_logits)).cpu())
    if not torch.allclose(logits, scripted_logits, rtol=1e-5, atol=1e-5):
        raise AssertionError(
            "Eager and TorchScript CPU model outputs differ "
            f"(max_abs_difference={max_abs_difference})"
        )

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "stft_shape": list(interleaved.shape),
        "logits_shape": list(logits.shape),
        "torchscript_logits_shape": list(scripted_logits.shape),
        "torchscript_max_abs_difference": max_abs_difference,
    }


def main() -> None:
    result = run_cpu_compatibility()
    print("CPU compatibility passed: " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
