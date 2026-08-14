import numpy as np
import torch

from src.spectrogram import (
    DroneClassifier,
    apply_inference_rule,
    apply_thresholds,
    enforce_count_constraint,
    iq_to_spectrogram,
)


def test_iq_to_spectrogram_returns_finite_target_shape():
    rng = np.random.default_rng(2026)
    raw = rng.integers(-100, 100, size=2048, dtype=np.int16)

    result = iq_to_spectrogram(
        raw,
        sample_rate=125_000_000,
        n_fft=64,
        hop=16,
        target_freq=33,
        target_time=48,
    )

    assert result is not None
    assert result.shape == (33, 48)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_iq_to_spectrogram_rejects_too_short_input():
    raw = np.zeros(64, dtype=np.int16)
    assert iq_to_spectrogram(raw, sample_rate=1.0, n_fft=64, hop=16) is None


def test_legacy_iq_to_spectrogram_preserves_zero_rate_fallback():
    raw = np.arange(128, dtype=np.int16)

    fallback = iq_to_spectrogram(raw, sample_rate=0.0, n_fft=32, hop=8)
    explicit = iq_to_spectrogram(raw, sample_rate=125_000_000.0, n_fft=32, hop=8)

    assert fallback is not None
    assert explicit is not None
    np.testing.assert_array_equal(fallback, explicit)


def test_classifier_forward_is_cpu_compatible_without_weights():
    model = DroneClassifier(num_classes=9, arch="resnet18", pretrained=False, dropout=0.0)
    model.eval()
    inputs = torch.randn(2, 3, 65, 64)

    with torch.inference_mode():
        logits = model(inputs)

    assert logits.shape == (2, 9)
    assert torch.isfinite(logits).all()


def test_inference_rules_keep_predictions_in_expected_range():
    probs = np.asarray([[0.9, 0.8, 0.1], [0.2, 0.3, 0.1]], dtype=np.float32)
    thresholded = apply_thresholds(probs, [0.5, 0.5, 0.5])
    constrained = enforce_count_constraint(thresholded, probs, max_labels=2)
    top2 = apply_inference_rule(
        probs,
        {"method": "top2_second_threshold", "second_threshold": 0.75},
    )

    assert thresholded.tolist() == [[1, 1, 0], [0, 0, 0]]
    assert constrained.tolist() == [[1, 1, 0], [0, 1, 0]]
    assert top2.tolist() == [[1, 1, 0], [0, 1, 0]]
