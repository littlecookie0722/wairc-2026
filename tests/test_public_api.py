import numpy as np
import pytest

import src.spectrogram as legacy_spectrogram
from src import __version__ as legacy_version
from src.spectrogram import iq_to_spectrogram as legacy_iq_to_spectrogram
from wairc_rf import (
    STFT_V1_PROFILE,
    STFTConfig,
    __version__,
    complex_iq_to_spectrogram,
    iq_to_spectrogram,
    label_to_multihot,
    multihot_to_signature,
    normalize_label_signature,
    parse_label_signature,
)


def test_public_version_and_stft_defaults_are_explicit():
    config = STFTConfig()

    assert __version__ == legacy_version
    assert config.profile == STFT_V1_PROFILE
    assert config.n_fft == 512
    assert config.hop == 128
    assert config.target_freq is None
    assert config.target_time is None


def test_public_stft_v1_is_numerically_identical_to_legacy_transform():
    rng = np.random.default_rng(2026)
    raw = rng.integers(-100, 100, size=2048, dtype=np.int16)
    config = STFTConfig(n_fft=64, hop=16, target_freq=33, target_time=48)

    public = iq_to_spectrogram(raw, sample_rate=125_000_000, config=config)
    legacy = legacy_iq_to_spectrogram(
        raw,
        sample_rate=125_000_000,
        n_fft=64,
        hop=16,
        target_freq=33,
        target_time=48,
    )

    assert public is not None
    assert legacy is not None
    np.testing.assert_array_equal(public, legacy)


def test_public_stft_kernel_is_independent_of_legacy_workflow(monkeypatch):
    raw = np.arange(128, dtype=np.int16)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("public transform should not call the legacy workflow")

    monkeypatch.setattr(legacy_spectrogram, "iq_to_spectrogram", fail_if_called)

    result = iq_to_spectrogram(raw, sample_rate=2_000_000.0, config=STFTConfig(n_fft=32, hop=8))

    assert result is not None
    assert result.dtype == np.float32


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_public_complex_iq_is_exactly_equivalent_to_interleaved_stft_v1(dtype):
    rng = np.random.default_rng(2027)
    values = (rng.normal(size=1024) + 1j * rng.normal(size=1024)).astype(dtype)
    original = values.copy()
    interleaved = np.empty(values.size * 2, dtype=np.float32)
    interleaved[0::2] = values.real
    interleaved[1::2] = values.imag
    config = STFTConfig(n_fft=64, hop=16, target_freq=33, target_time=48)

    expected = iq_to_spectrogram(interleaved, sample_rate=2_000_000.0, config=config)
    actual = complex_iq_to_spectrogram(values, sample_rate=2_000_000.0, config=config)

    assert expected is not None
    assert actual is not None
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, original)


def test_public_stft_v1_matches_frozen_golden_output():
    raw = np.array(
        [
            3,
            -2,
            5,
            1,
            -4,
            6,
            8,
            -3,
            2,
            7,
            -6,
            -1,
            4,
            9,
            -8,
            5,
            1,
            -7,
            6,
            2,
            -3,
            4,
            7,
            -5,
            9,
            0,
            -2,
            8,
            5,
            -6,
            0,
            3,
        ],
        dtype=np.int16,
    )
    expected = np.array(
        [
            [-0.34134042, -0.41174626, -0.48215207, -0.29211906, -0.10208602],
            [0.58148110, 1.09166586, 1.60185063, 1.46377039, 1.32569015],
            [-0.48273170, -0.87106830, -1.25940490, -0.55556285, 0.14827925],
            [-0.80323100, -1.31877267, -1.83431435, -1.23591328, -0.63751233],
        ],
        dtype=np.float32,
    )

    actual = iq_to_spectrogram(
        raw,
        sample_rate=8_000.0,
        config=STFTConfig(n_fft=8, hop=4, target_freq=4, target_time=5),
    )
    complex_actual = complex_iq_to_spectrogram(
        raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32),
        sample_rate=8_000.0,
        config=STFTConfig(n_fft=8, hop=4, target_freq=4, target_time=5),
    )

    assert actual is not None
    assert complex_actual is not None
    assert actual.shape == (4, 5)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(complex_actual, actual)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"profile": "future-profile"}, "Unsupported STFT profile"),
        ({"n_fft": 1}, "n_fft"),
        ({"n_fft": 64, "hop": 65}, "hop"),
        ({"target_freq": 0}, "target_freq"),
        ({"target_time": 0}, "target_time"),
    ],
)
def test_public_stft_config_rejects_invalid_contracts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        STFTConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"n_fft": 64.5}, {"hop": True}, {"target_freq": "257"}])
def test_public_stft_config_rejects_ambiguous_numeric_types(kwargs):
    with pytest.raises(TypeError, match="must be an integer"):
        STFTConfig(**kwargs)


def test_public_stft_rejects_ambiguous_iq_and_sample_rate():
    with pytest.raises(ValueError, match="complete I/Q pairs"):
        iq_to_spectrogram(np.zeros(129, dtype=np.int16), 1.0, STFTConfig(n_fft=64, hop=16))
    with pytest.raises(ValueError, match="finite positive"):
        iq_to_spectrogram(np.zeros(128, dtype=np.int16), 0.0, STFTConfig(n_fft=64, hop=16))
    with pytest.raises(TypeError, match="real number"):
        iq_to_spectrogram(np.zeros(128, dtype=np.int16), True, STFTConfig(n_fft=64, hop=16))


def test_public_complex_iq_validates_shape_dtype_and_short_recordings():
    config = STFTConfig(n_fft=64, hop=16)

    assert complex_iq_to_spectrogram(np.zeros(63, dtype=np.complex64), 1.0, config) is None
    with pytest.raises(ValueError, match="one-dimensional"):
        complex_iq_to_spectrogram(np.zeros((2, 64), dtype=np.complex64), 1.0, config)
    with pytest.raises(TypeError, match="complex numeric"):
        complex_iq_to_spectrogram(np.zeros(64, dtype=np.float32), 1.0, config)
    with pytest.raises(ValueError, match="finite positive"):
        complex_iq_to_spectrogram(np.zeros(64, dtype=np.complex64), 0.0, config)
    with pytest.raises(TypeError, match="real number"):
        complex_iq_to_spectrogram(np.zeros(64, dtype=np.complex64), True, config)


def test_public_label_helpers_preserve_validated_legacy_contract():
    assert parse_label_signature("2|0", num_classes=4) == [0, 2]
    assert normalize_label_signature("2|0", num_classes=4) == "0|2"
    assert label_to_multihot("2|0", num_classes=4) == [1, 0, 1, 0]
    assert multihot_to_signature([1, 0, 1, 0], num_classes=4) == "0|2"

    with pytest.raises(ValueError, match="Duplicate labels"):
        parse_label_signature("2|2", num_classes=4)


def test_public_multihot_to_signature_accepts_boolean_vectors():
    assert multihot_to_signature([True, False, True, False], num_classes=4) == "0|2"
    assert multihot_to_signature(np.array([True, False, True, False]), num_classes=4) == "0|2"


@pytest.mark.parametrize("multihot", [[0, 0, 0, 0], [1, 0, 2, 0], [1.0, 0, 0, 0]])
def test_public_multihot_to_signature_rejects_invalid_values(multihot):
    expected_error = TypeError if any(isinstance(value, float) for value in multihot) else ValueError
    with pytest.raises(expected_error):
        multihot_to_signature(multihot, num_classes=4)
