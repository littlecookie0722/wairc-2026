"""Versioned public transforms that preserve the released STFT behavior."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ._stft import compute_stft_v1


STFT_V1_PROFILE = "stft-v1"


@dataclass(frozen=True, slots=True)
class STFTConfig:
    """Configuration for the released STFT transform.

    The profile name versions behavior that is not fully expressed by numeric
    parameters, including DC removal, log-magnitude conversion, standardization,
    and linear resizing. Optional output sizes leave that axis at its native
    length.
    """

    profile: str = STFT_V1_PROFILE
    n_fft: int = 512
    hop: int = 128
    target_freq: int | None = None
    target_time: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str):
            raise TypeError("profile must be a string")
        if self.profile != STFT_V1_PROFILE:
            raise ValueError(f"Unsupported STFT profile: {self.profile!r}")

        for name in ("n_fft", "hop"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))

        for name in ("target_freq", "target_time"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer or None")
            object.__setattr__(self, name, int(value))

        if self.n_fft < 2:
            raise ValueError("n_fft must be at least 2")
        if self.hop < 1 or self.hop > self.n_fft:
            raise ValueError("hop must be between 1 and n_fft")
        if self.target_freq is not None and self.target_freq < 1:
            raise ValueError("target_freq must be positive when provided")
        if self.target_time is not None and self.target_time < 1:
            raise ValueError("target_time must be positive when provided")


def iq_to_spectrogram(
    interleaved_iq: np.ndarray,
    sample_rate: float,
    config: STFTConfig | None = None,
) -> np.ndarray | None:
    """Convert a one-dimensional ``I,Q,I,Q,...`` array to an STFT tensor.

    Valid inputs delegate to the released implementation without changing its
    numerical operations. ``None`` is returned when the recording is shorter
    than one configured FFT window, matching the legacy ``src`` API.
    """

    resolved = STFTConfig() if config is None else config
    if not isinstance(resolved, STFTConfig):
        raise TypeError("config must be an STFTConfig instance")

    values = np.asarray(interleaved_iq)
    if values.ndim != 1:
        raise ValueError("interleaved_iq must be a one-dimensional array")
    if values.size % 2 != 0:
        raise ValueError("interleaved_iq must contain complete I/Q pairs")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
        raise TypeError("interleaved_iq must contain real numeric values")

    if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(sample_rate, Real):
        raise TypeError("sample_rate must be a real number")
    rate = float(sample_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sample_rate must be a finite positive value")

    return compute_stft_v1(
        values,
        rate,
        n_fft=resolved.n_fft,
        hop=resolved.hop,
        target_freq=resolved.target_freq,
        target_time=resolved.target_time,
    )


def complex_iq_to_spectrogram(
    complex_iq: np.ndarray,
    sample_rate: float,
    config: STFTConfig | None = None,
) -> np.ndarray | None:
    """Convert one-dimensional complex IQ samples with the ``stft-v1`` profile.

    Real and imaginary components are converted to the same float32
    ``I,Q,I,Q,...`` representation consumed by :func:`iq_to_spectrogram`.
    This preserves the existing transform and its short-recording behavior.
    """

    values = np.asarray(complex_iq)
    if values.ndim != 1:
        raise ValueError("complex_iq must be a one-dimensional array")
    if not np.issubdtype(values.dtype, np.complexfloating):
        raise TypeError("complex_iq must contain complex numeric values")

    interleaved = np.empty(values.size * 2, dtype=np.float32)
    interleaved[0::2] = values.real
    interleaved[1::2] = values.imag
    return iq_to_spectrogram(interleaved, sample_rate, config)
