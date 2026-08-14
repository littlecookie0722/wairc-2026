"""Shared numerical kernel for the released ``stft-v1`` transform."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import stft as scipy_stft


def compute_stft_v1(
    interleaved_iq: np.ndarray,
    sample_rate: float,
    *,
    n_fft: int,
    hop: int,
    target_freq: int | None,
    target_time: int | None,
    fallback_sample_rate: float | None = None,
) -> np.ndarray | None:
    """Compute the released STFT values without model or dataset dependencies.

    Validation of the public API happens in ``wairc_rf.transforms``. The
    optional fallback exists only for the legacy competition wrapper, which
    historically substituted 125 MHz for a non-positive sample rate.
    """

    if interleaved_iq.size < n_fft * 2:
        return None

    raw = interleaved_iq.astype(np.float32, copy=False)
    complex_iq = raw[0::2] + 1j * raw[1::2]
    complex_iq = complex_iq - complex_iq.mean()

    resolved_sample_rate = sample_rate if sample_rate > 0 else fallback_sample_rate
    if resolved_sample_rate is None:
        raise ValueError("sample_rate must be positive")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input data is complex")
        _, _, spec = scipy_stft(
            complex_iq,
            fs=resolved_sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            boundary=None,
            padded=False,
        )
    mag = np.log1p(np.abs(spec).astype(np.float32))
    mag = (mag - mag.mean()) / (mag.std() + 1e-6)

    if target_freq is not None and mag.shape[0] != target_freq:
        mag = _resize_axis(mag, target_freq, axis=0)
    if target_time is not None and mag.shape[1] != target_time:
        mag = _resize_axis(mag, target_time, axis=1)
    return mag.astype(np.float32)


def _resize_axis(arr: np.ndarray, target: int, axis: int) -> np.ndarray:
    current = arr.shape[axis]
    if current == target:
        return arr
    xp = np.linspace(0.0, 1.0, current)
    x = np.linspace(0.0, 1.0, target)
    return np.apply_along_axis(lambda values: np.interp(x, xp, values), axis=axis, arr=arr).astype(np.float32)
