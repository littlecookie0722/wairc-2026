from pathlib import Path

import numpy as np

from .config import CACHE_DIR, FFT_SIZE, MAX_IQ_PAIRS
from .data import resolve_iq_path


def _safe_stats(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    ]


def _spectral_features(complex_iq: np.ndarray, fft_size: int = FFT_SIZE) -> list[float]:
    if complex_iq.size < 16:
        return [0.0] * 13

    n = min(int(fft_size), int(complex_iq.size))
    signal = complex_iq[:n].astype(np.complex64, copy=False)
    signal = signal - np.mean(signal)
    window = np.hanning(n).astype(np.float32)
    spectrum = np.fft.fft(signal * window)
    power = np.abs(spectrum).astype(np.float32) ** 2
    total = float(np.sum(power)) + 1e-12
    norm_power = power / total

    peak_idx = int(np.argmax(power))
    peak_power = float(power[peak_idx])
    entropy = float(-np.sum(norm_power * np.log(norm_power + 1e-12)) / np.log(power.size))

    bands = np.array_split(power, 8)
    band_features = [float(np.log1p(np.mean(band))) for band in bands]
    return [
        peak_idx / max(1, power.size - 1),
        float(np.log1p(peak_power)),
        float(np.log1p(np.mean(power))),
        float(np.log1p(np.std(power))),
        entropy,
        *band_features,
    ]


def extract_node_features(
    raw: np.ndarray,
    sample_rate: float,
    has_node: int,
    max_pairs: int = MAX_IQ_PAIRS,
    fft_size: int = FFT_SIZE,
) -> list[float]:
    if has_node == 0 or raw.size == 0:
        return [0.0] * node_feature_count()

    pair_count = raw.size // 2
    if pair_count == 0:
        return [0.0] * node_feature_count()

    usable = raw[: pair_count * 2]
    i_raw = usable[0::2]
    q_raw = usable[1::2]
    step = max(1, pair_count // max_pairs)
    i = i_raw[::step].astype(np.float32) / 32768.0
    q = q_raw[::step].astype(np.float32) / 32768.0
    complex_iq = i.astype(np.complex64) + 1j * q.astype(np.complex64)
    magnitude = np.sqrt(i * i + q * q)
    power = magnitude * magnitude

    percentiles = np.percentile(magnitude, [10, 50, 90]).astype(np.float32)
    features = [
        1.0,
        float(sample_rate) / 1e8,
        float(raw.size) / 1e6,
        float(pair_count) / 1e6,
        *_safe_stats(i),
        *_safe_stats(q),
        *_safe_stats(magnitude),
        float(np.mean(power)),
        float(np.std(power)),
        float(np.max(power)),
        *[float(v) for v in percentiles],
        *_spectral_features(complex_iq, fft_size=fft_size),
    ]
    return features


def node_feature_count() -> int:
    return 35


def extract_sample_features(
    root: Path,
    row: dict,
    max_pairs: int = MAX_IQ_PAIRS,
    fft_size: int = FFT_SIZE,
) -> np.ndarray:
    path = resolve_iq_path(root, row)
    features: list[float] = []
    with np.load(path) as data:
        for node in range(3):
            raw = data[f"iq_node{node}"]
            sample_rate = float(data[f"sample_rate_node{node}"])
            has_node = int(row[f"has_node{node}"])
            features.extend(
                extract_node_features(
                    raw,
                    sample_rate=sample_rate,
                    has_node=has_node,
                    max_pairs=max_pairs,
                    fft_size=fft_size,
                )
            )
    return np.asarray(features, dtype=np.float32)


def extract_features_for_rows(
    root: Path,
    rows: list[dict],
    cache_path: Path | None = None,
    max_pairs: int = MAX_IQ_PAIRS,
    fft_size: int = FFT_SIZE,
    force: bool = False,
    progress_every: int = 250,
) -> np.ndarray:
    if cache_path and cache_path.exists() and not force:
        cached = np.load(cache_path)
        x = cached["features"]
        sample_ids = cached["sample_ids"].astype(np.int64).tolist()
        expected_ids = [int(row["sample_id"]) for row in rows]
        if sample_ids == expected_ids:
            print(f"Loaded feature cache: {cache_path}")
            return x.astype(np.float32, copy=False)
        print(f"Ignoring stale feature cache with mismatched sample IDs: {cache_path}")

    all_features = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        all_features.append(
            extract_sample_features(
                root,
                row,
                max_pairs=max_pairs,
                fft_size=fft_size,
            )
        )
        if progress_every and (idx % progress_every == 0 or idx == total):
            print(f"Extracted features: {idx}/{total}")

    x = np.vstack(all_features).astype(np.float32)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=x,
            sample_ids=np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64),
        )
        print(f"Saved feature cache: {cache_path}")
    return x


def cache_path(name: str, max_pairs: int = MAX_IQ_PAIRS, max_samples: int | None = None) -> Path:
    suffix = f"{name}_features_pairs{max_pairs}"
    if max_samples:
        suffix += f"_n{max_samples}"
    return CACHE_DIR / f"{suffix}.npz"

