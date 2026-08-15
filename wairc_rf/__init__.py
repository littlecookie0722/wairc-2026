"""Stable public API for reusable WAIRC RF/IQ utilities."""

from src import __version__

from .datasets import CompetitionDatasetAdapter, RFDatasetAdapter, RFNode, RFSample, SyntheticDatasetAdapter
from .labels import label_to_multihot, multihot_to_signature, normalize_label_signature, parse_label_signature
from .reproducibility import set_reproducible_seed
from .transforms import STFT_V1_PROFILE, STFTConfig, complex_iq_to_spectrogram, iq_to_spectrogram

__all__ = [
    "CompetitionDatasetAdapter",
    "RFDatasetAdapter",
    "RFNode",
    "RFSample",
    "SyntheticDatasetAdapter",
    "STFT_V1_PROFILE",
    "STFTConfig",
    "__version__",
    "complex_iq_to_spectrogram",
    "iq_to_spectrogram",
    "label_to_multihot",
    "multihot_to_signature",
    "normalize_label_signature",
    "parse_label_signature",
    "set_reproducible_seed",
]
