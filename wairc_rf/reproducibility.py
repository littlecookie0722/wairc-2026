"""Shared random-state helpers for reproducible research workflows."""

from __future__ import annotations

import random
from numbers import Integral

import numpy as np
import torch


_MAX_NUMPY_SEED = 2**32 - 1


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    value = int(seed)
    if value < 0 or value > _MAX_NUMPY_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_NUMPY_SEED}")
    return value


def set_reproducible_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch CPU/CUDA random generators.

    ``deterministic=True`` requests deterministic cuDNN kernel selection and
    disables cuDNN benchmarking. It does not guarantee bit-for-bit equality
    across every CUDA operator, device, or library version.
    """

    value = _validate_seed(seed)
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a boolean")

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(_worker_id: int) -> None:
    """Seed Python and NumPy in a PyTorch DataLoader worker."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
