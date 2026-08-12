"""Validated label helpers shared by public RF workflows."""

from collections.abc import Iterable
from numbers import Integral

import numpy as np

from src.config import NUM_CLASSES
from src.data import (
    label_to_multihot as _label_to_multihot,
    normalize_label_signature as _normalize_label_signature,
    parse_label_signature as _parse_label_signature,
)


def _validate_num_classes(num_classes: int) -> int:
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral):
        raise TypeError("num_classes must be an integer")
    resolved = int(num_classes)
    if resolved < 1:
        raise ValueError("num_classes must be positive")
    return resolved


def parse_label_signature(signature: str, num_classes: int = NUM_CLASSES) -> list[int]:
    """Parse a pipe-delimited label signature into sorted unique indices."""

    if not isinstance(signature, str):
        raise TypeError("signature must be a string")
    return _parse_label_signature(signature, num_classes=_validate_num_classes(num_classes))


def normalize_label_signature(signature: str, num_classes: int = NUM_CLASSES) -> str:
    """Return the canonical sorted representation of a label signature."""

    if not isinstance(signature, str):
        raise TypeError("signature must be a string")
    return _normalize_label_signature(signature, num_classes=_validate_num_classes(num_classes))


def label_to_multihot(signature: str, num_classes: int = NUM_CLASSES) -> list[int]:
    """Convert a validated label signature to a binary multi-hot list."""

    if not isinstance(signature, str):
        raise TypeError("signature must be a string")
    return _label_to_multihot(signature, num_classes=_validate_num_classes(num_classes))


def multihot_to_signature(multihot: Iterable[int], num_classes: int = NUM_CLASSES) -> str:
    """Convert a strict binary multi-hot iterable to a canonical signature."""

    resolved_classes = _validate_num_classes(num_classes)
    if isinstance(multihot, (str, bytes)):
        raise TypeError("multihot must be an iterable of binary integers")
    try:
        values = list(multihot)
    except TypeError as exc:
        raise TypeError("multihot must be an iterable of binary integers") from exc
    if len(values) != resolved_classes:
        raise ValueError(f"Expected {resolved_classes} values, got {len(values)}")

    labels: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, (bool, np.bool_, Integral)):
            raise TypeError("multihot values must be integer 0 or 1")
        if value not in (0, 1):
            raise ValueError("multihot values must be 0 or 1")
        if value == 1:
            labels.append(str(index))
    if not labels:
        raise ValueError("Cannot convert all-zero multi-hot label to a signature")
    return "|".join(labels)
