import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .config import CACHE_DIR
from .config import NUM_CLASSES
from .data import label_to_multihot, resolve_iq_path


def _iq_to_fixed_channels(raw: np.ndarray, has_node: int, sequence_pairs: int) -> np.ndarray:
    channels = np.zeros((2, sequence_pairs), dtype=np.float32)
    if has_node == 0 or raw.size < 2:
        return channels

    pair_count = raw.size // 2
    if pair_count == 0:
        return channels

    usable = raw[: pair_count * 2]
    i_raw = usable[0::2]
    q_raw = usable[1::2]

    if pair_count > sequence_pairs:
        indices = np.linspace(0, pair_count - 1, num=sequence_pairs, dtype=np.int64)
        i = i_raw[indices]
        q = q_raw[indices]
    else:
        i = i_raw
        q = q_raw

    length = min(sequence_pairs, i.size)
    channels[0, :length] = i[:length].astype(np.float32) / 32768.0
    channels[1, :length] = q[:length].astype(np.float32) / 32768.0

    if length > 0:
        channels[:, :length] -= channels[:, :length].mean(axis=1, keepdims=True)
    return channels


def _sample_to_tensors(
    root: Path,
    row: dict[str, Any],
    sequence_pairs: int,
    has_labels: bool,
) -> dict[str, np.ndarray]:
    path = resolve_iq_path(root, row)

    node_channels = []
    node_mask = []
    sample_rates = []
    with np.load(path) as data:
        for node in range(3):
            has_node = int(row[f"has_node{node}"])
            raw = data[f"iq_node{node}"]
            node_channels.append(_iq_to_fixed_channels(raw, has_node, sequence_pairs))
            node_mask.append(float(has_node))
            sample_rates.append(float(data[f"sample_rate_node{node}"]) / 1e8 if has_node else 0.0)

    arrays = {
        "iq": np.concatenate(node_channels, axis=0).astype(np.float32, copy=False),
        "meta": np.asarray([*node_mask, *sample_rates], dtype=np.float32),
        "sample_id": np.asarray(int(row["sample_id"]), dtype=np.int64),
    }
    if has_labels:
        arrays["target"] = np.asarray(label_to_multihot(row["label_signature"], NUM_CLASSES), dtype=np.float32)
    return arrays


class IQDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: list[dict[str, Any]],
        sequence_pairs: int,
        has_labels: bool,
    ) -> None:
        if sequence_pairs <= 0:
            raise ValueError("sequence_pairs must be positive")
        self.root = Path(root)
        self.rows = rows
        self.sequence_pairs = int(sequence_pairs)
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        arrays = _sample_to_tensors(self.root, row, self.sequence_pairs, self.has_labels)
        item = {
            "iq": torch.from_numpy(arrays["iq"]),
            "meta": torch.from_numpy(arrays["meta"]),
            "sample_id": torch.from_numpy(arrays["sample_id"]),
        }

        if self.has_labels:
            item["target"] = torch.from_numpy(arrays["target"])
        return item


def iq_cache_base_path(
    name: str,
    sequence_pairs: int,
    max_samples: int | None = None,
    cache_dir: Path = CACHE_DIR,
) -> Path:
    suffix = f"{name}_iq_pairs{sequence_pairs}"
    if max_samples:
        suffix += f"_n{max_samples}"
    return Path(cache_dir) / suffix


def _cache_paths(base_path: Path) -> dict[str, Path]:
    base = Path(base_path)
    return {
        "metadata": base.with_suffix(".json"),
        "sample_ids": base.with_suffix(".sample_ids.npy"),
        "iq": base.with_suffix(".iq.npy"),
        "meta": base.with_suffix(".meta.npy"),
        "target": base.with_suffix(".target.npy"),
    }


def _load_cache_metadata(base_path: Path) -> dict[str, Any] | None:
    path = _cache_paths(base_path)["metadata"]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_matches(
    base_path: Path,
    rows: list[dict[str, Any]],
    sequence_pairs: int,
    has_labels: bool,
) -> bool:
    paths = _cache_paths(base_path)
    required = [paths["metadata"], paths["sample_ids"], paths["iq"], paths["meta"]]
    if has_labels:
        required.append(paths["target"])
    if any(not path.exists() for path in required):
        return False

    metadata = _load_cache_metadata(base_path)
    if not metadata:
        return False
    if int(metadata.get("sequence_pairs", -1)) != int(sequence_pairs):
        return False
    if bool(metadata.get("has_labels")) != bool(has_labels):
        return False
    if int(metadata.get("num_samples", -1)) != len(rows):
        return False

    expected_ids = np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64)
    cached_ids = np.load(paths["sample_ids"], mmap_mode="r")
    return cached_ids.shape == expected_ids.shape and bool(np.array_equal(cached_ids, expected_ids))


def build_iq_tensor_cache(
    root: Path,
    rows: list[dict[str, Any]],
    sequence_pairs: int,
    has_labels: bool,
    base_path: Path,
    dtype: str = "float16",
    force: bool = False,
    progress_every: int = 250,
) -> Path:
    if dtype not in {"float16", "float32"}:
        raise ValueError("cache dtype must be 'float16' or 'float32'")
    base_path = Path(base_path)
    if not force and _cache_matches(base_path, rows, sequence_pairs, has_labels):
        print(f"Loaded IQ tensor cache: {base_path}")
        return base_path

    paths = _cache_paths(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    np_dtype = np.float16 if dtype == "float16" else np.float32
    num_rows = len(rows)
    iq = np.lib.format.open_memmap(paths["iq"], mode="w+", dtype=np_dtype, shape=(num_rows, 6, sequence_pairs))
    meta = np.lib.format.open_memmap(paths["meta"], mode="w+", dtype=np.float32, shape=(num_rows, 6))
    sample_ids = np.lib.format.open_memmap(paths["sample_ids"], mode="w+", dtype=np.int64, shape=(num_rows,))
    target = None
    if has_labels:
        target = np.lib.format.open_memmap(paths["target"], mode="w+", dtype=np.float32, shape=(num_rows, NUM_CLASSES))

    total = len(rows)
    for idx, row in enumerate(rows):
        arrays = _sample_to_tensors(root, row, sequence_pairs, has_labels)
        iq[idx] = arrays["iq"].astype(np_dtype, copy=False)
        meta[idx] = arrays["meta"]
        sample_ids[idx] = arrays["sample_id"]
        if target is not None:
            target[idx] = arrays["target"]
        item_no = idx + 1
        if progress_every and (item_no % progress_every == 0 or item_no == total):
            print(f"Cached IQ tensors: {item_no}/{total}")

    iq.flush()
    meta.flush()
    sample_ids.flush()
    if target is not None:
        target.flush()

    metadata = {
        "root": str(Path(root).resolve()),
        "num_samples": num_rows,
        "sequence_pairs": int(sequence_pairs),
        "has_labels": bool(has_labels),
        "iq_dtype": dtype,
        "iq_shape": [num_rows, 6, int(sequence_pairs)],
        "meta_shape": [num_rows, 6],
        "target_shape": [num_rows, NUM_CLASSES] if has_labels else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved IQ tensor cache: {base_path}")
    return base_path


class CachedIQDataset(Dataset):
    def __init__(
        self,
        base_path: Path,
        rows: list[dict[str, Any]],
        has_labels: bool,
    ) -> None:
        self.base_path = Path(base_path)
        self.rows = rows
        self.has_labels = has_labels
        self.paths = _cache_paths(self.base_path)
        self.metadata = _load_cache_metadata(self.base_path)
        if not self.metadata:
            raise FileNotFoundError(f"Missing IQ tensor cache metadata: {self.paths['metadata']}")
        if bool(self.metadata.get("has_labels")) != bool(has_labels):
            raise ValueError("Cached IQ tensor labels setting does not match dataset request")

        cached_ids = np.load(self.paths["sample_ids"], mmap_mode="r")
        id_to_cache_index = {int(sample_id): idx for idx, sample_id in enumerate(cached_ids.tolist())}
        self.cache_indices = []
        for row in rows:
            sample_id = int(row["sample_id"])
            if sample_id not in id_to_cache_index:
                raise ValueError(f"Sample {sample_id} is missing from IQ tensor cache: {self.base_path}")
            self.cache_indices.append(id_to_cache_index[sample_id])
        self._iq = None
        self._meta = None
        self._target = None

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self._iq is None:
            self._iq = np.load(self.paths["iq"], mmap_mode="r")
            self._meta = np.load(self.paths["meta"], mmap_mode="r")
            if self.has_labels:
                self._target = np.load(self.paths["target"], mmap_mode="r")
        return self._iq, self._meta, self._target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        iq, meta, target = self._arrays()
        cache_index = self.cache_indices[index]
        row = self.rows[index]
        item = {
            "iq": torch.from_numpy(np.array(iq[cache_index], dtype=np.float32, copy=True)),
            "meta": torch.from_numpy(np.array(meta[cache_index], dtype=np.float32, copy=True)),
            "sample_id": torch.tensor(int(row["sample_id"]), dtype=torch.long),
        }
        if self.has_labels:
            if target is None:
                raise RuntimeError("Cached IQ target array is not loaded")
            item["target"] = torch.from_numpy(np.array(target[cache_index], dtype=np.float32, copy=True))
        return item


class IQCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 6,
        meta_features: int = 6,
        width: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.config = {
            "num_classes": num_classes,
            "in_channels": in_channels,
            "meta_features": meta_features,
            "width": width,
            "dropout": dropout,
        }
        self.signal_encoder = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Conv1d(width, width * 2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.SiLU(),
            nn.Conv1d(width * 2, width * 2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.SiLU(),
            nn.Conv1d(width * 2, width * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_features, width),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(width * 4 + width, width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, num_classes),
        )

    def forward(self, iq: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        signal_features = self.signal_encoder(iq)
        meta_features = self.meta_encoder(meta)
        return self.classifier(torch.cat([signal_features, meta_features], dim=1))


def predictions_from_probabilities(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    predictions = (probabilities >= threshold).astype(np.int64)
    empty_rows = np.where(predictions.sum(axis=1) == 0)[0]
    if empty_rows.size:
        best = np.argmax(probabilities[empty_rows], axis=1)
        predictions[empty_rows, best] = 1
    return predictions


def exact_match_accuracy_multihot(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.shape != predictions.shape:
        raise ValueError(f"Shape mismatch: {targets.shape} vs {predictions.shape}")
    return float(np.all(targets == predictions, axis=1).mean())


def macro_f1_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.shape != predictions.shape:
        raise ValueError(f"Shape mismatch: {targets.shape} vs {predictions.shape}")
    scores = []
    for class_idx in range(targets.shape[1]):
        y_true = targets[:, class_idx].astype(bool)
        y_pred = predictions[:, class_idx].astype(bool)
        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else float((2 * tp) / denom))
    return float(np.mean(scores))


def find_best_threshold(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {"exact_match_accuracy": -1.0, "macro_f1": -1.0}
    for threshold in np.linspace(0.05, 0.95, num=19):
        predictions = predictions_from_probabilities(probabilities, float(threshold))
        exact = exact_match_accuracy_multihot(targets, predictions)
        macro_f1 = macro_f1_score(targets, predictions)
        if (exact, macro_f1) > (best_metrics["exact_match_accuracy"], best_metrics["macro_f1"]):
            best_threshold = float(threshold)
            best_metrics = {"exact_match_accuracy": exact, "macro_f1": macro_f1}
    return best_threshold, best_metrics
