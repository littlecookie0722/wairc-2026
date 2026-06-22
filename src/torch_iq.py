from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

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
        path = resolve_iq_path(self.root, row)

        node_channels = []
        node_mask = []
        sample_rates = []
        with np.load(path) as data:
            for node in range(3):
                has_node = int(row[f"has_node{node}"])
                raw = data[f"iq_node{node}"]
                node_channels.append(_iq_to_fixed_channels(raw, has_node, self.sequence_pairs))
                node_mask.append(float(has_node))
                sample_rates.append(float(data[f"sample_rate_node{node}"]) / 1e8 if has_node else 0.0)

        iq = np.concatenate(node_channels, axis=0)
        meta = np.asarray([*node_mask, *sample_rates], dtype=np.float32)
        item = {
            "iq": torch.from_numpy(iq),
            "meta": torch.from_numpy(meta),
            "sample_id": torch.tensor(int(row["sample_id"]), dtype=torch.long),
        }

        if self.has_labels:
            target = np.asarray(label_to_multihot(row["label_signature"], NUM_CLASSES), dtype=np.float32)
            item["target"] = torch.from_numpy(target)
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
