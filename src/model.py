import pickle
from pathlib import Path

import numpy as np


class NearestCentroidClassifier:
    """A small dependency-light baseline classifier for label signatures."""

    def __init__(self) -> None:
        self.labels_: list[str] = []
        self.centroids_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray, labels: list[str]) -> "NearestCentroidClassifier":
        if x.ndim != 2:
            raise ValueError("x must be a 2D feature matrix")
        if len(x) != len(labels):
            raise ValueError("x and labels must have the same length")

        self.mean_ = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
        self.std_ = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
        self.std_[self.std_ < 1e-6] = 1.0
        x_scaled = self._scale(x)

        unique_labels = sorted(set(labels), key=lambda value: (value.count("|"), value))
        centroids = []
        for label in unique_labels:
            mask = np.asarray([item == label for item in labels], dtype=bool)
            centroids.append(np.mean(x_scaled[mask], axis=0))

        self.labels_ = unique_labels
        self.centroids_ = np.vstack(centroids).astype(np.float32)
        return self

    def predict(self, x: np.ndarray, batch_size: int = 512) -> list[str]:
        if self.centroids_ is None:
            raise ValueError("Model is not fitted")
        x_scaled = self._scale(x)
        predictions: list[str] = []
        centroid_norm = np.sum(self.centroids_ * self.centroids_, axis=1)

        for start in range(0, x_scaled.shape[0], batch_size):
            batch = x_scaled[start : start + batch_size]
            batch_norm = np.sum(batch * batch, axis=1, keepdims=True)
            distances = batch_norm + centroid_norm[None, :] - 2.0 * batch @ self.centroids_.T
            best = np.argmin(distances, axis=1)
            predictions.extend(self.labels_[int(idx)] for idx in best)
        return predictions

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Model is not fitted")
        return ((x.astype(np.float32) - self.mean_) / self.std_).astype(np.float32)


def save_model(model: NearestCentroidClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> NearestCentroidClassifier:
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    with Path(path).open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, NearestCentroidClassifier):
        raise TypeError(f"Unexpected model type in {path}: {type(model)!r}")
    return model

