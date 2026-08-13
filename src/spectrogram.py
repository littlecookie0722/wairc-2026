from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from scipy.signal import stft as scipy_stft
from torch.utils.data import Dataset

from .config import NUM_CLASSES
from .cache_artifact import load_cache_artifact, write_cache_artifact

warnings.filterwarnings("ignore", message="Input data is complex")


def label_signature_to_multihot(signature: str, num_classes: int = NUM_CLASSES) -> np.ndarray:
    target = np.zeros(num_classes, dtype=np.float32)
    for item in str(signature).split("|"):
        item = item.strip()
        if item:
            target[int(item)] = 1.0
    return target


def iq_to_spectrogram(
    iq_int16: np.ndarray,
    sample_rate: float,
    n_fft: int = 512,
    hop: int = 128,
    target_freq: int | None = None,
    target_time: int | None = None,
) -> np.ndarray | None:
    if iq_int16.size < n_fft * 2:
        return None

    raw = iq_int16.astype(np.float32, copy=False)
    complex_iq = raw[0::2] + 1j * raw[1::2]
    complex_iq = complex_iq - complex_iq.mean()

    _, _, spec = scipy_stft(
        complex_iq,
        fs=sample_rate if sample_rate > 0 else 125e6,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary=None,
        padded=False,
    )
    mag = np.log1p(np.abs(spec).astype(np.float32))
    mag = (mag - mag.mean()) / (mag.std() + 1e-6)

    if target_freq is not None and mag.shape[0] != target_freq:
        mag = resize_axis(mag, target_freq, axis=0)
    if target_time is not None and mag.shape[1] != target_time:
        mag = resize_axis(mag, target_time, axis=1)
    return mag.astype(np.float32)


def resize_axis(arr: np.ndarray, target: int, axis: int) -> np.ndarray:
    current = arr.shape[axis]
    if current == target:
        return arr
    xp = np.linspace(0.0, 1.0, current)
    x = np.linspace(0.0, 1.0, target)
    return np.apply_along_axis(lambda values: np.interp(x, xp, values), axis=axis, arr=arr).astype(np.float32)


def time_crop(spec: np.ndarray, target_time: int, augment: bool) -> np.ndarray:
    _, time = spec.shape
    if time == target_time:
        return spec
    if time > target_time:
        start = np.random.randint(0, time - target_time + 1) if augment else (time - target_time) // 2
        return spec[:, start : start + target_time]
    reps = (target_time + time - 1) // max(time, 1)
    return np.tile(spec, (1, reps))[:, :target_time]


def spec_augment(
    spec: np.ndarray,
    time_mask_param: int,
    freq_mask_param: int,
    p: float,
) -> np.ndarray:
    if np.random.rand() > p:
        return spec
    out = spec.copy()
    freq_dim = out.shape[-2]
    time_dim = out.shape[-1]

    if time_dim > time_mask_param:
        width = np.random.randint(1, time_mask_param + 1)
        start = np.random.randint(0, max(1, time_dim - width + 1))
        out[..., start : start + width] = 0

    if freq_dim > freq_mask_param:
        width = np.random.randint(1, freq_mask_param + 1)
        start = np.random.randint(0, max(1, freq_dim - width + 1))
        out[..., start : start + width, :] = 0

    return out


class DroneSpectrogramDataset(Dataset):
    def __init__(
        self,
        dataframe: Any,
        data_root: str | Path,
        num_classes: int = NUM_CLASSES,
        n_fft: int = 512,
        hop: int = 128,
        target_freq: int = 257,
        target_time: int = 768,
        cache_time: int = 1536,
        cache_dir: str | Path | None = None,
        cache_split: str = "train",
        is_train: bool = True,
        augment: bool = True,
        time_mask_param: int = 40,
        freq_mask_param: int = 25,
        specaug_p: float = 0.5,
        tta_crops: int = 0,
    ) -> None:
        self.df = dataframe.copy().reset_index(drop=True)
        self.data_root = Path(data_root)
        self.num_classes = int(num_classes)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.target_freq = int(target_freq)
        self.target_time = int(target_time)
        self.cache_time = max(int(cache_time), self.target_time)
        self.is_train = bool(is_train)
        self.augment = bool(augment and is_train)
        self.time_mask_param = int(time_mask_param)
        self.freq_mask_param = int(freq_mask_param)
        self.specaug_p = float(specaug_p)
        self.tta_crops = int(tta_crops)

        self.cache_dir = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir) / cache_split
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def _cache_path(self, sample_id: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{sample_id:08d}.npz"

    def _load_or_compute(self, npz_path: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
        cache_path = self._cache_path(sample_id)
        if cache_path is not None and cache_path.exists():
            if cache_path.stat().st_size > 0:
                cached = load_cache_artifact(
                    cache_path,
                    n_fft=self.n_fft,
                    hop=self.hop,
                    target_freq=self.target_freq,
                    cache_time=self.cache_time,
                )
                if cached is not None:
                    return cached
            try:
                cache_path.unlink()
            except FileNotFoundError:
                pass

        specs = []
        node_mask = []
        with np.load(npz_path) as data:
            for node_idx in range(3):
                raw = data[f"iq_node{node_idx}"]
                sample_rate = float(data[f"sample_rate_node{node_idx}"])
                if raw.size < self.n_fft * 2 or sample_rate <= 0:
                    spec = np.zeros((self.target_freq, self.cache_time), dtype=np.float32)
                    node_mask.append(0.0)
                else:
                    computed = iq_to_spectrogram(
                        raw,
                        sample_rate,
                        n_fft=self.n_fft,
                        hop=self.hop,
                        target_freq=self.target_freq,
                        target_time=self.cache_time,
                    )
                    if computed is None:
                        spec = np.zeros((self.target_freq, self.cache_time), dtype=np.float32)
                        node_mask.append(0.0)
                    else:
                        spec = computed
                        node_mask.append(1.0)
                specs.append(spec)

        x = np.stack(specs, axis=0).astype(np.float32)
        mask = np.asarray(node_mask, dtype=np.float32)
        if cache_path is not None:
            tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp{cache_path.suffix}")
            try:
                write_cache_artifact(
                    tmp_path,
                    x=x,
                    node_mask=mask,
                    n_fft=self.n_fft,
                    hop=self.hop,
                    target_freq=self.target_freq,
                    cache_time=self.cache_time,
                )
                os.replace(tmp_path, cache_path)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except FileNotFoundError:
                        pass
        return x, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        sample_id = int(row["sample_id"])
        rel_path = str(row["iq_npz_relpath"]).replace("\\", "/")
        x, node_mask = self._load_or_compute(self.data_root / rel_path, sample_id)

        if self.tta_crops > 0:
            crops_by_node = []
            for node_idx in range(3):
                crops = [time_crop(x[node_idx], self.target_time, augment=True) for _ in range(self.tta_crops)]
                crops_by_node.append(np.stack(crops, axis=0))
            x_out = np.stack(crops_by_node, axis=0)
        else:
            x_out = np.stack(
                [time_crop(x[node_idx], self.target_time, augment=self.augment) for node_idx in range(3)],
                axis=0,
            )
            if self.augment:
                x_out = spec_augment(x_out, self.time_mask_param, self.freq_mask_param, self.specaug_p)

        if self.is_train and "label_signature" in row:
            label = label_signature_to_multihot(row["label_signature"], self.num_classes)
        else:
            label = np.zeros(self.num_classes, dtype=np.float32)

        return {
            "x": torch.from_numpy(x_out.astype(np.float32)),
            "node_mask": torch.from_numpy(node_mask),
            "label": torch.from_numpy(label),
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
        }


class DroneClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        arch: str = "resnet34",
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)
        self.dropout = float(dropout)

        if arch == "resnet18":
            backbone = tvm.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))
        elif arch == "resnet34":
            backbone = tvm.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))
        elif arch == "resnet50":
            backbone = tvm.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))
        elif arch == "efficientnet_b0":
            backbone = tvm.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))
        elif arch == "efficientnet_b2":
            backbone = tvm.efficientnet_b2(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))
        elif arch == "convnext_tiny":
            backbone = tvm.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = backbone.classifier[2].in_features
            backbone.classifier[2] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes),
            )
        elif arch == "densenet121":
            backbone = tvm.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = backbone.classifier.in_features
            backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        self.backbone = backbone
        if not pretrained:
            self._init_head()

    def _init_head(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.out_features == self.num_classes:
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            batch, nodes, crops, freq, time = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(batch * crops, nodes, freq, time)
            logits = self.backbone(x)
            return logits.view(batch, crops, -1).mean(dim=1)
        return self.backbone(x)


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
        disable_torch_grad_focal_loss: bool = True,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ASL contains logarithms close to zero, so keep this block in FP32 even
        # when the surrounding model forward pass uses AMP.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits_fp32 = logits.float()
            targets_fp32 = targets.float()
            probs_pos = torch.sigmoid(logits_fp32)
            probs_neg = 1.0 - probs_pos
            if self.clip and self.clip > 0:
                probs_neg = (probs_neg + self.clip).clamp(max=1.0)

            loss = targets_fp32 * torch.log(probs_pos.clamp(min=self.eps))
            loss = loss + (1.0 - targets_fp32) * torch.log(probs_neg.clamp(min=self.eps))
            loss = -loss

            if self.gamma_neg > 0 or self.gamma_pos > 0:
                def focal_weight() -> torch.Tensor:
                    pt = probs_pos * targets_fp32 + probs_neg * (1.0 - targets_fp32)
                    gamma = self.gamma_pos * targets_fp32 + self.gamma_neg * (1.0 - targets_fp32)
                    return torch.pow(1.0 - pt, gamma)

                if self.disable_torch_grad_focal_loss:
                    with torch.no_grad():
                        weight = focal_weight()
                else:
                    weight = focal_weight()
                loss = loss * weight

            if self.reduction == "sum":
                return loss.sum()
            if self.reduction == "none":
                return loss
            return loss.mean()


class BCEWithLogitsPosWeight(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        ce = -(targets * probs.log() + (1.0 - targets) * (1.0 - probs).log())
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha * (1.0 - pt).pow(self.gamma) * ce).mean()


def compute_pos_weight(labels: np.ndarray, max_weight: float = 8.0) -> torch.Tensor:
    labels = np.asarray(labels, dtype=np.float32)
    pos = labels.sum(axis=0).astype(np.float32)
    neg = labels.shape[0] - pos
    weight = neg / np.maximum(pos, 1.0)
    weight = np.clip(weight, 1.0, max_weight)
    return torch.tensor(weight, dtype=torch.float32)


def make_loss(name: str, pos_weight: torch.Tensor, asymmetric_gamma_neg: float = 4, asymmetric_gamma_pos: float = 1) -> nn.Module:
    if name == "bce_pos":
        return BCEWithLogitsPosWeight(pos_weight)
    if name == "asymmetric":
        return AsymmetricLoss(gamma_neg=asymmetric_gamma_neg, gamma_pos=asymmetric_gamma_pos)
    if name == "focal":
        return FocalLoss()
    raise ValueError(f"Unknown loss: {name}")


def exact_match(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions.astype(np.int32) == labels.astype(np.int32)).all(axis=1).mean())


def apply_thresholds(probs: np.ndarray, thresholds: list[float] | np.ndarray) -> np.ndarray:
    thresholds_array = np.asarray(thresholds, dtype=np.float32)
    return (probs > thresholds_array.reshape(1, -1)).astype(np.int32)


def enforce_count_constraint(preds: np.ndarray, probs: np.ndarray, max_labels: int = 2) -> np.ndarray:
    fixed = preds.astype(np.int32, copy=True)
    for row_idx in range(len(fixed)):
        count = int(fixed[row_idx].sum())
        if count == 0:
            fixed[row_idx, int(np.argmax(probs[row_idx]))] = 1
        elif count > max_labels:
            top = np.argsort(probs[row_idx])[-max_labels:]
            fixed[row_idx] = 0
            fixed[row_idx, top] = 1
    return fixed


def apply_top2_threshold_rule(probs: np.ndarray, second_threshold: float) -> np.ndarray:
    preds = np.zeros_like(probs, dtype=np.int32)
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    rows = np.arange(len(probs))
    preds[rows, top1] = 1
    use_second = probs[rows, top2] >= float(second_threshold)
    preds[rows[use_second], top2[use_second]] = 1
    return preds


def apply_inference_rule(probs: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    method = rule.get("method", "per_class_thresholds")
    if method == "top2_second_threshold":
        return apply_top2_threshold_rule(probs, float(rule["second_threshold"]))
    if method == "per_class_thresholds":
        preds = apply_thresholds(probs, rule["thresholds"])
        return enforce_count_constraint(preds, probs)
    raise ValueError(f"Unknown inference rule: {method}")


def search_global_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    base = exact_match(apply_thresholds(probs, [0.5] * probs.shape[1]), labels)
    best_t, best_acc = 0.5, base
    for threshold in np.arange(0.20, 0.80, 0.005):
        preds = apply_thresholds(probs, [float(threshold)] * probs.shape[1])
        preds = enforce_count_constraint(preds, probs)
        acc = exact_match(preds, labels)
        if acc > best_acc:
            best_t, best_acc = float(threshold), float(acc)
    return best_t, best_acc, base


def search_per_class_thresholds(probs: np.ndarray, labels: np.ndarray, num_classes: int = NUM_CLASSES) -> tuple[list[float], float]:
    global_t, _, _ = search_global_threshold(probs, labels)
    thresholds = np.full(num_classes, global_t, dtype=np.float32)
    preds = enforce_count_constraint(apply_thresholds(probs, thresholds), probs)
    best_acc = exact_match(preds, labels)
    grid = np.arange(0.20, 0.80, 0.01)

    for _ in range(3):
        for cls_idx in range(num_classes):
            best_cls_t = thresholds[cls_idx]
            for threshold in grid:
                trial_thresholds = thresholds.copy()
                trial_thresholds[cls_idx] = float(threshold)
                trial = enforce_count_constraint(apply_thresholds(probs, trial_thresholds), probs)
                acc = exact_match(trial, labels)
                if acc > best_acc:
                    best_acc = acc
                    best_cls_t = float(threshold)
            thresholds[cls_idx] = best_cls_t

    return thresholds.astype(float).tolist(), best_acc


def search_top2_second_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_acc = -1.0
    for threshold in np.arange(0.05, 0.95, 0.005):
        preds = apply_top2_threshold_rule(probs, float(threshold))
        acc = exact_match(preds, labels)
        if acc > best_acc:
            best_t = float(threshold)
            best_acc = acc
    return best_t, best_acc


def search_best_inference_rule(probs: np.ndarray, labels: np.ndarray, num_classes: int = NUM_CLASSES) -> dict[str, Any]:
    global_t, global_acc, base_acc = search_global_threshold(probs, labels)
    thresholds, per_class_acc = search_per_class_thresholds(probs, labels, num_classes)
    second_t, top2_acc = search_top2_second_threshold(probs, labels)

    candidates = [
        {
            "method": "per_class_thresholds",
            "thresholds": thresholds,
            "accuracy": per_class_acc,
            "global_threshold": global_t,
            "global_accuracy": global_acc,
            "base_acc_05": base_acc,
        },
        {
            "method": "top2_second_threshold",
            "second_threshold": second_t,
            "accuracy": top2_acc,
            "base_acc_05": base_acc,
        },
    ]
    best = max(candidates, key=lambda item: float(item["accuracy"]))
    return {
        "selected": best,
        "candidates": candidates,
    }


def load_inference_rule(path: Path, num_classes: int = NUM_CLASSES) -> dict[str, Any]:
    from .rule_artifact import load_rule_artifact

    return load_rule_artifact(path, num_classes=num_classes)


class WarmupCosineLR:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int, eta_max: float, eta_min: float = 1e-6) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = max(1, int(total_epochs))
        self.eta_max = float(eta_max)
        self.eta_min = float(eta_min)
        self.epoch = 0

    def step(self) -> float:
        self.epoch += 1
        if self.warmup_epochs and self.epoch <= self.warmup_epochs:
            lr = self.eta_max * self.epoch / self.warmup_epochs
        else:
            progress = (self.epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            progress = min(1.0, max(0.0, progress))
            lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return float(lr)
