import csv
import random
from collections import defaultdict
from pathlib import Path

from .config import NUM_CLASSES, RANDOM_SEED, VAL_RATIO


def load_index(root: Path, has_labels: bool) -> list[dict]:
    root = Path(root)
    index_path = root / "index.csv"
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index.csv: {index_path}")

    with index_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "sample_id",
        "iq_npz_relpath",
        "has_node0",
        "has_node1",
        "has_node2",
        "sample_rate_node0",
        "sample_rate_node1",
        "sample_rate_node2",
    }
    if has_labels:
        required.add("label_signature")
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{index_path} is missing required columns: {sorted(missing)}")

    for row in rows:
        row["sample_id"] = int(row["sample_id"])
        for node in range(3):
            row[f"has_node{node}"] = int(row[f"has_node{node}"])
            row[f"sample_rate_node{node}"] = float(row[f"sample_rate_node{node}"])
        if has_labels:
            row["label_signature"] = normalize_label_signature(row["label_signature"])
    return rows


def resolve_iq_path(root: Path, row: dict) -> Path:
    relpath = str(row["iq_npz_relpath"]).replace("\\", "/")
    path = Path(root) / relpath
    if not path.exists():
        raise FileNotFoundError(f"Missing IQ file for sample {row['sample_id']}: {path}")
    return path


def parse_label_signature(signature: str, num_classes: int = NUM_CLASSES) -> list[int]:
    if signature is None:
        raise ValueError("label_signature is missing")
    parts = [part.strip() for part in str(signature).split("|") if part.strip() != ""]
    if not parts:
        raise ValueError("label_signature is empty")

    labels: list[int] = []
    for part in parts:
        try:
            label = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid label value in label_signature {signature!r}") from exc
        if label < 0 or label >= num_classes:
            raise ValueError(f"Label {label} is outside 0..{num_classes - 1}")
        labels.append(label)

    unique = sorted(set(labels))
    if len(unique) != len(labels):
        raise ValueError(f"Duplicate labels in label_signature {signature!r}")
    return unique


def normalize_label_signature(signature: str, num_classes: int = NUM_CLASSES) -> str:
    labels = parse_label_signature(signature, num_classes)
    return "|".join(str(label) for label in labels)


def label_to_multihot(signature: str, num_classes: int = NUM_CLASSES) -> list[int]:
    labels = parse_label_signature(signature, num_classes)
    multihot = [0] * num_classes
    for label in labels:
        multihot[label] = 1
    return multihot


def multihot_to_signature(multihot) -> str:
    values = [int(v) for v in multihot]
    if len(values) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} values, got {len(values)}")
    labels = [str(idx) for idx, value in enumerate(values) if value == 1]
    if not labels:
        raise ValueError("Cannot convert all-zero multi-hot label to a signature")
    return "|".join(labels)


def stratified_split(
    rows: list[dict],
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["label_signature"]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for label, group_rows in sorted(groups.items()):
        shuffled = list(group_rows)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        val_rows.extend(shuffled[:val_count])
        train_rows.extend(shuffled[val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows
