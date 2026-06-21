from pathlib import Path

from .config import NUM_CLASSES
from .data import label_to_multihot


def signature_predictions_to_multihot(predictions: list[str]) -> list[list[int]]:
    return [label_to_multihot(signature, NUM_CLASSES) for signature in predictions]


def write_submission(rows: list[dict], predictions: list[list[int]], output_path: Path) -> Path:
    if len(rows) != len(predictions):
        raise ValueError(f"Got {len(rows)} rows but {len(predictions)} predictions")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for row, pred in sorted(zip(rows, predictions), key=lambda item: int(item[0]["sample_id"])):
            values = [int(v) for v in pred]
            if len(values) != NUM_CLASSES:
                raise ValueError(f"Sample {row['sample_id']} prediction length is {len(values)}")
            if any(v not in (0, 1) for v in values):
                raise ValueError(f"Sample {row['sample_id']} prediction is not binary: {values}")
            f.write(f"{int(row['sample_id'])}: {values}\n")
    return output_path

