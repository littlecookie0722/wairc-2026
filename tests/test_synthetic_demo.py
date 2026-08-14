import json

import joblib

from src.synthetic_demo import run_demo
from src.validate_submission import validate_submission


def test_synthetic_demo_runs_end_to_end(tmp_path):
    output_dir = tmp_path / "demo"

    result = run_demo(output_dir, train_samples_per_class=2)

    assert result.train_samples == 27
    assert result.test_samples == 14
    assert 0.0 <= result.exact_match_accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert len(result.per_class_recall) == 9
    assert all(0.0 <= value <= 1.0 for value in result.per_class_recall)
    assert validate_submission(output_dir / "submission.txt", output_dir / "data" / "test") == []
    assert joblib.load(output_dir / "model.joblib") is not None

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["train_samples"] == result.train_samples
    assert metrics["test_samples"] == result.test_samples
    assert metrics["threshold"] == result.threshold
