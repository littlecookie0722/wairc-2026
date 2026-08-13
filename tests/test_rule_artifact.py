import json

import pytest

from src.rule_artifact import (
    RULE_SCHEMA,
    load_rule_artifact,
    make_rule_payload,
    write_rule_artifact,
)


def per_class_rule():
    return {"method": "per_class_thresholds", "thresholds": [0.5] * 9, "accuracy": 0.75}


def test_rule_writer_and_loader_preserve_selected_rule(tmp_path):
    path = tmp_path / "best_rule.json"
    payload = make_rule_payload(per_class_rule(), candidates=[per_class_rule()], num_classes=9)
    write_rule_artifact(path, payload)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schemaVersion"] == RULE_SCHEMA
    assert saved["artifactType"] == "inference-rule"
    assert saved["numClasses"] == 9
    assert "private" not in json.dumps(saved)
    assert load_rule_artifact(path) == per_class_rule()


def test_top2_rule_round_trips(tmp_path):
    path = tmp_path / "top2.json"
    selected = {"method": "top2_second_threshold", "second_threshold": 0.65, "accuracy": 0.8}
    write_rule_artifact(path, make_rule_payload(selected, num_classes=9))

    assert load_rule_artifact(path) == selected


@pytest.mark.parametrize(
    "payload, expected",
    [
        (per_class_rule(), {"method": "per_class_thresholds", "thresholds": [0.5] * 9, "accuracy": None}),
        ({"thresholds": [0.4] * 9, "per_class_acc": 0.5}, {"method": "per_class_thresholds", "thresholds": [0.4] * 9, "accuracy": 0.5}),
    ],
)
def test_loader_accepts_legacy_rule_shapes(tmp_path, payload, expected):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_rule_artifact(path) == expected


def test_missing_rule_uses_legacy_default(tmp_path):
    assert load_rule_artifact(tmp_path / "missing.json")["thresholds"] == [0.5] * 9


@pytest.mark.parametrize(
    "rule, message",
    [
        ({"method": "future"}, "Unknown inference rule"),
        ({"method": "per_class_thresholds", "thresholds": [0.5]}, "contain 9"),
        ({"method": "per_class_thresholds", "thresholds": [2.0] * 9}, r"in \[0, 1\]"),
        ({"method": "top2_second_threshold"}, "second_threshold"),
    ],
)
def test_loader_rejects_invalid_rule_values(tmp_path, rule, message):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(make_rule_payload(per_class_rule(), num_classes=9) | {"selected": rule}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_rule_artifact(path)


def test_v1_loader_rejects_wrong_class_count_and_writer_sanitizes_source_paths(tmp_path):
    path = tmp_path / "v1.json"
    payload = make_rule_payload(per_class_rule(), num_classes=9, source_files=["C:/private/oof_fold0.npz"])
    write_rule_artifact(path, payload)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["source_files"] == ["oof_fold0.npz"]

    saved["numClasses"] = 4
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="numClasses"):
        load_rule_artifact(path)


def test_writer_rejects_invalid_payload_and_candidates(tmp_path):
    path = tmp_path / "invalid.json"
    payload = make_rule_payload(per_class_rule(), num_classes=9)
    payload["candidates"] = [{"method": "future"}]

    with pytest.raises(ValueError, match="Unknown inference rule"):
        write_rule_artifact(path, payload)

    payload = make_rule_payload(per_class_rule(), num_classes=9)
    payload["schemaVersion"] = "rule-v9"
    with pytest.raises(ValueError, match="rule-v1"):
        write_rule_artifact(path, payload)

    payload = make_rule_payload(per_class_rule(), num_classes=9, source_files="oof_fold0.npz")
    with pytest.raises(ValueError, match="source_files"):
        write_rule_artifact(path, payload)
