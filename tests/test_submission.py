from src.submission import signature_predictions_to_multihot, write_submission
from src.validate_submission import validate_submission


def write_public_index(root, sample_ids):
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        "sample_id,iq_npz_relpath,has_node0,has_node1,has_node2,"
        "sample_rate_node0,sample_rate_node1,sample_rate_node2"
    ]
    for sample_id in sample_ids:
        lines.append(f"{sample_id},iq_sample/{sample_id}.npz,1,0,1,125000000,0,122880000")
    (root / "index.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_submission_writer_sorts_ids_and_converts_signatures(tmp_path):
    assert signature_predictions_to_multihot(["2|0"]) == [[1, 0, 1, 0, 0, 0, 0, 0, 0]]

    output = write_submission(
        [{"sample_id": 2}, {"sample_id": 1}],
        [[0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]],
        tmp_path / "submission.txt",
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "1: [1, 0, 0, 0, 0, 0, 0, 0, 0]",
        "2: [0, 1, 0, 0, 0, 0, 0, 0, 0]",
    ]


def test_submission_validator_accepts_valid_file_and_rejects_boolean(tmp_path):
    test_root = tmp_path / "test"
    write_public_index(test_root, [1, 2])
    valid_path = tmp_path / "valid.txt"
    valid_path.write_text(
        "1: [1, 0, 0, 0, 0, 0, 0, 0, 0]\n"
        "2: [0, 1, 0, 0, 0, 0, 0, 0, 0]\n",
        encoding="utf-8",
    )
    invalid_path = tmp_path / "invalid.txt"
    invalid_path.write_text(
        "1: [True, 0, 0, 0, 0, 0, 0, 0, 0]\n"
        "2: [0, 1, 0, 0, 0, 0, 0, 0, 2]\n",
        encoding="utf-8",
    )

    assert validate_submission(valid_path, test_root) == []
    errors = validate_submission(invalid_path, test_root)
    assert any("integer 0 or 1" in error for error in errors)
