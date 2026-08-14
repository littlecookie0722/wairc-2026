from src.cpu_compatibility import run_cpu_compatibility


def test_cpu_compatibility_probe_uses_public_api_and_cpu_device():
    result = run_cpu_compatibility()

    assert result["device"] == "cpu"
    assert result["stft_shape"] == [65, 64]
    assert result["logits_shape"] == [1, 9]
    assert result["torchscript_logits_shape"] == [1, 9]
    assert result["torchscript_max_abs_difference"] <= 1e-5
