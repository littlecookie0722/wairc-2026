import numpy as np

from src.cache_artifact import CACHE_SCHEMA, load_cache_artifact, write_cache_artifact


def arrays():
    return np.zeros((3, 5, 7), dtype=np.float32), np.asarray([1, 0, 1], dtype=np.float32)


def test_cache_writer_and_loader_round_trip_with_metadata(tmp_path):
    path = tmp_path / "cache.npz"
    x, node_mask = arrays()
    x[0, 1, 2] = 0.75
    write_cache_artifact(path, x=x, node_mask=node_mask, n_fft=8, hop=2, target_freq=5, cache_time=7)

    with np.load(path, allow_pickle=False) as saved:
        assert saved["schemaVersion"].item() == CACHE_SCHEMA
        assert saved["artifactType"].item() == "stft-cache"
        assert saved["stftProfile"].item() == "stft-v1"

    loaded = load_cache_artifact(path, n_fft=8, hop=2, target_freq=5, cache_time=7)
    assert loaded is not None
    actual_x, actual_mask = loaded
    np.testing.assert_allclose(actual_x, x, atol=1e-3)
    np.testing.assert_array_equal(actual_mask, node_mask)


def test_loader_accepts_legacy_cache_and_rejects_stale_metadata(tmp_path):
    path = tmp_path / "legacy.npz"
    x, node_mask = arrays()
    np.savez_compressed(path, x=x.astype(np.float16), node_mask=node_mask)

    loaded = load_cache_artifact(path, n_fft=8, hop=2, target_freq=5, cache_time=7)
    assert loaded is not None
    np.testing.assert_allclose(loaded[0], x, atol=1e-3)

    versioned = tmp_path / "versioned.npz"
    write_cache_artifact(versioned, x=x, node_mask=node_mask, n_fft=8, hop=2, target_freq=5, cache_time=7)
    assert load_cache_artifact(versioned, n_fft=16, hop=2, target_freq=5, cache_time=7) is None


def test_loader_returns_none_for_corrupt_or_invalid_cache(tmp_path):
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not a zip")
    assert load_cache_artifact(corrupt, n_fft=8, hop=2, target_freq=5, cache_time=7) is None

    invalid = tmp_path / "invalid.npz"
    x, node_mask = arrays()
    node_mask[1] = 2
    np.savez_compressed(invalid, x=x, node_mask=node_mask)
    assert load_cache_artifact(invalid, n_fft=8, hop=2, target_freq=5, cache_time=7) is None

    missing_metadata = tmp_path / "missing-metadata.npz"
    np.savez_compressed(
        missing_metadata,
        schemaVersion=np.asarray("cache-v1"),
        stftProfile=np.asarray("stft-v1"),
        n_fft=np.asarray(8, dtype=np.int32),
        hop=np.asarray(2, dtype=np.int32),
        target_freq=np.asarray(5, dtype=np.int32),
        cache_time=np.asarray(7, dtype=np.int32),
        node_count=np.asarray(3, dtype=np.int32),
        x=x,
        node_mask=np.asarray([1, 0, 1], dtype=np.float32),
    )
    assert load_cache_artifact(missing_metadata, n_fft=8, hop=2, target_freq=5, cache_time=7) is None


def test_writer_rejects_invalid_transform_metadata(tmp_path):
    x, node_mask = arrays()
    try:
        write_cache_artifact(tmp_path / "invalid.npz", x=x, node_mask=node_mask, n_fft=0, hop=2, target_freq=5, cache_time=7)
    except ValueError as error:
        assert "n_fft" in str(error)
    else:
        raise AssertionError("invalid cache metadata was accepted")
