import numpy as np
import pandas as pd

from src.spectrogram import DroneSpectrogramDataset


def test_dataset_handles_missing_nodes_and_returns_fixed_shapes(tmp_path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    npz_path = data_root / "sample.npz"
    rng = np.random.default_rng(2026)
    np.savez(
        npz_path,
        iq_node0=rng.integers(-20, 20, size=512, dtype=np.int16),
        iq_node1=np.asarray([], dtype=np.int16),
        iq_node2=rng.integers(-20, 20, size=512, dtype=np.int16),
        sample_rate_node0=np.float32(125_000_000),
        sample_rate_node1=np.float32(0),
        sample_rate_node2=np.float32(122_880_000),
    )
    frame = pd.DataFrame(
        [
            {
                "sample_id": 42,
                "iq_npz_relpath": "sample.npz",
                "label_signature": "0|2",
            }
        ]
    )

    dataset = DroneSpectrogramDataset(
        dataframe=frame,
        data_root=data_root,
        n_fft=16,
        hop=4,
        target_freq=9,
        target_time=12,
        cache_time=12,
        cache_dir=tmp_path / "cache",
        is_train=True,
        augment=False,
    )
    sample = dataset[0]

    assert sample["x"].shape == (3, 9, 12)
    assert sample["node_mask"].tolist() == [1.0, 0.0, 1.0]
    assert sample["label"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert sample["sample_id"].item() == 42
