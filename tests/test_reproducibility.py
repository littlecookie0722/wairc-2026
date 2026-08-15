import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.train_spectrogram import make_loader
from wairc_rf import set_reproducible_seed
from wairc_rf.reproducibility import seed_worker


def test_set_reproducible_seed_repeats_python_numpy_and_torch_streams():
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    try:
        set_reproducible_seed(2028, deterministic=True)
        first = (random.random(), np.random.random(), torch.rand(4))
        set_reproducible_seed(2028, deterministic=True)
        second = (random.random(), np.random.random(), torch.rand(4))

        assert first[0] == second[0]
        assert first[1] == second[1]
        torch.testing.assert_close(first[2], second[2])
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic


@pytest.mark.parametrize("seed", [True, 1.5, -1, 2**32])
def test_set_reproducible_seed_rejects_ambiguous_or_unsupported_seeds(seed):
    expected = TypeError if isinstance(seed, (bool, float)) else ValueError
    with pytest.raises(expected, match="seed"):
        set_reproducible_seed(seed)


def test_set_reproducible_seed_rejects_non_boolean_deterministic_flag():
    with pytest.raises(TypeError, match="deterministic"):
        set_reproducible_seed(2028, deterministic=1)


def test_seed_worker_derives_python_and_numpy_seed_from_torch_worker_seed(monkeypatch):
    monkeypatch.setattr(torch, "initial_seed", lambda: 2**32 + 17)
    calls = []
    monkeypatch.setattr(random, "seed", lambda seed: calls.append(("python", seed)))
    monkeypatch.setattr(np.random, "seed", lambda seed: calls.append(("numpy", seed)))

    seed_worker(3)

    assert calls == [("python", 17), ("numpy", 17)]


def test_training_loader_registers_worker_seed_hook():
    loader = make_loader(
        SimpleNamespace(num_workers=0),
        TensorDataset(torch.arange(2)),
        batch_size=1,
        shuffle=False,
        device=torch.device("cpu"),
    )

    assert loader.worker_init_fn is seed_worker
