# AI Radio Drone Identification Baseline

This repository contains a beginner-friendly baseline for the AI + radio drone identification competition.

## Data

- Training data: `data_and_code/ai_radio_2026_qualifying_release/train/`
- Public test data: `data_and_code_patch-1/test_public_v1.1/`
- Competition docs: `docs/`

The public test set has no labels. Local validation is created by splitting the labeled training set.

## Environment

```powershell
cd D:\Study\wairc-2026
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train

```powershell
python -m src.train_baseline
```

This writes:

- `outputs/models/baseline.pkl`
- `outputs/metrics/baseline_metrics.json`
- feature caches under `outputs/cache/`

For a quick smoke test:

```powershell
python -m src.train_baseline --max-samples 300 --max-pairs 8192
```

## Predict

```powershell
python -m src.predict
```

This writes:

```text
outputs/submissions/submission_baseline.txt
```

For a quick smoke test after a smoke-test model:

```powershell
python -m src.predict --max-samples 20 --max-pairs 8192 --output-path outputs/submissions/submission_smoke.txt
```

## Validate Submission

```powershell
python -m src.validate_submission outputs/submissions/submission_baseline.txt
```

Smoke-test submissions can be checked with:

```powershell
python -m src.validate_submission outputs/submissions/submission_smoke.txt --allow-partial
```

## Baseline Method

The first baseline extracts fixed-length features from each available IQ node:

- node availability
- sample rate and raw length
- I/Q statistics
- magnitude and power statistics
- simple FFT-based spectral features

It then trains a dependency-light nearest centroid classifier over observed label combinations such as `3` or `1|7`. Predictions are converted back to the required 9-dimensional multi-hot submission format.

This is meant to be a reliable starting point. After the full pipeline runs, improve it with richer spectral features, cross-validation, stronger tree models, or neural networks.

## GPU Train With RTX 4060

The GPU path is a separate PyTorch pipeline. It reads raw IQ samples directly and trains a multi-label 1D CNN, while the baseline files remain unchanged.

Install dependencies in the PyCharm interpreter environment:

```powershell
D:/Develop/Miniconda/envs/deepl/python.exe -m pip install -r requirements.txt
D:/Develop/Miniconda/envs/deepl/python.exe -m pip install -r requirements-gpu.txt
```

Check CUDA:

```powershell
D:/Develop/Miniconda/envs/deepl/python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Quick smoke test:

```powershell
D:/Develop/Miniconda/envs/deepl/python.exe -m src.train_torch_iq --max-samples 300 --sequence-pairs 8192 --epochs 2 --batch-size 16
```

Full GPU training:

```powershell
D:/Develop/Miniconda/envs/deepl/python.exe -m src.train_torch_iq --sequence-pairs 32768 --epochs 30 --batch-size 32
```

This writes:

- `outputs/models/iq_cnn.pt`
- `outputs/metrics/iq_cnn_metrics.json`

If the RTX 4060 runs out of memory, reduce `--batch-size` first, then reduce `--sequence-pairs`.

By default, GPU training now builds a preprocessed IQ tensor cache under `outputs/cache/`.
The first run spends time converting `.npz` files into cached tensors; later runs with the
same `--sequence-pairs` and sample limit reuse that cache and avoid repeated `.npz` decompression.

Useful cache options:

```powershell
# Force direct .npz reads
D:/Develop/Miniconda/envs/deepl/python.exe -m src.train_torch_iq --no-cache

# Rebuild the tensor cache
D:/Develop/Miniconda/envs/deepl/python.exe -m src.train_torch_iq --sequence-pairs 65536 --rebuild-cache

# Use a larger batch after cache is built
D:/Develop/Miniconda/envs/deepl/python.exe -m src.train_torch_iq --sequence-pairs 65536 --batch-size 64 --num-workers 2
```

The PyTorch trainer supports two model types:

- `time`: the original 1D CNN over time-domain IQ samples.
- `timefreq`: a two-branch model that combines the time-domain CNN with a GPU STFT frequency branch.

Run a first time+frequency experiment:

```powershell
D:/Develop/Miniconda/envs/deepl/python.exe -m src.train_torch_iq --model-type timefreq --sequence-pairs 65536 --epochs 45 --batch-size 16 --width 32 --num-workers 2 --model-path outputs/models/iq_timefreq_w32.pt --metrics-path outputs/metrics/iq_timefreq_w32.json
```

The `timefreq` model is slower and uses more memory than `time`, so start with a smaller
`--batch-size` and `--width`, then scale up after checking GPU memory.

Predict with the trained PyTorch model:

```powershell
D:/Develop/Miniconda/envs/deepl/python.exe -m src.predict_torch_iq
```

This writes:

```text
outputs/submissions/submission_iq_cnn.txt
```

## High-Score STFT Spectrogram Pipeline

The stronger pipeline follows the high-scoring approach:

- convert 3-node IQ samples into cached STFT spectrograms
- train an ImageNet-pretrained 2D backbone such as ResNet34
- use BCE with per-class `pos_weight`
- search the best validation inference rule
- run test-time augmentation with multiple time crops
- optionally ensemble k-fold and multi-architecture models

Install the extra CPU/GPU dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

Train one strong model:

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram --epochs 40 --batch-size 16 --num-workers 2
```

This writes:

- `outputs/spectrogram/best_model.pth`
- `outputs/spectrogram/best_rule.json`
- STFT caches under `outputs/cache/stft/`

Predict and validate:

```powershell
.\.venv\Scripts\python.exe -m src.predict_spectrogram
.\.venv\Scripts\python.exe -m src.validate_submission outputs/submissions/submission_spectrogram.txt
```

For a higher-score k-fold ensemble, train each fold:

```powershell
0..4 | ForEach-Object {
  .\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --fold $_ --tag r34 --epochs 40 --batch-size 16 --num-workers 2
}
```

Search the OOF inference rule and predict with the ensemble:

```powershell
.\.venv\Scripts\python.exe -m src.search_spectrogram_kfold_thresholds --tags r34
.\.venv\Scripts\python.exe -m src.predict_spectrogram_kfold --tags r34
.\.venv\Scripts\python.exe -m src.validate_submission outputs/submissions/submission_spectrogram_kfold.txt
```

To push beyond the single ResNet34 ensemble, train additional tags and average them:

```powershell
0..4 | ForEach-Object {
  .\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --fold $_ --tag b0 --arch efficientnet_b0 --epochs 40 --batch-size 24 --num-workers 2
}

.\.venv\Scripts\python.exe -m src.search_spectrogram_kfold_thresholds --tags r34 b0
.\.venv\Scripts\python.exe -m src.predict_spectrogram_kfold --tags r34 b0
```
