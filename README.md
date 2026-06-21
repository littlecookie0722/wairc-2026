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

