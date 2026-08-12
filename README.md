# WAIRC-2026

A public, competition-originated deep-learning pipeline for RF drone
identification from multi-node IQ signals, being prepared for open-source
release.

This repository documents a real 2026 AI radio competition project. The main
pipeline converts raw interleaved IQ samples into STFT spectrograms and uses
pretrained computer-vision backbones, k-fold training, out-of-fold threshold
search, ensemble inference, and submission validation.

The project is maintained as a reproducible research codebase. It does not
claim production adoption, download volume, leaderboard results, or community
size that are not backed by repository evidence.

## Pipeline

Raw IQ NPZ -> STFT spectrogram -> pretrained vision backbone ->
multi-label probabilities -> k-fold OOF threshold search ->
ensemble inference -> 9-value submission

## Current capabilities

- Three-node IQ loading with missing-node handling.
- STFT spectrogram generation, resizing, caching, and SpecAugment.
- ResNet, EfficientNet, ConvNeXt, and DenseNet classifier options.
- Single-model and five-fold training entry points.
- OOF threshold/rule search and weighted model ensemble.
- Public-test prediction and strict submission-format validation.
- Archived early baselines under archived_baselines.

The current public repository is intentionally still a competition-oriented
research project. Engineering work is being added incrementally; see the
architecture and reproducibility documents for current limitations.

## Quick start

### Install

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-gpu.txt
    python -m pip install -r requirements-dev.txt

The GPU requirements use the CUDA 12.8 PyTorch index. For a CPU-only
environment, install a compatible CPU build of torch and torchvision instead.

### Verify without competition data

    python -m pytest
    python scripts/smoke_test.py

The tests and smoke test use synthetic inputs and do not require the
competition dataset, checkpoints, CUDA, or external credentials. A local
environment still needs torch, torchvision, NumPy, SciPy, pandas,
scikit-learn, and tqdm installed.

### Train the main five-fold pipeline

Place the labeled training set at:

    data_and_code/ai_radio_2026_qualifying_release/train/

Then run:

    python -m src.train_spectrogram_kfold --tag r34 --arch resnet34 --epochs 40 --batch-size 16 --num-workers 2
    python -m src.search_spectrogram_kfold_thresholds --tags r34
    python -m src.predict_spectrogram_kfold --tags r34
    python -m src.validate_submission outputs/submissions/submission_spectrogram_kfold.txt

Omit --fold to train all five folds. The public test set is unlabeled, so its
score cannot be calculated locally.

### Optional model groups

    python -m src.train_spectrogram_kfold --tag b0 --arch efficientnet_b0 --epochs 40 --batch-size 24 --num-workers 2
    python -m src.train_spectrogram_kfold --tag cnx --arch convnext_tiny --epochs 40 --batch-size 12 --num-workers 2
    python -m src.search_spectrogram_kfold_thresholds --tags r34 b0 cnx
    python -m src.predict_spectrogram_kfold --tags r34 b0 cnx
    python -m src.validate_submission outputs/submissions/submission_spectrogram_kfold.txt

For a smaller environment check:

    python -m src.train_spectrogram --epochs 3 --batch-size 8 --num-workers 0 --max-samples 300

## Data and licensing boundary

The competition dataset is not redistributed by this repository. Obtain it
through the original competition channel and follow its rules and licensing
terms. Do not commit raw data, checkpoints, caches, or private test labels.

The repository currently has no declared software license. Public visibility
does not by itself grant permission to reuse the code. The maintainer must
confirm the license choice before a LICENSE file is added.

## Project structure

    src/config.py
    src/data.py
    src/spectrogram.py
    src/train_spectrogram.py
    src/train_spectrogram_kfold.py
    src/search_spectrogram_kfold_thresholds.py
    src/predict_spectrogram_kfold.py
    src/submission.py
    src/validate_submission.py
    tests/
    scripts/smoke_test.py
    docs/architecture.md
    docs/reproducibility.md
    archived_baselines/

## Documentation

- docs/冲高分STFT频谱图方案说明.md: current competition workflow.
- docs/数据集说明.md and docs/Dataset_Guide_EN.md: dataset and submission format.
- docs/architecture.md: current module and data-flow boundaries.
- docs/reproducibility.md: reproducibility limits and required records.

## Contributing

Read CONTRIBUTING.md and AGENTS.md before changing the pipeline. Changes to
STFT parameters, label mapping, checkpoint metadata, fold splitting,
threshold rules, ensemble weights, or submission format require regression
tests and an explicit compatibility note.

## Security and privacy

Do not put API keys, account or organization identifiers, email addresses,
private dataset links, or credentials in the repository. See SECURITY.md for
reporting guidance.
