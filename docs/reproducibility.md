# Reproducibility

WAIRC-2026 is a competition-originated machine-learning project. The raw
competition data and trained checkpoints are not stored in this repository,
so a full training or public-test reproduction requires obtaining the data
through the original competition channel.

## Current defaults

The current script defaults are defined in the source, not copied into this
document:

- 9 output classes with label indices 0 through 8.
- Random seed 2026.
- Validation ratio 0.2 for the single-model workflow.
- STFT n_fft 512 and hop 128.
- Target frequency width n_fft / 2 + 1 and target time width 768.
- Cache time width 1536.
- Five folds for the recommended k-fold workflow.
- ResNet34 as the default backbone.

When a run uses non-default values, record the complete command and preserve
the generated config JSON next to the outputs.

## Environment record

At minimum, record:

- operating system and Python version;
- NumPy, SciPy, pandas, scikit-learn, tqdm, torch, and torchvision versions;
- CPU/GPU model, CUDA availability, and AMP setting;
- exact command, model tag, fold count, seed, and data revision;
- Git commit and dirty working-tree state;
- checkpoint, OOF, rule, submission, and validation output paths.

The current training scripts already write configuration JSON, history JSON,
checkpoint metadata, OOF files, and rule JSON. Complete environment and Git
manifest capture is a follow-up improvement; do not describe it as fully
automatic until it is implemented.

## Verification layers

Data-free checks:

    python -m pytest
    python scripts/smoke_test.py
    ruff check src tests scripts
    wairc demo

The synthetic demo writes its generated IQ data, fitted lightweight model,
metrics, and validated submission under `outputs/demo/`. Its exact-match value
only checks deterministic synthetic separability and must not be reported as a
real-data benchmark or competition result.

Data-backed checks:

    python -m src.train_spectrogram_kfold --tag r34 --arch resnet34 --fold 0 --epochs 1 --num-workers 0
    python -m src.search_spectrogram_kfold_thresholds --tags r34
    python -m src.predict_spectrogram_kfold --tags r34
    python -m src.validate_submission outputs/submissions/submission_spectrogram_kfold.txt

The data-backed commands are intentionally manual. They require the
competition dataset, model downloads when pretrained weights are missing,
and enough CPU/GPU resources.

## Compatibility rule

Before changing STFT, labels, fold splitting, thresholds, ensemble weighting,
checkpoint metadata, or submission serialization, compare the old and new
behavior on the same synthetic or saved input. Record shape, dtype, maximum
absolute difference where meaningful, and the resulting submission diff.
