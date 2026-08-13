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

The training scripts write configuration JSON, history JSON, checkpoint
metadata, OOF files, and rule JSON. Model checkpoints written by the current
training scripts use the `checkpoint-v1` schema and retain the original flat
metadata fields for compatibility. The prediction entry points accept both
this schema and older unversioned checkpoint dictionaries, normalizing the
older case as `legacy-unversioned`. They now also write a `run-manifest.json`
(or a tag-specific `run-manifest_<tag>.json` for k-fold runs) using the
`run-manifest-v1` schema. The manifest records a Git commit and dirty state,
selected runtime versions, device information, transform/model/training
parameters, the sanitized command arguments, and output filenames. It is
finalized as `completed` or `failed` so an interrupted run is distinguishable
from a finished run.

After the training index is loaded, each training manifest also records a
`dataset-fingerprint-v1` summary. It uses SHA-256 over normalized index
metadata and the content of each referenced IQ file. The summary records only
aggregate counts, node-presence counts, and digests; it excludes the dataset
root, filenames, sample IDs, and raw IQ values. Moving an unchanged dataset to
another local directory therefore preserves its fingerprint, while changing
index semantics, the train/test label scope, or referenced file content changes
the relevant digest.

Each k-fold OOF file uses the `oof-v1` schema. It retains the historical
`probs`, `labels`, `indices`, `fold`, `sample_ids`, and `metrics` arrays while
adding scalar schema metadata. The rule-search reader accepts legacy
unversioned files, validates both formats, and rejects shape/range errors or
conflicting labels and sample IDs for the same original row. It does not alter
the existing average or tag-weighted probability formulas.

The `checkpoint-v1` envelope requires the model state dictionary, architecture,
class count, positive STFT dimensions, and `stftProfile: "stft-v1"`. Prediction
fails before model construction when these fields are missing or incompatible.
Legacy unversioned checkpoints keep the original fallback behavior for optional
metadata, and are labeled as legacy in memory only; loading them does not
rewrite the source file.

The manifest deliberately excludes absolute input/output paths, usernames,
secrets, and raw dataset identifiers. A Git commit of `unknown` or a dirty
state may occur when Git is unavailable or the run starts from a modified
working tree. Seed capture does not promise bit-for-bit determinism across all
CUDA operators and hardware.

## Verification layers

Data-free checks:

    python -m pytest
    python scripts/smoke_test.py
    ruff check src wairc_rf tests scripts
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
