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

On successful completion, current training entry points also add an optional
`artifact-index-v1` block. It covers only manifest-declared checkpoints, OOF
files, and inference rules, recording each filename, artifact type, schema,
byte size, SHA-256 digest, and fold/tag metadata when present. It does not
record absolute paths. Existing `run-manifest-v1` files without this optional
block remain valid.

After the training index is loaded, each training manifest also records a
`dataset-fingerprint-v1` summary. It uses SHA-256 over normalized index
metadata and the content of each referenced IQ file. The summary records only
aggregate counts, node-presence counts, and digests; it excludes the dataset
root, filenames, sample IDs, and raw IQ values. Moving an unchanged dataset to
another local directory therefore preserves its fingerprint, while changing
index semantics, the train/test label scope, or referenced file content changes
the relevant digest.

## Public dataset adapter

`wairc_rf.CompetitionDatasetAdapter` is an additive, lazy reader for the
competition `index.csv` plus three-node IQ NPZ schema. It returns `RFSample`
objects containing fixed-order `RFNode` values, normalized labels when the
adapter is created with `has_labels=True`, and `labels=None` for public-test
rows. It validates sample-ID uniqueness, node flags, sample rates, NPZ field
types, index/NPZ rate agreement, complete interleaved I/Q pairs, and path
containment. These checks make malformed or ambiguous inputs fail before a
caller converts them into features.

The adapter does not replace the current training dataframe path, alter the
dataset fingerprint algorithm, or reinterpret missing nodes. Existing
`src.data.load_index`, `resolve_iq_path`, and `DroneSpectrogramDataset` entry
points remain supported while callers migrate deliberately.

`SyntheticDatasetAdapter` stores already-generated `RFSample` values in memory
with stable order and unique IDs. It is intended for data-free tests and future
benchmark fixtures; it does not claim to reproduce real RF recordings.

The experimental SigMF parser and recording adapter are read-only. The
synthetic fixture under `tests/fixtures/sigmf/` records the supported datatype,
sample-rate, capture, and annotation fields without adding a raw recording to
the repository. The adapter loads one local recording on sample access and
remains separate from the competition label and submission paths.

Each k-fold OOF file uses the `oof-v1` schema. It retains the historical
`probs`, `labels`, `indices`, `fold`, `sample_ids`, and `metrics` arrays while
adding scalar schema metadata. The rule-search reader accepts legacy
unversioned files, validates both formats, and rejects shape/range errors or
conflicting labels and sample IDs for the same original row. It does not alter
the existing average or tag-weighted probability formulas.

Rule files written by training and OOF search use `rule-v1`. They retain the
selected rule and candidate list, record the nine-class contract, and support
the existing `per_class_thresholds` and `top2_second_threshold` methods. The
shared reader accepts the prior top-level rule shapes and missing-file default,
validates thresholds before prediction, and strips local directory components
from recorded OOF source filenames.

STFT cache files written by `DroneSpectrogramDataset` use `cache-v1` metadata
for the `stft-v1` profile, transform dimensions, node count, tensor shape, and
node mask. A cache with mismatched metadata, invalid arrays, or corrupted bytes
is discarded and recomputed. Older cache files containing only `x` and
`node_mask` remain readable when their shape is compatible.

## Artifact inspection

The unified CLI can inspect or validate an artifact without rewriting it:

    wairc artifact inspect outputs/models/model.pth --json
    wairc artifact validate outputs/oof/oof_fold0.npz --json

The command detects checkpoint, OOF, inference-rule, and STFT cache files,
reuses their existing compatibility loaders, reports the schema and a small
path-redacted summary, and exits non-zero when validation fails. Legacy
unversioned checkpoints, OOF files, rules, and shape-compatible caches are
reported as `legacy-unversioned`; the command does not upgrade or mutate them.

For a completed training output directory, validate the run manifest and its
linked artifacts together:

    wairc artifact validate-run outputs/models/run-manifest.json --json

This read-only check resolves only manifest-declared filenames inside the
manifest directory. It checks that declared outputs exist, linked checkpoint,
OOF, and rule files use the expected artifact types, and that public class,
STFT, fold, and rule `source_files` metadata agree. When an artifact index is
present, it additionally requires exact declared-artifact coverage and checks
the recorded type, schema, fold/tag metadata, byte size, and SHA-256 digest.
Config, history, summary, and other auxiliary outputs are existence-checked
without being interpreted. Absolute paths and parent-directory traversal are
rejected, and the JSON summary contains filenames only.

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

For a machine-readable synthetic check, run:

    wairc benchmark run --profile cpu-smoke

The benchmark writes `benchmark-manifest.json` and `benchmark-report.json`
under `outputs/benchmark/` by default. The manifest captures the generator
version, seed, sample rate, node count, class mapping, noise and missing-node
pattern, and the exact `stft-v1` feature parameters. The report records
synthetic exact-match metrics and a SHA-256 deterministic signature over the
manifest and deterministic metrics; runtime is intentionally not part of that
signature. This verifies the public CPU workflow only and does not estimate
real RF or competition performance.

The controlled `robustness-small` profile uses the same clean training split
for six test conditions: the default baseline, a higher noise standard
deviation of `0.20`, node 0 missing, a `180 Hz` frequency offset, a 32-sample
timing offset, and a `0.5` signal gain. Its report keeps the metrics and
relative artifacts nested under each condition, so the comparison does not
depend on local absolute paths.

For a human-readable review artifact, render the report without copying local
paths into the document:

    wairc benchmark summarize outputs/benchmark/benchmark-report.json

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
