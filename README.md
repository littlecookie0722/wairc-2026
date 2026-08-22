# WAIRC-2026

[![CI](https://github.com/littlecookie0722/wairc-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/littlecookie0722/wairc-2026/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/littlecookie0722/wairc-2026)](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.14.0)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A public research implementation of multi-node RF drone identification from
IQ signals, originating from a 2026 AI radio competition.

This repository documents a real 2026 AI radio competition project. The main
pipeline converts raw interleaved IQ samples into STFT spectrograms and uses
pretrained computer-vision backbones, k-fold training, out-of-fold threshold
search, ensemble inference, and submission validation.

The project is maintained as a reproducible research codebase. It does not
claim production adoption, download volume, leaderboard results, or community
size that are not backed by repository evidence.

The latest public release is [v0.14.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.14.0).

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
- Versioned `run-manifest-v1` provenance records for data-backed training runs.
- Sanitized `dataset-fingerprint-v1` summaries in training manifests.
- Versioned `checkpoint-v1` model metadata with legacy checkpoint loading.
- Versioned `oof-v1` per-fold predictions with legacy-compatible validation.
- Versioned `oof-aggregate-v1` rule-search probabilities with aggregation
  provenance and legacy-compatible validation.
- Versioned `rule-v1` inference rules with legacy-compatible loading.
- Versioned `cache-v1` STFT cache metadata with legacy cache loading.
- `wairc artifact inspect` and `wairc artifact validate` commands for
  machine-readable artifact summaries and compatibility checks.
- `wairc artifact validate-run` for read-only run-manifest linkage checks.
- Optional `artifact-index-v1` references in newly completed training manifests,
  with path-free SHA-256, size, schema, fold, and tag metadata.
- Public `RFNode`/`RFSample` contracts, `SyntheticDatasetAdapter`, and a strict
  competition CSV/NPZ adapter that preserves node order, sample rates, labels,
  and missing nodes.
- Installable `wairc` command-line interface.
- Synthetic CPU end-to-end demonstration that requires no competition data.
- Reproducible `wairc benchmark run --profile cpu-smoke` and controlled
  `robustness-small` checks with path-safe manifest/report files.
- `wairc benchmark summarize` for compact Markdown review summaries from
  machine-readable benchmark reports.
- `wairc doctor` for a lightweight Python, Torch, Torchvision, CUDA, and
  package-version environment check without accessing competition data.
- Public `wairc_rf.set_reproducible_seed` support for shared Python, NumPy, and
  PyTorch CPU/CUDA seed handling, including DataLoader worker initialization.
- Archived early baselines under archived_baselines.

The current public repository is intentionally still a competition-oriented
research project. Engineering work is being added incrementally; see the
architecture and reproducibility documents for current limitations.

## Quick start

### Install

For an editable CPU development environment, first install compatible CPU
builds of torch and torchvision, then install this repository:

    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    python -m pip install -e ".[dev]"

The GPU requirements use the CUDA 12.8 PyTorch index. For a CPU-only
environment, use the command above instead of `requirements-gpu.txt`.

The existing requirements-file workflow remains supported:

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-gpu.txt
    python -m pip install -r requirements-dev.txt

### Run the synthetic end-to-end demo

    wairc demo

This deterministic CPU workflow generates self-contained synthetic multi-node
IQ samples, extracts STFT features, trains a lightweight multi-label
classifier, runs inference, writes a submission, and validates it. Artifacts
are written under `outputs/demo/`.

Example output from a verified default run (paths abbreviated):

```text
Synthetic CPU demo passed
Train samples: 45
Test samples: 14
Synthetic exact-match accuracy: 0.786
Submission: <output-dir>/submission.txt
Metrics: <output-dir>/metrics.json
```

The reported synthetic exact-match accuracy is only a functional check of the
workflow. It is not a competition score or evidence of performance on real RF
data, and it may vary if the seed or dependency versions change.

### Run the synthetic benchmark

    wairc benchmark run --profile cpu-smoke

This writes `benchmark-manifest.json` and `benchmark-report.json` under
`outputs/benchmark/`. The report includes a deterministic signature over the
fixed synthetic inputs and metrics; runtime is recorded separately. See the
[synthetic benchmark guide](docs/benchmark.md) for the profiles, report schema,
and synthetic-only boundary.

List the unified commands with:

    wairc --help

Check the installed runtime without reading competition data or running a
training job:

    wairc doctor
    wairc doctor --json

Inspect a model, OOF, inference-rule, or STFT cache artifact without exposing
its local path in the summary:

    wairc artifact inspect outputs/models/model.pth --json
    wairc artifact validate outputs/rules/best_rule.json
    wairc artifact validate-run outputs/models/run-manifest.json --json

The `validate-run` command checks manifest-declared relative outputs, validates
linked checkpoint/OOF/rule artifacts, and compares their public class, STFT,
fold, and OOF-source metadata. When an `artifact-index-v1` block is present, it
also verifies exact artifact coverage, type, schema, fold/tag metadata, byte
size, and SHA-256 digest. Auxiliary files such as configs and histories are
checked for existence but are not interpreted as model artifacts.

Reusable transforms and label helpers are available from the stable Python
namespace:

```python
import numpy as np

from wairc_rf import STFTConfig, complex_iq_to_spectrogram, iq_to_spectrogram

interleaved_iq = np.fromfile("recording.int16", dtype=np.int16)
config = STFTConfig(n_fft=512, hop=128, target_freq=257)
spectrogram = iq_to_spectrogram(interleaved_iq, sample_rate=125_000_000, config=config)

native_complex_iq = np.fromfile("recording.complex64", dtype=np.complex64)
same_profile = complex_iq_to_spectrogram(
    native_complex_iq,
    sample_rate=125_000_000,
    config=config,
)
```

See the [public Python API](docs/public-api.md) for the `stft-v1` compatibility
contract, complex/interleaved input requirements, and exact-equivalence rule.
The numerical kernel is independent of the competition model and dataset
workflow; existing `src.*` entry points remain compatible wrappers.

### Verify without competition data

    python -m pytest
    python scripts/smoke_test.py
    python scripts/cpu_compatibility.py
    wairc demo
    wairc benchmark run --profile cpu-smoke

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

The repository software is available under the MIT License; see `LICENSE`.
This license does not grant rights to the competition dataset, pretrained
weights, or third-party dependencies. Those assets remain subject to their
original terms.

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
    src/cli.py
    src/synthetic_demo.py
    src/benchmark.py
    docs/architecture.md
    docs/benchmark.md
    docs/reproducibility.md
    archived_baselines/

## Documentation

- [Current STFT competition workflow](docs/冲高分STFT频谱图方案说明.md).
- Dataset and submission format: [Chinese](docs/数据集说明.md) and
  [English](docs/Dataset_Guide_EN.md).
- [Architecture](docs/architecture.md): current module and data-flow boundaries.
- [Public Python API](docs/public-api.md): stable imports and the versioned
  `stft-v1` transform contract.
- [SigMF interoperability](docs/sigmf.md): experimental metadata subset and
  explicit raw-data compatibility boundary.
- [Synthetic benchmark](docs/benchmark.md): reproducible CPU profile and
  machine-readable manifest/report contract.
- [Redistributable benchmark fixture](tests/fixtures/benchmark/synthetic_iq_v1.json):
  repository-authored `cpu-smoke` parameter manifest without raw recordings or
  model data.
- [Robustness benchmark fixture](tests/fixtures/benchmark/synthetic_iq_robustness_v1.json):
  repository-authored ten-condition manifest without generated artifacts.
- `wairc benchmark verify-fixture` replays a redistributable fixture and checks
  its manifest and deterministic report signature.
- [CPU compatibility probe](docs/reproducibility.md): data-free public-transform,
  eager CPU-model, and TorchScript CPU checks across the supported Python matrix.
- [Reproducibility](docs/reproducibility.md): current limits and required records.
- [Release checklist](docs/release-checklist.md): the v0.14.0 release record and
  future release gates.
- [v0.14.0 release notes](docs/releases/v0.14.0.md).
- [v0.13.0 release notes](docs/releases/v0.13.0.md).
- [v0.12.0 release notes](docs/releases/v0.12.0.md).
- [v0.11.0 release notes](docs/releases/v0.11.0.md).
- [v0.10.0 release notes](docs/releases/v0.10.0.md).
- [v0.9.0 release notes](docs/releases/v0.9.0.md).
- [v0.8.0 release notes](docs/releases/v0.8.0.md).
- [v0.7.0 release notes](docs/releases/v0.7.0.md).
- [v0.6.0 release notes](docs/releases/v0.6.0.md).
- [v0.5.0 release notes](docs/releases/v0.5.0.md).
- [v0.4.0 release notes](docs/releases/v0.4.0.md).
- [v0.3.0 release notes](docs/releases/v0.3.0.md).
- [v0.2.0 release notes](docs/releases/v0.2.0.md).
- [v0.1.0 release notes](docs/releases/v0.1.0.md).
- [Roadmap](ROADMAP.md): released foundations and planned interoperability work.
- [Citation metadata](CITATION.cff).

## License

The repository software is licensed under the MIT License. Competition data,
model weights, and third-party dependencies are not relicensed by this
repository.

## Contributing

Read CONTRIBUTING.md and AGENTS.md before changing the pipeline. Changes to
STFT parameters, label mapping, checkpoint metadata, fold splitting,
threshold rules, ensemble weights, or submission format require regression
tests and an explicit compatibility note.

For usage questions, troubleshooting, and guidance on where to ask for help,
see [SUPPORT.md](SUPPORT.md).

## Security and privacy

Do not put API keys, account or organization identifiers, email addresses,
private dataset links, or credentials in the repository. See SECURITY.md for
reporting guidance.
