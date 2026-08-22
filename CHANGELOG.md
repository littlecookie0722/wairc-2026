# Changelog

This project follows a lightweight Keep a Changelog format. Entries describe
verified repository changes; they do not imply a release has been published.

## [Unreleased]

### Added

- `artifact-index-v2` can content-address single-model validation probability
  artifacts in addition to checkpoints, OOF files, and inference rules.
- Single-model training now writes v2 indexes so `validate-run` detects size or
  SHA-256 changes to `best_val_probs.npz`.

### Compatibility

- `artifact-index-v1` keeps its exact historical coverage, K-fold training
  continues to write v1, and manifests with v1 or no artifact index remain
  valid. Training, validation probabilities, inference, and submission behavior
  are unchanged.

## [0.15.0] - 2026-08-22

### Added

- Single-model training now writes `validation-predictions-v1` metadata with
  sample IDs, best epoch, selection metric, and metric value while preserving
  the historical validation probability arrays and filename.
- Artifact inspection and run-manifest validation recognize versioned and
  historical validation prediction files.

### Compatibility

- Training, validation splitting, probability values, checkpoint selection,
  threshold search, and submission behavior are unchanged. Historical files
  containing only `probs` and `labels` remain readable, and
  `artifact-index-v1` coverage is unchanged.

## [0.14.0] - 2026-08-22

### Added

- Rule search now writes `oof-aggregate-v1` metadata alongside its historical
  probability, label, and sample-ID arrays. The artifact records mean or
  tag-weighted aggregation, normalized weights, and path-free source filenames.
- `wairc artifact inspect|validate` recognizes both versioned and historical
  aggregate OOF probability files.

### Compatibility

- The aggregate output filename, arrays, dtypes, average and weighted formulas,
  threshold search, inference rules, and submission behavior are unchanged.

## [0.13.0] - 2026-08-16

### Added

- `wairc doctor` checks Python, Torch, Torchvision, CUDA availability, and
  package version without reading competition data or starting training.
- The command provides human-readable diagnostics and a machine-readable
  `doctor-v1` JSON report for automation.

### Compatibility

- Doctor diagnostics are read-only and do not change the competition data
  schema, label mapping, STFT behavior, fold splitting, inference rules,
  checkpoint fields, or submission format.
- Import failures expose only their exception type in structured output; local
  environment paths and exception messages are not included.

## [0.12.0] - 2026-08-15

### Added

- A shared `wairc_rf.set_reproducible_seed` helper seeds Python, NumPy, and
  PyTorch CPU/CUDA generators, while legacy training and inference helpers keep
  their existing entry points. Training DataLoader workers now initialize
  Python and NumPy from PyTorch's worker seed.

### Compatibility

- Training keeps the existing cuDNN benchmark preference by default, while
  k-fold inference requests deterministic cuDNN selection through the shared
  helper.
- Fold split inputs, loader shuffle policy, STFT behavior, labels, checkpoint
  fields, inference rules, and submission serialization are unchanged.
- Seed handling remains a reproducibility aid and does not promise bit-for-bit
  equality across all CUDA operators, devices, or dependency versions.

## [0.11.0] - 2026-08-14

### Added

- `robustness-small` now includes an explicit `all-nodes-present` control
  alongside the periodic missing-node baseline and dedicated single-node
  absence conditions.
- The redistributable robustness fixture pins the ten-condition manifest and
  its new deterministic report signature without including generated artifacts.

### Compatibility

- The existing `cpu-smoke` profile, prior robustness conditions, competition
  data path, `stft-v1`, labels, training/inference entry points, checkpoint
  fields, and submission format remain unchanged.
- The new condition and signature remain synthetic-only checks; they do not
  claim real-data robustness or competition performance.

## [0.10.0] - 2026-08-14

### Added

- `wairc benchmark verify-fixture` replays a repository-authored
  `benchmark-fixture-v1` manifest and verifies its generated manifest fields,
  report schema, and deterministic SHA-256 signature.
- CI now replays both redistributable synthetic benchmark fixtures in the
  quality job without requiring competition data or trained checkpoints.

### Compatibility

- The command writes generated benchmark artifacts only to the selected local
  output directory and does not change competition data loading, STFT,
  labels, training/inference entry points, checkpoints, or submission format.
- Fixture verification remains a synthetic reproducibility check; it does not
  claim real-data benchmark results or competition performance.

## [0.9.0] - 2026-08-14

### Added

- `robustness-small` now includes dedicated `node1-missing` and
  `node2-missing` conditions, completing receiver-specific systematic
  missing-node coverage while retaining all existing conditions.

### Compatibility

- The existing `cpu-smoke` profile, seven prior robustness condition names and
  parameters, competition data path, `stft-v1`, labels, training/inference
  entry points, checkpoint fields, and submission format remain unchanged.
- The additional conditions and updated fixture remain synthetic-only checks;
  they do not claim real-data robustness or competition performance.

## [0.8.0] - 2026-08-14

### Added

- A redistributable `benchmark-fixture-v1` parameter manifest now covers the
  seven-condition `robustness-small` profile, including its expected report
  signature. It contains no generated IQ, model outputs, or private labels.

### Compatibility

- The existing `cpu-smoke` fixture and benchmark profile remain unchanged.
- The new fixture is an input contract only; generated benchmark artifacts,
  competition data, checkpoints, and private labels remain outside the source
  distribution and wheel.

## [0.7.0] - 2026-08-14

### Added

- `robustness-small` now includes a deterministic `combined-stress` condition
  that applies the documented noise, frequency-offset, timing-offset, and
  signal-gain controls together. The condition remains a synthetic interaction
  check and does not claim real-data robustness.
- The data-free CPU compatibility probe now executes the no-weights model
  through TorchScript and compares its finite logits with the eager CPU path;
  the competition training and inference entry points are unchanged.

### Compatibility

- `robustness-small` keeps the existing `cpu-smoke` profile and the six prior
  condition names and parameters; `combined-stress` is an additive synthetic
  condition.
- The probe-only TorchScript check does not change the competition training or
  inference entry points, checkpoint fields, or serialized submissions.

## [0.6.0] - 2026-08-14

### Added

- A data-free CPU compatibility probe checks public interleaved/native-complex
  transform equality and a no-weights CPU model forward on Python 3.10, 3.11,
  and 3.12 in CI.
- A `benchmark-fixture-v1` parameter manifest documents the synthetic
  `cpu-smoke` profile, its repository-authored provenance, and its expected
  deterministic signature. The fixture contains no raw IQ, external recording,
  model weight, or private-label data.

### Changed

- The reusable `stft-v1` numerical kernel is isolated from model, dataset,
  and training code. The legacy `src.spectrogram` entry point reuses the same
  kernel while retaining its historical non-positive sample-rate fallback.

## [0.5.0] - 2026-08-14

### Added

- `wairc benchmark summarize` validates `benchmark-report-v1` and renders a
  compact path-safe Markdown summary for CPU and robustness profiles.
- `robustness-small` now includes controlled frequency-offset, timing-offset,
  and signal-gain conditions in addition to noise and missing-node checks.

### Compatibility

- Existing interleaved-IQ transforms, `src.*` entry points, STFT defaults, and
  competition training/inference behavior are unchanged.
- Benchmark summaries validate report fields and reject absolute or parent-
  traversal paths without changing generated data or model behavior.

## [0.4.0] - 2026-08-14

### Added

- Stable `complex_iq_to_spectrogram` API for one-dimensional complex IQ arrays,
  with exact `stft-v1` equivalence to the corresponding float32 interleaved
  representation.
- Public `RFNode`/`RFSample` contracts, the `RFDatasetAdapter` sequence
  protocol, `SyntheticDatasetAdapter`, and a strict lazy
  `CompetitionDatasetAdapter` for the existing three-node CSV/NPZ schema.
- Experimental metadata-only SigMF subset parser and a synthetic fixture for
  single-channel complex recordings.
- Read-only `SigMFDatasetAdapter` support for one local single-channel recording
  with supported complex integer or complex float IQ datatypes.
- Reproducible `wairc benchmark run --profile cpu-smoke` with
  `benchmark-manifest-v1` inputs and `benchmark-report-v1` results.
- Controlled `robustness-small` benchmark conditions for higher noise and
  node-0 absence, with macro-F1 and per-class-recall metrics.

### Compatibility

- Existing interleaved-IQ transforms, `src.*` entry points, STFT defaults, and
  competition training/inference behavior are unchanged.
- The adapter preserves sample order, node order, sample-rate metadata, the
  0..8 label mapping, and empty-IQ/zero-rate missing-node semantics. It is an
  additive public path and does not rewrite legacy dataset files.
- The synthetic benchmark uses the existing CPU demo path and reports only
  synthetic functional metrics; it does not claim real-data or competition
  performance.

## [0.3.0] - 2026-08-14

### Added

- `wairc artifact inspect` and `wairc artifact validate` commands with JSON
  summaries, legacy compatibility reporting, and non-zero failure status for
  invalid checkpoint, OOF, inference-rule, and STFT cache artifacts.
- `wairc artifact validate-run` for read-only manifest output, artifact type,
  class/STFT/fold, and rule-to-OOF linkage checks with path-safe summaries.
- Optional `artifact-index-v1` metadata for newly completed training runs,
  recording path-free SHA-256, size, type, schema, fold, and tag references for
  declared checkpoint, OOF, and rule artifacts.

### Compatibility

- Existing `run-manifest-v1` files without an artifact index remain valid, and
  training, STFT, label, fold, inference, ensemble, checkpoint-consumer, and
  submission behavior are unchanged.

## [0.2.0] - 2026-08-13

### Added

- Versioned `cache-v1` STFT cache metadata with transform/shape validation and
  legacy cache loading.
- Versioned `rule-v1` inference-rule artifacts with legacy-compatible loading,
  threshold validation, and local source-filename sanitization.
- Versioned `oof-v1` per-fold prediction artifacts with legacy-compatible
  loading and validation of shapes, ranges, row identities, and cross-file
  label/sample-ID consistency.
- Sanitized `dataset-fingerprint-v1` summaries for training manifests, covering
  normalized index metadata and referenced IQ file content without recording
  local paths or raw sample data.
- Versioned `checkpoint-v1` metadata for newly written model checkpoints and a
  shared loader that accepts legacy unversioned checkpoint dictionaries.
- Versioned `run-manifest-v1` provenance records for single-model and k-fold
  training runs, including sanitized Git, runtime, transform, model, and
  output metadata.
- Stable `wairc_rf` Python namespace with a versioned `STFTConfig` and validated
  label helpers.
- Exact compatibility tests and documentation for the released `stft-v1`
  transform behavior.
- Citation metadata, support guidance, and a project code of conduct.

### Compatibility

- Existing `src.*` entry points, legacy unversioned checkpoints, OOF files,
  inference rules, and STFT caches remain supported where their validation
  rules allow.
- The 0..8 label mapping, interleaved IQ parsing, missing-node semantics, STFT
  behavior, fold splitting, inference rules, ensemble weighting, and submission
  text format are unchanged.

### Changed

- CI now tests Python 3.10, 3.11, and 3.12, verifies the built wheel from an
  isolated directory, and checks that the source distribution carries its
  public API documentation.
- Contributor setup is CPU-first, with CUDA dependencies documented as
  optional.
- Architecture, roadmap, release, and README documentation now describe the
  artifact governance milestone and its remaining interoperability work.

## [0.1.0] - 2026-08-12

### Added

- Source-grounded architecture and reproducibility documents.
- CPU synthetic regression tests and a data-free smoke test.
- Installable `wairc` CLI and a synthetic IQ end-to-end CPU demonstration.
- Public roadmap and first-release checklist.
- Contributor, security, agent, issue, pull request, and CI guidance.
- MIT license for the repository software.

### Changed

- README now describes the current STFT/k-fold pipeline and its limitations.
- IDE metadata and generated submissions are excluded from source tracking.

### Fixed

- Submission validation now rejects boolean values as malformed binary
  predictions.
