# Changelog

This project follows a lightweight Keep a Changelog format. Entries describe
verified repository changes; they do not imply a release has been published.

## [Unreleased]

Future unreleased changes will be listed here before the next release.

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
