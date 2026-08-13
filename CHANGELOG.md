# Changelog

This project follows a lightweight Keep a Changelog format. Entries describe
verified repository changes; they do not imply a release has been published.

## [Unreleased]

Future changes will be listed here before the next release.

### Added

- `wairc artifact inspect` and `wairc artifact validate` commands with JSON
  summaries, legacy compatibility reporting, and non-zero failure status for
  invalid checkpoint, OOF, inference-rule, and STFT cache artifacts.

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
