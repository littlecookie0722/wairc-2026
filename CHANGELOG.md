# Changelog

This project follows a lightweight Keep a Changelog format. Entries describe
verified repository changes; they do not imply a release has been published.

## [Unreleased]

### Added

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

### Changed

- CI now tests Python 3.10, 3.11, and 3.12, verifies the built wheel from an
  isolated directory, and checks that the source distribution carries its
  public API documentation.
- Contributor setup is CPU-first, with CUDA dependencies documented as
  optional.
- Architecture, roadmap, release, and README documentation now reflect the
  published v0.1.0 state.

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
