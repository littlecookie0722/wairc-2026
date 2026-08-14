# Roadmap

WAIRC-2026 is evolving from a competition implementation into a reusable RF
machine-learning research project. Work is prioritized around reproducibility,
clear compatibility boundaries, and workflows that contributors can run
without access to private data.

## Released: v0.3.0 artifact inspection and integrity

[v0.3.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.3.0)
was released on 2026-08-14 with the following verified capabilities:

- Path-redacted inspection and validation for supported checkpoint, OOF,
  inference-rule, and STFT cache artifacts.
- Read-only run-manifest validation across declared outputs, artifact types,
  class/STFT/fold metadata, and rule-to-OOF source linkage.
- Optional `artifact-index-v1` references with exact output coverage, SHA-256,
  byte-size, schema, fold, and tag validation.
- Backward compatibility for supported legacy artifacts and
  `run-manifest-v1` files without an artifact index.

## Released: v0.2.0 reproducible artifact foundation

[v0.2.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.2.0)
was released on 2026-08-13 with the following verified capabilities:

- `run-manifest-v1` provenance records with sanitized runtime, command, Git,
  model, transform, and output metadata.
- `dataset-fingerprint-v1` summaries that link a run to normalized dataset
  content without recording local paths or raw samples.
- Legacy-compatible `checkpoint-v1`, `oof-v1`, `rule-v1`, and `cache-v1`
  artifact schemas with focused validation and regression coverage.
- Python 3.10, 3.11, and 3.12 CI, source/wheel builds, isolated wheel checks,
  synthetic CPU checks, and documented compatibility boundaries.

## Released: v0.1.0 foundation

[v0.1.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.1.0)
was released on 2026-08-12 with the following foundations:

- Installable Python project with a unified command-line entry point.
- Synthetic CPU demonstration covering IQ generation, feature extraction,
  training, inference, submission writing, and validation.
- Data-free tests and CI for the stable data, STFT, inference-rule, and
  submission boundaries.
- Architecture, reproducibility, contribution, and security documentation.

## Unreleased: stable API and data interoperability

- Expand the initial stable transform and label API while preserving the
  existing `src.*` module entry points during migration.
- Public `RFNode`/`RFSample` contracts, `SyntheticDatasetAdapter`, and a strict
  competition CSV/NPZ adapter now provide an additive generic sequence path
  while preserving the legacy loader and training dataset.

## Next: broader public interoperability and benchmarks

- Evaluate SigMF import as an optional interoperability path.

## Later: public interoperability and benchmarks

- Separate reusable RF transformations from competition-specific workflows.
- Add documented benchmark fixtures that can be redistributed legally.
- Expand CPU compatibility tests across supported Python versions.
- Add robustness protocols for signal-to-noise ratio, frequency offset,
  timing offset, gain changes, and missing receiver nodes.

Only the released sections describe completed capability. All other items are
intentions rather than completion claims; use GitHub issues and pull requests
to discuss and implement individual items.
