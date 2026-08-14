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

## Released: v0.4.0 stable API, interoperability, and benchmarks

[v0.4.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.4.0)
was released on 2026-08-14 with the following verified capabilities:

- Stable transform and label APIs now coexist with the existing `src.*` module
  entry points.
- Public `RFNode`/`RFSample` contracts, `SyntheticDatasetAdapter`, and a strict
  competition CSV/NPZ adapter now provide an additive generic sequence path
  while preserving the legacy loader and training dataset.
- Experimental SigMF metadata-subset parsing and a synthetic fixture now define
  the supported single-channel complex-recording boundary.
- A read-only `SigMFDatasetAdapter` now maps one supported local recording to
  one unlabeled `RFSample` without changing competition labels or submission
  behavior.
- Deterministic `cpu-smoke` and `robustness-small` synthetic benchmark profiles
  now record fixed generator, data, transform, training, and evaluation inputs
  in `benchmark-manifest-v1`, and write path-safe `benchmark-report-v1` records
  with deterministic signatures. The robustness profile covers higher noise
  and node-0 absence without making real-data claims.

## Released: v0.5.0 benchmark maintenance and robustness

[v0.5.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.5.0)
was released on 2026-08-14 with the following verified capabilities:

- `wairc benchmark summarize` validates `benchmark-report-v1` and renders a
  compact path-safe Markdown summary from machine-readable results.
- `robustness-small` covers controlled noise, missing-node, frequency-offset,
  timing-offset, and signal-gain conditions while keeping synthetic results
  separate from real-data claims.

## Released: v0.6.0 interoperability and CPU reproducibility

- [v0.6.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.6.0)
  was released on 2026-08-14 with the following verified capabilities:
- The reusable `stft-v1` numerical kernel lives independently of model,
  dataset, and training workflows. The existing `src.spectrogram` wrapper and
  the stable `wairc_rf` API share that kernel, with regression coverage for
  frozen output and legacy sample-rate fallback behavior.
- `benchmark-fixture-v1` provides a repository-authored parameter manifest for
  the synthetic `cpu-smoke` profile. It is included in the source distribution
  and contains no raw IQ, external recordings, model weights, or private
  labels; its expected deterministic signature is test-verified.
- The CPU compatibility probe runs public transform equality and a no-weights
  CPU model forward on Python 3.10, 3.11, and 3.12 in CI.

## Released: v0.7.0 robustness and CPU execution compatibility

- [v0.7.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.7.0)
  was released on 2026-08-14 with the following verified capabilities:
- `robustness-small` covers seven controlled synthetic conditions, including a
  deterministic `combined-stress` interaction of noise, frequency offset,
  timing offset, and signal gain. The results remain synthetic-only checks.
- The CPU compatibility probe executes the no-weights model through both eager
  PyTorch and TorchScript on CPU, compares finite logits, and continues to run
  on Python 3.10, 3.11, and 3.12 in CI.

## Released: v0.8.0 redistributable robustness fixture

- [v0.8.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.8.0)
  was released on 2026-08-14 with the following verified
  capability:
- A repository-authored `benchmark-fixture-v1` manifest covers all seven
  `robustness-small` conditions and pins the expected deterministic report
  signature without distributing generated IQ, model outputs, or private
  labels.

## Released: v0.9.0 receiver-specific robustness coverage

- [v0.9.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.9.0)
  was released on 2026-08-14 with the following capability:
- The `robustness-small` profile adds dedicated `node1-missing` and
  `node2-missing` conditions to the existing protocol. The redistributable
  parameter manifest and deterministic signature remain synthetic-only and do
  not change the competition workflow.

## Later: public interoperability and benchmarks

- Extend the robustness protocol with additional evidence-backed conditions or
  redistributable fixtures without changing the competition workflow.

Only the released sections describe completed capability. All other items are
intentions rather than completion claims; use GitHub issues and pull requests
to discuss and implement individual items.
