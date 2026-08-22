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

## Released: v0.10.0 executable benchmark fixture verification

- [v0.10.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.10.0)
  was released on 2026-08-14 with the following verified
  capability:
- `wairc benchmark verify-fixture` replays either repository-authored
  `benchmark-fixture-v1` manifest and verifies the generated manifest fields,
  report schema, and deterministic signature without requiring competition data
  or trained checkpoints.

## Released: v0.11.0 complete-node robustness baseline

- [v0.11.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.11.0)
  was released on 2026-08-14 with the following verified
  capability:
- `robustness-small` adds an `all-nodes-present` condition that isolates the
  complete three-receiver state from the periodic missing-node baseline. The
  condition uses the existing synthetic missing-node semantics and does not
  change the competition workflow.
- The redistributable robustness fixture records all ten condition controls and
  the updated deterministic signature without generated IQ, model outputs, or
  private labels.

Future benchmark extensions, if needed, require a separately scoped,
evidence-backed proposal; no additional condition is currently part of this
roadmap.

## Released: v0.12.0 reproducible seed utility

[v0.12.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.12.0)
was released on 2026-08-15 with the following verified capabilities:

- The public `wairc_rf.set_reproducible_seed` helper consolidates Python,
  NumPy, and PyTorch CPU/CUDA seed handling.
- Training DataLoader workers initialize Python and NumPy state from the
  PyTorch worker seed without changing fold splitting, loader shuffle policy,
  or competition data behavior.
- Focused regression coverage and reproducibility documentation define the
  deterministic-mode boundary and CUDA limitations.

## Released: v0.13.0 environment doctor

[v0.13.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.13.0)
was released on 2026-08-16 with the following verified capabilities:

- `wairc doctor` reports Python, Torch, Torchvision, CUDA availability, and
  package version without reading competition data or starting training.
- Human-readable diagnostics and `doctor-v1` JSON output are available for
  local checks and automation.
- Structured import failures expose an exception type without copying local
  environment paths or exception messages into the report.

## Released: v0.14.0 aggregated OOF provenance

[v0.14.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.14.0)
was released on 2026-08-22 with the following verified capabilities:

- Rule search writes `oof-aggregate-v1` metadata while preserving the existing
  probability, label, and sample-ID arrays and output filename.
- Mean and tag-weighted outputs record their aggregation method, normalized
  weights, and path-free source filenames without changing their formulas.
- The artifact CLI validates and summarizes versioned aggregate files and
  remains compatible with historical unversioned aggregate outputs.

## Released: v0.15.0 validation prediction provenance

[v0.15.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.15.0)
was released on 2026-08-22 with the following verified capabilities:

- Single-model training writes `validation-predictions-v1` metadata while
  preserving the historical validation probability arrays, dtypes, and output
  filename.
- Validation artifacts record sample IDs, best epoch, selection metric, metric
  value, class count, schema, and artifact type.
- The artifact CLI validates versioned and historical validation predictions
  and checks their class, epoch, and metric linkage to `run-manifest-v1`.

## Released: v0.16.0 validation artifact integrity

[v0.16.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.16.0)
was released on 2026-08-22 with the following verified capabilities:

- `artifact-index-v2` adds content-addressed coverage for single-model
  validation probabilities, including artifact type, schema, byte size, and
  SHA-256 digest validation.
- Single-model training writes v2 indexes, while K-fold training and the
  default finalizer retain exact `artifact-index-v1` coverage.
- `wairc artifact validate-run` accepts v1, v2, and historical no-index
  manifests while rejecting unknown index schemas.

Only the released sections describe completed capability. All other items are
intentions rather than completion claims; use GitHub issues and pull requests
to discuss and implement individual items.
