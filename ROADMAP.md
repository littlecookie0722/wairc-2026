# Roadmap

WAIRC-2026 is evolving from a competition implementation into a reusable RF
machine-learning research project. Work is prioritized around reproducibility,
clear compatibility boundaries, and workflows that contributors can run
without access to private data.

## Released: v0.1.0 foundation

[v0.1.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.1.0)
was released on 2026-08-12 with the following foundations:

- Installable Python project with a unified command-line entry point.
- Synthetic CPU demonstration covering IQ generation, feature extraction,
  training, inference, submission writing, and validation.
- Data-free tests and CI for the stable data, STFT, inference-rule, and
  submission boundaries.
- Architecture, reproducibility, contribution, and security documentation.

## Next: reproducible artifacts and stable interfaces

- Capture Git, Python, dependency, device, and command provenance for every
  training run.
- Version checkpoint, OOF, rule, and cache metadata, and add deterministic
  compatibility checks before artifacts are consumed.
- Expand the initial stable transform and label API while preserving the
  existing `src.*` module entry points during migration.
- Define a generic public dataset adapter while preserving the competition
  CSV/NPZ adapter.
- Evaluate SigMF import as an optional interoperability path.

## Later: public interoperability and benchmarks

- Separate reusable RF transformations from competition-specific workflows.
- Add documented benchmark fixtures that can be redistributed legally.
- Expand CPU compatibility tests across supported Python versions.
- Add robustness protocols for signal-to-noise ratio, frequency offset,
  timing offset, gain changes, and missing receiver nodes.

Only the first section describes released capability. All other items are
intentions rather than completion claims; use GitHub issues and pull requests
to discuss and implement individual items.
