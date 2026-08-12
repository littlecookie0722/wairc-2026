# Roadmap

WAIRC-2026 is evolving from a competition implementation into a reusable RF
machine-learning research project. Work is prioritized around reproducibility,
clear compatibility boundaries, and workflows that contributors can run
without access to private data.

## 0.1 foundation

- Installable Python project with a unified command-line entry point.
- Synthetic CPU demonstration covering IQ generation, feature extraction,
  training, inference, submission writing, and validation.
- Data-free tests and CI for the stable data, STFT, inference-rule, and
  submission boundaries.
- Architecture, reproducibility, contribution, and security documentation.

## Next

- Capture Git, Python, dependency, device, and command provenance for every
  training run.
- Define a generic public dataset adapter while preserving the competition
  CSV/NPZ adapter.
- Evaluate SigMF import as an optional interoperability path.
- Add deterministic checkpoint compatibility checks.
- Publish the first versioned release after the release checklist and CI are
  complete.

## Later

- Separate reusable RF transformations from competition-specific workflows.
- Add documented benchmark fixtures that can be redistributed legally.
- Expand CPU compatibility tests across supported Python versions.

Roadmap items describe intended work, not released capability. Use GitHub
issues and pull requests to discuss and implement individual items.
