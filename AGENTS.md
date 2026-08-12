# WAIRC-2026 Agent Rules

## Scope

This repository is a Python machine-learning project whose compatibility
boundaries are the competition data schema, STFT transformation, label
mapping, checkpoint metadata, inference rules, ensemble behavior, and
submission text format.

## Before editing

1. Read README.md and the relevant document under docs/.
2. Inspect the current call chain and existing tests.
3. Check git status and preserve unrelated working-tree changes.
4. Do not read or commit files under data_and_code/ or data_and_code_patch-1/.

## High-risk boundaries

Do not silently change:

- NUM_CLASSES or the 0..8 label mapping.
- Interleaved IQ parsing or missing-node semantics.
- STFT defaults, normalization, resize, or crop behavior.
- Fold splitting, threshold search, ensemble weighting, or submission format.
- Checkpoint fields consumed by prediction scripts.

If one of these must change, add a focused regression test and document the
before/after compatibility result.

## Development rules

- Keep existing script entry points working while introducing helpers.
- Prefer small, explicit functions over speculative architecture layers.
- Keep generated models, caches, metrics, and datasets out of git.
- Do not add credentials or private identifiers.
- Do not claim stars, downloads, benchmarks, users, or scores without evidence.
- Do not choose or change the software license without maintainer confirmation.
- Preserve archived_baselines unless the maintainer explicitly requests removal.
- Use English identifiers and concise comments for non-obvious logic.

## Verification

Run the narrowest relevant checks first:

    python -m pytest
    python scripts/smoke_test.py
    ruff check src wairc_rf tests scripts
    python -m compileall -q src wairc_rf archived_baselines

GPU training and full-dataset inference are manual checks. They are not
required for the CPU test suite or GitHub Actions.

