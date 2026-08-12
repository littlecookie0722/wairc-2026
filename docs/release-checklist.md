# Release Checklist

This document records the v0.1.0 release state and provides a baseline for
future releases. WAIRC-2026 v0.1.0 was published on 2026-08-12; unchecked items
below are explicit post-release follow-up work, not claims that the release is
still pending.

## Maintainer decisions

- [x] Confirm the MIT software license and add the `LICENSE` file.
- [x] Confirm that repository-authored software is released under MIT.
- [x] Keep competition data, checkpoints, and generated outputs outside the
  release artifact.

## Verification

- [ ] Install the built wheel in a clean environment with runtime dependencies,
  then run `wairc --version`, `wairc --help`, `wairc demo`, and `pip check`.
  This clean-wheel verification remains pending after v0.1.0.
- [x] Run `python -m pytest`.
- [x] Run `ruff check src tests scripts` for v0.1.0. Future releases also
  include the stable `wairc_rf` package in this check.
- [x] Run `python scripts/smoke_test.py`.
- [x] Run `wairc demo` and validate its generated submission.
- [x] Build the source and wheel distributions and inspect their contents.

## Release metadata

- [x] Update `CHANGELOG.md` with the release date and verified changes.
- [x] Confirm the version in `pyproject.toml` and `src/__init__.py`.
- [x] Confirm CI passes on release-candidate commit `007ac2f` (GitHub Actions
  CI run 31568590301).
- [x] Create and push annotated tag `v0.1.0` for commit `12f70ff`.
- [x] Publish [v0.1.0 release notes](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.1.0)
  that distinguish synthetic checks from real-data or competition results.

For future releases, complete every maintainer-decision item and document any
remaining verification gap before publishing.
