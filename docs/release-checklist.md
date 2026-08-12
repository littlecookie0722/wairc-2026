# Release Checklist

Use this checklist before publishing the first versioned release.

## Maintainer decisions

- [x] Confirm the MIT software license and add the `LICENSE` file.
- [x] Confirm that repository-authored software is released under MIT.
- [x] Keep competition data, checkpoints, and generated outputs outside the
  release artifact.

## Verification

- [ ] Install the project in a clean environment with `pip install -e .`.
- [x] Run `python -m pytest`.
- [x] Run `ruff check src tests scripts`.
- [x] Run `python scripts/smoke_test.py`.
- [x] Run `wairc demo` and validate its generated submission.
- [x] Build the source and wheel distributions and inspect their contents.

## Release metadata

- [x] Update `CHANGELOG.md` with the release date and verified changes.
- [x] Confirm the version in `pyproject.toml` and `src/__init__.py`.
- [x] Confirm CI passes on release-candidate commit `007ac2f` (GitHub Actions
  CI run 31568590301).
- [x] Create and push annotated tag `v0.1.0` for commit `12f70ff`.
- [ ] Publish release notes that distinguish synthetic checks from real-data
  or competition results.

Do not publish a release until every maintainer-decision item is complete.
