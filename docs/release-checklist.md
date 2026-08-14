# Release Checklist

This document records the current `v0.3.0` release candidate and the completed
`v0.2.0` gate. A checked item is backed by local or GitHub verification; it is
not a claim about real-data performance.

## Maintainer decisions

- [x] Repository-authored software remains under the MIT License.
- [x] Competition data, model weights, generated caches, private labels, and
  generated outputs are outside the release artifact.
- [x] The release preserves the 0..8 label mapping, interleaved IQ parsing,
  missing-node semantics, `stft-v1`, fold behavior, inference rules, ensemble
  weighting, checkpoint compatibility, and submission format.

## v0.3.0 verification

- [x] `python -m pytest` passes locally with 101 tests.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, and the synthetic demo pass locally.
- [x] Source and wheel distributions build successfully.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, `wairc --help`, and `wairc demo` pass there.
- [ ] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit, including package and isolated-wheel checks on Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.3.0 release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.3.0`.
- [x] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.3.0.md` are prepared.
- [x] README and `ROADMAP.md` release text and links are prepared for
  `v0.3.0` publication.
- [ ] An annotated `v0.3.0` tag points to the verified release commit.
- [ ] GitHub release `v0.3.0` is published with verified notes and no private
  assets.

## Previous v0.2.0 release record

### Verification

- [x] `python -m pytest` passes locally.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, and the synthetic demo pass locally.
- [x] Source and wheel distributions build successfully.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, `wairc --help`, and `wairc demo` pass there.
- [x] GitHub Actions passes on Python 3.10, 3.11, and 3.12, including package
  and isolated-wheel checks on Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

### Release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.2.0`.
- [x] `CHANGELOG.md`, `ROADMAP.md`, README links, migration boundaries, and
  `docs/releases/v0.2.0.md` are updated.
- [x] Annotated tag `v0.2.0` is created and pushed for the verified release
  commit `dfd7895`.
- [x] [GitHub release v0.2.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.2.0)
  distinguishes synthetic checks from real-data or competition results and
  lists remaining follow-up work.

## Future release gate

For future releases, repeat every maintainer-decision and verification item,
record any remaining gap, and do not publish capabilities that are still only
planned in `ROADMAP.md`.
