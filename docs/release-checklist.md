# Release Checklist

This document records the `v0.5.0`, `v0.4.0`, `v0.3.0`, and `v0.2.0` release gates. A
checked item is backed by local or GitHub verification; it is not a claim about
real-data performance.

## Maintainer decisions

- [x] Repository-authored software remains under the MIT License.
- [x] Competition data, model weights, generated caches, private labels, and
  generated outputs are outside the release artifact.
- [x] The release preserves the 0..8 label mapping, interleaved IQ parsing,
  missing-node semantics, `stft-v1`, fold behavior, inference rules, ensemble
  weighting, checkpoint compatibility, and submission format.

## v0.5.0 verification

- [x] `python -m pytest` passes locally with 152 tests.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, both synthetic benchmark profiles,
  the summary command, and the synthetic demo pass locally.
- [x] Source and wheel distributions build successfully and include the v0.5.0
  release documentation.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, `wairc --help`, the demo, and summary command
  pass there. The benchmark profiles pass in the release matrix.
- [x] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit and tag, including package and isolated-wheel checks on Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.5.0 release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.5.0`.
- [x] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.5.0.md` are prepared.
- [x] README and `ROADMAP.md` release text and links are prepared for
  `v0.5.0` publication.
- [x] Annotated tag `v0.5.0` points to verified release commit
  `4d4831e1a762f26bf4c870d15480ca34e0407409`.
- [x] GitHub release [`v0.5.0`](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.5.0)
  is published with verified notes, wheel/sdist
  assets, and no private assets.

The tag CI run was `31771142148`. Published assets were downloaded again and
matched these SHA-256 digests:

- `wairc_rf-0.5.0-py3-none-any.whl`:
  `f7a519129f3e283d0b1d46e038103721b749aab3779cc94115b3190ebfe8441e`
- `wairc_rf-0.5.0.tar.gz`:
  `65a4ca52d2c15cda2398145bcfe0bebef406abca979c2f3ad239b260b0332ceb`

## v0.4.0 verification

- [x] `python -m pytest` passes locally with 151 tests.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, both synthetic benchmark profiles,
  and the synthetic demo pass locally.
- [x] Source and wheel distributions build successfully and include the public
  benchmark documentation.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, `wairc --help`, and the synthetic demo pass
  there. The two benchmark profiles pass in the release matrix.
- [x] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit and tag, including package and isolated-wheel checks on Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.4.0 release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.4.0`.
- [x] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.4.0.md` are prepared.
- [x] README and `ROADMAP.md` release text and links are prepared for
  `v0.4.0` publication.
- [x] Annotated tag `v0.4.0` points to verified release commit `0390fc36`.
- [x] GitHub release `v0.4.0` is published with verified notes, wheel/sdist
  assets, and no private assets.

The tag CI run was `31769714221`. Published assets were downloaded again and
matched these SHA-256 digests:

- `wairc_rf-0.4.0-py3-none-any.whl`:
  `4a4b047e714c57f60a68dcb33b8b072e2616a9d8d4cc8d0c9c570088a7eb3d79`
- `wairc_rf-0.4.0.tar.gz`:
  `1ecfc49f3e7b2061bec0496509d265b638c65f897e673fdac0f72c66a9498c`

## v0.3.0 verification

- [x] `python -m pytest` passes locally with 101 tests.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, and the synthetic demo pass locally.
- [x] Source and wheel distributions build successfully.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, `wairc --help`, and `wairc demo` pass there.
- [x] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit and tag, including package and isolated-wheel checks on Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.3.0 release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.3.0`.
- [x] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.3.0.md` are prepared.
- [x] README and `ROADMAP.md` release text and links are prepared for
  `v0.3.0` publication.
- [x] Annotated tag `v0.3.0` points to verified release commit `ae158b68`.
- [x] [GitHub release v0.3.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.3.0)
  is published with verified notes, wheel/sdist assets, and no private assets.

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
