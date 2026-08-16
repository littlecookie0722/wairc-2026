# Release Checklist

This document records the `v0.13.0`, `v0.12.0`, `v0.11.0`, `v0.10.0`, `v0.9.0`, `v0.8.0`, `v0.7.0`, `v0.6.0`, `v0.5.0`, `v0.4.0`, `v0.3.0`, and `v0.2.0` release gates. A
checked item is backed by local or GitHub verification; it is not a claim about
real-data performance.

## Maintainer decisions

- [x] Repository-authored software remains under the MIT License.
- [x] Competition data, model weights, generated caches, private labels, and
  generated outputs are outside the release artifact.
- [x] The release preserves the 0..8 label mapping, interleaved IQ parsing,
  missing-node semantics, `stft-v1`, fold behavior, inference rules, ensemble
  weighting, checkpoint compatibility, and submission format.

## v0.13.0 verification

- [ ] `python -m pytest` passes locally with 171 tests.
- [ ] `python scripts/smoke_test.py` passes locally.
- [ ] `python scripts/cpu_compatibility.py` passes locally, including the
  eager/TorchScript CPU comparison.
- [ ] `ruff check src wairc_rf tests scripts` passes locally.
- [ ] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [ ] `wairc --version`, `wairc --help`, human-readable and JSON doctor checks,
  both synthetic benchmark profiles, both fixture verification commands, the
  summary command, and the synthetic demo pass locally.
- [ ] Source and wheel distributions build successfully, include the v0.13.0
  release documentation, and include both benchmark fixtures in the source
  distribution.
- [ ] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, both doctor output modes, the demo, summary
  command, both fixture verification commands, and CPU compatibility probe pass
  there. Both synthetic profiles pass in the release matrix.
- [ ] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit and tag, including the doctor command, package, source documentation,
  and isolated-wheel checks on Python 3.12.
- [ ] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.13.0 release metadata

- [ ] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.13.0`.
- [ ] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.13.0.md` are prepared.
- [ ] README and `ROADMAP.md` release text and links are prepared for
  `v0.13.0` publication.
- [ ] Annotated tag `v0.13.0` points to the verified release commit.
- [ ] GitHub release `v0.13.0` is published with verified notes, wheel/sdist
  assets, and no private assets.

## v0.12.0 verification

- [x] `python -m pytest` passes locally with 169 tests.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `python scripts/cpu_compatibility.py` passes locally, including the
  eager/TorchScript CPU comparison.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, both synthetic benchmark profiles,
  both `verify-fixture` commands, the summary command, and the synthetic demo
  pass locally.
- [x] Source and wheel distributions build successfully, include the v0.12.0
  release documentation, and include both benchmark fixtures in the source
  distribution.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, the public seed helper, `wairc --version`, the demo, summary
  command, both fixture verification commands, and CPU compatibility probe pass
  there. Both synthetic profiles pass in the release matrix.
- [x] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit and tag, including the public seed helper, package, source
  documentation, and isolated-wheel checks on Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.12.0 release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.12.0`.
- [x] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.12.0.md` are prepared.
- [x] README and `ROADMAP.md` release text and links are prepared for
  `v0.12.0` publication.
- [x] Annotated tag `v0.12.0` points to the verified release commit
  `7add45d57769ffc5fa59868d77687e47a1182063`; tag CI run
  [`31883739671`](https://github.com/littlecookie0722/wairc-2026/actions/runs/31883739671)
  passed.
- [x] GitHub release [`v0.12.0`](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.12.0)
  is published with verified notes, wheel/sdist assets, and no private assets.
  The wheel SHA-256 is
  `6da2a1d1afb5c254307ccea26e7d5dbeb4d71e78ed433fd1e49d5c9b8fdc2d48`; the
  source distribution SHA-256 is
  `05cd264340b8843ad114b0dd290c6d39aabdd41ef63a043f61e5bf99721389ef`.

## v0.11.0 verification

- [x] `python -m pytest` passes locally with 161 tests.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `python scripts/cpu_compatibility.py` passes locally, including the
  eager/TorchScript CPU comparison.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, both synthetic benchmark profiles,
  both `verify-fixture` commands, the summary command, and the synthetic demo
  pass locally.
- [x] Source and wheel distributions build successfully, include the v0.11.0
  release documentation, and include both benchmark fixtures in the source
  distribution.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, the demo, summary command, both fixture
  verification commands, and CPU compatibility probe pass there. The
  ten-condition robustness profile passes in the release matrix.
- [x] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit and tag, including the ten-condition robustness fixture, package,
  source documentation, and isolated-wheel checks on Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.11.0 release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.11.0`.
- [x] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.11.0.md` are prepared.
- [x] README and `ROADMAP.md` release text and links are updated for the
  `v0.11.0` release.
- [x] Annotated tag `v0.11.0` points to the verified release commit
  `b3038dd47347f5d6c7ce164cffc4267d16d0a253`; tag CI run `31784155351`
  completed successfully.
- [x] [GitHub release v0.11.0](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.11.0)
  is published with verified notes, wheel/sdist assets, and no private assets.
  The wheel SHA-256 is
  `d6eaed941d520efe5fd92f85745b75a8808be57d755f0b913a946bc4fb328956`; the
  sdist SHA-256 is
  `be0fdfba2d743566491e75fa12cb0bf25f8a196cc473d941848322c119d4bb80`.

## v0.10.0 verification

- [x] `python -m pytest` passes locally with 161 tests.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `python scripts/cpu_compatibility.py` passes locally, including the
  eager/TorchScript CPU comparison.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, both synthetic benchmark profiles,
  both `verify-fixture` commands, the summary command, and the synthetic demo
  pass locally.
- [x] Source and wheel distributions build successfully, include the v0.10.0
  release documentation, and include both benchmark fixtures in the source
  distribution.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, the demo, summary command, both fixture
  verification commands, and CPU compatibility probe pass there. Both
  benchmark profiles pass in the release matrix.
- [x] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit and tag, including both fixture verification, package, source
  documentation, and isolated-wheel checks on Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.10.0 release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.10.0`.
- [x] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.10.0.md` are prepared.
- [x] README and `ROADMAP.md` release text and links are prepared for
  `v0.10.0` publication.
- [x] Annotated tag `v0.10.0` points to the verified release commit
  `31d8acd2f57023f514648f61cb4c4bcb99093b19`; tag CI run `31781923226`
  completed successfully.
- [x] GitHub release [`v0.10.0`](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.10.0)
  is published with verified notes, wheel/sdist assets, and no private assets.
  The wheel SHA-256 is
  `04b6fe04bd17030cea08f93b87bca5304b49c0fb67dbfddcb4565f1bd5de7694`; the
  sdist SHA-256 is
  `3791638cb7a0e43446e37d7fc89eee3495fcd99083345cf8222f0e97251f7107`.

## v0.9.0 verification

- [x] `python -m pytest` passes locally with 157 tests.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `python scripts/cpu_compatibility.py` passes locally, including the
  eager/TorchScript CPU comparison.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, both synthetic benchmark profiles,
  the summary command, and the synthetic demo pass locally.
- [x] Source and wheel distributions build successfully, include the v0.9.0
  release documentation, and include both benchmark fixtures in the source
  distribution.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, the demo, summary command, and CPU
  compatibility probe pass there. Both benchmark profiles pass in the release
  matrix.
- [x] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit and tag, including package, both fixtures, and isolated-wheel checks
  on Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.9.0 release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.9.0`.
- [x] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.9.0.md` are prepared.
- [x] README and `ROADMAP.md` release text and links are prepared for
  `v0.9.0` publication.
- [x] Annotated tag `v0.9.0` points to the verified release commit
  `5735e957dd87d691ee6069496673f2f88939714c`; tag CI run `31778972929`
  completed successfully.
- [x] GitHub release [`v0.9.0`](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.9.0)
  is published with verified notes, wheel/sdist assets, and no private assets.
  The wheel SHA-256 is
  `caf2bf8bf5391b7e0325e2a7cc7563a0d937451eaef6237aa48ba2ba1b82f3d3`; the
  sdist SHA-256 is
  `3e3ad1e080af248c0e7cb97c882f7d49456140d2ddabde12cfc8c574824ec852`.

## v0.8.0 verification

- [x] `python -m pytest` passes locally with 157 tests.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `python scripts/cpu_compatibility.py` passes locally, including the
  eager/TorchScript CPU comparison.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, both synthetic benchmark profiles,
  the summary command, and the synthetic demo pass locally.
- [x] Source and wheel distributions build successfully, include the v0.8.0
  release documentation, and include both benchmark fixtures in the source
  distribution.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, the demo, summary command, and CPU
  compatibility probe pass there. Both benchmark profiles pass in the release
  matrix.
- [x] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit and tag, including package, both fixtures, and isolated-wheel checks
  on Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.8.0 release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.8.0`.
- [x] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.8.0.md` are prepared.
- [x] README and `ROADMAP.md` release text and links are prepared for
  `v0.8.0` publication.
- [x] Annotated tag `v0.8.0` points to the verified release commit
  `776b31222777767fbf412313d8324972f6d305d9`; tag CI run `31776985711`
  completed successfully.
- [x] GitHub release [`v0.8.0`](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.8.0)
  is published with verified notes, wheel/sdist assets, and no private assets.
  The wheel SHA-256 is
  `c5a37b58e0f7de41796dfe40bb6c332177d5c15197cf25070c33cba3acb9a443`; the
  sdist SHA-256 is
  `a070fdce9e117371409ee8acb9f514f6bc46c10cfd64af298356db555838374b`.

## v0.7.0 verification

- [x] `python -m pytest` passes locally with 156 tests.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `python scripts/cpu_compatibility.py` passes locally, including the
  eager/TorchScript CPU comparison.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, both synthetic benchmark profiles,
  the summary command, and the synthetic demo pass locally.
- [x] Source and wheel distributions build successfully, include the v0.7.0
  release documentation, and include the benchmark fixture in the source
  distribution.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, `wairc --help`, the demo, summary command,
  and CPU compatibility probe pass there. The benchmark profiles pass in the
  release matrix.
- [x] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit and tag, including package, fixture, and isolated-wheel checks on
  Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.7.0 release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.7.0`.
- [x] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.7.0.md` are prepared.
- [x] README and `ROADMAP.md` release text and links are prepared for
  `v0.7.0` publication.
- [x] Annotated tag `v0.7.0` points to the verified release commit
  `ec5fd2f9d80065b0856ee77bf30d1c9db21a014b`.
- [x] GitHub release [`v0.7.0`](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.7.0)
  is published with verified notes, wheel/sdist
  assets, and no private assets.

The tag CI run was `31775456268`. Published assets were downloaded again and
matched these SHA-256 digests:

- `wairc_rf-0.7.0-py3-none-any.whl`:
  `635fc9dde5d86eaafeae04b64af6db607100a8feaef63fcaecc428ccebcce9fa`
- `wairc_rf-0.7.0.tar.gz`:
  `9edaba3e2eece1e96293a511582c56fef4dd6a72fe93a714cc07dd3fdadbc7b4`


## v0.6.0 verification

- [x] `python -m pytest` passes locally with 156 tests.
- [x] `python scripts/smoke_test.py` passes locally.
- [x] `python scripts/cpu_compatibility.py` passes locally.
- [x] `ruff check src wairc_rf tests scripts` passes locally.
- [x] `python -m compileall -q src wairc_rf archived_baselines` passes locally.
- [x] `wairc --version`, `wairc --help`, both synthetic benchmark profiles,
  the summary command, and the synthetic demo pass locally.
- [x] Source and wheel distributions build successfully, include the v0.6.0
  release documentation, and include the benchmark fixture in the source
  distribution.
- [x] The wheel installs in an isolated environment with runtime dependencies;
  `pip check`, `wairc --version`, `wairc --help`, the demo, summary command,
  and CPU compatibility probe pass there. The benchmark profiles pass in the
  release matrix.
- [x] GitHub Actions passes on Python 3.10, 3.11, and 3.12 for the release
  commit and tag, including package, fixture, and isolated-wheel checks on
  Python 3.12.
- [x] Release contents are inspected for data, model weights, caches, secrets,
  and private paths.

## v0.6.0 release metadata

- [x] `pyproject.toml`, `src/__init__.py`, and `CITATION.cff` use `0.6.0`.
- [x] `CHANGELOG.md`, migration boundaries, and
  `docs/releases/v0.6.0.md` are prepared.
- [x] README and `ROADMAP.md` release text and links are prepared for
  `v0.6.0` publication.
- [x] Annotated tag `v0.6.0` points to verified release commit
  `c86313798d5936cd105df6b89ea9703703d0b16c`.
- [x] GitHub release [`v0.6.0`](https://github.com/littlecookie0722/wairc-2026/releases/tag/v0.6.0)
  is published with verified notes, wheel/sdist
  assets, and no private assets.

The tag CI run was `31773189061`. Published assets were downloaded again and
matched these SHA-256 digests:

- `wairc_rf-0.6.0-py3-none-any.whl`:
  `9902a45b8b13f8f40cba7a7d66ecf9d0bfb6b8ea4432524ffc2df90e08e43ace`
- `wairc_rf-0.6.0.tar.gz`:
  `79fb35b26bca20094baea1bff6138b5ad33aaa0077c386ff16ab008cd42f68e2`

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
