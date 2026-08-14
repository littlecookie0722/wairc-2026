# Synthetic benchmark

The repository includes a small, data-free benchmark for checking that the
public synthetic IQ path remains runnable and reproducible on CPU. It is a
functional compatibility check, not a measurement of performance on real RF
recordings or competition data.

## Run

After installing the project, run the default profile with:

    wairc benchmark run --profile cpu-smoke

The output directory defaults to `outputs/benchmark/`. It can be changed
without changing the benchmark inputs:

    wairc benchmark run --profile cpu-smoke --output-dir outputs/my-benchmark --seed 2026

The command generates a self-contained synthetic demo under the selected
directory and validates its submission. Generated datasets, models, and
reports are local outputs and should not be committed.

The controlled robustness profile runs the same clean training workflow
against ten test conditions:

    wairc benchmark run --profile robustness-small

- `baseline`: the default noise and periodic missing-node pattern;
- `all-nodes-present`: the default noise with all three receiver nodes present
  for every test sample;
- `high-noise`: noise standard deviation `0.20` with the default node pattern;
- `node0-missing`: the first receiver node is absent for every test sample;
- `node1-missing`: the second receiver node is absent for every test sample;
- `node2-missing`: the third receiver node is absent for every test sample;
- `frequency-offset`: a `180 Hz` test frequency offset;
- `timing-offset`: a `32`-sample test window offset;
- `low-gain`: a `0.5` test signal gain;
- `combined-stress`: `0.20` noise, `180 Hz` frequency offset, a `32`-sample
  timing offset, and `0.5` signal gain applied together.

Each condition is evaluated independently with its own local demo artifacts.

## Redistributable fixture

[`tests/fixtures/benchmark/synthetic_iq_v1.json`](../tests/fixtures/benchmark/synthetic_iq_v1.json)
is a small `benchmark-fixture-v1` parameter manifest for the `cpu-smoke`
profile. It is repository-authored and distributed under the repository's MIT
terms. It contains generator parameters and an expected deterministic
signature only; it does not contain raw IQ, external recordings, model
weights, or private labels. The source distribution includes this JSON fixture
so consumers can verify the documented benchmark boundary without obtaining
competition data.

[`tests/fixtures/benchmark/synthetic_iq_robustness_v1.json`](../tests/fixtures/benchmark/synthetic_iq_robustness_v1.json)
is the corresponding parameter manifest for `robustness-small`. It records all
ten condition controls and the expected report signature without including
generated IQ, model outputs, or private labels. Both fixtures are input
contracts; generated benchmark artifacts remain local outputs.

## Verify a redistributable fixture

Replay a repository-authored fixture and verify that the generated manifest,
report schema, and deterministic signature match the fixture contract:

    wairc benchmark verify-fixture \
      tests/fixtures/benchmark/synthetic_iq_v1.json \
      --output-dir outputs/benchmark-fixture

Use the robustness fixture in the same way when checking all ten controlled
conditions. The command writes ordinary local benchmark outputs under the
selected directory; it does not modify the fixture, download competition
data, or include generated IQ and model artifacts in the repository.

## Render a Markdown summary

Turn a machine-readable report into a compact review document with:

    wairc benchmark summarize outputs/benchmark/benchmark-report.json

The default output is `benchmark-summary.md` next to the report. Use
`--output` to choose another local path. The summary validates the
`benchmark-report-v1` schema, displays the CPU or per-condition metrics, keeps
the deterministic signature, and lists only the report's relative artifact
names.

## Machine-readable files

`benchmark-manifest.json` records the inputs and compatibility boundary:

- `schemaVersion`: `benchmark-manifest-v1`;
- fixed seed, generator name/version, sample rate, three-node layout, class
  mapping, class frequencies, signal length, noise, and missing-node pattern;
- the `stft-v1` feature parameters used by the demo;
- the lightweight classifier configuration and the synthetic-only evaluation
  metric.

`benchmark-report.json` records the result:

- `schemaVersion`: `benchmark-report-v1`;
- profile, seed, `passed` status, exact-match accuracy, macro F1, per-class
  recall, threshold, and sample counts;
- for `robustness-small`, a `metrics.conditions` list with the same metrics for
  each named condition;
- a SHA-256 `deterministic_signature` over the manifest and deterministic
  metrics;
- runtime and relative artifact names for local inspection.

The fixture manifest is intentionally separate from generated benchmark output:
it is a reviewable input contract, while `benchmark-manifest.json` records the
full run inputs and `benchmark-report.json` records the result.

The runtime is deliberately excluded from the signature because it depends on
the machine. Re-running the same profile and seed with the same dependency
behavior should produce the same deterministic metrics and signature. A
signature match does not establish real-data accuracy, hardware equivalence,
or competition performance.

Both profiles use two training samples per class so they remain suitable for a
CPU smoke check. All condition values are synthetic controls and do not imply
real-world robustness or competition performance. The combined condition is a
controlled interaction check; it is not a calibrated model of a field
recording.
