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
- profile, seed, `passed` status, exact-match accuracy, threshold, and sample
  counts;
- a SHA-256 `deterministic_signature` over the manifest and deterministic
  metrics;
- runtime and relative artifact names for local inspection.

The runtime is deliberately excluded from the signature because it depends on
the machine. Re-running the same profile and seed with the same dependency
behavior should produce the same deterministic metrics and signature. A
signature match does not establish real-data accuracy, hardware equivalence,
or competition performance.

The initial profile uses two training samples per class so it remains suitable
for a CPU smoke check. Robustness profiles for controlled SNR, offsets, gain,
and missing-node perturbations are follow-up work.
