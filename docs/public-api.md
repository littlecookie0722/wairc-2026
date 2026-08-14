# Public Python API

The `wairc_rf` package is the stable import surface for reusable project
utilities. Existing `src.*` module entry points remain supported for backward
compatibility while implementation details are migrated incrementally.

## Stable symbols

| Symbol | Stable signature and defaults |
| --- | --- |
| `STFTConfig` | `profile="stft-v1", n_fft=512, hop=128, target_freq=None, target_time=None` |
| `iq_to_spectrogram` | `(interleaved_iq, sample_rate, config=None)` |
| `complex_iq_to_spectrogram` | `(complex_iq, sample_rate, config=None)` |
| `parse_label_signature` | `(signature, num_classes=9)` |
| `normalize_label_signature` | `(signature, num_classes=9)` |
| `label_to_multihot` | `(signature, num_classes=9)` |
| `multihot_to_signature` | `(multihot, num_classes=9)` |
| `RFNode` | `(iq, sample_rate, present=True)` |
| `RFSample` | `(sample_id, nodes, labels=None)` |
| `RFDatasetAdapter` | sequence protocol returning `RFSample` |
| `SyntheticDatasetAdapter` | `(samples)` |
| `CompetitionDatasetAdapter` | `(root, has_labels=...)` |

The dataset symbols provide the first public interoperability contract. The
competition adapter reads the existing `index.csv` and three-node NPZ format
without changing the legacy `src.data` or training entry points. It loads IQ
files lazily and returns nodes in fixed `node0`, `node1`, `node2` order.

`SyntheticDatasetAdapter` is the in-memory implementation for generated or
fixture `RFSample` values. It keeps sample order, rejects duplicate IDs, and
does not write data or infer a file format, so it can be used in CPU tests and
benchmarks without competition data.

```python
from wairc_rf import CompetitionDatasetAdapter

dataset = CompetitionDatasetAdapter("path/to/train", has_labels=True)
sample = dataset[0]

assert isinstance(sample.sample_id, int)  # the actual ID comes from index.csv
assert sample.labels is not None
for node in sample.nodes:
    print(node.iq_format, node.sample_rate, node.present, node.iq.shape)
```

`RFNode` accepts one-dimensional real interleaved IQ arrays or one-dimensional
complex IQ arrays. A missing node must use an empty array, `sample_rate=0`, and
`present=False`; the competition adapter validates this against both the CSV
row and NPZ fields. Labeled samples expose sorted, unique zero-based label
indices, while public-test samples expose `labels=None`. Relative IQ paths are
required to remain inside the dataset root, including after symlink resolution.

## STFT profile v1

```python
import numpy as np

from wairc_rf import STFTConfig, iq_to_spectrogram

raw_iq = np.fromfile("recording.int16", dtype=np.int16)
config = STFTConfig(
    profile="stft-v1",
    n_fft=512,
    hop=128,
    target_freq=257,
    target_time=1536,
)
spectrogram = iq_to_spectrogram(raw_iq, sample_rate=125_000_000, config=config)
```

`stft-v1` names the released transform behavior:

1. obtain complex IQ from one-dimensional interleaved `I,Q,I,Q,...` values or
   from a one-dimensional complex array;
2. convert to complex float values and remove the complex mean;
3. compute SciPy STFT with no boundary extension or padding;
4. use standardized `log1p(abs(STFT))` magnitude;
5. optionally resize frequency and time axes with linear interpolation;
6. return `float32`, or `None` when the input is shorter than one FFT window.

The interleaved IQ input must be a one-dimensional, even-length array of real
numeric values. `sample_rate` must be a finite positive real number.
`STFTConfig()` leaves both target axes as `None`; because the transform operates
on complex IQ, this preserves the native two-sided STFT frequency dimension
instead of forcing the competition's 257 bins, and the native time dimension
depends on input length.

### Native complex IQ

Use `complex_iq_to_spectrogram` when samples are already represented as a
one-dimensional NumPy complex array:

```python
import numpy as np

from wairc_rf import STFTConfig, complex_iq_to_spectrogram

complex_iq = np.fromfile("recording.complex64", dtype=np.complex64)
spectrogram = complex_iq_to_spectrogram(
    complex_iq,
    sample_rate=2_000_000.0,
    config=STFTConfig(n_fft=256, hop=64),
)
```

The input dtype must be complex numeric, such as `complex64` or `complex128`.
The function converts real and imaginary components to float32
`I,Q,I,Q,...` values and delegates to `iq_to_spectrogram`; it does not mutate
the input. For corresponding values and the same config, both entry points are
exactly equal. Both return `None` when there are fewer than `n_fft` complex
samples and otherwise return a float32 array.

The profile does not include training-time crop sampling, SpecAugment, cache
layout, label mapping, or inference rules. Those remain separate compatibility
boundaries. The competition dataset currently requests 257 frequency bins and
a 1536-frame cache tensor before selecting a 768-frame training or evaluation
crop; generic users should choose output sizes for their own data.

The public implementations delegate valid inputs to the legacy transform and
have exact interleaved/complex/delegate equality tests plus an independent
frozen-output regression. Introducing phase channels, alternative
normalization, or frequency-axis alignment requires a new profile instead of a
silent change to `stft-v1`.

## Labels

The package also exposes validated helpers for pipe-delimited label signatures:

```python
from wairc_rf import label_to_multihot, multihot_to_signature, normalize_label_signature

assert normalize_label_signature("2|0", num_classes=4) == "0|2"
assert label_to_multihot("2|0", num_classes=4) == [1, 0, 1, 0]
assert multihot_to_signature([1, 0, 1, 0], num_classes=4) == "0|2"
```

These helpers reject empty, duplicate, non-integer, and out-of-range labels,
as well as non-binary multi-hot values and an all-zero multi-hot vector.
Multi-hot vectors may use integer `0`/`1` or Boolean values.
The default class count remains the competition contract; pass `num_classes`
explicitly for a different public dataset.
