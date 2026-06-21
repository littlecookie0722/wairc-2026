# Dataset Description

In an urban environment, unauthorized drones operating within the ISM public frequency bands (2.4GHz/5.8GHz) are about to enter the detection range of three radio sensors deployed at different locations. To accurately identify these unauthorized drones, participating teams are required to complete a drone model identification task by annotating the corresponding drone model category label for each sample.

The released dataset contains 7,500 labeled training samples and 1,000 unlabeled public test samples. During the competition, participants can use the training set to train their models, make predictions on the public test set, and submit their results to receive score feedback. The final ranking will be evaluated based on performance on a private test set.

## 1. Task Definition

The label space consists of 9 classes of drones (category indices `0~8`), and the task is multi-hot multi-label recognition.

Data Composition:

1. Approximately `50%` are single-target samples (only 1 category is set to 1).
2. Approximately `50%` are dual-target samples, and the two categories are distinct (2 different categories are set to 1).

## 2. Directory Structure

```text
<dataset_root>/
  index.csv
  iq_sample/
    00000000_xxxxxxxx.npz
    00000001_xxxxxxxx.npz
    ...
```

## 3. index.csv Field Formats and Meanings

Each row in `index.csv` corresponds to a single sample, with the following fields:

1. `sample_id`
Field Type: `int`
Meaning: Unique ID of the sample. Predictions must be aligned according to this ID upon submission.

2. `iq_npz_relpath`
Field Type: `str`
Meaning: Relative path of the IQ file (relative to `dataset_root`).

3. `label_signature`
Field Type: `str`
Meaning: Label signature. Active category indices are concatenated with a `|`, e.g., `3`, `2|7`.

4. `has_node0` / `has_node1` / `has_node2`
Field Type: `0/1`
Meaning: Indicates whether the IQ data for the corresponding node exists.

5. `sample_rate_node0` / `sample_rate_node1` / `sample_rate_node2`
Field Type: `float`
Meaning: Sampling rate of the corresponding node (in Hz).

## 4. NPZ File Format

Each `npz` file contains:

1. `iq_node0`: `int16` 1D array, interleaved IQ format.
2. `iq_node1`: `int16` 1D array, interleaved IQ format.
3. `iq_node2`: `int16` 1D array, interleaved IQ format.
4. `sample_rate_node0`: `float32`.
5. `sample_rate_node1`: `float32`.
6. `sample_rate_node2`: `float32`.

The interleaved IQ format is: `I0, Q0, I1, Q1, ...`

Important Notes:

1. Data from certain nodes may be missing (the `has_nodeX` field is `0`). In such cases, the corresponding `iq_nodeX` array is empty, and the sampling rate is `0`.
2. Data sampling rates may vary and must be handled according to the `sample_rate_nodeX` field.
3. Data across different nodes are not strictly aligned in time; there may be a time difference of a few seconds, but they correspond to the same time window of the same drone activity.

## 5. File Reading Example (Python)

```python
from pathlib import Path
import pandas as pd
import numpy as np

root = Path("your_dataset_root")
index_df = pd.read_csv(root / "index.csv")
row = index_df.iloc[0]

sample_id = int(row["sample_id"])
npz_path = root / row["iq_npz_relpath"]

with np.load(npz_path) as data:
    iq0 = data["iq_node0"]   # int16 interleaved IQ
    iq1 = data["iq_node1"]
    iq2 = data["iq_node2"]
    sr0 = float(data["sample_rate_node0"])
    sr1 = float(data["sample_rate_node1"])
    sr2 = float(data["sample_rate_node2"])

label_signature = row["label_signature"]
```

## 6. Submission File Format (Required)

Submissions must be in plain text `txt` format, containing one sample prediction per line. The format is fixed as follows:

```text
0: [0, 1, 0, 0, 0, 0, 0, 1, 0]
1: [0, 0, 0, 1, 0, 0, 0, 0, 0]
2: [...]
```

Rules:

1. The left side is the `sample_id`.
2. The right side is the predicted multi-hot array.
3. The array length must be exactly 9.
4. If the length is less than 9: Pad the end with `0`s to reach 9 elements.
5. If the length exceeds 9: Truncate to the first 9 elements.
6. Lines that do not conform to the formatting rules (e.g. "foo: {-2, 3}") will score 0 points.
7. If the total number of lines is insufficient, the missing samples will score 0 points.
8. If the total number of lines exceeds the required amount, the excess samples will not be scored.

## 7. Scoring Criteria

Scoring is based on exact matching (strict matching):

1. A single sample is counted as correct *only* if the 9-dimensional prediction array is completely identical to the 9-dimensional ground truth array.
2. The final score is the exact match accuracy rate across all samples.