# Codex Development Brief: AI + Radio Drone Identification Competition（历史归档）

> 注意：本文是早期 baseline 开发指导。当前项目已整理为 STFT 频谱图冲高分主线，详见 `docs/冲高分STFT频谱图方案说明.md`。早期 baseline 代码已移动到 `archived_baselines/`。

## Background

This project is for an AI + radio competition. The task is to identify drone model labels from multi-node IQ radio samples collected in a real urban environment.

The repository currently contains:

- Competition documents in `docs/`.
- Labeled training data in `data_and_code/ai_radio_2026_qualifying_release/train/`.
- Unlabeled public test data in `data_and_code_patch-1/test_public_v1.1/`.

Important clarification: `data_and_code_patch-1/test_public_v1.1/` is the public test set, not a local validation set. It has no labels. Local validation must be created by splitting the labeled training set.

## Goals

- Build a reproducible baseline pipeline for the competition.
- Train a model on the 7,500 labeled samples.
- Create a local validation split from the training set.
- Predict labels for the 1,000 public test samples.
- Generate a valid submission `.txt` file.
- Keep the first version simple enough for a beginner to understand and extend.

## User Roles

| Role | Action | Concern |
| --- | --- | --- |
| Beginner competitor | Run scripts, train baseline, submit predictions | Needs clear commands and low ambiguity |
| Codex developer | Implement code and docs | Needs concrete file paths, modules, checks, and stop conditions |
| Competition evaluator | Run submitted code | Needs reproducible environment and readable implementation |

## Scope

### In Scope

- Read and inspect competition docs.
- Read `index.csv` and `.npz` IQ files.
- Convert `label_signature` into 9-dimensional multi-hot labels.
- Create a local train/validation split from the labeled training data.
- Implement a first baseline using hand-crafted IQ/statistical/spectral features.
- Train a multi-label or label-combination classifier.
- Evaluate exact-match accuracy on local validation.
- Predict public test labels.
- Write `submission.txt` in the required format.
- Add `requirements.txt` and usage instructions.

### Out Of Scope

- Building a web UI.
- Changing raw dataset files.
- Assuming public test labels exist.
- Using private test labels.
- Optimizing for maximum leaderboard rank before a working baseline exists.
- Large refactors unrelated to the competition pipeline.

## Competition Rules To Encode

- There are 9 drone classes: `0` through `8`.
- Each sample is a multi-hot multi-label classification target.
- About 50% of samples contain one active class.
- About 50% of samples contain two active classes.
- Public submission format must be plain text:

```text
0: [0, 1, 0, 0, 0, 0, 0, 1, 0]
1: [0, 0, 0, 1, 0, 0, 0, 0, 0]
```

- Left side is `sample_id`.
- Right side is a length-9 integer list.
- Exact-match accuracy is used: a sample is correct only if all 9 predicted values match the ground truth.
- The public test set contains 1,000 unlabeled samples.
- The final ranking is based on a private test set.
- Daily public submissions are limited, so local validation is required before submitting.

## Existing Data Paths

Use these paths relative to repository root:

```text
docs/数据集说明.md
docs/Dataset_Guide_EN.md
docs/# 比赛摘要.md
data_and_code/ai_radio_2026_qualifying_release/train/index.csv
data_and_code/ai_radio_2026_qualifying_release/train/iq_sample/
data_and_code_patch-1/test_public_v1.1/index.csv
data_and_code_patch-1/test_public_v1.1/iq_sample/
```

Observed dataset facts:

- Training rows: 7,500.
- Public test rows: 1,000.
- Training columns:
  - `sample_id`
  - `iq_npz_relpath`
  - `label_signature`
  - `has_node0`
  - `has_node1`
  - `has_node2`
  - `sample_rate_node0`
  - `sample_rate_node1`
  - `sample_rate_node2`
- Public test columns are the same except there is no `label_signature`.
- `iq_node0` and `iq_node1` use sample rate `125000000` when present.
- `iq_node2` uses sample rate `122880000` when present.
- Missing nodes have empty IQ arrays and sample rate `0`.

## NPZ Data Structure

Each `.npz` file contains:

- `iq_node0`: `int16` 1D interleaved IQ array.
- `iq_node1`: `int16` 1D interleaved IQ array.
- `iq_node2`: `int16` 1D interleaved IQ array.
- `sample_rate_node0`: `float32`.
- `sample_rate_node1`: `float32`.
- `sample_rate_node2`: `float32`.

IQ arrays are interleaved:

```text
I0, Q0, I1, Q1, ...
```

For feature extraction, reshape a non-empty IQ array as:

```python
iq = raw.astype("float32").reshape(-1, 2)
i = iq[:, 0]
q = iq[:, 1]
complex_iq = i + 1j * q
```

## Recommended Project Structure

Codex should create a small, readable pipeline:

```text
src/
  config.py
  data.py
  features.py
  train_baseline.py
  predict.py
  submission.py
  validate_submission.py
requirements.txt
README.md
outputs/
  models/
  submissions/
  metrics/
```

The exact structure may be adjusted if the existing project later grows, but keep the first implementation simple.

## Backend Design

### `src/config.py`

Define central paths and constants:

- `NUM_CLASSES = 9`
- `TRAIN_ROOT`
- `TEST_ROOT`
- `OUTPUT_DIR`
- `RANDOM_SEED`
- validation split ratio, default `0.2`

### `src/data.py`

Responsibilities:

- Load `index.csv`.
- Resolve `.npz` file paths.
- Parse `label_signature`.
- Convert labels to multi-hot arrays.
- Create local train/validation split.
- Load one sample's node arrays.

Required functions:

```python
def parse_label_signature(signature: str, num_classes: int = 9) -> list[int]:
    ...

def label_to_multihot(signature: str, num_classes: int = 9) -> list[int]:
    ...

def load_index(root: Path, has_labels: bool) -> pandas.DataFrame:
    ...
```

### `src/features.py`

Responsibilities:

- Convert interleaved IQ arrays into numeric features.
- Handle missing nodes.
- Keep feature length fixed for every sample.

Baseline feature set per node:

- Node present flag.
- Sample rate.
- Raw length.
- I mean/std/min/max.
- Q mean/std/min/max.
- Magnitude mean/std/min/max.
- Power mean/std/max.
- Simple FFT/spectral features from a downsampled or capped segment:
  - spectral peak bin
  - spectral peak value
  - mean spectral energy
  - energy quantiles or fixed-band energies

Implementation constraints:

- Do not FFT the full raw array if it is too slow. Cap or stride the signal first.
- Return zeros for missing nodes.
- Use the same feature order for train, validation, and test.

### `src/train_baseline.py`

Responsibilities:

- Load training index.
- Extract or cache features.
- Split training data into local train/validation.
- Train a baseline classifier.
- Evaluate exact-match accuracy.
- Save model and metrics.

Recommended first modeling approach:

- Treat each `label_signature` as one class.
- Train a multi-class classifier over observed label combinations.
- Convert predicted label combination back to 9-dimensional multi-hot.

Reason: the official metric is strict exact-match, and the dataset contains only single-label or two-label combinations. This approach is easier for a beginner and aligns well with the scoring rule.

Recommended model options:

- `ExtraTreesClassifier`
- `RandomForestClassifier`
- `HistGradientBoostingClassifier`
- Optional later: LightGBM/XGBoost if installed.

### `src/predict.py`

Responsibilities:

- Load saved model.
- Load public test index.
- Extract features using the same feature pipeline.
- Predict label signatures or multi-hot labels.
- Call submission writer.

### `src/submission.py`

Responsibilities:

- Convert predictions into required text format.
- Sort or align by `sample_id`.
- Write to `outputs/submissions/submission_baseline.txt`.

Required line format:

```text
<sample_id>: [v0, v1, v2, v3, v4, v5, v6, v7, v8]
```

### `src/validate_submission.py`

Responsibilities:

- Check the generated `.txt` before upload.
- Verify:
  - exactly 1,000 lines for public test submission
  - every sample ID from public `index.csv` appears once
  - every prediction has exactly 9 values
  - values are only `0` or `1`
  - no malformed lines

## API Contract

This project does not need a web API. The interface is command-line scripts.

Expected commands:

```powershell
python -m src.train_baseline
python -m src.predict
python -m src.validate_submission outputs/submissions/submission_baseline.txt
```

Optional command-line arguments may be added:

```powershell
python -m src.train_baseline --train-root data_and_code/ai_radio_2026_qualifying_release/train --val-ratio 0.2
python -m src.predict --test-root data_and_code_patch-1/test_public_v1.1 --model outputs/models/baseline.joblib
```

## Data And Persistence

Generated artifacts:

```text
outputs/models/baseline.joblib
outputs/metrics/baseline_metrics.json
outputs/submissions/submission_baseline.txt
outputs/cache/train_features.npy
outputs/cache/test_features.npy
```

Cache files are optional but recommended because IQ feature extraction may take time.

Metrics JSON should include:

- model name
- validation split ratio
- random seed
- local exact-match accuracy
- number of train samples
- number of validation samples
- timestamp

## Configuration

Use command-line arguments or constants for:

- training root
- public test root
- output directory
- validation ratio
- random seed
- model type
- FFT segment length or stride

No secrets or credentials are needed.

## Security And Permissions

- Do not modify raw files under `data_and_code/` or `data_and_code_patch-1/`.
- Write all generated outputs under `outputs/`.
- Do not use destructive filesystem commands.
- Do not assume internet access is required for baseline development.
- If external libraries are added, document them in `requirements.txt`.

## Error Handling

Codex should implement clear errors for:

- missing dataset directories
- missing `index.csv`
- missing `.npz` file referenced by `iq_npz_relpath`
- malformed `label_signature`
- unexpected label outside `0..8`
- malformed or incomplete submission line
- model file not found during prediction

When a node is missing, the code should not fail. It should return fixed zero features for that node and include the missing-node flag.

## Beginner Workflow

Follow this sequence:

1. Install dependencies.

```powershell
cd D:\Study\wairc-2026
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Read docs first:

```text
docs/数据集说明.md
docs/# 比赛摘要.md
```

3. Run a small data inspection command or script to confirm:

- training row count is 7,500
- public test row count is 1,000
- first `.npz` can be loaded
- label conversion works

4. Train baseline:

```powershell
python -m src.train_baseline
```

5. Check local validation score in:

```text
outputs/metrics/baseline_metrics.json
```

6. Generate public test predictions:

```powershell
python -m src.predict
```

7. Validate submission:

```powershell
python -m src.validate_submission outputs/submissions/submission_baseline.txt
```

8. Upload the validated `.txt` file to the competition platform.

9. Record public score and exact code/model version used.

## Improvement Roadmap

Implement only after the baseline is working.

1. Add richer spectral features:
   - fixed-band FFT energies
   - spectral entropy
   - peak count
   - occupied bandwidth approximation

2. Improve validation:
   - stratified split by `label_signature`
   - multiple random seeds
   - cross-validation

3. Try multi-label classifiers:
   - one binary classifier per class
   - tune class thresholds
   - enforce one or two active labels if useful

4. Add deep learning:
   - 1D CNN on IQ sequences
   - spectrogram or STFT image CNN
   - per-node feature extraction followed by fusion

5. Ensemble models:
   - average probabilities
   - vote between feature model and neural model
   - calibrate predictions using local validation

6. Use public test feedback carefully:
   - daily submissions are limited
   - keep a changelog for every submission
   - do not overfit blindly to public leaderboard

## Acceptance Criteria

- [ ] `requirements.txt` exists and installs the required baseline dependencies.
- [ ] Training script runs from repository root.
- [ ] Code reads all 7,500 training rows.
- [ ] Code creates a local validation split from training data.
- [ ] Label conversion from `label_signature` to length-9 multi-hot is tested or checked.
- [ ] Feature extraction returns fixed-length numeric vectors for all samples.
- [ ] Missing nodes do not crash feature extraction.
- [ ] Baseline model trains successfully.
- [ ] Local exact-match accuracy is computed and saved.
- [ ] Prediction script reads all 1,000 public test rows.
- [ ] Submission file is generated in the required format.
- [ ] Submission validator passes.
- [ ] README explains how to run train, predict, and validation commands.

## Risks And Open Questions

- The public test set has no labels, so local validation quality is important.
- Hand-crafted features may produce a low first score, but they are useful for a reliable first submission.
- Full IQ arrays are large, so naive FFT over every full array may be slow.
- Labels may contain small noise or missing labels, according to the competition FAQ.
- The final private test distribution may differ from the public test distribution.
- Hardware constraints from the competition FAQ: inference should fit within CPU memory 16 GB and GPU memory 24 GB.

## Codex Task Checklist

- [ ] Read `docs/数据集说明.md` and `docs/# 比赛摘要.md`.
- [ ] Inspect `index.csv` schemas for train and public test.
- [ ] Create `requirements.txt`.
- [ ] Create `src/` package.
- [ ] Implement data loading and label conversion.
- [ ] Implement robust IQ feature extraction.
- [ ] Implement baseline training and local validation.
- [ ] Implement prediction on public test.
- [ ] Implement submission writer.
- [ ] Implement submission validator.
- [ ] Add README run instructions.
- [ ] Run the full baseline workflow once.
- [ ] Report local validation score and output file path.

## Cutover Rule

Start implementation as soon as this document has been read once and no high-risk open questions remain. Do not create additional planning documents, review-only stages, or extra process gates before building the baseline. If a detail is ambiguous but low risk, implement with a documented assumption and continue.
