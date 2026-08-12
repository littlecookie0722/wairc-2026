# Contributing to WAIRC-2026

Thank you for improving the project. Contributions should keep the
competition pipeline understandable and behaviorally stable.

## Development setup

For a CPU development environment, install a compatible CPU build of
torch/torchvision, followed by the project and development tools:

    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    python -m pip install -e ".[dev]"

GPU development is optional. When CUDA 12.8 is available, the existing
requirements-file workflow remains supported:

    python -m pip install -r requirements-gpu.txt
    python -m pip install -r requirements-dev.txt

## Before opening a pull request

    python -m pytest
    python scripts/smoke_test.py
    ruff check src wairc_rf tests scripts

If the change affects training, run the smallest available data-backed check
and report the command, data assumptions, device, and result. Do not upload
competition data or checkpoints in a pull request.

## Pull requests

Describe:

- what changed and why;
- which pipeline boundary is affected;
- whether STFT, labels, checkpoints, or submission output can change;
- tests and manual checks run;
- known limitations or reproducibility differences.

New experiments should record their configuration, seed, model tag, data
assumptions, and metrics. Do not present a local validation result as a public
test or official competition result.

## Contribution license

By submitting a contribution, you agree that it may be distributed under the
MIT License used by this repository. Do not contribute competition data,
model weights, copied code, or other material that you do not have permission
to distribute.
