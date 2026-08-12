# Contributing to WAIRC-2026

Thank you for improving the project. Contributions should keep the
competition pipeline understandable and behaviorally stable.

## Development setup

Install the runtime requirements, a compatible torch/torchvision build, and
the development tools:

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-gpu.txt
    python -m pip install -r requirements-dev.txt

## Before opening a pull request

    python -m pytest
    python scripts/smoke_test.py
    ruff check src tests scripts

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

