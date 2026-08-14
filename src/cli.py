from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib import import_module


COMMAND_MODULES = {
    "train": "src.train_spectrogram",
    "train-kfold": "src.train_spectrogram_kfold",
    "search-rules": "src.search_spectrogram_kfold_thresholds",
    "predict": "src.predict_spectrogram",
    "predict-kfold": "src.predict_spectrogram_kfold",
    "validate": "src.validate_submission",
    "demo": "src.synthetic_demo",
    "artifact": "src.artifact_cli",
}


def format_help() -> str:
    commands = "\n".join(
        [
            "  demo           Run a synthetic CPU end-to-end example",
            "  train          Train one spectrogram model",
            "  train-kfold    Train the k-fold spectrogram pipeline",
            "  search-rules   Search OOF inference rules",
            "  predict        Predict with one spectrogram model",
            "  predict-kfold  Predict with a k-fold ensemble",
            "  validate       Validate a submission text file",
            "  artifact       Inspect artifacts or validate a run manifest",
        ]
    )
    return (
        "WAIRC-2026 command-line interface\n\n"
        "Usage:\n"
        "  wairc <command> [options]\n\n"
        "Commands:\n"
        f"{commands}\n\n"
        "Run 'wairc <command> --help' for command-specific options."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(format_help())
        return
    if args[0] in {"-V", "--version"}:
        from . import __version__

        print(__version__)
        return

    command = args[0]
    module = COMMAND_MODULES.get(command)
    if module is None:
        available = ", ".join(COMMAND_MODULES)
        raise SystemExit(f"Unknown command {command!r}. Available commands: {available}")

    original_argv = sys.argv
    try:
        sys.argv = [f"wairc {command}", *args[1:]]
        import_module(module).main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
