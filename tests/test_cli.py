import sys

import pytest

from src import __version__
from src.cli import main


def test_cli_prints_help_and_version(capsys):
    main([])
    help_output = capsys.readouterr().out
    assert "wairc <command>" in help_output
    assert "demo" in help_output
    assert "validate" in help_output
    assert "benchmark" in help_output
    assert "artifact" in help_output

    main(["--version"])
    assert capsys.readouterr().out.strip() == __version__


def test_cli_rejects_unknown_command():
    with pytest.raises(SystemExit, match="Unknown command"):
        main(["unknown"])


def test_cli_restores_process_arguments_after_dispatch(monkeypatch):
    original = ["pytest", "sentinel"]
    monkeypatch.setattr(sys, "argv", original)

    with pytest.raises(SystemExit) as exc_info:
        main(["validate", "--help"])

    assert exc_info.value.code == 0
    assert sys.argv is original
