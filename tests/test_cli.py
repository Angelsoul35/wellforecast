"""Tests for the command-line interface."""

from __future__ import annotations

import pytest

from wellforecast import cli


def test_demo_runs_end_to_end(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["wellforecast", "demo", "--well", "F-11 H"])
    cli.main()
    out = capsys.readouterr().out
    assert "Well F-11 H" in out
    assert "Arps fit" in out
    assert "ML correction" in out
    assert "EUR-style" in out
    assert "verdict:" in out  # the fleet benchmark prints too


def test_demo_defaults_to_the_first_producer(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["wellforecast", "demo"])
    cli.main()
    out = capsys.readouterr().out
    assert "post-peak producing days" in out


def test_cli_requires_a_subcommand(monkeypatch):
    monkeypatch.setattr("sys.argv", ["wellforecast"])
    with pytest.raises(SystemExit):
        cli.main()
