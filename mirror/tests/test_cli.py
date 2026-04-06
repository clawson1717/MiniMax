"""Tests for MIRROR CLI."""

from click.testing import CliRunner

import mirror
from mirror.cli import cli


class TestCLI:
    """Tests for the Click CLI interface."""

    def test_version_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert mirror.__version__ in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "MIRROR" in result.output

    def test_probe_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["probe", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--input" in result.output
        assert "--cot" in result.output
        assert "--answer" in result.output

    def test_probe_missing_args(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["probe"])
        assert result.exit_code != 0
        assert "Missing" in result.output or "required" in result.output.lower() or "Error" in result.output
