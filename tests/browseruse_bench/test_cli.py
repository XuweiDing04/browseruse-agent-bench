"""Tests for browseruse_bench.utils.cli_utils module."""

import argparse
import subprocess
import sys

import pytest

import browseruse_bench.cli as cli_pkg
from browseruse_bench.cli.viz import configure_viz_parser
from browseruse_bench.utils import REPO_ROOT, create_eval_parser, create_run_parser


class TestCreateRunParser:
    """Tests for create_run_parser function."""

    def test_parser_has_required_arguments(self):
        """Test parser has all required arguments."""
        parser = create_run_parser()
        
        # Parse with minimal args
        args = parser.parse_args([])
        
        # Check default values exist
        assert hasattr(args, 'mode')
        assert hasattr(args, 'count')


@pytest.mark.parametrize("command", ["run", "eval"])
def test_runtime_commands_reject_non_mapping_config_before_parser(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_pkg, "load_config_file", lambda _: ["not", "a", "mapping"])

    with pytest.raises(SystemExit, match="top-level YAML mapping"):
        cli_pkg.main([command])


@pytest.mark.parametrize("command", ["run", "eval"])
def test_runtime_command_help_works_with_non_mapping_config(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_pkg, "load_config_file", lambda _: ["not", "a", "mapping"])

    with pytest.raises(SystemExit) as exc_info:
        cli_pkg.main([command, "--help"])

    assert exc_info.value.code == 0


class TestConfigureVizParser:
    def test_defaults(self):
        parser = argparse.ArgumentParser()
        configure_viz_parser(parser)
        args = parser.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 8080
        assert args.watch is False
        assert args.watch_interval == 3.0
        assert args.generate_only is False

    def test_all_flags(self):
        parser = argparse.ArgumentParser()
        configure_viz_parser(parser)
        args = parser.parse_args(
            ["--host", "0.0.0.0", "--port", "9090", "--watch", "--watch-interval", "5", "--generate-only"]
        )
        assert args.host == "0.0.0.0"
        assert args.port == 9090
        assert args.watch is True
        assert args.watch_interval == 5.0
        assert args.generate_only is True

    def test_scripts_viz_runs_directly(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "viz.py"), "--generate-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_visualization_serve_runs_directly(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "browseruse_bench" / "visualization" / "serve.py"), "--generate-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


class TestCreateEvalParser:
    """Tests for create_eval_parser function."""

    def test_eval_parser_has_required_arguments(self):
        """Test eval parser has all required arguments."""
        parser = create_eval_parser()
        
        args = parser.parse_args([])
        
        assert hasattr(args, 'model')
        assert hasattr(args, 'score_threshold')
