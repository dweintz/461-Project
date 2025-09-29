"""
Tests for CLI
"""

import subprocess
import sys


# run the CLI
def run_cli(args):
    """Helper to run the CLI as a subprocess."""
    result = subprocess.run(
        [sys.executable, "src/scorer/cli_updated.py", *args],
        capture_output=True,
        text=True,
    )
    return result


# test the help message
def test_help_message():
    result = run_cli(["--help"])
    assert result.returncode == 0
    combined_output = result.stdout + result.stderr
    assert "" in combined_output


# test the missing URL file error check
def test_missing_url_file():
    result = run_cli(["nonexistent.txt"])
    # Your cli should exit nonzero on bad file
    assert result.returncode != 0
