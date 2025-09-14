'''
Tests for CLI
'''

import subprocess
import sys
from pathlib import Path
import tempfile

# run the CLI
def run_cli(args):
    """Helper to run the CLI as a subprocess."""
    result = subprocess.run(
        [sys.executable, "cli.py", *args],
        capture_output=True,
        text=True,
    )
    return result

# test the help message
def test_help_message():
    result = run_cli(["--help"])
    assert result.returncode == 0
    assert "CLI for scoring models" in result.stdout

# test the missing URL file error check
def test_missing_url_file():
    result = run_cli(["nonexistent.txt"])
    # Your cli should exit nonzero on bad file
    assert result.returncode != 0
    assert "does not exist" in result.stderr or "Error" in result.stdout

# test a valid URL and see if correct list output
def test_valid_url_file(tmp_path: Path):
    # create a temporary file with dummy URLs
    url_file = tmp_path / "urls.txt"
    url_file.write_text("http://example.com/model1\nhttp://example.com/dataset1\n")

    result = run_cli([str(url_file)])
    assert result.returncode == 0
    assert "Processing 2 URLs" in result.stdout
    assert "http://example.com/model1" in result.stdout
    assert "http://example.com/dataset1" in result.stdout
