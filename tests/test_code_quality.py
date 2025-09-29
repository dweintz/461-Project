"""
Test code_quality.py
"""

import os
import pytest
from unittest.mock import patch
from pathlib import Path

from src.scorer.metrics.code_quality import (
    get_code_quality,
    _check_code_repo_quality,
    run_radon,
    run_lizard,
    score_from_lizard_totals,
    docstring_ratio,
)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
    return os.getenv("HF_TOKEN", "")


def test_code_url(load_env):
    url_code = "https://github.com/google-research/bert"
    url_type_code = "code"
    score, latency = get_code_quality(url_code, url_type_code)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert isinstance(latency, int)
    assert latency > 0


@patch("src.scorer.metrics.code_quality.Repo.clone_from")
@patch("src.scorer.metrics.code_quality.os.walk")
@patch("src.scorer.metrics.code_quality.os.path.exists")
@patch("src.scorer.metrics.code_quality.run_radon")
@patch("src.scorer.metrics.code_quality.run_lizard")
@patch("src.scorer.metrics.code_quality.docstring_ratio")
def test_check_code_repo_quality_all_branches(
    mock_docstring, mock_lizard, mock_radon, mock_exists, mock_walk, mock_clone
):
    # Mock repo clone does nothing
    mock_clone.return_value = None

    # Mock files in repo to cover different branches
    mock_walk.return_value = [
        ("/tmp/mockrepo", [], ["README.md", "test_file.py", "setup.py"])
    ]

    # Simulate that .github exists and Dockerfile exists
    def exists_side(path):
        return any(
            x in path
            for x in [".github", "Dockerfile", "requirements.txt", "README.md"]
        )

    mock_exists.side_effect = exists_side

    # Radon score branch
    mock_radon.return_value = 0.8
    # Lizard branch
    mock_lizard.return_value = {"Avg CCN": 6, "Avg NLOC": 25, "Warning Count": 1}
    # Docstring ratio
    mock_docstring.return_value = 0.6

    score = _check_code_repo_quality("https://fake.repo")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_from_lizard_totals_various():
    # CCN branches
    totals_list = [
        {"Avg CCN": 3, "Avg NLOC": 20, "Warning Count": 0},  # best case
        {"Avg CCN": 8, "Avg NLOC": 35, "Warning Count": 2},  # mid case
        {"Avg CCN": 15, "Avg NLOC": 80, "Warning Count": 5},  # lower score
        {"Avg CCN": 25, "Avg NLOC": 120, "Warning Count": 10},  # worst
    ]
    for totals in totals_list:
        score = score_from_lizard_totals(totals)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_docstring_ratio(tmp_path):
    # create Python file with functions and classes with/without docstrings
    file1 = tmp_path / "a.py"
    file1.write_text('''def foo():\n    """doc"""\n    pass\nclass Bar:\n    pass''')
    score = docstring_ratio(str(tmp_path))
    assert 0.0 <= score <= 1.0


def test_run_radon_and_lizard(monkeypatch):
    # Patch subprocess.run to simulate radon output
    class Result:
        stdout = "a.py - A\nb.py - B\n"
        returncode = 0

    monkeypatch.setattr("subprocess.run", lambda *a, **k: Result())
    score = run_radon(".")
    assert 0.0 <= score <= 1.0

    # Lizard totals output
    monkeypatch.setattr("subprocess.run", lambda *a, **k: Result())
    totals = run_lizard(".")
    assert totals is None or isinstance(totals, dict)
