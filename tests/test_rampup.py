import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os
import json
from pathlib import Path
from src.scorer.metrics.rampup import (
    get_ramp_up,
    _read_first_readme,
    _top_level_summary,
    _ask_llm,
    README_CANDIDATES,
    SKIP_DIRS
)

def test_read_first_readme_found(tmp_path):
    f = tmp_path / "README.md"
    f.write_text("Hello World")
    content = _read_first_readme(str(tmp_path))
    assert content == "Hello World"

def test_read_first_readme_missing(tmp_path):
    # No README files
    content = _read_first_readme(str(tmp_path))
    assert content == ""

@patch("src.scorer.metrics.rampup.InferenceClient")
def test_ask_llm_json_and_regex(mock_client):
    # Mock JSON response
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps({"score": 0.77})))]
    mock_client.return_value.chat.completions.create.return_value = mock_resp

    os.environ["RAMPUP_LLM_MODEL"] = "test-model"
    os.environ["HF_TOKEN"] = "token"

    score = _ask_llm("readme", "tree")
    assert 0.0 <= score <= 1.0
    assert score == 0.77

    # Mock fallback regex
    mock_resp.choices[0].message.content = "score is 0.55"
    score2 = _ask_llm("readme", "tree")
    assert score2 == 0.55

# Helper to create dummy repo structure
def create_dummy_repo_files(tmpdir, files=None, readme_content="Example README"):
    files = files or ["README.md", "train.py", "requirements.txt"]
    for f in files:
        path = Path(tmpdir) / f
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(readme_content if f.lower().startswith("readme") else "print('hello')")
    return tmpdir

@patch("src.scorer.metrics.rampup.Repo.clone_from")
@patch("src.scorer.metrics.rampup._ask_llm")
def test_get_ramp_up_success(mock_ask_llm, mock_clone_from):
    """Test get_ramp_up returns the score from mocked LLM and computes latency."""
    mock_ask_llm.return_value = 0.85

    # Use a temporary directory to simulate cloned repo
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_repo_files(tmpdir)
        mock_clone_from.side_effect = lambda url, to_path: tmpdir

        score, latency = get_ramp_up("https://huggingface.co/mock/repo", "model")

    assert 0.0 <= score <= 1.0
    assert score == 0.85
    assert isinstance(latency, int)
    mock_ask_llm.assert_called_once()
    mock_clone_from.assert_called_once()

@patch("src.scorer.metrics.rampup.Repo.clone_from")
@patch("src.scorer.metrics.rampup._ask_llm")
def test_get_ramp_up_clone_fail(mock_ask_llm, mock_clone_from):
    """Simulate clone error; should return score 0."""
    mock_clone_from.side_effect = Exception("clone failed")
    score, latency = get_ramp_up("https://huggingface.co/fake/repo", "model")
    assert score == 0.0
    assert isinstance(latency, int)

@patch("src.scorer.metrics.rampup.Repo.clone_from")
@patch("src.scorer.metrics.rampup._ask_llm")
def test_get_ramp_up_llm_fail(mock_ask_llm, mock_clone_from):
    """Simulate LLM error; should return score 0."""
    mock_ask_llm.side_effect = Exception("LLM failure")

    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_repo_files(tmpdir)
        mock_clone_from.side_effect = lambda url, to_path: tmpdir

        score, latency = get_ramp_up("https://huggingface.co/mock/repo", "model")

    assert score == 0.0
    assert isinstance(latency, int)
    mock_clone_from.assert_called_once()
    mock_ask_llm.assert_called_once()

