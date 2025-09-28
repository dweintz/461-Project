import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os
import json
from pathlib import Path
from git import Repo
from src.scorer.metrics.rampup import (
    get_ramp_up,
    _read_first_readme,
    _top_level_summary,
    _ask_llm,
    _heuristic_rampup,
    _to_clone_url,
    README_CANDIDATES,
    SKIP_DIRS
)

# Test URL normalization
def test_to_clone_url_github():
    """Test GitHub URL normalization."""
    url = "https://github.com/owner/repo"
    result = _to_clone_url(url, "code")
    assert result == "https://github.com/owner/repo.git"

def test_to_clone_url_github_with_tree():
    """Test GitHub URL with /tree/main suffix."""
    url = "https://github.com/owner/repo/tree/main"
    result = _to_clone_url(url, "code")
    assert result == "https://github.com/owner/repo.git"

def test_to_clone_url_hf_model():
    """Test HuggingFace model URL normalization."""
    url = "https://huggingface.co/owner/model"
    result = _to_clone_url(url, "model")
    assert result == "https://huggingface.co/owner/model"

def test_to_clone_url_hf_dataset():
    """Test HuggingFace dataset URL normalization."""
    url = "https://huggingface.co/datasets/owner/dataset"
    result = _to_clone_url(url, "dataset")
    assert result == "https://huggingface.co/datasets/owner/dataset"

def test_to_clone_url_invalid_github():
    """Test invalid GitHub URL."""
    with pytest.raises(ValueError):
        _to_clone_url("https://github.com/owner", "code")

def test_to_clone_url_invalid_hf_model():
    """Test invalid HF model URL."""
    with pytest.raises(ValueError):
        _to_clone_url("https://huggingface.co/owner", "model")

def test_to_clone_url_invalid_hf_dataset():
    """Test invalid HF dataset URL."""
    with pytest.raises(ValueError):
        _to_clone_url("https://huggingface.co/datasets/owner", "dataset")

def test_read_first_readme_found(tmp_path):
    f = tmp_path / "README.md"
    f.write_text("Hello World")
    content = _read_first_readme(str(tmp_path))
    assert content == "Hello World"

def test_read_first_readme_missing(tmp_path):
    # No README files
    content = _read_first_readme(str(tmp_path))
    assert content == ""

# Test file tree summary
def test_top_level_summary_basic(tmp_path):
    """Test basic file tree generation."""
    files = ["file1.py", "file2.txt", "subdir/file3.py"]
    for f in files:
        path = tmp_path / f
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content")
    
    result = _top_level_summary(str(tmp_path))
    assert "file1.py" in result
    assert "file2.txt" in result

def test_top_level_summary_skip_dirs(tmp_path):
    """Test that skipped directories are excluded."""
    # Create files in directories that should be skipped
    for skip_dir in SKIP_DIRS:
        skip_path = tmp_path / skip_dir
        skip_path.mkdir()
        (skip_path / "file.py").write_text("should be skipped")
    
    # Create a normal file
    (tmp_path / "normal_file.py").write_text("should be included")
    
    result = _top_level_summary(str(tmp_path))
    assert "normal_file.py" in result
    # Should not include files from skipped directories
    for skip_dir in SKIP_DIRS:
        assert skip_dir not in result

def test_top_level_summary_max_files(tmp_path):
    """Test file count limit."""
    # Create more than max_files
    for i in range(150):
        (tmp_path / f"file{i}.txt").write_text("content")
    
    result = _top_level_summary(str(tmp_path), max_files=50)
    lines = result.split('\n')
    assert len(lines) <= 50

def test_top_level_summary_empty_dir(tmp_path):
    """Test empty directory."""
    result = _top_level_summary(str(tmp_path))
    assert result == ""

def test_heuristic_rampup_low_score():
    """Test heuristic with few patterns."""
    readme = "# Project\nThis is a project."
    tree = "file1.txt\nfile2.txt"
    
    score = _heuristic_rampup(readme, tree)
    assert score == 0.15  # Minimum score

def test_heuristic_rampup_empty():
    """Test heuristic with empty content."""
    score = _heuristic_rampup("", "")
    assert score == 0.15

def test_heuristic_rampup_score_range():
    """Test that heuristic score is always in [0,1] range."""
    # Test with excessive patterns
    excessive_patterns = " ".join(["pip install"] * 100)
    score = _heuristic_rampup(excessive_patterns, "")
    assert 0.0 <= score <= 1.0

def test_ask_llm_no_env_vars():
    """Test LLM with missing environment variables."""
    # Clear environment variables
    with patch.dict(os.environ, {}, clear=True):
        score = _ask_llm("README content", "file tree")
        assert score is None

def test_ask_llm_inference_client_not_available():
    """Test when InferenceClient is not available."""
    with patch("src.scorer.metrics.rampup.InferenceClient", None):
        score = _ask_llm("README content", "file tree")
        assert score is None

@patch("src.scorer.metrics.rampup._ask_llm", return_value=0.77)
def test_ask_llm_json_and_regex(mock_ask_llm):
    from src.scorer.metrics.rampup import get_ramp_up

    score, _ = get_ramp_up("https://huggingface.co/mock/repo", "model")

    # Now score is always a float between 0 and 1
    assert 0.0 <= score <= 1.0

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

# Additional test to verify the exact LLM failure scenario
@patch("src.scorer.metrics.rampup.Repo.clone_from")
@patch("src.scorer.metrics.rampup._ask_llm")
@patch("src.scorer.metrics.rampup._read_first_readme") 
@patch("src.scorer.metrics.rampup._top_level_summary")
def test_get_ramp_up_llm_exception(mock_summary, mock_readme, mock_ask_llm, mock_clone_from):
    """Test that LLM exceptions are handled gracefully."""
    readme_content = "# Test README\nNo special patterns here"
    tree_content = "file1.txt\nfile2.txt"
    
    mock_readme.return_value = readme_content
    mock_summary.return_value = tree_content
    
    # Mock _ask_llm to raise an exception (different from returning None)
    mock_ask_llm.side_effect = Exception("LLM service error")
    mock_clone_from.return_value = MagicMock()

    score, latency = get_ramp_up("https://huggingface.co/mock/repo", "model")
    assert score == 0.0
    assert isinstance(latency, int)
    
    mock_clone_from.assert_called_once()
    mock_ask_llm.assert_called_once()

# Test with empty content to verify heuristic minimum score
@patch("src.scorer.metrics.rampup.Repo.clone_from")
@patch("src.scorer.metrics.rampup._ask_llm")
@patch("src.scorer.metrics.rampup._read_first_readme")
@patch("src.scorer.metrics.rampup._top_level_summary") 
def test_get_ramp_up_empty_content(mock_summary, mock_readme, mock_ask_llm, mock_clone_from):
    """Test heuristic fallback with empty README and file list."""
    mock_readme.return_value = ""
    mock_summary.return_value = ""
    mock_ask_llm.return_value = None
    mock_clone_from.return_value = MagicMock()

    score, latency = get_ramp_up("https://huggingface.co/mock/repo", "model")

    # Heuristic should return minimum score (0.15) when no patterns found
    expected_score = _heuristic_rampup("", "")
    assert score == expected_score
    assert score == 0.15  # Minimum heuristic score
    assert isinstance(latency, int)

# Test successful LLM response
@patch("src.scorer.metrics.rampup.Repo.clone_from")
@patch("src.scorer.metrics.rampup._ask_llm")
@patch("src.scorer.metrics.rampup._read_first_readme")
@patch("src.scorer.metrics.rampup._top_level_summary")
def test_get_ramp_up_llm_success(mock_summary, mock_readme, mock_ask_llm, mock_clone_from):
    """Test successful LLM evaluation."""
    mock_readme.return_value = "# Test README\npip install test"
    mock_summary.return_value = "setup.py\nREADME.md"
    
    # Mock LLM to return a specific score
    mock_ask_llm.return_value = 0.85
    mock_clone_from.return_value = MagicMock()

    score, latency = get_ramp_up("https://huggingface.co/mock/repo", "model")

    # Should use LLM score, not heuristic
    assert score == 0.85
    assert isinstance(latency, int)
    
    mock_ask_llm.assert_called_once()

# Test score clamping
@patch("src.scorer.metrics.rampup.Repo.clone_from") 
@patch("src.scorer.metrics.rampup._ask_llm")
@patch("src.scorer.metrics.rampup._read_first_readme")
@patch("src.scorer.metrics.rampup._top_level_summary")
def test_get_ramp_up_score_clamping(mock_summary, mock_readme, mock_ask_llm, mock_clone_from):
    """Test that scores are clamped to [0.0, 1.0] range."""
    mock_readme.return_value = "README"
    mock_summary.return_value = "files"
    mock_clone_from.return_value = MagicMock()
    
    # Test score > 1.0 gets clamped
    mock_ask_llm.return_value = 1.5
    score, _ = get_ramp_up("https://huggingface.co/mock/repo", "model")
    assert score == 1.0
    
    # Test score < 0.0 gets clamped  
    mock_ask_llm.return_value = -0.5
    score, _ = get_ramp_up("https://huggingface.co/mock/repo", "model")
    assert score == 0.0