import pytest
from unittest.mock import patch, MagicMock
from src.scorer.metrics.busfactor import get_bus_factor

def make_mock_repo(file_structure=None):
    """Helper to create a mock Repo with dummy commits/files."""
    mock_repo = MagicMock()
    
    # Setup iter_commits
    mock_commit = MagicMock()
    mock_commit.author.email = "example@example.com"
    mock_commit.author.name = "example"
    mock_commit.stats.files = file_structure or {"file1.py": {}}
    mock_repo.iter_commits.return_value = [mock_commit]
    
    # Patch git log for _first_author_email
    mock_repo.git.log.return_value = "example@example.com\n"
    
    return mock_repo

@patch("src.scorer.metrics.busfactor.Repo.clone_from")
@patch("src.scorer.metrics.busfactor.Repo", autospec=True)
def test_get_bus_factor_basic(mock_repo_class, mock_clone_from):
    """Test normal repo behavior with one file/author."""
    # Mock the repo instance returned by Repo()
    mock_repo = make_mock_repo()
    mock_repo_class.return_value = mock_repo

    url = "https://huggingface.co/datasets/xlangai/AgentNet"
    score, latency = get_bus_factor(url, "dataset", since_days=1)

    # Assertions
    assert 0 <= score <= 1
    assert isinstance(latency, int)
    mock_clone_from.assert_called_once()

@patch("src.scorer.metrics.busfactor.Repo.clone_from")
@patch("src.scorer.metrics.busfactor.Repo", autospec=True)
def test_get_bus_factor_no_files(mock_repo_class, mock_clone_from):
    """Repo with no commits/files should return score 0."""
    mock_repo = make_mock_repo(file_structure={})
    mock_repo.iter_commits.return_value = []
    mock_repo_class.return_value = mock_repo

    url = "https://huggingface.co/google/gemma-3-270m"
    score, latency = get_bus_factor(url, "model")
    assert score == 0.0

@patch("src.scorer.metrics.busfactor.Repo.clone_from")
@patch("src.scorer.metrics.busfactor.Repo", autospec=True)
def test_get_bus_factor_exception(mock_repo_class, mock_clone_from):
    """Simulate Repo clone raising an exception; should return 0 score."""
    mock_clone_from.side_effect = Exception("clone failed")

    url = "https://huggingface.co/invalid/repo"
    score, latency = get_bus_factor(url, "model")
    assert score == 0.0
    assert isinstance(latency, int)
