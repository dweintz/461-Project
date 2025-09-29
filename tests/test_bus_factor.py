import pytest
from unittest.mock import patch, MagicMock
from src.scorer.metrics.busfactor import (
    get_bus_factor,
    _is_code_like,
    _compute_bus_factor,
    _authors_by_file,
    _doa,
    _hf_kind_and_repo_id,
    _normalize_github_clone,
    _resolve_code_repo_for_target
)


def test_hf_kind_and_repo_id():
    """Test parsing HF URLs into kind and repo ID."""
    # Model URLs
    result = _hf_kind_and_repo_id("https://huggingface.co/google/bert-base-uncased")
    assert result == ("model", "google/bert-base-uncased")

    # Dataset URLs
    result = _hf_kind_and_repo_id("https://huggingface.co/datasets/squad")
    assert result is None

    # Invalid URLs
    assert _hf_kind_and_repo_id("https://huggingface.co/google") is None
    assert _hf_kind_and_repo_id("https://huggingface.co/datasets") is None
    assert _hf_kind_and_repo_id("https://github.com/owner/repo") is None


def test_normalize_github_clone():
    """Test GitHub URL normalization."""
    # Basic GitHub URL
    result = _normalize_github_clone("https://github.com/owner/repo")
    assert result == "https://github.com/owner/repo.git"

    # GitHub URL with tree
    result = _normalize_github_clone("https://github.com/owner/repo/tree/main")
    assert result == "https://github.com/owner/repo.git"

    # Invalid GitHub URL
    with pytest.raises(ValueError):
        _normalize_github_clone("https://github.com/owner")


@patch("src.scorer.metrics.busfactor.HF")
def test_resolve_code_repo_for_target_github(mock_hf):
    """Test GitHub URL resolution (should return normalized GitHub URL)."""
    url = "https://github.com/owner/repo"
    result = _resolve_code_repo_for_target(url, "code")
    assert result == "https://github.com/owner/repo.git"


@patch("src.scorer.metrics.busfactor.HF")
def test_resolve_code_repo_for_target_hf_model_with_gh_link(mock_hf):
    """Test HF model URL with GitHub link in card data."""
    mock_info = MagicMock()
    mock_info.cardData = {
        "repository": "https://github.com/owner/code-repo",
        "summary": "Some summary"
    }
    mock_hf.model_info.return_value = mock_info

    url = "https://huggingface.co/google/bert-base-uncased"
    result = _resolve_code_repo_for_target(url, "model")

    assert result == "https://github.com/owner/code-repo.git"
    mock_hf.model_info.assert_called_once_with("google/bert-base-uncased",
                                               files_metadata=False)


def make_mock_repo(file_structure=None, authors=None):
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
    """Normal repo behavior with one file/author."""
    mock_repo = make_mock_repo()
    mock_repo_class.return_value = mock_repo

    url = "https://huggingface.co/datasets/xlangai/AgentNet"
    score, latency = get_bus_factor(url, "dataset", since_days=1)

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
def test_get_bus_factor_clone_exception(mock_clone_from):
    """Simulate Repo clone raising an exception; should return 0 score."""
    mock_clone_from.side_effect = Exception("clone failed")

    url = "https://huggingface.co/invalid/repo"
    score, latency = get_bus_factor(url, "model")
    assert score == 0.0
    assert isinstance(latency, int)
    mock_clone_from.assert_called_once()


def test_is_code_like_and_binary_skip(tmp_path):
    # code-like file
    file1 = tmp_path / "script.py"
    file1.write_text("print('hello')")
    assert _is_code_like(str(file1))

    # binary-like file
    file2 = tmp_path / "model.pt"
    file2.write_bytes(b"\x00\x01\x02")
    assert not _is_code_like(str(file2))


def test_compute_bus_factor_edge_cases():
    # No files
    bf, removed = _compute_bus_factor({})
    assert bf == 0
    assert removed == []

    # All files abandoned
    authors_of_file = {"f1": set(), "f2": set()}
    bf, removed = _compute_bus_factor(authors_of_file)
    assert bf == 0
    assert removed == []

    # Simple case with authors
    authors_of_file = {"f1": {"a1", "a2"}, "f2": {"a1"}}
    bf, removed = _compute_bus_factor(authors_of_file)
    assert bf >= 0
    assert isinstance(removed, list)


def test_authors_by_file_and_doa_logic():
    dl = {"file1.py": {"a1": 5, "a2": 2}}
    total_by_file = {"file1.py": 7}
    contributors = {"file1.py": {"a1", "a2"}}
    creators = {"file1.py": "a1@example.com"}

    # _doa computation
    val_a1 = _doa("a1@example.com",
                  "file1.py",
                  dl,
                  total_by_file,
                  contributors,
                  creators)
    val_a2 = _doa("a2", "file1.py", dl, total_by_file, contributors, creators)
    assert val_a1 > val_a2

    authors_of_file = _authors_by_file(dl, total_by_file, contributors, creators)
    assert "file1.py" in authors_of_file
    assert authors_of_file["file1.py"]  # at least one author kept


@patch("src.scorer.metrics.busfactor.Repo.clone_from")
@patch("src.scorer.metrics.busfactor.Repo", autospec=True)
def test_get_bus_factor_multiple_files_authors(mock_repo_class, mock_clone_from):
    """Repo with multiple files and authors."""
    mock_repo = MagicMock()
    commit1 = MagicMock()
    commit1.author.email = "a1@example.com"
    commit1.author.name = "a1"
    commit1.stats.files = {"f1.py": {}, "f2.py": {}}
    commit2 = MagicMock()
    commit2.author.email = "a2@example.com"
    commit2.author.name = "a2"
    commit2.stats.files = {"f1.py": {}}
    mock_repo.iter_commits.return_value = [commit1, commit2]
    mock_repo.git.log.return_value = "a1@example.com\n"

    mock_repo_class.return_value = mock_repo

    url = "https://github.com/org/repo"
    score, latency = get_bus_factor(url, "code")
    assert 0 <= score <= 1
    assert isinstance(latency, int)
