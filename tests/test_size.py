import types
from unittest.mock import patch, MagicMock
from src.scorer.metrics import size

def test_score_for_hardware_under_limit():
    limit = 1000000000 # 1GB
    total_bytes = 100000000  # 100MB
    score = size.score_for_hardware(total_bytes, limit)
    assert 0 < score <= 1
    assert score > 0.5 

def test_score_for_hardware_over_limit():
    limit = 1000000000
    total_bytes = 10_000_000_000  # 10GB
    score = size.score_for_hardware(total_bytes, limit)
    assert 0 <= score < 1
    assert score < 0.5

@patch("src.scorer.metrics.size.get_repo_id", return_value="mock/repo")
def test_get_size_score_model(mock_get_repo_id):
    # Mock HF_API.model_info
    fake_file = types.SimpleNamespace(size=100_000_000)
    fake_info = types.SimpleNamespace(siblings=[fake_file, fake_file])
    with patch.object(size.HF_API, "model_info", return_value=fake_info):
        scores, latency = size.get_size_score("fake_url", "model")

    assert isinstance(scores, dict)
    assert all(hw in scores for hw in size.hardware_limits)
    assert all(0 <= v <= 1 for v in scores.values())
    assert isinstance(latency, int)

@patch("src.scorer.metrics.size.get_repo_id", return_value="mock/repo")
def test_get_size_score_dataset(mock_get_repo_id):
    fake_file = types.SimpleNamespace(size=50_000_000)
    fake_info = types.SimpleNamespace(siblings=[fake_file])
    with patch.object(size.HF_API, "dataset_info", return_value=fake_info):
        scores, latency = size.get_size_score("fake_url", "dataset")

    assert isinstance(scores, dict)
    assert "raspberry_pi" in scores

@patch("src.scorer.metrics.size.get_repo_id", return_value="mock/repo")
def test_get_size_score_code(mock_get_repo_id):
    fake_response = {"size": 1024}  # GitHub "size" is KB
    with patch("src.scorer.metrics.size.requests.get", return_value=MagicMock(json=lambda: fake_response)):
        scores, latency = size.get_size_score("fake_url", "code")

    assert isinstance(scores, dict)
    assert "aws_server" in scores
