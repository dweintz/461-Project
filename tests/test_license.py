from unittest.mock import patch, MagicMock
from src.scorer.metrics import license


def test_is_compatible_valid_licenses():
    assert license.is_compatible("Apache 2.0")
    assert license.is_compatible("MIT")
    assert license.is_compatible("BSD 2-Clause")
    assert license.is_compatible("BSD 3-Clause")
    assert license.is_compatible("LGPL v2.1")


def test_is_compatible_invalid_license():
    assert not license.is_compatible("Proprietary")
    assert not license.is_compatible(None)
    assert not license.is_compatible("")


@patch("src.scorer.metrics.license.get_repo_id", return_value="mock/repo")
def test_get_license_score_model(mock_get_repo_id):
    fake_info = MagicMock(license="mit")
    with patch.object(license.HF_API, "model_info", return_value=fake_info):
        score, latency = license.get_license_score("fake_url", "model")
    assert score == 1
    assert isinstance(latency, int)


@patch("src.scorer.metrics.license.get_repo_id", return_value="mock/repo")
def test_get_license_score_dataset(mock_get_repo_id):
    fake_info = MagicMock(license="apache-2.0")
    with patch.object(license.HF_API, "dataset_info", return_value=fake_info):
        score, latency = license.get_license_score("fake_url", "dataset")
    assert score == 1
    assert isinstance(latency, int)


@patch("src.scorer.metrics.license.get_repo_id", return_value="mock/repo")
def test_get_license_score_code(mock_get_repo_id):
    fake_license_response = {"license": {"name": "MIT"}}
    with patch(
        "src.scorer.metrics.license.requests.get",
        return_value=MagicMock(json=lambda: fake_license_response),
    ):
        score, latency = license.get_license_score("fake_url", "code")
    assert score == 1
    assert isinstance(latency, int)


@patch("src.scorer.metrics.license.get_repo_id", return_value="mock/repo")
def test_get_license_score_incompatible(mock_get_repo_id):
    fake_info = MagicMock(license="proprietary")
    with patch.object(license.HF_API, "model_info", return_value=fake_info):
        score, latency = license.get_license_score("fake_url", "model")
    assert score == 0
    assert isinstance(latency, int)
