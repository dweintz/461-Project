import pytest
from unittest.mock import patch, MagicMock

def test_dataset_and_code_score_model():
    with patch("src.scorer.metrics.dataset_and_code.login") as mock_login, \
         patch("src.scorer.metrics.dataset_and_code.HF_API") as mock_hf_api_instance, \
         patch("src.scorer.metrics.dataset_and_code.get_repo_id", return_value="mock/repo"):

        # Make login do nothing
        mock_login.return_value = None

        # Mock siblings/files and README
        mock_siblings = [
            MagicMock(rfilename="train.py"),
            MagicMock(rfilename="README.md"),
            MagicMock(rfilename="requirements.txt")
        ]
        mock_card = {"datasets": ["Example dataset"], "model-index": [{"name": "Submodel"}]}

        # HF_API.model_info returns mock object
        mock_hf_api_instance.model_info.return_value = MagicMock(
            siblings=mock_siblings,
            cardData=mock_card
        )

        # Import inside the patch context so login is patched
        from src.scorer.metrics.dataset_and_code import get_dataset_and_code_score

        score, latency = get_dataset_and_code_score("https://huggingface.co/mock/repo", "model")
        assert score == 0.5
