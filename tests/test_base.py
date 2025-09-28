import pytest
from src.scorer.metrics.base import get_repo_id

def test_get_model_repo_id():
    url = "https://huggingface.co/google/gemma-3-270m/tree/main"
    assert get_repo_id(url, "model") == "google/gemma-3-270m"

def test_get_dataset_repo_id_full():
    url = "https://huggingface.co/datasets/xlangai/AgentNet"
    assert get_repo_id(url, "dataset") == "xlangai/AgentNet"

def test_get_dataset_repo_id_single():
    url = "https://huggingface.co/datasets/xlangai"
    assert get_repo_id(url, "dataset") == "xlangai"

def test_get_code_repo_id():
    url = "https://github.com/SkyworkAI/Matrix-Game"
    assert get_repo_id(url, "code") == "SkyworkAI/Matrix-Game"
