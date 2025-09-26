'''
Test code_quality.py
'''

import os
import pytest
from dotenv import load_dotenv
from pathlib import Path
from src.scorer.metrics.code_quality import get_code_quality

@pytest.fixture(scope = "session", autouse = True)
def load_env():
    '''
    Load enviornment files once per test session.
    '''

    load_dotenv(dotenv_path = Path(__file__).resolve().parents[1] / ".env")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found")
    return hf_token

def test_code_url(load_env):
    '''
    Test performance claims on a code URL
    '''

    url_code = "https://github.com/google-research/bert"
    url_type_code = "code"
    score, latency = get_code_quality(url_code, url_type_code)

    print(f"Code: Score = {score:.2f}, Latency = {latency}ms")

    assert isinstance(score, (float))
    assert isinstance(latency, (int))
    assert 0.0 <= score <= 1.0
    assert latency > 0
