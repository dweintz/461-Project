'''
Implementing size metric based of model size 
and hardware capabilities
'''

import math
import os
import requests
import time
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from .base import get_repo_id
from typing import Tuple, Dict, Optional

# suppress logging from Hugging Face
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

load_dotenv()
# HF_TOKEN = os.getenv("HF_Token")
HF_API = HfApi()
# login(token=HF_TOKEN)

hardware_limits = {
    "raspberry_pi": 4000000000, # 4GB
    "jetson_nano": 4000000000, # 4GB
    "desktop_pc": 32000000000, # 32GB
    "aws_server": 512000000000 # 512GB
}


def _hf_total_bytes(model_url: str) -> int:
    repo_id = get_repo_id(model_url, "model")  # returns 'google-bert/bert-base-uncased'
    total = 0
    for f in HF_API.list_files_info(repo_id=repo_id, revision="main"):
        # f.size is bytes (int). Some entries may lack size; guard with 0.
        total += int(getattr(f, "size", 0) or 0)
    return total


def _maybe_login() -> None:
    """
    Log in non-interactively only if a token is present.
    Never prompt, never run at import time.
    """
    token = (
        os.getenv("HF_TOKEN")           # preferred
        or os.getenv("HF_Token")        # be forgiving if someone used this
        or os.getenv("HUGGINGFACE_TOKEN")  # extra alias, optional
    )
    if not token:
        return
    try:
        # No interactive questions, no new session popups
        login(
            token=token,
            add_to_git_credential=False,
            write_permission=False,
            new_session=False,
        )
    except Exception:
        # Swallow login issues; callers should still work anonymously where possible
        pass

def score_for_hardware(total_bytes: int, limit: int) -> float:
    """
    Returns a score between 0 and 1 based on how well the model size fits.
    1 = fits easily, 0 = too big.
    """
    if total_bytes == 0:
        return 0.0
    
    ratio = total_bytes / limit
    if ratio <= 1:
        # Perfect fit at small sizes, degrades as you approach limit
        return 1 - 0.5 * ratio  
    else:
        # Penalize oversize models strongly
        return max(0, 1 - math.log10(ratio))

# Pass in the url and its type from the URL handler cli code
def get_size_score(url: str, url_type: str) -> Tuple[Optional[Dict[str, float]], int]:
    _maybe_login()
    start_time = time.time()

    # Get repo id
    try: 
        repo_id = get_repo_id(url, url_type)
    except Exception as e:
        print(f"Error getting repo id {e}")
        latency = int((time.time() - start_time) * 1000)
        return None, latency

    total_bytes = 0
    if url_type == "model":
        # Get model info and get size
        info = HF_API.model_info(repo_id=repo_id, files_metadata=True)
        total_bytes = _hf_total_bytes(url)
                
    elif url_type == "dataset":
        info = HF_API.dataset_info(repo_id=repo_id, files_metadata=True)
        for file in info.siblings:
            total_bytes += file.size or 0
        
    elif url_type == "code":
        base_url = f"https://api.github.com/repos/{repo_id}"
        code_info = requests.get(base_url).json()
        total_bytes = (code_info.get("size", 0) or 0) * 1024 # Convert to bytes

    print(f"Total bytes for {url} is {total_bytes}")

    size_dict = {}
    for hardware, limit in hardware_limits.items():
        score = score_for_hardware(total_bytes, limit)
        size_dict[hardware] = score

    latency = int((time.time() - start_time) * 1000)

    return size_dict, latency
