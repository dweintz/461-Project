'''
Implementing size metric based of model size
'''

import math
import os
import requests
import time
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from .base import get_repo_id

load_dotenv()
HF_TOKEN = os.getenv("HF_Token")
HF_API = HfApi()
login(token=HF_TOKEN)

target_size_bytes = 1000000000 # 1 billion bytes = 1 GB

# Pass in the url and its type from the URL handler cli code
def get_size_score(url: str, url_type: str) -> float:
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
        for file in info.siblings:
            total_bytes += file.size or 0
                
    elif url_type == "dataset":
        info = HF_API.dataset_info(repo_id=repo_id, files_metadata=True)
        for file in info.siblings:
            total_bytes += file.size or 0
        
    elif url_type == "code":
        base_url = f"https://api.github.com/repos/{repo_id}"
        code_info = requests.get(base_url).json()
        total_bytes = (code_info.get("size", 0) or 0) * 1024 # Convert to bytes

    # Normalize this score
    print(f"Total bytes for {url} is {total_bytes}")
    diff = abs(math.log10(total_bytes) - math.log10(target_size_bytes))
    score = max(0, 1 - diff / 3) # Tolerance of 3 is forgiving for size differences to target

    latency = int((time.time() - start_time) * 1000)

    return score, latency
