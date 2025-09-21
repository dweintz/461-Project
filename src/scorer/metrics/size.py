'''
Implementing size metric based of model size
'''

from huggingface_hub import HfApi
from base import get_repo_id
import math

HF_API = HfApi()

target_size_bytes = 1000000000 # 1 billion bytes = 1 GB

# Pass in the url and its type from the URL handler cli code
def get_size_score(url: str, type: str) -> float:
    # Any logic to determine size from model/dataset/code url?
    
    # Get repo id
    try: 
        repo_id = get_repo_id(url)
    except Exception as e:
        print(f"Error getting repo id {e}")
        return None

    # Get model info
    info = HF_API.model_info(repo_id=repo_id, files_metadata=True)
    total_bytes = 0

    # Go through the model's files
    for file in info.siblings:
        size = file.size or 0
        total_bytes += size

    # Normalize this score
    diff = abs(math.log10(total_bytes) - math.log10(target_size_bytes))
    score = max(0, 1 - diff / 3) # Tolerance of 3 is forgiving for size differences to target

    return score
