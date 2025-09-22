'''
Implementing dataset quality metric scoring by looking at
the number of downloads, likes
'''

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from .base import get_repo_id
import math

load_dotenv()
HF_TOKEN = os.getenv("HF_Token")
HF_API = HfApi()
login(token=HF_TOKEN)

# Downloads and likes targets for top tier quality
max_downloads = 1000000 # 1 million downloads
max_likes = 2000 

def normalize(value: int, target: int) -> float:
    if value <= 0:
        return 0.0
    return min(1.0, math.log10(value + 1) / math.log10(target + 1))

def get_dataset_quality_score(url: str, url_type: str) -> float:
    if url_type != "dataset":
        print("Dataset quality score is only applicable to datasets")
        return None

    # Get repo id
    try: 
        repo_id = get_repo_id(url, url_type)
    except Exception as e:
        print(f"Error getting repo id {e}")
        return None
    
    dataset_info = HF_API.dataset_info(repo_id=repo_id, files_metadata=False)

    # Look at number of downloads and likes
    downloads = getattr(dataset_info, "downloads", 0) or 0
    likes = getattr(dataset_info, "likes", 0) or getattr(dataset_info, "stars", 0) or 0

    # Normalize these scores
    downloads_score = normalize(downloads, max_downloads)
    likes_score = normalize(likes, max_likes)

    # If likes/stars don't exist, return downloads score
    if likes == 0:
        return round(downloads_score, 2)
    
    # Weighted sum of downloads and likes scores
    score = 0.8 * downloads_score + 0.2 * likes_score

    return round(score, 2)
