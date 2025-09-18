'''
Implementing license metric scoring by seeing if 
the license is compatible with LGPLv2.1
'''

from huggingface_hub import HfApi
from base import get_repo_id

HF_API = HfApi()

def get_license_score(url: str, type: str) -> float:
    # Any logic to determine license from model/dataset/code url?
    
    # Get repo id
    try: 
        repo_id = get_repo_id(url)
    except Exception as e:
        print(f"Error getting repo id {e}")
        return None

    info = HfApi.model_info(repo_id=repo_id)
    license = getattr(info, "license", None)
    if license is None:
        return 0.0
    
    # Check if it's compatible with LGPLv2.1

    return
