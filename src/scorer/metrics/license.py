'''
Implementing license metric scoring by seeing if 
the license is compatible with LGPLv2.1
'''

import os
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from .base import get_repo_id

load_dotenv()
HF_TOKEN = os.getenv("HF_Token")
HF_API = HfApi()
login(token=HF_TOKEN)

compatible_licenses = [
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "lgpl-2.1"
]

# Normalize license names from HF/GitHub API
def is_compatible(license: str) -> bool:
    if not license:
        return False

    normalized = license.lower().strip()

    if normalized.startswith("apache"):
        normalized = "apache-2.0"
    elif normalized.startswith("mit"):
        normalized = "mit"
    elif normalized.startswith("bsd 2-clause"):
        normalized = "bsd-2-clause"
    elif normalized.startswith("bsd 3-clause"):
        normalized = "bsd-3-clause"
    elif normalized.startswith("lgpl v2.1"):
        normalized = "lgpl-2.1"

    return normalized in compatible_licenses

def get_license_score(url: str, url_type: str) -> int:
    # Get repo id
    try: 
        repo_id = get_repo_id(url, url_type)
    except Exception as e:
        print(f"Error getting repo id {e}")
        return None

    license = None
    if url_type == "model":
        info = HF_API.model_info(repo_id=repo_id)
        license = getattr(info, "license", None)
    
    elif url_type == "dataset":
        info = HF_API.dataset_info(repo_id=repo_id)
        license = getattr(info, "license", None)

    elif url_type == "code":
        base_url = f"https://api.github.com/repos/{repo_id}"
        license_info = requests.get(f"{base_url}/license").json()
        license = license_info.get("license", {}).get("name")
    
    print(f"License for {url} is {license}")

    # Check if it's compatible with LGPLv2.1
    normalized = is_compatible(license)

    if normalized:
        return 1
    else: 
        return 0
