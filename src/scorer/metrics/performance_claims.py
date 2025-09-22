'''
Evaluate model performance claims.
'''

import os
import shutil
import tempfile
import time
from typing import Tuple
from git import Repo
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from pathlib import Path

def get_performance_claims(url: str, url_type: str) -> Tuple[float, int]:
    '''
    Function to get model or code performance claims based on URL type. 
    '''

    start_time = time.time()
    score = 0.0

    if url_type == 'code':
        # clone GitHub repo and check readme for performance claims
        score = _check_code_repo_performance(url)
    elif url_type == 'model':
        # check Hugging face model card for performance claims
        score = _check_model_card_performance(url)
    
    latency = int((time.time() - start_time) * 1000)
    
    return score, latency
   
def _check_code_repo_performance(code_url: str) -> float:
    '''
    Function to check the code repo for performance claims.
    '''

    score = 0.0
    temp_dir = tempfile.mkdtemp()

    try:
        # clone the repo
        try:
            Repo.clone_from(code_url, temp_dir)
        except Exception as e:
            print(f"Cannot clone repo: {e}")
            return 0.0
        
        # check README file
        readme_path = os.path.join(temp_dir, "README.md")
        text = ""
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", encoding = "utf-8", errors = "ignore") as f:
                    text = f.read().lower()
            except Exception:
                print("Cannot open readme")
        keywords = ["benchmark", "evaluation", "performance"]
        score = _keyword_score(text, keywords)

        # check for any test or evaluation scripts
        for filename in os.listdir(temp_dir):
            if "test" in filename.lower() or "eval" in filename.lower():
                score = max(score, 0.7)
    
    # remove the repo
    finally:
        shutil.rmtree(temp_dir, ignore_errors = True)
    
    return score
     
def _check_model_card_performance(model_url: str) -> float:
    '''
    Function to check the model card/README on Hugging Face for performance claims.
    '''

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")
    hf_token = os.getenv("HF_TOKEN")

    score = 0.0

    try:
        # extract repo_id from url
        if "huggingface.co/" not in model_url:
            raise ValueError(f"Invalid HuggingFace URL: {model_url}")
        model_id = model_url.split("huggingface.co/")[-1].strip("/")

        # download README.md from the repo
        readme_path = hf_hub_download(
            repo_id=model_id,
            filename="README.md",
            token=hf_token
        )

        # read full text
        with open(readme_path, "r", encoding="utf-8") as f:
            text = f.read().lower()
        keywords = ["benchmark", "evaluation", "performance"]
        score = _keyword_score(text, keywords)

    except Exception as e:
        print(f"Error checking model card: {e}")

    return score

def _keyword_score(text: str, keywords: list[str]) -> float:
    '''
    Function to count keywords in a string and compute score.
    '''

    if not text:
        return 0.0
    text = text.lower()
    matches = 0
    for keyword in keywords:
        if keyword in text:
            matches += 1        
    score = min(1.0, matches / len(keywords))
    return score
    
# TEST (DELETE THIS)
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found") 
    print("Loaded token starts with:", hf_token[:10])

    # model URL test
    url_model = f"https://huggingface.co/bert-base-uncased"
    url_type_model = "model"
    score, latency = get_performance_claims(url_model, url_type_model)
    print(f"Model: Score = {score:.2f}, Latency={latency}ms")

    # code URL test
    url_code = "https://github.com/pallets/flask"
    url_type_code = "code"
    score, latency = get_performance_claims(url_code, url_type_code)
    print(f"Code: Score = {score:.2f}, Latency= {latency}ms")