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
import re


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
                with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
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
        shutil.rmtree(temp_dir, ignore_errors=True)

    return score


def _check_model_card_performance(model_url: str) -> float:
    '''
    Function to check the model card/README on Hugging Face for performance claims.
    '''

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")
    hf_token = os.getenv("HF_TOKEN")

    score = 0.0

    try:
        # # extract repo_id from url
        # if "huggingface.co/" not in model_url:
        #     raise ValueError(f"Invalid HuggingFace URL: {model_url}")
        # model_id = model_url.split("huggingface.co/")[-1].strip("/")

        # extract repo_id from URL
        if "huggingface.co/" not in model_url:
            raise ValueError(f"Invalid HuggingFace URL: {model_url}")

        model_id = model_url.split("huggingface.co/")[-1].strip("/")

        # Remove any /tree/main, /blob/... etc
        model_id = model_id.split("/tree")[0]
        model_id = model_id.split("/blob")[0]

        # download README.md from the repo
        readme_path = hf_hub_download(
            repo_id=model_id,
            filename="README.md",
            token=hf_token
        )

        # read full text
        with open(readme_path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        # define keywords to check in the readme
        sentences = re.split(r"[.!?]", text)
        keywords = [
            "benchmark", "evaluation", "performance", "metric", "score", "result",
            "outcome", "effectiveness", "efficacy", "validation", "accuracy", "f1",
            "precision", "recall", "auc", "roc", "top-1", "top-5", "mse", "mae", "rmse",
            "loss", "cross-entropy", "log-loss", "bleu", "rouge", "meteor",
            "perplexity", "iou", "ap", "map", "precision-recall", "latency",
            "throughput", "fps", "speed", "memory", "params", "size", "parameter",
            "parameters", "recognition", "beneficial"
        ]

        # Keep track of keywords that have already been counted
        counted_keywords = set()
        keyword_count = 0

        for sent in sentences:
            sent_lower = sent.lower()
            for kw in keywords:
                # match whole word only using \b for word boundaries
                if re.search(rf"\b{re.escape(kw)}\b", sent_lower):
                    if kw not in counted_keywords:
                        # bonus if numeric value present
                        if re.search(r"\b\d+(\.\d+)?%?\b", sent_lower):
                            keyword_count += 2
                        else:
                            keyword_count += 1
                        counted_keywords.add(kw)
                        # print(kw)

        score = min(keyword_count / 10, 1.0)

        # print(f"Number of performance keywords = {keyword_count}")

    except Exception as e:
        print(f"Error checking model card: {e}")

    return round(score, 2)


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
