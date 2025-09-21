'''
Evaluate model performance claims.
'''

import os
import shutil
import tempfile
import time
from git import Repo
from typing import Tuple
from dotenv import load_dotenv
from pathlib import Path

def get_performance_claims_score(url : str, url_type : str) -> Tuple[float, int]:
    # mark start time for latency calculation
    start_time = time.time()

    # make temporary directory to locally clone repo to
    temp_dir = tempfile.mkdtemp()

    try:
        # clone the repo to temporary directory
        print("Cloning model repo to local directory...")
        Repo.clone_from(url, temp_dir)
        print("Directory cloned successfully.")

        # try to find README file
        print("Finding README file...")
        readme_path = os.path.join(temp_dir, "README.md")
        score = 0.0
        print("README found successfully.")

        # if README exists, open it and analyze
        if os.path.exists(readme_path):
            print("Opening README...")
            with open(readme_path, "r", encoding = "utf-8", errors = "ignore") as f:
                text = f.read().lower()
            print("README opened successfully.")
            
            # define keywords related to performance
            keywords = ["accuracy", "precision", "f1", "recall", "benchmark", "evaluation", "performance"]
            
            print("Searching README for keywords...")
            # count the matches to keywords
            matches = 0
            for keyword in keywords:
                if keyword in text:
                    matches += 1
            
            print("Computing score...")
            # normalize score
            score = min(1.0, matches / len(keywords))

        print("Checking for test and evaluation files...")
        # check for evaluation and test scripts
        for filename in os.listdir(temp_dir):
            if "test" in filename.lower() or "eval" in filename.lower():
                score = max(score, 0.7)
            print("     Found files, updating score...")

        print("Calculating latency...")
        # calculate latency to nearest millisecond
        performance_claims_latency = int((time.time() - start_time) * 1000)

        print("Finished.")
        return score, performance_claims_latency
    
    # remove local repo
    finally:
        print("Removing local model repo...")
        shutil.rmtree(temp_dir, ignore_errors = True)
        print("Repo removed.")

# TEMPORARY MAIN FUNCTION
if __name__ == "__main__":
    load_dotenv(dotenv_path = Path(__file__).resolve().parents[3] / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not found")
    print("Loaded token starts with:", token[:10])

    url = f"https://hf:{token}@huggingface.co/bert-base-uncased"
    url_type = ""
    score, latency = get_performance_claims_score(url, url_type)
    print(f"Performance claims score: {score}, latency: {latency} ms")