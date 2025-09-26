'''
Evaluates the quality of the code.
'''

import os
import shutil
import tempfile
import time
import subprocess
from git import Repo
from typing import Tuple, Dict
import ast
from dotenv import load_dotenv
from pathlib import Path
import json
import re

# function to run radon, a Python tool that analyzes source code complexity and maintainability
def run_radon(path: str) -> float:
    try:
        result = subprocess.run(
            ["radon", "mi", "-s", path],
            capture_output = True,
            text = True,
            check = True
        )
    except subprocess.CalledProcessError as e:
        print(f"Radon failed:\n{e.stderr}")
        exit(1)

    # define corresponding score to each grade
    score_dict = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "F": 0.0}

    scores = []
    for line in result.stdout.splitlines():
        if " - " in line:
            score = line.split(" -")[-1].strip()
            scores.append(score_dict.get(score[0], 0.0))
            print(score)
            print(score_dict.get(score[0], 0.0))
    if scores:
        final_score = sum(scores) / len(scores)
    else:
        final_score = 0.0  
    
    return final_score
    
# path = "461-Project/src/scorer/metrics"
# print("Files in path:", os.listdir(path))
# run_radon(path)


# NLOC = non-comment lines of code
# CNN = cyclomatic complexity number - measures number of paths through a function
# token = number of syntactic tokens (operators, keywords, identifiers, etc)
# PARAM = number of parameters
# length = total lines of code, including comments
# location = path where function is defined



# function to run lizard, a multi-language code analysis tool that analyzes function complexity
def run_lizard(path: str) -> Dict:
    try:
        result = subprocess.run(
            ["lizard", path],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Lizard failed:\n{e.stderr}")
        exit(1)

    print(result.stdout)

    # find the last summary row using regex
    total_row = None
    for line in result.stdout.splitlines():
        # match a line that starts with spaces then numbers/floats
        if re.match(r"^\s*\d+", line.strip()):
            total_row = line.strip()
    
    if not total_row:
        print("Could not find totals in Lizard output")
        exit(1)

    # extract numeric values from the totals row
    parts = total_row.split()
    if len(parts) < 8:
        print("Unexpected totals format:", total_row)
        exit(1)

    # put values into a dict
    totals = {
        "Total NLOC": float(parts[0]),
        "Avg NLOC": float(parts[1]),
        "Avg CCN": float(parts[2]),
        "Avg Tokens": float(parts[3]),
        "Function Count": int(parts[4]),
        "Warning Count": int(parts[5]),
        "Function Rate": float(parts[6]),
        "NLOC Rate": float(parts[7]),
    }
    return totals


path = "461-Project/src/scorer/metrics"
print("Files in path:", os.listdir(path))
# run_radon(path)
totals = run_lizard(path)
print(totals)


# # function to count how many Python functions or classes have docstrings
# def docstring_ratio(path: str) -> float:
#     total = 0
#     documented = 0

#     for root, _, files in os.walk(path):
#         for file in files:
#             if file.endswith(".py"):
#                 with open(os.path.join(root, file), "r", encoding = "utf-8", errors = "ignore") as fh:
#                     try:
#                         tree = ast.parse(fh.read())
#                         for node in ast.walk(tree):
#                             if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
#                                 total += 1
#                                 if ast.get_docstring(node):
#                                     documented += 1
#                     except Exception:
#                         continue
#     return documented / total if total > 0 else 0.0

# # function to analyze the quality of the code
# def get_code_quality(url: str, url_type: str) -> Tuple[float, int]:
#     start_time = time.time()
#     temp_dir = tempfile.mkdtemp()

#     try:
#         print("Cloning repo from Hugging Face...")
#         Repo.clone_from(url, temp_dir)
#         print("Repo successfully cloned.")

#         # first reliability check - check for the word test in the files
#         print("Checking reliability...")
#         print("     Checking for keyword 'test'...")
#         reliability = 0.0
#         for root, _, files in os.walk(temp_dir):
#             for file in files:
#                 if "test" in file.lower():
#                     reliability = 0.7
#                     break
        
#         # second reliability check - check for testing frameworks
#         print("     Checking for testing frameworks...")
#         file_names = " ".join(os.listdir(temp_dir))
#         if any(x in file_names.lower() for x in ["pytest", "unittest", "mocha", "jest"]):
#             reliability = 1.0

#         # check extendability (number of files and classes, complexity, multi-language use, etc)
#         print("Checking extendability...")
#         print("     Check radon score...")
#         radon_score = run_radon(temp_dir)
#         print("     Check lizard score...")
#         lizard_score = run_lizard(temp_dir)
#         extendibility = max(radon_score, lizard_score)

#         # check testabilty (CI/CD configs)
#         print("Checking extendability...")
#         testability = 0.0
#         for ci in [".github", ".gitlab-ci.yml", "azure-pipelines.yml"]:
#             if os.path.exists(os.path.join(temp_dir, ci)):
#                 testability = 1.0
#                 break

#         # check portability (check enviornment files)
#         print("Checking portability...")
#         portability = 0.0
#         if os.path.exists(os.path.join(temp_dir, "Dockerfile")):
#             portability += 0.5
#         if os.path.exists(os.path.join(temp_dir, "requirements.txt")) or \
#            os.path.exists(os.path.join(temp_dir, "environment.yml")):
#             portability += 0.5

#         # reusability (check for README)
#         print("Checking reusability...")
#         reusability = max(docstring_ratio(temp_dir), 0.5 if os.path.exists(os.path.join(temp_dir, "README.md")) else 0)

#         # compute weighted score
#         print("Computing weighted score...")
#         final_score = (
#             reliability * 0.35 +
#             extendibility * 0.30 +
#             testability * 0.1 +
#             portability * 0.1 +
#             reusability * 0.15
#         )

#         latency = int((time.time() - start_time) * 1000)
#         return min(1.0, final_score), latency

#     # remove the local copy of the repo
#     finally:
#         shutil.rmtree(temp_dir, ignore_errors = True)

# # TEMPORARY MAIN FUNCTION
# if __name__ == "__main__":
#     print("Starting program...")
#     load_dotenv(dotenv_path = Path(__file__).resolve().parents[3] / ".env")
#     token = os.getenv("HF_TOKEN")
#     if not token:
#         raise RuntimeError("HF_TOKEN not found")
#     print("Loaded token starts with:", token[:10])

#     url = f"https://hf:{token}@huggingface.co/bert-base-uncased"
#     url_type = 'model'
#     score, latency = get_code_quality(url, url_type)
#     print(f"Code quality score: {score}, latency: {latency} ms")