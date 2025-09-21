'''
Evaluates the quality of the code.
'''

import os
import shutil
import tempfile
import time
import subprocess
from git import Repo
from typing import Tuple
import ast


def run_radon(path: str) -> float:
    return 0.0

def run_lizard(path: str) -> float:
    return 0.0

def docstring_ratio(path: str) -> float:
    return 0.0

def code_quality(url: str) -> Tuple[float, int]:
    return 0.0, 0

if __name__ == "__main__":
    url = "https://github.com/some/repo.git"
    score, latency = code_quality(url)
    print(f"Code quality score: {score:.2f}, latency: {latency} ms")