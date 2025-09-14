'''
CLI for scoring tool.
'''

from __future__ import annotations
import argparse
from typing import Dict, Any
from pathlib import Path
import sys
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "CLI for scoring models, datasets, and code."
    )
    parser.add_argument(
        "url_file",
        type = Path,
        help = "Path to a newline-delimited file containing URLS of type model, dataset, and/or code"
    )
    parser.add_argument(
        "--parallel",
        action = "store true",
        help = "Calculate metrics in parallel"
    )
    parser.add_argument(
        "--log-file",
        type = Path,
        default = None,
        help = "Path to write log file (if not set, logs go to stdout)"
    )
    return parser.parse_args()