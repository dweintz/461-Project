'''
CLI for scoring tool.
'''

from __future__ import annotations
import argparse
from typing import List
from pathlib import Path
import sys
from url_handler.base import classify_url
from url_handler.model import handle_model_url
from url_handler.dataset import handle_dataset_url
from url_handler.code import handle_code_url

# ONCE THEY ARE CODED, WILL IMPORT URL HANDLERS HERE AND CALL THEM LATER


def parse_args() -> argparse.Namespace:
    '''
    Parse CLI arguments
    '''
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
        action = "store_true",
        help = "Calculate metrics in parallel"
    )
    parser.add_argument(
        "--log-file",
        type = Path,
        default = None,
        help = "Path to write log file (if not set, logs go to stdout)"
    )
    return parser.parse_args()

def read_urls(file_path: Path) -> List[str]:
    '''
    read newline-delimited URLs from a file
    '''
    # check if file path exists
    if not file_path.exists():
        raise FileNotFoundError(f"URL file {file_path} does not exist.")
    
    # open file and add URLs to a list
    with file_path.open("r", encoding = "utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
def main() -> None:
    # get CLI arguments
    args = parse_args()

    # open URL file
    try:
        urls = read_urls(args.url_file)
    except Exception as e:
        print(f"Error reading URL file {e}", file = sys.stderr)
        sys.exit(1)
    
    # show that URLs are being processed
    print(f"Processing {len(urls)} URLs...")

    # TODO: INITIALIZE HANDLERS AND METRICS

    # Classify URLs by type (model, dataset, code)
    url_type = classify_url(urls)
    if url_type == "unknown":
        print(f"Error: Unknown URL type for {urls}", file = sys.stderr)
        sys.exit(1)
    if url_type == "model":
        handle_model_url(urls)
    elif url_type == "dataset":
        handle_dataset_url(urls)
    elif url_type == "code":
        handle_code_url(urls)

    # TEMPORARY OUTPUT, REPLACE LATER
    for url in urls:
        print(f"{url} -> NetScore: 0.0")

if __name__  == "__main__":
    main()