'''
CLI for scoring tool.
'''

from __future__ import annotations
import argparse
from typing import List
from pathlib import Path
import sys, os
from utils.logging import setup_logging, set_run_id, get_logger, set_url
from url_handler.base import classify_url
from url_handler.model import handle_model_url
from url_handler.dataset import handle_dataset_url
from url_handler.code import handle_code_url
from metrics.size import get_size_score
from metrics.license import get_license_score
from metrics.dataset_quality import get_dataset_quality_score
from metrics.code_quality import *
from metrics.performance_claims import get_performance_claims_score

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
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="File log level (default INFO)"
    )
    parser.add_argument(
        "--log-text",
        action="store_true",
        help="Use plain text logs instead of JSON Lines"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id to correlate logs across processes"
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

    # Configure the log destination first
    if args.log_file:
        os.environ["LOG_FILE"] = str(args.log_file)

    # Init logging
    setup_logging(level=args.log_level, json_lines=not args.log_text)
    run_id = set_run_id(args.run_id)
    log = get_logger("cli")

    # open URL file
    try:
        urls = read_urls(args.url_file)
    except Exception as e:
        print(f"Error reading URL file {e}", file = sys.stderr)
        sys.exit(1)
    
    # show that URLs are being processed
    print(f"Processing {len(urls)} URLs...")

    # Classify URLs by type (model, dataset, code)
    classifications = {}
    for url in urls:
        url_type = classify_url(url)
        if url_type == "unknown":
            print(f"Error: Unknown URL type for {urls}", file = sys.stderr)
            sys.exit(1)
        if url_type == "model":
            handle_model_url(url)
            classifications[url] = url_type
        elif url_type == "dataset":
            handle_dataset_url(url)
            classifications[url] = url_type
        elif url_type == "code":
            handle_code_url(url)
            classifications[url] = url_type

        # Calculate metrics
        size_score = get_size_score(url, url_type)
        license_score = get_license_score(url, url_type)
        dataset_quality_score = get_dataset_quality_score(url, url_type)
        code_quality_score = get_code_quality_score(url, url_type)
        performance_score = get_performance_claims_score(url, url_type)
        

    # TEMPORARY OUTPUT, REPLACE LATER
    for url in urls:
        print(f"{url} -> NetScore: 0.0")

if __name__  == "__main__":
    main()