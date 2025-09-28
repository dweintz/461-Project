'''
CLI for scoring tool.
'''

from __future__ import annotations
import argparse
from typing import List
from pathlib import Path
import time
import sys, os
import json
from utils.logging import setup_logging, set_run_id, get_logger, set_url
from url_handler.base import classify_url
from url_handler.model import handle_model_url
from url_handler.dataset import handle_dataset_url
from url_handler.code import handle_code_url
from metrics.size import get_size_score
from metrics.license import get_license_score
from metrics.dataset_quality import get_dataset_quality_score
from metrics.code_quality import get_code_quality
from metrics.performance_claims import get_performance_claims
from metrics.rampup import get_ramp_up
from metrics.busfactor import get_bus_factor

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
        help = "Path to write log file (if not set, uses $LOG_FILE or logs/scorer.log)"
    )
    parser.add_argument(
        "--log-level",
        type=int,
        choices=[0, 1, 2],
        default=int(os.environ.get("LOG_LEVEL", "0")),
        help="Verbosity: 0=silent, 1=info, 2=debug (default 0 or $LOG_LEVEL)"
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
    
    urls = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            # Separate commas and strip whitespace
            parts = [url.strip() for url in line.split(",") if url.strip()]
            
            urls.append(parts)
    return urls
    
def main() -> None:
    # get CLI arguments
    args = parse_args()

    url_file_path = args.url_file.resolve()
    
    # check GitHub token
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
    if not GITHUB_TOKEN:
        print("Warning: GITHUB_TOKEN environment variable is not set or empty.", file=sys.stderr)
        sys.exit(1)
        
    # Configure the log destination first
    if args.log_file:
        os.environ["LOG_FILE"] = str(args.log_file)
    else:
        # ensure a default is present so the file is always produced
        os.environ.setdefault("LOG_FILE", "logs/scorer.log")
    
    # Init logging
    os.environ["LOG_LEVEL"] = str(args.log_level)
    setup_logging(level=args.log_level, json_lines=not args.log_text)
    run_id = set_run_id(args.run_id)
    log = get_logger("cli")

    start_ns = time.perf_counter_ns()
    log.info("run started", extra={"phase": "run", "function": "main", "run_id": run_id})

    # open URL file
    try:
        urls = read_urls(url_file_path)
        log.info("read urls", extra={"phase": "run", "count": len(urls), "file": str(url_file_path)})
    except Exception as e:
        print(f"Error reading URL file {e}", file = sys.stderr)
        log.exception("failed to read url file", extra={"phase": "run"})
        sys.exit(1)
    
    # Classify URLs by type (model, dataset, code)
    classifications = []
    for line in urls:
        line_classifications = {}
        for url in line:
            log.info("processing url", extra={"phase": "controller", "url": url})
            try:
                url_type = classify_url(url)
                log.info("classified", extra={"phase": "controller", "type": url_type})
            except Exception:
                log.exception("classification failed", extra={"phase": "controller"})
                print(f"{url} -> ERROR: classification failed")
                continue

            if url_type == "unknown":
                log.warning("unknown url type", extra={"phase": "controller"})
                print(f"Error: Unknown URL type for {urls}", file = sys.stderr)
                sys.exit(1)
            if url_type == "model":
                handle_model_url(url)
                line_classifications[url] = url_type
            elif url_type == "dataset":
                handle_dataset_url(url)
                line_classifications[url] = url_type
            elif url_type == "code":
                handle_code_url(url)
                line_classifications[url] = url_type
        classifications.append(line_classifications)
    
    # Calculate metrics
    start_time = time.time()

    for line in classifications:
        # intialize all fields to zero

        # string fields
        name = ""
        category = ""

        # float scores (0–1)
        net_score = 0.0
        ramp_up = 0.0
        bus_factor = 0.0
        performance_claims = 0.0
        license = 0.0
        dataset_and_code_score = 0.0
        dataset_quality = 0.0
        code_quality = 0.0

        # latencies (milliseconds)
        net_score_latency = 0
        ramp_up_latency = 0
        bus_factor_latency = 0
        performance_claims_latency = 0
        license_latency = 0
        size_latency = 0
        dataset_and_code_score_latency = 0
        dataset_quality_latency = 0
        code_quality_latency = 0

        # object field (dict)
        size_dict = {
            "raspberry_pi": 0.0,
            "jetson_nano": 0.0,
            "desktop_pc": 0.0,
            "aws_server": 0.0
        }

        # update fields based on URL type
        for url, url_type in zip(line.keys(), line.values()):
            name = url.split("/")[-1]
            category = url_type.upper()

            if url_type == 'code':
                code_quality, code_quality_latency = get_code_quality(url, url_type)
            elif url_type == 'dataset':
                dataset_quality, dataset_quality_latency = get_dataset_quality_score(url, url_type)
                dataset_and_code_score, dataset_and_code_score_latency = 0.0, 0.0
            elif url_type == 'model':
                size_dict, size_latency = get_size_score(url, url_type)
                license, license_latency = get_license_score(url, url_type)
                performance_claims, performance_claims_latency = get_performance_claims(url, url_type)
                bus_factor, bus_factor_latency = get_bus_factor(url, url_type)
                ramp_up, ramp_up_latency = get_ramp_up(url, url_type)
            log.info("url done", extra = {"phase": "controller", "url": url})

        # Compute net score
        size_score = 0.0
        if size_dict:
            size_score = sum(size_dict.values()) / len(size_dict)

        net_score = 0.15 * size_score + \
                    0.15 * license + \
                    0.10 * ramp_up + \
                    0.10 * bus_factor + \
                    0.15 * dataset_quality + \
                    0.10 * code_quality + \
                    0.15 * performance_claims + \
                    0.10 * dataset_and_code_score

        # Compute net score latency in milliseconds
        net_score_latency = int((time.time() - start_time) * 1000)

        # Build NDJSON output
        output = {
            "name":name,
            "category":category,
            "net_score":round(net_score, 2),
            "net_score_latency":net_score_latency,
            "ramp_up_time":round(ramp_up, 2),
            "ramp_up_time_latency":ramp_up_latency,
            "bus_factor":round(bus_factor, 2),
            "bus_factor_latency":bus_factor_latency,
            "performance_claims":round(performance_claims, 2),
            "performance_claims_latency":performance_claims_latency,
            "license":round(license, 2),
            "license_latency": license_latency,
            "size_score":{k: round(v, 2) for k, v in size_dict.items()} if size_dict else {},
            "size_score_latency":size_latency,
            "dataset_and_code_score":round(dataset_and_code_score, 2),
            "dataset_and_code_score_latency":dataset_and_code_score_latency,
            "dataset_quality":round(dataset_quality, 2),
            "dataset_quality_latency":dataset_quality_latency,
            "code_quality":round(code_quality, 2),
            "code_quality_latency":code_quality_latency
        }

        print(json.dumps(output, separators=(',', ':')))

    dur_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
    log.info("run finished", extra={"phase": "run", "function": "main", "latency_ms": dur_ms})
    exit(0)
    
if __name__  == "__main__":
    main()