'''
This code implements the ramp up feasibility of the hugging face model by utilizing an LLM prompt.
'''

from __future__ import annotations

import os
import re
import time
import json
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, List

from git import Repo

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

README_CANDIDATES = [
    "README.md", "readme.md", "README.rst", "readme.rst", "README", "Readme.md",
    "docs/README.md", "docs/index.md"
]
SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "node_modules", "dist", "build"}

def _read_first_readme(repo_dir: str) -> str:
    for rel in README_CANDIDATES:
        p = Path(repo_dir) / rel
        if p.exists() and p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return ""

def _top_level_summary(repo_dir: str, max_files: int = 120) -> str:
    """Simple, compact snapshot of the repo tree to give the LLM context."""
    entries: List[str] = []
    root = Path(repo_dir)
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # prune noisy dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel_dir = os.path.relpath(dirpath, root)
        for f in filenames:
            rel = os.path.normpath(os.path.join(rel_dir, f))
            if rel.startswith("./"):
                rel = rel[2:]
            entries.append(rel)
            count += 1
            if count >= max_files:
                break
        if count >= max_files:
            break
    return "\n".join(entries)


# Prompt to the LLM
SYSTEM_PROMPT = """You are a precise software onboarding evaluator.
Given a repository README and a brief repo file listing, rate how fast a new engineer could ramp up.
Consider ONLY: installation clarity, prerequisites, quickstart/usage examples, runnable commands, troubleshooting,
links to docs/tutorials, and overall coherence/structure of the README.

Return STRICT JSON with two fields:
{"score": <float between 0 and 1>, "rationale": "<<=200 chars explanation>"}

Do NOT include anything else.
"""

USER_PROMPT_TEMPLATE = """REPO SUMMARY (first {n_files} files)
----------------
{tree}

README (truncated if very long)
----------------
{readme}
"""

def _ask_llm(readme: str, tree: str) -> float:
    if InferenceClient is None:
        raise RuntimeError("huggingface_hub not installed. `pip install huggingface_hub`")

    model_id = os.getenv("RAMPUP_LLM_MODEL", "").strip()
    token = os.getenv("HF_TOKEN", "").strip()
    if not model_id or not token:
        raise RuntimeError("RAMPUP_LLM_MODEL and HF_TOKEN must be set for LLM-only ramp-up scoring.")

    # Cap README to keep prompt small
    readme = (readme or "").strip()
    if len(readme) > 20000:
        readme = readme[:20000] + "\n\n[TRUNCATED]"

    user_prompt = USER_PROMPT_TEMPLATE.format(n_files=120, tree=tree[:8000], readme=readme)

    client = InferenceClient(model=model_id, token=token, timeout=90)

    # For serverless text-generation models; adjusted for concise JSON output
    completion = client.text_generation(
        prompt=f"{SYSTEM_PROMPT}\n{user_prompt}",
        max_new_tokens=200,
        temperature=0.2,
        do_sample=False,
        return_full_text=False,
    )

    # Try strict JSON first
    txt = completion.strip()
    try:
        data = json.loads(txt)
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
        # Fallback: find a float 0..1 in output
        m = re.search(r"([01](?:\.\d+)?|\.\d+)", txt)
        if m:
            try:
                val = float(m.group(1))
                return max(0.0, min(1.0, val))
            except Exception:
                pass
    return 0.0

# Public API
def ramp_up(url: str) -> Tuple[float, int]:
    """
    LLM-only ramp-up metric.
    - Clones repo
    - Sends README + repo summary to LLM
    - Returns LLM-provided score in [0,1]
    """
    start = time.time()
    temp_dir = tempfile.mkdtemp()
    try:
        Repo.clone_from(url, temp_dir)
        readme = _read_first_readme(temp_dir)
        tree = _top_level_summary(temp_dir, max_files=120)
        score = _ask_llm(readme, tree)
        latency_ms = int((time.time() - start) * 1000)
        return score, latency_ms
    except Exception as e:
        print(f"[rampup] Error: {e}")
        return 0.0, int((time.time() - start) * 1000)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Local test main
if __name__ == "__main__":
    token = os.getenv("HF_TOKEN")
    model = os.getenv("RAMPUP_LLM_MODEL")
    if not token:
        print("Not token")
    if not model:
        print("Not model")
    if not token or not model:
        print("Set HF_TOKEN and RAMPUP_LLM_MODEL before running this module directly.")
    url = f"https://hf:{token}@huggingface.co/bert-base-uncased" if token else "https://huggingface.co/bert-base-uncased"
    s, ms = ramp_up(url)
    print(f"Ramp-Up (LLM-only) score: {s:.3f}, latency: {ms} ms")