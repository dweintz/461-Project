# This code implements the ramp up feasibility of the hugging face model by utilizing an LLM prompt.

from __future__ import annotations

import os
import re
import time
import json
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional

from git import Repo
from dotenv import load_dotenv

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

# NEW: normalize clone targets (works for GitHub & HF, strips /tree/main)
from urllib.parse import urlparse

def _to_clone_url(url: str, url_type: str) -> str:
    p = urlparse(url)
    host = p.netloc.lower()
    parts = [x for x in p.path.split("/") if x]

    if "github.com" in host:
        if len(parts) < 2:
            raise ValueError("GitHub URL must be /owner/repo")
        return f"https://github.com/{parts[0]}/{parts[1]}.git"

    if "huggingface.co" in host:
        # models:  /<ns>/<name>[/...]
        # datasets:/datasets/<ns>/<name>[/...]
        if parts and parts[0].lower() == "datasets":
            if len(parts) < 3:
                raise ValueError("HF dataset URL must be /datasets/<ns>/<name>")
            repo_id = f"{parts[1]}/{parts[2]}"
            return f"https://huggingface.co/datasets/{repo_id}"
        else:
            if len(parts) < 2:
                raise ValueError("HF model URL must be /<ns>/<name>")
            repo_id = f"{parts[0]}/{parts[1]}"
            return f"https://huggingface.co/{repo_id}"

    # fallback (may still work if it's plain git)
    return url

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
    entries: List[str] = []
    root = Path(repo_dir)
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
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

SYSTEM_PROMPT = (
    "You are a precise software onboarding evaluator.\n"
    "Given a repository README and a brief repo file listing, rate how fast a new engineer could ramp up.\n"
    "Consider ONLY: installation clarity, prerequisites, quickstart/usage examples, runnable commands, troubleshooting,\n"
    "links to docs/tutorials, and overall coherence/structure of the README.\n\n"
    "Return STRICT JSON with two fields:\n"
    '{"score": <float between 0 and 1>, "rationale": "<<=200 chars explanation>"}\n\n'
    "Do NOT include anything else."
)
USER_PROMPT_TEMPLATE = "REPO SUMMARY (first {n_files} files)\n----------------\n{tree}\n\nREADME (truncated if very long)\n----------------\n{readme}\n"

# --- NEW: small heuristic fallback if LLM not available or fails
_HEUR_PATTERNS = [
    r"\bpip install\b", r"\bconda (?:create|install)\b", r"\bgit clone\b", r"\bpython (?:-m )?\w+\.py\b",
    r"\busage\b", r"\bquick\s*start\b", r"\bexample\b", r"\brequirements\.txt\b", r"\benvironment\.yml\b",
    r"```", r"\btroubleshoot", r"\bfaq\b", r"\bdocs?\b", r"\btutorial\b"
]
def _heuristic_rampup(readme: str, tree: str) -> float:
    txt = (readme or "") + "\n" + (tree or "")
    hits = sum(1 for pat in _HEUR_PATTERNS if re.search(pat, txt, flags=re.IGNORECASE))
    # Cap at 1.0, tuned so good repos land ~0.8-0.95
    return max(0.0, min(1.0, 0.15 + 0.06 * hits))

def _ask_llm(readme: str, tree: str) -> Optional[float]:
    if InferenceClient is None:
        return None

    model_id = os.getenv("RAMPUP_LLM_MODEL", "").strip()
    token = os.getenv("HF_TOKEN", "").strip()
    if not model_id or not token:
        return None

    readme = (readme or "").strip()
    if len(readme) > 20000:
        readme = readme[:20000] + "\n\n[TRUNCATED]"
    user_prompt = USER_PROMPT_TEMPLATE.format(n_files=120, tree=tree[:8000], readme=readme)

    client = InferenceClient(model=model_id, token=token, timeout=90)

    # 1) Try chat-completions (OpenAI-style)
    try:
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.2,
        )
        txt = (resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict)
               else resp.choices[0].message.content)
        data = json.loads((txt or "").strip())
        return float(data.get("score", 0.0))
    except Exception:
        pass

    # 2) Try provider's conversational task (the one your error mentions)
    try:
        conv = client.conversational(
            inputs=user_prompt,
            parameters={"max_new_tokens": 200, "temperature": 0.2},
            system_prompt=SYSTEM_PROMPT,
        )
        # conv can be dict or string depending on provider
        if isinstance(conv, dict):
            txt = conv.get("generated_text") or (
                conv.get("conversation", {}).get("generated_responses", [""]) or [""]
            )[-1]
        else:
            txt = str(conv)
        data = json.loads((txt or "").strip())
        return float(data.get("score", 0.0))
    except Exception:
        pass

    # 3) Fallback to raw text-generation
    try:
        gen = client.text_generation(
            prompt=f"{SYSTEM_PROMPT}\n{user_prompt}",
            max_new_tokens=200,
            temperature=0.2,
            do_sample=False,
            return_full_text=False,
        )
        data = json.loads((gen or "").strip())
        return float(data.get("score", 0.0))
    except Exception:
        pass

    # If all LLM attempts fail
    return None

# Public API
def get_ramp_up(url: str, url_type: str) -> Tuple[float, int]:
    """
    Clone a normalized repo URL, send README + repo summary to an LLM when available,
    otherwise use a heuristic. Returns (score, latency_ms).
    """
    start = time.time()
    temp_dir = tempfile.mkdtemp()
    try:
        kind = "dataset" if url_type.lower() == "dataset" else "model" if url_type.lower() == "model" else "code"
        clone_url = _to_clone_url(url, kind)

        env = os.environ.copy()
        env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
        repo = Repo.clone_from(clone_url, temp_dir, multi_options=["--depth=100"], env=env)

        readme = _read_first_readme(temp_dir)
        tree = _top_level_summary(temp_dir, max_files=120)

        score = _ask_llm(readme, tree)
        if score is None:
            score = _heuristic_rampup(readme, tree)

        latency_ms = int((time.time() - start) * 1000)
        return max(0.0, min(1.0, float(score))), latency_ms

    except Exception:
        # Don’t print to stdout; return neutral-low fallback
        return 0.0, int((time.time() - start) * 1000)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
