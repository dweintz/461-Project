# loggin config
'''
Centralized logging for files
'''

from __future__ import annotations
import json, logging, os, time, traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional, Dict
import contextvars
from datetime import datetime, timezone
import uuid

# Adding context so that every logging line can be consistent and grepable
_run_id_var = contextvars.ContextVar("run_id", default=None)
_url_var    = contextvars.ContextVar("url", default=None)
_metric_var = contextvars.ContextVar("metric", default=None)

def set_run_id(run_id: Optional[str] = None) -> str:
    rid = run_id or str(uuid.uuid4())
    _run_id_var.set(rid)
    return rid

def set_url(url: Optional[str]) -> None:
    _url_var.set(url)

def set_metric(metric_name: Optional[str]) -> None:
    _metric_var.set(metric_name)

def _extra(**kw) -> Dict[str, Any]:
    ex = dict(run_id=_run_id_var.get(),
              url=_url_var.get(),
              metric=_metric_var.get())
    ex.update({k: v for k, v in kw.items() if v is not None})
    return ex

# Formatting
class JSONLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        payload = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage()
        }

        for k in ("run_id","url","metric","latency_ms","function","phase"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
            elif record.__dict__.get("extra", {}).get(k) is not None:
                payload[k] = record.__dict__["extra"][k]
        if record.exc_info:
            payload["exc"] = "".join(traceback.format_exception(*record.exc_info))[-4000:]
        return json.dumps(payload, ensure_ascii=False)
    
# For text formatting
class TextFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(name)s | run=%(run_id)s url=%(url)s metric=%(metric)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

# For first-time setup
_INITIALIZED = False

def setup_logging(*, level: str = "INFO", json_lines: bool = True) -> Path:
    """
    Initialize project logging.
    - File path from $LOG_FILE, else ./logs/scorer.log
    - File-only; no console handler (keeps stdout clean).
    - Rotates at 5 MB, keep 2 backups.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return Path(os.environ.get("LOG_FILE", "logs/scorer.log")).resolve()

    log_path = Path(os.environ.get("LOG_FILE", "logs/scorer.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("scorer")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
    handler.setLevel(logger.level)
    handler.setFormatter(JSONLineFormatter() if json_lines else TextFormatter())
    logger.addHandler(handler)

    logging.getLogger().handlers.clear()

    _INITIALIZED = True
    return log_path.resolve()

def get_logger(name: Optional[str] = None) -> logging.LoggerAdapter:
    base = logging.getLogger("scorer" + ("" if not name else f".{name}"))
    return logging.LoggerAdapter(base, _extra())

# Decorator for logs
def log_call(phase: str = "metric") -> Callable:
    def deco(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            log = get_logger(fn.__module__)
            start = time.perf_counter_ns()
            log.info(f"start {fn.__name__}", extra=_extra(phase=phase, function=fn.__name__))
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception:
                dur_ms = (time.perf_counter_ns() - start) // 1_000_000
                log.exception(f"error {fn.__name__}", extra=_extra(phase=phase, function=fn.__name__, latency_ms=dur_ms))
                raise
            finally:
                dur_ms = (time.perf_counter_ns() - start) // 1_000_000
                log.info(f"end {fn.__name__}", extra=_extra(phase=phase, function=fn.__name__, latency_ms=dur_ms))
        return wrapper
    return deco