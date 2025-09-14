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
from datetime import datetime
import uuid

# Adding context so that every logging line can be consistent
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

