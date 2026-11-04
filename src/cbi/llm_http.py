"""Shared LLM HTTP helper: process-local concurrency cap + connection retry."""
from __future__ import annotations

import os
import random
import threading
import time

import requests

_LIMIT = max(1, int(os.environ.get("CBI_LLM_MAX_CONCURRENCY", "4")))
_SEM = threading.BoundedSemaphore(_LIMIT)


def post(url: str, *, headers: dict | None = None, timeout: int = 120, **kwargs):
    headers = dict(headers or {})
    headers.setdefault("Connection", "close")
    last: Exception | None = None
    for attempt in range(6):
        try:
            with _SEM:
                return requests.post(url, headers=headers, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last = exc
            time.sleep(min(30.0, 1.5 ** attempt) + random.random())
    assert last is not None
    raise last
