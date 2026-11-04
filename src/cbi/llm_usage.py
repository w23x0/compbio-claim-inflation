"""Process-local token accounting for LLM calls."""
from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.Lock()
_TOTALS: dict[str, int] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "requests": 0,
}


def record(usage: dict[str, Any] | None) -> None:
    if not isinstance(usage, dict):
        return
    with _LOCK:
        _TOTALS["requests"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            try:
                _TOTALS[key] += int(usage.get(key) or 0)
            except (TypeError, ValueError):
                pass


def reset() -> None:
    with _LOCK:
        for key in _TOTALS:
            _TOTALS[key] = 0


def snapshot() -> dict[str, int]:
    with _LOCK:
        return dict(_TOTALS)
