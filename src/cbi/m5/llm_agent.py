"""LLM red-team agent decision layer.

The model chooses the batch strategy family from the registered families and the
shared log; deterministic code still executes every analysis and writes every
number.  This keeps the MD's "LLM agent + code execution" split without letting
the model invent effects.
"""
from __future__ import annotations

import json
import os
import time
from urllib.parse import urlparse

import requests
from cbi.llm_http import post as llm_post
from cbi.llm_usage import record as record_usage

from cbi.m5.strategies import FAMILIES, family


def _config() -> tuple[str, str, str]:
    endpoint = os.environ.get("CBI_LLM_ENDPOINT") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("CBI_LLM_MODEL") or os.environ.get("OPENAI_MODEL")
    key = os.environ.get("CBI_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not endpoint or not model:
        raise RuntimeError("M5 LLM agent requires CBI_LLM_ENDPOINT and CBI_LLM_MODEL")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" and os.environ.get("CBI_LLM_ALLOW_HTTP") != "1":
        raise RuntimeError("M5 LLM endpoint must use https")
    base = endpoint.rstrip("/")
    if not base.endswith(("/chat/completions", "/completions")):
        base = base + ("/v1/chat/completions" if not base.endswith("/v1") else "/chat/completions")
    return base, model, key


def choose_family(log_entries: list[dict], quota: float, requested: str | None = None) -> tuple[str, str]:
    if requested:
        return requested, "operator override"
    endpoint, model, key = _config()
    families = {
        name: {"description": f.description, "registry_choices": f.registry_choices,
               "n_comparisons": f.n_comparisons}
        for name, f in FAMILIES.items()
    }
    history = [
        {"family": e.get("strategy_family"), "score": e.get("score"),
         "flag_rate": e.get("flag_rate"), "rejected_by": e.get("rejected_by")}
        for e in log_entries[-30:]
    ]
    prompt = (
        "You are the red-team planner for a prognostic-claim inflation harness. "
        "Choose ONE strategy family for the next batch. You will not write numbers; "
        "downstream code executes every analysis. Prefer a family that is under quota "
        "and has not been exhausted; use history scores and flag rates.\n"
        f"Family quota: {quota:.2f}. History: {json.dumps(history)}\n"
        f"Families: {json.dumps(families)}\n"
        "Return JSON: {\"family\": <one family name>, \"rationale\": <short string>}."
    )
    headers = {"Content-Type": "application/json", "x-opencode-session": os.environ.get("CBI_LLM_SESSION", "cbi-m5")}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": model, "temperature": 0.2,
        "max_tokens": int(os.environ.get("CBI_M5_MAX_TOKENS", "2048")),
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    last: Exception | None = None
    for attempt in range(5):
        try:
            resp = llm_post(endpoint, headers=headers, json=payload, timeout=60)
        except requests.RequestException as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
            continue
        if resp.status_code in {429, 500, 502, 503, 504}:
            last = RuntimeError(f"HTTP {resp.status_code}")
            time.sleep(1.5 * (attempt + 1))
            continue
        if resp.status_code >= 400 and "response_format" in payload:
            payload.pop("response_format")
            last = RuntimeError(f"HTTP {resp.status_code}")
            continue
        resp.raise_for_status()
        data = resp.json()
        record_usage(data.get("usage"))
        content = (data["choices"][0]["message"].get("content") or "").strip()
        if not content:
            last = RuntimeError("empty content")
            payload.pop("response_format", None)
            payload["max_tokens"] = min(int(payload["max_tokens"]) * 2, 8192)
            continue
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            obj = json.loads(content)
            name = str(obj.get("family") or "")
            if name not in FAMILIES:
                raise ValueError(f"unknown family: {name}")
            return name, str(obj.get("rationale") or "")[:500]
        except Exception as exc:
            last = exc
    raise RuntimeError(f"M5 LLM agent failed to choose a family: {last}")
