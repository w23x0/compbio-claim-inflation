"""Probe the gateway's stable concurrency with tiny chat requests.

This is a cheap ramp test: it does not extract papers.  Use it before raising
the P1 worker count, then validate the chosen level on a real 100-200 paper
batch.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def _one(url: str, key: str, model: str) -> tuple[int, float]:
    start = time.time()
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "Say OK"}],
                  "max_tokens": 2, "temperature": 0},
            timeout=30,
        )
        return resp.status_code, time.time() - start
    except Exception:
        return 0, time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="5,10,15,20,30")
    parser.add_argument("--requests", type=int, default=30)
    args = parser.parse_args()
    base = os.environ["CBI_LLM_ENDPOINT"].rstrip("/")
    url = base + "/v1/chat/completions"
    key = os.environ["CBI_LLM_API_KEY"]
    model = os.environ.get("CBI_LLM_MODEL", "gpt-6-astra")
    print(f"endpoint={url} model={model} requests_per_level={args.requests}")
    for level in [int(x) for x in args.levels.split(",") if x.strip()]:
        with ThreadPoolExecutor(max_workers=level) as pool:
            results = list(pool.map(lambda _: _one(url, key, model), range(args.requests)))
        ok = sum(1 for status, _ in results if status == 200)
        statuses = {}
        for status, _ in results:
            statuses[status] = statuses.get(status, 0) + 1
        lat = [dt for status, dt in results if status == 200]
        print(f"concurrency={level:>2} ok={ok}/{args.requests} "
              f"status={statuses} "
              f"p50={statistics.median(lat):.2f}s" if lat else
              f"concurrency={level:>2} ok={ok}/{args.requests} status={statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
