"""Wait for the LLM gateway, then run the resumable P4 retrain."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import requests

BASE = os.environ["CBI_LLM_ENDPOINT"].rstrip("/")
URL = BASE + "/v1/chat/completions"
KEY = os.environ["CBI_LLM_API_KEY"]
MODEL = os.environ.get("CBI_LLM_MODEL", "gpt-6-astra")
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def ready() -> bool:
    try:
        resp = requests.post(URL, headers=HEADERS, timeout=30, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "OK"}],
            "max_tokens": 5,
        })
        print(f"probe {resp.status_code}", flush=True)
        return resp.status_code == 200
    except Exception as exc:
        print(f"probe error {type(exc).__name__}", flush=True)
        return False


for i in range(80):
    if ready():
        break
    time.sleep(30)
else:
    raise SystemExit("gateway did not recover within 40 minutes")

env = dict(os.environ)
env["CBI_P4_WORKERS"] = os.environ.get("CBI_P4_WORKERS", "4")
env["CBI_LLM_MAX_CONCURRENCY"] = os.environ.get("CBI_LLM_MAX_CONCURRENCY", "4")
for attempt in range(3):
    print(f"P4 attempt {attempt + 1}", flush=True)
    proc = subprocess.run([sys.executable, "-m", "cbi.cli", "p4"], env=env)
    if proc.returncode == 0:
        raise SystemExit(0)
    time.sleep(60)
raise SystemExit(2)
