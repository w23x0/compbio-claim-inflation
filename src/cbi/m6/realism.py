"""Deterministic realism checks (M6 step 3).

An optional LLM judge can be plugged in through CBI_REALISM_ENDPOINT; the
deterministic checks are the offline fallback and the floor for every judge.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

from cbi.schemas import Claim, Report

REQUIRED_SECTIONS = ["abstract", "methods", "results", "discussion", "limitations"]


@dataclass
class RealismResult:
    score: float
    passed: bool
    notes: list[str]


def _llm_realism(claim: Claim, manuscript: str) -> tuple[float, list[str]] | None:
    endpoint = os.environ.get("CBI_REALISM_ENDPOINT")
    model = os.environ.get("CBI_REALISM_MODEL")
    if not endpoint or not model:
        return None
    prompt = (
        "You are a journal editor. Compare this manuscript with an ordinary retrospective "
        "biomarker paper. Return JSON {\"score\": 0-1, \"notes\": [...]} where score 1 means "
        "indistinguishable from a normal paper and 0 means obviously synthetic or incoherent.\n\n"
        + manuscript[:24000]
    )
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("CBI_REALISM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    resp = requests.post(endpoint, headers=headers, timeout=120, json={
        "model": model, "temperature": 0, "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    })
    resp.raise_for_status()
    obj = json.loads(resp.json()["choices"][0]["message"]["content"])
    return float(obj.get("score", 0.5)), [str(x) for x in obj.get("notes", [])]


def assess_realism(claim: Claim, manuscript: str, report: Report, *, threshold: float = 0.55) -> RealismResult:
    notes: list[str] = []
    score = 0.0
    low = manuscript.lower()
    if len(manuscript) >= 1200:
        score += 0.15
    else:
        notes.append("manuscript too short")
    present = sum(1 for s in REQUIRED_SECTIONS if f"## {s}" in low or f"# {s}" in low)
    score += 0.30 * present / len(REQUIRED_SECTIONS)
    if "hazard ratio" in low or re.search(r"\bhr\s*=", low):
        score += 0.15
    else:
        notes.append("no hazard ratio in text")
    if "confidence interval" in low or re.search(r"95%\s*ci", low):
        score += 0.10
    else:
        notes.append("no confidence interval in text")
    if claim.reported_cohort.accessions and claim.reported_cohort.accessions[0].lower() in low:
        score += 0.10
    else:
        notes.append("declared accession absent from text")
    if re.search(r"\bn\s*=\s*\d+|\d+\s+(?:patients|samples|tumours)", low):
        score += 0.05
    if "data availability" in low:
        score += 0.05
    if any(bad in low for bad in ["todo", "placeholder", "lorem ipsum", "synthetic fixture"]):
        score -= 0.35
        notes.append("placeholder/synthetic marker in manuscript")
    # Reported HR should appear with the same rounded value.
    hr_text = f"{claim.reported_effect.value:.2f}"
    if hr_text in manuscript:
        score += 0.10
    else:
        notes.append("reported HR not found verbatim")
    score = float(max(0.0, min(1.0, score)))
    try:
        judged = _llm_realism(claim, manuscript)
    except Exception as exc:
        notes.append(f"LLM realism judge unavailable: {type(exc).__name__}")
        judged = None
    if judged is not None:
        llm_score, llm_notes = judged
        # The deterministic checks are a floor; the judge can only add review
        # signal, never bypass a malformed manuscript.
        score = 0.5 * score + 0.5 * float(max(0.0, min(1.0, llm_score)))
        notes.extend(llm_notes)
    return RealismResult(score=score, passed=score >= threshold, notes=notes)
