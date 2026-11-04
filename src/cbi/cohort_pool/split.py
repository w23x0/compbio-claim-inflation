"""Frozen cohort split helpers."""
from __future__ import annotations

from cbi.config import Config
from cbi.util import sha256_obj


def validate_split(cfg: Config) -> dict[str, str]:
    roles = cfg.cohort_split.get("roles") or {}
    if not roles:
        raise RuntimeError("cohort_split.yaml is missing")
    expected = {c["cohort_id"] for c in cfg.cohort_pool.get("cohorts", [])}
    if set(roles) != expected:
        raise RuntimeError("split does not cover the cohort pool exactly")
    by_cancer: dict[str, dict[str, int]] = {}
    for cid, role in roles.items():
        cancer = next(c["cancer_type"] for c in cfg.cohort_pool["cohorts"] if c["cohort_id"] == cid)
        by_cancer.setdefault(cancer, {}).setdefault(role, 0)
        by_cancer[cancer][role] += 1
    for cancer, counts in by_cancer.items():
        for role in ["dev", "visible_eval", "heldout"]:
            if counts.get(role, 0) < 1:
                raise RuntimeError(f"cancer {cancer} lacks a {role} cohort")
    stored = cfg.cohort_split.get("split_hash")
    if stored and stored != sha256_obj(roles):
        raise RuntimeError("cohort_split hash does not match roles; the frozen split was edited")
    return roles
