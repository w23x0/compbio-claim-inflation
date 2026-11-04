"""Offline GEO accession existence cache (ported from comp01).

A local cache lets M1 distinguish "not in the curated cohort pool" from "the
accession does not exist / was mistyped".  The cache is optional: when it is
empty the check is skipped instead of rejecting every claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

_CACHE: set[str] = set()
_LOADED = False


def load_geo_cache(paths: Iterable[str | Path] | None = None) -> set[str]:
    global _CACHE, _LOADED
    candidates = [p for p in (paths or []) if p]
    env_path = os.environ.get("GEO_CACHE_PATH")
    if env_path:
        candidates.insert(0, env_path)
    root = Path(os.environ.get("CBI_ROOT", "."))
    candidates += [root / "data" / "raw" / "geo_cache.json",
                   root / "m1_extract" / "geo_cache.json",
                   Path(__file__).with_name("geo_cache.json")]
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, list):
            _CACHE.update(str(x).upper() for x in data)
        elif isinstance(data, dict):
            _CACHE.update(str(k).upper() for k, ok in data.items() if ok)
        _LOADED = True
    return set(_CACHE)


def set_geo_cache(accessions: Iterable[str] | dict[str, bool] | None) -> None:
    global _CACHE, _LOADED
    if isinstance(accessions, dict):
        _CACHE = {str(k).upper() for k, ok in accessions.items() if ok}
        _LOADED = True
    elif accessions is None:
        _CACHE = set()
        _LOADED = False
    else:
        _CACHE = {str(x).upper() for x in accessions}
        _LOADED = True


def cache_loaded() -> bool:
    return _LOADED or bool(_CACHE)


def geo_exists(accession: str) -> bool | None:
    """True/False when a cache is loaded; None when existence is unknown."""
    if not cache_loaded():
        return None
    return str(accession).strip().upper() in _CACHE
