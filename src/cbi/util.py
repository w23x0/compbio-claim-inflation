"""Small, dependency-free helpers shared across modules."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

WATERMARK = "INTERNAL RED-TEAM ARTIFACT - NOT REAL RESEARCH"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def atomic_write_text(path: str | Path, text: str, *, mode: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def append_jsonl(path: str | Path, obj: Any, *, lock: bool = True) -> None:
    """Append one JSON line; uses flock so parallel workers can share the log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        if lock:
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
        fh.write(line)
        fh.flush()
        if lock:
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def watermark_header(marker: str = "#") -> str:
    line = f"{marker} {WATERMARK}"
    return line + "\n" + marker + " " + "-" * len(WATERMARK) + "\n"


def ensure_watermark(text: str, marker: str = "#") -> str:
    if WATERMARK in text:
        return text
    return watermark_header(marker) + text


def safe_join(base: str | Path, relative: str) -> Path:
    base = Path(base).resolve()
    target = (base / relative).resolve()
    if base != target and base not in target.parents:
        raise ValueError(f"path escapes base: {relative}")
    return target


def percentile_rank(values: Iterable[float], x: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float((arr <= x).mean())


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def slug(text: str, max_len: int = 80) -> str:
    out = _SLUG_RE.sub("_", text).strip("_")
    return out[:max_len] or "item"


def fmt_float(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:.{digits}f}"
