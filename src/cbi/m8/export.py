"""Weight export helpers (JSON/CSV/Parquet).

The Python API is the primary M8 interface; these helpers make the same weights
usable by downstream tools that read tabular files.  pandas/pyarrow are imported
only for Parquet so the core package keeps its small dependency set.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

_FORMATS = {"json", "csv", "parquet", "pq"}


def records(claim_weights: Iterable[dict[str, Any]], edges: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in claim_weights:
        out.append({"type": "claim", **row})
    for row in edges:
        out.append({"type": "edge", **row})
    return out


def export(rows: Iterable[dict[str, Any]], path: str | Path, fmt: str | None = None) -> Path:
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".") or "json").lower()
    if fmt not in _FORMATS:
        raise ValueError(f"unsupported format: {fmt}")
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
    if fmt == "csv":
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pandas/pyarrow are required for Parquet export") from exc
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
