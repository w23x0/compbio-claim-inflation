"""Gene symbol and cancer-type normalization for M1.

The resolver reads the complete HGNC TSV at data/raw/hgnc/hgnc_complete.tsv and
uses approved symbols, aliases, and previous symbols.  No synthetic symbol map
is part of the build.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CANCER_MAP = {
    "hepatocellular": "LIHC",
    "hepatocellular carcinoma": "LIHC",
    "liver cancer": "LIHC",
    "hcc": "LIHC",
    "breast invasive": "BRCA",
    "breast cancer": "BRCA",
    "brca": "BRCA",
    "lung adenocarcinoma": "LUAD",
    "luad": "LUAD",
    "kidney renal clear cell": "KIRC",
    "clear cell renal": "KIRC",
    "kirc": "KIRC",
}


class HGNCResolver:
    def __init__(self, path: str | Path | None = None):
        self.symbol_to_id: dict[str, str] = {}
        self.id_to_symbol: dict[str, str] = {}
        self.alias_to_id: dict[str, str] = {}
        if path is not None and Path(path).is_file():
            df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
            def norm(column: str) -> str:
                return column.lower().replace("_", " ").strip()
            symbol_col = next((c for c in df.columns if norm(c) in {"symbol", "approved symbol"}), None)
            id_col = next((c for c in df.columns if norm(c).startswith("hgnc id")), None)
            if symbol_col and id_col:
                for sym, hid in zip(df[symbol_col], df[id_col]):
                    if isinstance(sym, str) and isinstance(hid, str):
                        self.symbol_to_id[sym.upper()] = hid
                        self.id_to_symbol[hid] = sym.upper()
            alias_cols = [c for c in df.columns
                          if norm(c) in {"alias symbols", "aliases", "alias symbol", "prev symbol"}]
            for alias_col in alias_cols:
                for aliases, hid in zip(df[alias_col], df[id_col]):
                    if not isinstance(aliases, str) or not isinstance(hid, str):
                        continue
                    for alias in aliases.replace("|", ",").split(","):
                        alias = alias.strip().upper()
                        if alias:
                            self.alias_to_id.setdefault(alias, hid)

    def normalize(self, token: str) -> tuple[str | None, str | None]:
        token = token.strip()
        up = token.upper()
        if up in self.symbol_to_id:
            return up, self.symbol_to_id[up]
        if up in self.alias_to_id:
            hid = self.alias_to_id[up]
            return self.id_to_symbol.get(hid, up), hid
        return None, None


def map_cancer(text: str) -> str | None:
    low = text.lower()
    best: tuple[int, str] | None = None
    for phrase, code in CANCER_MAP.items():
        idx = low.find(phrase)
        if idx >= 0:
            length = len(phrase)
            if best is None or length > best[0]:
                best = (length, code)
    return best[1] if best else None
