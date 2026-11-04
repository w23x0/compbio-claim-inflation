"""Context snapshot loader (OpenAlex/Crossref-shaped local JSON).

The snapshot is produced from OpenAlex/Crossref/PubPeer exports; the feature
code only sees the normalized shape documented in README and tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cbi.config import Config
from cbi.util import read_json


@dataclass
class ContextSnapshot:
    raw: dict

    @property
    def papers(self) -> list[dict]:
        return list(self.raw.get("papers", []))

    @property
    def citations(self) -> list[dict]:
        return list(self.raw.get("citations", []))

    @property
    def authors(self) -> dict[str, dict]:
        return dict(self.raw.get("authors", {}))

    @property
    def support_links(self) -> list[dict]:
        return list(self.raw.get("support_links", []))

    def paper_for_claim(self, claim_id: str) -> dict | None:
        for paper in self.papers:
            if paper.get("claim_id") == claim_id:
                return paper
        return None


def context_snapshot_path(cfg: Config) -> Path:
    return cfg.paths.raw / "context" / "context.json"


def load_context_snapshot(cfg: Config) -> ContextSnapshot:
    path = context_snapshot_path(cfg)
    if not path.is_file():
        return ContextSnapshot({"papers": [], "citations": [], "authors": {}, "support_links": []})
    return ContextSnapshot(read_json(path))
