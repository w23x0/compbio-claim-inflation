"""Coauthorship graph and independent-cluster counting."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Iterable

from cbi.m3.asof import is_visible


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> dict[str, str]:
        groups: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        out: dict[str, str] = {}
        for i, members in enumerate(sorted(groups.values(), key=lambda m: min(m)), start=1):
            label = f"ac_{i:04d}"
            for m in members:
                out[m] = label
        return out


def author_clusters(papers: Iterable[dict], as_of: date) -> dict[str, str]:
    uf = UnionFind()
    for paper in papers:
        if not is_visible(paper.get("pub_date"), as_of):
            continue
        authors = [str(a) for a in paper.get("author_ids", [])]
        for a in authors:
            uf.find(a)
        for a in authors[1:]:
            uf.union(authors[0], a)
    return uf.clusters()


def cluster_count(clusters: Iterable[str]) -> int:
    return len({c for c in clusters if c})
