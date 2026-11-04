"""Role-guarded cohort access.

Red-team workers construct `CohortStore(allowed_roles={"dev"})`; requests for
visible-eval or held-out cohorts raise `AccessDenied`.  The grader additionally
needs the secret token from `secrets/grader.token` (or CBI_GRADER_TOKEN).
Bubblewrap hides the restricted directories from worker processes entirely.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pandas as pd

from cbi.config import Config
from cbi.util import read_json, sha256_text


class AccessDenied(PermissionError):
    pass


def grader_token_path(cfg: Config) -> Path:
    return cfg.paths.secrets / "grader.token"


def load_or_create_token(cfg: Config) -> str:
    """Return the provisioned grader token; never mint one implicitly."""
    path = grader_token_path(cfg)
    env = os.environ.get("CBI_GRADER_TOKEN")
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if env and env != token:
            raise AccessDenied("CBI_GRADER_TOKEN does not match secrets/grader.token")
        return token
    if env:
        return env
    raise AccessDenied(
        "grader token is not provisioned; set CBI_GRADER_TOKEN or create secrets/grader.token (0600)"
    )


class CohortStore:
    def __init__(self, cfg: Config, *, role: str = "worker", token: str | None = None,
                 allowed_roles: set[str] | None = None):
        self.cfg = cfg
        self.role = role
        self.token = token
        if allowed_roles is not None:
            requested = set(allowed_roles)
            if role != "grader" and not requested <= {"dev"}:
                raise AccessDenied("only the grader may request non-dev cohort roles")
            self.allowed_roles = requested
        elif role == "grader":
            self.allowed_roles = {"dev", "visible_eval", "heldout"}
        else:
            self.allowed_roles = {"dev"}
        self._meta = {c["cohort_id"]: c for c in cfg.cohort_pool.get("cohorts", [])}
        self._cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]] = {}

    def _check_role(self, cohort_id: str) -> str:
        if cohort_id not in self._meta:
            raise KeyError(f"unknown cohort: {cohort_id}")
        role = self.cfg.cohort_split.get("roles", {}).get(cohort_id)
        if role is None:
            raise AccessDenied(f"cohort has no split role: {cohort_id}")
        if role not in self.allowed_roles:
            raise AccessDenied(f"role {self.role} may not access {role} cohort {cohort_id}")
        if self.role == "grader":
            expected = load_or_create_token(self.cfg)
            if not self.token or self.token != expected:
                raise AccessDenied("grader token missing or invalid")
        return role

    def cohort_ids(self, *, cancer: str | None = None, role: str | None = None) -> list[str]:
        out = []
        for cid, meta in self._meta.items():
            cid_role = self.cfg.cohort_split["roles"].get(cid)
            if role is not None and cid_role != role:
                continue
            if cid_role not in self.allowed_roles:
                continue
            if cancer is not None and meta["cancer_type"] != cancer:
                continue
            out.append(cid)
        return sorted(out)

    def meta(self, cohort_id: str) -> dict:
        role = self._check_role(cohort_id)
        path = self.cfg.paths.cohort_data_dir(cohort_id, role) / "meta.json"
        return read_json(path)

    def clinical(self, cohort_id: str) -> pd.DataFrame:
        role = self._check_role(cohort_id)
        return pd.read_parquet(self.cfg.paths.cohort_data_dir(cohort_id, role) / "clinical.parquet")

    def expression(self, cohort_id: str) -> pd.DataFrame:
        role = self._check_role(cohort_id)
        return pd.read_parquet(self.cfg.paths.cohort_data_dir(cohort_id, role) / "expression.parquet")

    def gene_expression(self, cohort_id: str, gene: str) -> pd.Series | None:
        """Read only one expression column; avoids loading the full matrix."""
        role = self._check_role(cohort_id)
        path = self.cfg.paths.cohort_data_dir(cohort_id, role) / "expression.parquet"
        try:
            frame = pd.read_parquet(path, columns=[gene])
        except (KeyError, ValueError):
            return None
        if gene not in frame.columns:
            return None
        return frame[gene]

    def load(self, cohort_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        role = self._check_role(cohort_id)
        if cohort_id not in self._cache:
            base = self.cfg.paths.cohort_data_dir(cohort_id, role)
            clinical = pd.read_parquet(base / "clinical.parquet")
            expr = pd.read_parquet(base / "expression.parquet")
            meta = read_json(base / "meta.json")
            self._cache[cohort_id] = (clinical, expr, meta)
        return self._cache[cohort_id]

    def genes(self, cohort_id: str) -> list[str]:
        import pyarrow.parquet as pq
        role = self._check_role(cohort_id)
        path = self.cfg.paths.cohort_data_dir(cohort_id, role) / "expression.parquet"
        return list(pq.read_schema(path).names)

    def iter(self, *, cancer: str | None = None, role: str | None = None) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame, dict]]:
        for cid in self.cohort_ids(cancer=cancer, role=role):
            clinical, expr, meta = self.load(cid)
            yield cid, clinical, expr, meta

    def assert_dev_only(self) -> None:
        if self.allowed_roles != {"dev"}:
            raise AssertionError("worker view must contain only dev cohorts")
