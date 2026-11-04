"""Configuration and filesystem layout.

The project root is resolved from --root, CBI_ROOT, or the nearest parent of
this file containing configs/mvp.yaml.  All paths are absolute afterwards.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    pass


def find_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("CBI_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "configs" / "mvp.yaml").is_file():
            return parent
    return Path.cwd().resolve()


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True, width=100)
    tmp.replace(path)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass(frozen=True)
class Paths:
    root: Path
    snapshot: str

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        return self.data / "processed" / self.snapshot

    @property
    def restriction(self) -> Path:
        return self.data / "eval_restricted" / self.snapshot

    @property
    def visible_eval(self) -> Path:
        return self.restriction / "visible"

    @property
    def heldout(self) -> Path:
        return self.restriction / "heldout"

    @property
    def eval_root(self) -> Path:
        return self.data / "eval_restricted"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def grader_internal(self) -> Path:
        return self.logs / "grader_internal"

    @property
    def experiment_log(self) -> Path:
        return self.logs / "experiment_log.jsonl"

    @property
    def artifacts(self) -> Path:
        return self.root / "redteam_artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def secrets(self) -> Path:
        return self.root / "secrets"

    @property
    def cohort_pool(self) -> Path:
        return self.root / "cohort_pool"

    def processed_cohort(self, cohort_id: str) -> Path:
        return self.processed / "cohorts" / cohort_id

    def cohort_data_dir(self, cohort_id: str, role: str) -> Path:
        if role == "dev":
            return self.processed_cohort(cohort_id)
        if role == "visible_eval":
            return self.visible_eval / "cohorts" / cohort_id
        if role == "heldout":
            return self.heldout / "cohorts" / cohort_id
        raise ConfigError(f"unknown cohort role: {role}")


@dataclass
class Config:
    paths: Paths
    mvp: dict[str, Any]
    scoring: dict[str, Any]
    redteam: dict[str, Any]
    blue: dict[str, Any]
    registry: dict[str, Any]
    lexicon: dict[str, Any]
    cohort_pool: dict[str, Any] = field(default_factory=dict)
    cohort_split: dict[str, Any] = field(default_factory=dict)

    @property
    def snapshot(self) -> str:
        return self.paths.snapshot

    @property
    def cancers(self) -> list[str]:
        return list(self.mvp["cancers"])

    @property
    def weights(self) -> dict[str, float]:
        if self.scoring.get("use_calibrated_weights") and self.scoring.get("calibrated_weights"):
            return dict(self.scoring["calibrated_weights"])
        return dict(self.scoring["weights"])

    def choice(self, choice_id: str) -> dict[str, Any]:
        for item in self.registry.get("choices", []):
            if item["id"] == choice_id:
                return item
        raise ConfigError(f"choice not in registry: {choice_id}")

    def reference_choices(self, dimension: str) -> list[dict[str, Any]]:
        return [
            c
            for c in self.registry.get("choices", [])
            if c["dimension"] == dimension and c.get("in_reference_set")
        ]

    def all_families(self) -> list[str]:
        families = self.redteam.get("strategy_families")
        if families:
            return list(families)
        return [
            "optimal_cutpoint",
            "subgroup_selection",
            "endpoint_switching",
            "covariate_fishing",
            "followup_cap",
            "multiverse_fishing",
            "evasive_reporting",
            "honest_reference",
        ]


def load_config(root: str | Path | None = None, *, overrides: dict[str, Any] | None = None) -> Config:
    root_path = find_root(root)
    configs = root_path / "configs"
    mvp = read_yaml(configs / "mvp.yaml")
    if overrides:
        mvp = deep_merge(mvp, overrides)
    snapshot = str(mvp.get("snapshot", "2026-11-01"))
    paths = Paths(root=root_path, snapshot=snapshot)
    scoring_path = configs / "scoring.yaml"
    scoring = read_yaml(scoring_path)
    calibrated = configs / "scoring.calibrated.yaml"
    if calibrated.is_file():
        scoring = deep_merge(scoring, read_yaml(calibrated))
    redteam = read_yaml(configs / "redteam.yaml")
    blue = read_yaml(configs / "blue.yaml")
    registry = read_yaml(configs / "choice_registry.yaml")
    lexicon = read_yaml(configs / "detector_lexicon.yaml")
    pool = read_yaml(configs / "cohort_pool.yaml") if (configs / "cohort_pool.yaml").is_file() else {}
    split = read_yaml(configs / "cohort_split.yaml") if (configs / "cohort_split.yaml").is_file() else {}
    return Config(
        paths=paths,
        mvp=mvp,
        scoring=scoring,
        redteam=redteam,
        blue=blue,
        registry=registry,
        lexicon=lexicon,
        cohort_pool=pool,
        cohort_split=split,
    )


def ensure_layout(cfg: Config) -> None:
    """Create directories with conservative permissions.

    Eval and secret directories are 0700; the deployment guide explains how to
    move them to a dedicated grader OS user.  Bubblewrap additionally masks
    them from red-team workers at process level.
    """
    for path in [
        cfg.paths.configs,
        cfg.paths.raw,
        cfg.paths.processed,
        cfg.paths.logs,
        cfg.paths.artifacts,
        cfg.paths.reports,
        cfg.paths.models,
        cfg.paths.secrets,
        cfg.paths.eval_root,
        cfg.paths.visible_eval,
        cfg.paths.heldout,
        cfg.paths.grader_internal,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for path in [cfg.paths.eval_root, cfg.paths.secrets, cfg.paths.grader_internal]:
        try:
            path.chmod(0o700)
        except OSError:
            pass
