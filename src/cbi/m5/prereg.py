"""Pre-registration freezing (spec M5)."""
from __future__ import annotations

from datetime import datetime, timezone

from cbi.schemas import StrategyPrereg
from cbi.util import canonical_json, sha256_text


def build_prereg(*, submission_id: str, round: int, worker: int, family: str,
                 choices: dict, target_cancers: list[str], target_cohorts: list[str],
                 motivation: str, reports_planned: int) -> tuple[StrategyPrereg, str, str]:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prereg = StrategyPrereg(
        submission_id=submission_id, round=round, worker=worker, strategy_family=family,
        choices=choices, target_cancers=target_cancers, target_cohorts=target_cohorts,
        motivation=motivation, reports_planned=reports_planned, created_at=created,
    )
    text = canonical_json(prereg.model_dump(mode="json"))
    return prereg, text, sha256_text(text)


def check_prereg_text(text: str, expected_hash: str) -> bool:
    return sha256_text(text) == expected_hash
