"""Date truncation helpers for M3.  Every feature function takes as_of_date and
must ignore anything that happened later (spec R5 / §6)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Iterable


def parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        text = value[:10]
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None
    return None


def is_visible(value: Any, as_of: date) -> bool:
    d = parse_date(value)
    return d is not None and d <= as_of


def visible(records: Iterable[dict[str, Any]], date_key: str, as_of: date) -> list[dict[str, Any]]:
    return [r for r in records if is_visible(r.get(date_key), as_of)]


def add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # Feb 29
        return d.replace(year=d.year + years, day=28)
