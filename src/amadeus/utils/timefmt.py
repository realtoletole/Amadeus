"""Human-friendly time deltas ("3 days ago") for prompts."""

from __future__ import annotations

from datetime import datetime, timezone


def humanize_age(then: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    seconds = max((now - then).total_seconds(), 0.0)

    if seconds < 60:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        n = int(minutes)
        return f"{n} minute{'s' if n != 1 else ''} ago"
    hours = minutes / 60
    if hours < 24:
        n = int(hours)
        return f"{n} hour{'s' if n != 1 else ''} ago"
    days = hours / 24
    if days < 7:
        n = int(days)
        return f"{n} day{'s' if n != 1 else ''} ago"
    if days < 60:
        n = int(days / 7)
        return f"{n} week{'s' if n != 1 else ''} ago"
    n = int(days / 30)
    return f"{n} month{'s' if n != 1 else ''} ago"
