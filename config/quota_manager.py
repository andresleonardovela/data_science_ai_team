"""Quota manager — tracks IBM Cloud free-tier usage to prevent silent overruns."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# IBM Cloud Lite free-tier monthly limits
_LIMITS = {
    "tokens": 300_000,       # watsonx.ai tokens/month
    "actions": 100,          # watsonx Orchestrate actions/month
    "cuh": 20.0,             # Capacity Unit Hours/month
}

_USAGE_FILE = Path(".quota_usage.json")


def _load_usage() -> dict:
    if _USAGE_FILE.exists():
        with _USAGE_FILE.open() as f:
            return json.load(f)
    return {"tokens": 0, "actions": 0, "cuh": 0.0, "reset_month": _current_month()}


def _save_usage(usage: dict) -> None:
    with _USAGE_FILE.open("w") as f:
        json.dump(usage, f, indent=2)


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


class QuotaManager:
    """Tracks and enforces IBM Cloud Lite free-tier quota limits.

    Usage is persisted to .quota_usage.json (gitignored).
    Call check_limits() before any LLM/orchestration operation.
    """

    def __init__(self) -> None:
        self._usage = _load_usage()
        self._maybe_reset()

    def _maybe_reset(self) -> None:
        """Reset counters if we've entered a new billing month."""
        if self._usage.get("reset_month") != _current_month():
            self._usage = {"tokens": 0, "actions": 0, "cuh": 0.0, "reset_month": _current_month()}
            _save_usage(self._usage)

    def record(self, tokens: int = 0, actions: int = 0, cuh: float = 0.0) -> None:
        """Record resource consumption."""
        self._usage["tokens"] += tokens
        self._usage["actions"] += actions
        self._usage["cuh"] += cuh
        _save_usage(self._usage)

    def check_limits(self, raise_on_exceed: bool = False) -> None:
        """Print a quota summary table and warn/raise if limits are close."""
        table = Table(title=f"IBM Cloud Lite Quota — {_current_month()}")
        table.add_column("Resource", style="cyan")
        table.add_column("Used", justify="right")
        table.add_column("Limit", justify="right")
        table.add_column("Remaining", justify="right")
        table.add_column("Status", justify="center")

        exceeded = []
        for key, limit in _LIMITS.items():
            used = self._usage.get(key, 0)
            remaining = limit - used
            pct = used / limit * 100
            if pct >= 100:
                status = "[red]EXCEEDED[/red]"
                exceeded.append(key)
            elif pct >= 80:
                status = "[yellow]WARNING[/yellow]"
            else:
                status = "[green]OK[/green]"
            table.add_row(key, str(used), str(limit), str(max(0, remaining)), status)

        console.print(table)

        if exceeded and raise_on_exceed:
            raise RuntimeError(f"Free-tier quota exceeded for: {exceeded}. Upgrade your plan.")

    @property
    def usage(self) -> dict:
        return dict(self._usage)
