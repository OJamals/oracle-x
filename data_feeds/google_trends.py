"""Google Trends feed temporarily disabled.

Repeated 429 rate limits made the free (pytrends) retrieval unreliable in the
current environment. This module now stubs `fetch_google_trends` returning an
empty mapping so downstream code remains functional without conditional
imports. Re-enable by replacing this stub with a working implementation.
"""
from typing import Dict, Any, Iterable

DISABLED_REASON = "google_trends_disabled_due_to_rate_limits"

def fetch_google_trends(keywords: Iterable[str], timeframe: str = "now 7-d", geo: str = "US") -> Dict[str, Dict[str, int]]:
    """Return empty result with a sentinel reason (non-breaking stub)."""
    return {}

def trends_status() -> Dict[str, Any]:
    return {"status": "disabled", "reason": DISABLED_REASON}
