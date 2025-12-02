"""Per-feed real (non-mocked) data validation harness.

Runs each adapter / data source in isolation so we can inspect:
  - latency
  - basic shape / row counts
  - sample fields
  - quality scores recorded by the orchestrator (when available)

Usage:
  python scripts/test_individual_feeds.py                # run default suite
  python scripts/test_individual_feeds.py --only finviz  # single group
  python scripts/test_individual_feeds.py --list         # list available groups

Notes:
  - No network mocking: all calls hit live/free endpoints or scrapers.
  - Failures are captured; the script continues to next feed.
  - Output is newline-delimited JSON (one object per feed) for easy parsing.
  - Options flow currently returns placeholder synthetic data; flagged accordingly.
  - Reddit/Twitter sentiment will gracefully skip if credentials / rate limits block.

This script does NOT attempt exhaustive validation – it is a diagnostic aid to
pinpoint which specific feed causes overall health degradation (e.g. Reddit).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Conservative defaults used elsewhere
os.environ.setdefault("CACHE_DB_PATH", "data/databases/model_monitoring.db")
os.environ.setdefault("CONSERVE_FMP_BANDWIDTH", "true")

from data_feeds.data_feed_orchestrator import (
    get_orchestrator,
    get_quote,
    get_market_data,
    get_sentiment_data,
    get_advanced_sentiment,
)

ORCH = get_orchestrator()

TickerList = ["AAPL", "TSLA", "NVDA"]


def _capture(name: str, fn: Callable[[], Any]) -> Dict[str, Any]:
    start = time.time()
    try:
        result = fn()
        ok = True
        err = None
    except Exception as e:  # pragma: no cover - diagnostic script
        ok = False
        result = None
        err = f"{type(e).__name__}: {e}"
    dur_ms = round((time.time() - start) * 1000, 2)
    payload = {
        "feed": name,
        "ok": ok,
        "latency_ms": dur_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if ok:
        payload["data_summary"] = _summarize(result)
    else:
        payload["error"] = err
    return payload


def _summarize(obj: Any) -> Any:
    """Return a lightweight summary; avoid dumping huge DataFrames whole."""
    try:
        import pandas as pd  # local import
    except Exception:  # pragma: no cover
        pd = None

    if obj is None:
        return None
    # Sentiment map
    if isinstance(obj, dict):
        # If dict of SentimentData objects
        sample = {}
        for k, v in list(obj.items())[:5]:  # limit keys
            # Dataclasses have attributes; sentiment objects expected
            if hasattr(v, "sentiment_score"):
                sample[k] = {
                    "score": getattr(v, "sentiment_score", None),
                    "conf": getattr(v, "confidence", None),
                    "samples": getattr(v, "sample_size", None),
                    "source": getattr(v, "source", None),
                }
            elif isinstance(v, dict):
                sample[k] = {kk: v[kk] for kk in list(v.keys())[:5]}
            else:
                sample[k] = str(type(v))
        return {"type": "dict", "keys": len(obj), "sample": sample}
    # Pandas objects
    if pd is not None and isinstance(obj, pd.DataFrame):
        return {
            "type": "dataframe",
            "rows": int(len(obj)),
            "cols": list(obj.columns)[:15],
        }
    # Dataclass-like objects (Quote / MarketData / SentimentData)
    for attr in ("symbol", "price", "change_percent", "quality_score", "timeframe", "sentiment_score"):
        if hasattr(obj, attr):
            # Build limited snapshot of public attributes
            keys = ["symbol", "source"]
            if hasattr(obj, "price"):
                keys.extend(["price", "change_percent", "volume", "quality_score"])
            if hasattr(obj, "timeframe"):
                keys.append("timeframe")
            if hasattr(obj, "sentiment_score"):
                keys.extend(["sentiment_score", "confidence", "sample_size"])
            snap = {}
            for k in keys:
                if hasattr(obj, k):
                    snap[k] = getattr(obj, k)
            return snap
    # Fallback textual form truncated
    text = str(obj)
    return text[:400] + ("..." if len(text) > 400 else "")


def build_tasks() -> Dict[str, Callable[[], Dict[str, Any]]]:
    return {
        "quotes": lambda: {t: _summarize(get_quote(t)) for t in TickerList},
        "market_data_daily": lambda: _summarize(get_market_data("AAPL", period="6mo", interval="1d")),
        "market_data_hourly": lambda: _summarize(get_market_data("AAPL", period="1mo", interval="1h")),
        "sentiment_reddit_twitter": lambda: _summarize(get_sentiment_data("AAPL")),
        "advanced_sentiment": lambda: _summarize(get_advanced_sentiment("AAPL")),
        "finviz_market_breadth": lambda: _summarize(ORCH.get_market_breadth()),
        "finviz_sector_performance": lambda: _summarize(ORCH.get_sector_performance()),
        "finviz_news": lambda: _summarize(ORCH.get_finviz_news()),
        "finviz_insider_trading": lambda: _summarize(ORCH.get_finviz_insider_trading()),
        "finviz_earnings": lambda: _summarize(ORCH.get_finviz_earnings()),
        "finviz_forex": lambda: _summarize(ORCH.get_finviz_forex()),
        "finviz_crypto": lambda: _summarize(ORCH.get_finviz_crypto()),
        "google_trends": lambda: _summarize(ORCH.get_google_trends(["AAPL", "TSLA", "NVDA"], timeframe="now 7-d", geo="US")),
        "options_analytics": lambda: _summarize(ORCH.get_options_analytics("AAPL", include=["chain", "iv", "greeks", "gex", "max_pain"])),
        "options_flow_placeholder": lambda: _summarize(_run_options_flow_placeholder()),
    }


def _run_options_flow_placeholder() -> Any:
    try:
        from data_feeds.options_flow import fetch_options_flow
        return fetch_options_flow(["AAPL", "TSLA"])
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def list_feeds(tasks: Dict[str, Callable]):  # pragma: no cover - CLI helper
    for name in sorted(tasks):
        print(name)


def main():  # pragma: no cover - script entry
    parser = argparse.ArgumentParser(description="Run individual feed diagnostics.")
    parser.add_argument("--only", help="Comma-separated feed names to run (see --list)")
    parser.add_argument("--list", action="store_true", help="List available feed names and exit")
    args = parser.parse_args()

    tasks = build_tasks()
    if args.list:
        list_feeds(tasks)
        return

    selected: List[str]
    if args.only:
        selected = [s.strip() for s in args.only.split(",") if s.strip()]
    else:
        selected = list(tasks.keys())

    for name in selected:
        if name not in tasks:
            print(json.dumps({"feed": name, "ok": False, "error": "unknown feed"}))
            continue
        payload = _capture(name, tasks[name])
        print(json.dumps(payload, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
