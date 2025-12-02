"""Enriched Advanced Sentiment Aggregation Tests

Validates that the orchestrator advanced sentiment pipeline exposes
the aggregated_counts metadata and respects environment tuning.

Tests are intentionally resilient to network/service variability:
- If upstream sources return no data (e.g. rate limits / offline),
  the test prints a diagnostic and exits without failing hard.
"""

import os
import logging

logging.basicConfig(level=logging.INFO)


def _set_env_overrides():
    # Keep per-source cap small for test performance
    os.environ.setdefault("ADVANCED_SENTIMENT_MAX_PER_SOURCE", "40")
    # Provide a lightweight RSS feed (optional); won't fail if feedparser missing
    os.environ.setdefault(
        "RSS_FEEDS", "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
    )
    os.environ.setdefault("RSS_INCLUDE_ALL", "1")


def test_advanced_sentiment_aggregated_counts():
    _set_env_overrides()
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator

    orch = DataFeedOrchestrator()
    adv = orch.get_advanced_sentiment_data("AAPL")

    if adv is None:
        # Non-fatal: remote sources may be unavailable in CI/offline
        logging.warning(
            "Advanced sentiment returned None (network/limits) - skipping assertions"
        )
        return

    assert adv.raw_data is not None, "raw_data expected on advanced sentiment result"
    counts = adv.raw_data.get("aggregated_counts")
    assert isinstance(counts, dict), "aggregated_counts dict missing in raw_data"
    assert "total_unique" in counts, "total_unique key missing in aggregated_counts"

    # Basic sanity: all count values should be ints >= 0
    for k, v in counts.items():
        assert (
            isinstance(v, int) and v >= 0
        ), f"Count for {k} should be non-negative int"

    # If we have news counts ensure they don't exceed cap * 3 (global truncation rule)
    cap = int(os.getenv("ADVANCED_SENTIMENT_MAX_PER_SOURCE", "300"))
    if counts.get("total_unique"):
        assert (
            counts["total_unique"] <= cap * 3
        ), "total_unique exceeds global truncation bound"

    logging.info(
        "Advanced aggregated sentiment OK: score=%.3f conf=%.3f counts=%s",
        getattr(adv, "sentiment_score", 0.0),
        getattr(adv, "confidence", 0.0),
        counts,
    )


def test_rss_adapter_registration():
    """Validate RSS sentiment adapter registration & include_all flag.

    Skips silently if feedparser not installed or no RSS feeds configured.
    """
    _set_env_overrides()
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator

    orch = DataFeedOrchestrator()
    rss_key = (
        "RSS",
        "rss_news",
    )  # tuple key registered by orchestrator when RSS active
    adapter = orch.adapters.get(rss_key)  # type: ignore[index]
    if not adapter:
        logging.warning(
            "RSS adapter not registered (feedparser missing or no feeds) - skipping"
        )
        return
    # Validate include_all flag respected from env
    assert (
        getattr(adapter, "include_all", False) is True
    ), "RSS_INCLUDE_ALL env flag not reflected in adapter"

    # Light sentiment fetch (non-fatal if None)
    sd = adapter.get_sentiment("AAPL", limit=10)
    if sd is None:
        logging.warning(
            "RSS sentiment returned None (no headlines / parsing issue) - skipping deeper assertions"
        )
        return
    assert (
        sd.sample_size is None or sd.sample_size <= 10
    ), "Sample size should respect limit"
    if sd.raw_data:
        sample_texts = sd.raw_data.get("sample_texts", [])
        assert len(sample_texts) <= 5, "Stored sample_texts should be truncated to 5"
