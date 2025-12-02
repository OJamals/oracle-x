"""Regression test: Ensure FinViz news sentiment path no longer triggers
ValueError: The truth value of a DataFrame is ambiguous.

The previous implementation used chained boolean / `or` expressions that could
implicitly evaluate a pandas DataFrame in a truthy context, raising the
ambiguous truth-value error. This test exercises the FinViz news sentiment
adapter (if available) and fails if such an exception resurfaces.

Network / scraping environments can vary; if the adapter is unavailable or
no news is returned, the test will pass as long as no ambiguous ValueError
occurs. Any occurrence of the specific ValueError message is a failure.
"""

import logging
import os

import pytest

# Keep logs quieter
logging.basicConfig(level=logging.INFO)


def test_finviz_news_no_ambiguous_truth_value():
    # Small caps to keep processing light
    os.environ.setdefault("ADVANCED_SENTIMENT_MAX_PER_SOURCE", "20")

    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator, DataSource

    orch = DataFeedOrchestrator()

    # Candidate adapter keys (news sentiment adapter stored with tuple key)
    # Tuple key form is used when separate news sentiment adapter registered
    adapter = orch.adapters.get((DataSource.FINVIZ, "news")) or orch.adapters.get(DataSource.FINVIZ)  # type: ignore[index]

    if not adapter or not hasattr(adapter, "get_sentiment"):
        pytest.skip("FinViz news sentiment adapter not available in this environment")

    try:
        # Symbol choice arbitrary; any liquid ticker likely to have headlines
        _ = adapter.get_sentiment("AAPL")
    except ValueError as ve:  # Explicitly catch to inspect message
        msg = str(ve).lower()
        assert (
            "ambiguous" not in msg
        ), "Encountered pandas ambiguous truth-value error again"
        # Re-raise other ValueErrors for visibility
        raise
    except (
        Exception
    ) as e:  # Other exceptions are not considered regression for this specific issue
        logging.warning(
            "FinViz news sentiment fetch raised non-ambiguous exception: %s", e
        )
        # Do not fail; only the ambiguous truth-value regression is critical
        return

    # If we reach here without exception, regression is fixed / remains fixed
    assert True
