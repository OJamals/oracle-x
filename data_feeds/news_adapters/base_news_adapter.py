"""
Base News Adapter - DEPRECATED

This file is deprecated. Use data_feeds/news_adapter.py for unified news handling
with SourceAdapterProtocol compliance and consolidated sentiment analysis.

Specific news adapters should now inherit from the unified NewsAdapter class.
"""

import logging
import warnings

logger = logging.getLogger(__name__)

# Issue deprecation warning
warnings.warn(
    "BaseNewsAdapter is deprecated. Use data_feeds.news_adapter.NewsAdapter instead.",
    DeprecationWarning,
    stacklevel=2,
)


class BaseNewsAdapter:
    """
    DEPRECATED: Base class for news source adapters.

    This class is maintained for backward compatibility but should not be used
    for new development. Use data_feeds.news_adapter.NewsAdapter instead.
    """

    def __init__(
        self,
        source_name: str,
        rss_url: str = None,
        api_url: str = None,
        api_key: str = None,
    ):
        warnings.warn(
            f"BaseNewsAdapter is deprecated. Use NewsAdapter from data_feeds.news_adapter instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.source_name = source_name
        self.rss_url = rss_url
        self.api_url = api_url
        self.api_key = api_key

        # For backward compatibility, try to import old sentiment logic
        try:
            from sentiment.sentiment_engine import get_sentiment_engine

            self.sentiment_engine = get_sentiment_engine()
        except ImportError:
            self.sentiment_engine = None

    def fetch_news_articles(self, symbol: str, limit: int = 20):
        """DEPRECATED: Use NewsAdapter.fetch_news() instead"""
        warnings.warn(
            "fetch_news_articles is deprecated. Use NewsAdapter.fetch_news()",
            DeprecationWarning,
        )
        return []

    def get_sentiment(self, symbol: str, limit: int = 20):
        """DEPRECATED: Use NewsAdapter.fetch_sentiment() instead"""
        warnings.warn(
            "get_sentiment is deprecated. Use NewsAdapter.fetch_sentiment()",
            DeprecationWarning,
        )
        return None

    def get_health_status(self):
        """DEPRECATED: Use NewsAdapter.health() instead"""
        warnings.warn(
            "get_health_status is deprecated. Use NewsAdapter.health()",
            DeprecationWarning,
        )
        return {"status": "deprecated", "message": "Use NewsAdapter instead"}
