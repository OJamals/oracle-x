"""
MarketWatch News Adapter - DEPRECATED

This adapter is deprecated. Use data_feeds.news_adapter.NewsAdapter instead,
which provides unified news handling with MarketWatch included in the
consolidated source list.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "MarketWatchAdapter is deprecated. Use NewsAdapter from data_feeds.news_adapter instead.",
    DeprecationWarning,
    stacklevel=2,
)


class MarketWatchAdapter:
    """
    DEPRECATED: MarketWatch news adapter.

    Use NewsAdapter from data_feeds.news_adapter for unified news handling.
    MarketWatch is included in the consolidated news sources.
    """

    def __init__(self):
        warnings.warn(
            "MarketWatchAdapter is deprecated. Use NewsAdapter from data_feeds.news_adapter",
            DeprecationWarning,
            stacklevel=2,
        )

    def fetch_news_articles(self, symbol: str, limit: int = 20):
        """DEPRECATED: Use NewsAdapter.fetch_news()"""
        warnings.warn(
            "fetch_news_articles is deprecated. Use NewsAdapter.fetch_news()",
            DeprecationWarning,
        )
        return []

    def get_sentiment(self, symbol: str, limit: int = 20):
        """DEPRECATED: Use NewsAdapter.fetch_sentiment()"""
        warnings.warn(
            "get_sentiment is deprecated. Use NewsAdapter.fetch_sentiment()",
            DeprecationWarning,
        )
        return None
