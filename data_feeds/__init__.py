# data_feeds package
# Free, open-source scraping sources for market/financial data:
# - Yahoo Finance: price, volume, news, fundamentals (yfinance, requests, bs4)
# - Finviz: breadth, earnings, block trades (requests, bs4)
# - Reddit: sentiment, trending tickers (PRAW)
# - Twitter/X: sentiment, trending tickers (snscrape)
# - Google Trends: search interest (pytrends)
# - News scraping: headlines, sentiment (newspaper3k, requests, bs4)

# Public exports for adapter utilities
from .investiny_adapter import (
    search_investing_id,
    fetch_historical_by_id,
    get_history,
)
__all__ = [
    "search_investing_id",
    "fetch_historical_by_id",
    "get_history",
]

# Public exports for adapter utilities
from .investiny_adapter import (
    search_investing_id,
    fetch_historical_by_id,
    get_history,
)
