"""
Reuters News Adapter
Fetches financial news from Reuters RSS feeds with advanced sentiment analysis
"""

import logging
from typing import Dict, List, Optional, Any
from .base_news_adapter import BaseNewsAdapter

logger = logging.getLogger(__name__)


class ReutersAdapter(BaseNewsAdapter):
    """Reuters financial news adapter with enhanced sentiment analysis and fallback mechanisms"""

    def __init__(self):
        # Primary Reuters RSS feed
        primary_rss = "https://www.reuters.com/arc/outboundfeeds/rss/?outputType=xml"

        # Fallback RSS feeds for financial news
        self.fallback_feeds = [
            "https://feeds.npr.org/1006/rss.xml",  # NPR Business (financial news)
            "https://feeds.bloomberg.com/markets/news.rss",  # Bloomberg Markets
            "https://rss.cnn.com/rss/edition_business.rss",  # CNN Business
            "https://feeds.marketwatch.com/marketwatch/realtimeheadlines",  # MarketWatch
            "https://feeds.businesswire.com/BWTopStories",  # BusinessWire
        ]

        super().__init__(source_name="reuters", rss_url=primary_rss)

    def fetch_news_articles(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch Reuters articles with fallback to alternative financial news sources
        """
        # Try primary Reuters RSS feed first
        articles = self._fetch_from_rss(symbol, limit)

        if articles:
            # Additional Reuters-specific filtering
            filtered_articles = []
            for article in articles:
                if self._is_financial_article(article):
                    filtered_articles.append(article)

            logger.info(
                f"Filtered {len(filtered_articles)} financial articles from {len(articles)} Reuters articles for {symbol}"
            )
            return filtered_articles[:limit]

        # If primary feed fails, try fallback feeds
        logger.warning("Primary Reuters RSS feed failed, trying fallback feeds...")

        for fallback_url in self.fallback_feeds:
            try:
                logger.info(f"Trying fallback feed: {fallback_url}")

                # Temporarily change RSS URL
                original_url = self.rss_url
                self.rss_url = fallback_url

                fallback_articles = self._fetch_from_rss(symbol, limit)

                # Restore original URL
                self.rss_url = original_url

                if fallback_articles:
                    # Filter for financial content
                    filtered_articles = []
                    for article in fallback_articles:
                        if self._is_financial_article(article):
                            filtered_articles.append(article)

                    if filtered_articles:
                        logger.info(
                            f"Successfully fetched {len(filtered_articles)} articles from fallback feed for {symbol}"
                        )
                        return filtered_articles[:limit]

            except Exception as e:
                logger.warning(f"Fallback feed {fallback_url} failed: {e}")
                continue

        logger.error("All RSS feeds failed, no articles retrieved")
        return []

    def _is_financial_article(self, article: Dict[str, Any]) -> bool:
        """Check if article is finance-related with enhanced filtering"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()

        financial_keywords = [
            "stock",
            "stocks",
            "shares",
            "earnings",
            "revenue",
            "profit",
            "financial",
            "market",
            "trading",
            "investor",
            "investment",
            "analyst",
            "economy",
            "sec",
            "ipo",
            "dividend",
            "quarterly",
            "guidance",
            "forecast",
            "wall street",
            "nasdaq",
            "nyse",
            "sp 500",
            "s&p 500",
            "dow jones",
            "business",
            "company",
            "corporation",
            "corporate",
            "ceo",
            "cfo",
            "fund",
            "bank",
            "banking",
            "finance",
            "economics",
            "economic",
            "merger",
            "acquisition",
            "m&a",
            "valuation",
            "price",
            "trading",
            "bullish",
            "bearish",
            "volatility",
            "hedge fund",
            "etf",
            "mutual fund",
        ]

        return any(keyword in text for keyword in financial_keywords)

    def _is_relevant_to_symbol(self, article: Dict[str, Any], symbol: str) -> bool:
        """Enhanced symbol relevance check with fallback mechanisms"""
        # Use base class logic first (this includes the financial keywords fallback)
        if super()._is_relevant_to_symbol(article, symbol):
            return True

        # Additional Reuters-specific logic for stricter matching if needed
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()

        # Check for ticker format variations
        symbol_variations = [
            f"{symbol.lower()}",
            f"${symbol.lower()}",
            f"({symbol.lower()})",
            f"{symbol.upper()}",
            f"${symbol.upper()}",
            f"({symbol.upper()})",
        ]

        return any(variation in text for variation in symbol_variations)

    def get_health_status(self) -> Dict[str, Any]:
        """Get enhanced health status with fallback information"""
        base_status = super().get_health_status()
        base_status.update(
            {
                "fallback_feeds_available": len(self.fallback_feeds),
                "fallback_feeds": self.fallback_feeds,
                "resilient_mode": True,
            }
        )
        return base_status
