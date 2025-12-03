"""
FinViz Adapter for Oracle-X Data Feed Orchestrator

This adapter provides FinViz data integration using the finvizfinance library
and custom scraping functions from finviz_scraper.py.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Define minimal classes to avoid circular imports
class DataSource:
    FINVIZ = "finviz"

    def __init__(self, value):
        self.value = value


class SentimentData:
    def __init__(
        self,
        symbol: str,
        sentiment_score: float,
        confidence: float,
        source: str,
        timestamp: datetime,
        sample_size: Optional[int] = None,
        raw_data: Optional[Dict] = None,
    ):
        self.symbol = symbol
        self.sentiment_score = sentiment_score
        self.confidence = confidence
        self.source = source
        self.timestamp = timestamp
        self.sample_size = sample_size
        self.raw_data = raw_data


class FinVizAdapter:
    """
    FinViz data adapter providing market breadth, sector performance, news, and sentiment data.
    """

    def __init__(self, cache=None, rate_limiter=None, performance_tracker=None):
        """Initialize FinViz adapter with optional caching and rate limiting"""
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker
        self.source = DataSource("finviz")  # Create instance with value

    def get_news(self) -> Optional[Dict]:
        """
        Get news data from FinViz using finvizfinance library.

        Returns:
            Dict containing news DataFrames or None if failed
        """
        try:
            # Import here to avoid circular imports
            from data_feeds.finviz_scraper import fetch_finviz_news

            # Check rate limiting if available
            if self.rate_limiter and not self.rate_limiter.wait_if_needed(self.source):
                logger.warning("Rate limit exceeded for FinViz news")
                return None

            start_time = datetime.now()

            # Fetch news data
            news_data = fetch_finviz_news()

            # Track performance if available
            if self.performance_tracker:
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                quality_score = 80.0 if news_data else 0.0
                self.performance_tracker.record_success(
                    self.source.value, response_time, quality_score
                )

            return news_data

        except Exception as e:
            logger.error(f"FinViz news fetch failed: {e}")
            if self.performance_tracker:
                self.performance_tracker.record_error(self.source.value, str(e))
            return None

    def get_market_breadth(self) -> Optional[Dict]:
        """
        Get market breadth data (advancers/decliners) from FinViz.

        Returns:
            Dict with advancers, decliners, new_highs, new_lows or None if failed
        """
        try:
            # Import here to avoid circular imports
            from data_feeds.finviz_scraper import fetch_finviz_breadth

            # Check rate limiting if available
            if self.rate_limiter and not self.rate_limiter.wait_if_needed(self.source):
                logger.warning("Rate limit exceeded for FinViz breadth")
                return None

            start_time = datetime.now()

            # Fetch breadth data
            breadth_data = fetch_finviz_breadth()

            # Track performance if available
            if self.performance_tracker:
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                quality_score = (
                    85.0 if breadth_data and any(breadth_data.values()) else 0.0
                )
                self.performance_tracker.record_success(
                    self.source.value, response_time, quality_score
                )

            return breadth_data

        except Exception as e:
            logger.error(f"FinViz breadth fetch failed: {e}")
            if self.performance_tracker:
                self.performance_tracker.record_error(self.source.value, str(e))
            return None

    def get_sector_performance(self) -> Optional[List[Dict]]:
        """
        Get sector performance data from FinViz.

        Returns:
            List of sector performance dictionaries or None if failed
        """
        try:
            # Import here to avoid circular imports
            from data_feeds.finviz_scraper import fetch_finviz_sector_performance

            # Check rate limiting if available
            if self.rate_limiter and not self.rate_limiter.wait_if_needed(self.source):
                logger.warning("Rate limit exceeded for FinViz sectors")
                return None

            start_time = datetime.now()

            # Fetch sector data
            sector_data = fetch_finviz_sector_performance()

            # Track performance if available
            if self.performance_tracker:
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                quality_score = 80.0 if sector_data and len(sector_data) > 0 else 0.0
                self.performance_tracker.record_success(
                    self.source.value, response_time, quality_score
                )

            return sector_data

        except Exception as e:
            logger.error(f"FinViz sector performance fetch failed: {e}")
            if self.performance_tracker:
                self.performance_tracker.record_error(self.source.value, str(e))
            return None

    def get_insider_trading(self) -> Optional[object]:
        """
        Get insider trading data from FinViz.

        Returns:
            Pandas DataFrame with insider trading data or None if failed
        """
        try:
            # Import here to avoid circular imports
            from data_feeds.finviz_scraper import fetch_finviz_insider_trading

            # Check rate limiting if available
            if self.rate_limiter and not self.rate_limiter.wait_if_needed(self.source):
                logger.warning("Rate limit exceeded for FinViz insider trading")
                return None

            start_time = datetime.now()

            # Fetch insider trading data
            insider_data = fetch_finviz_insider_trading()

            # Track performance if available
            if self.performance_tracker:
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                quality_score = (
                    75.0 if insider_data is not None and not insider_data.empty else 0.0
                )
                self.performance_tracker.record_success(
                    self.source.value, response_time, quality_score
                )

            return insider_data

        except Exception as e:
            logger.error(f"FinViz insider trading fetch failed: {e}")
            if self.performance_tracker:
                self.performance_tracker.record_error(self.source.value, str(e))
            return None

    def get_earnings(self) -> Optional[Dict]:
        """
        Get earnings data from FinViz.

        Returns:
            Dict with earnings data partitioned by day or None if failed
        """
        try:
            # Import here to avoid circular imports
            from data_feeds.finviz_scraper import fetch_finviz_earnings

            # Check rate limiting if available
            if self.rate_limiter and not self.rate_limiter.wait_if_needed(self.source):
                logger.warning("Rate limit exceeded for FinViz earnings")
                return None

            start_time = datetime.now()

            # Fetch earnings data
            earnings_data = fetch_finviz_earnings()

            # Track performance if available
            if self.performance_tracker:
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                quality_score = (
                    70.0 if earnings_data and any(earnings_data.values()) else 0.0
                )
                self.performance_tracker.record_success(
                    self.source.value, response_time, quality_score
                )

            return earnings_data

        except Exception as e:
            logger.error(f"FinViz earnings fetch failed: {e}")
            if self.performance_tracker:
                self.performance_tracker.record_error(self.source.value, str(e))
            return None

    def get_forex_performance(self) -> Optional[object]:
        """
        Get forex performance data from FinViz.

        Returns:
            Pandas DataFrame with forex performance data or None if failed
        """
        try:
            # Import here to avoid circular imports
            from data_feeds.finviz_scraper import fetch_finviz_forex

            # Check rate limiting if available
            if self.rate_limiter and not self.rate_limiter.wait_if_needed(self.source):
                logger.warning("Rate limit exceeded for FinViz forex")
                return None

            start_time = datetime.now()

            # Fetch forex data
            forex_data = fetch_finviz_forex()

            # Track performance if available
            if self.performance_tracker:
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                quality_score = (
                    75.0 if forex_data is not None and not forex_data.empty else 0.0
                )
                self.performance_tracker.record_success(
                    self.source.value, response_time, quality_score
                )

            return forex_data

        except Exception as e:
            logger.error(f"FinViz forex fetch failed: {e}")
            if self.performance_tracker:
                self.performance_tracker.record_error(self.source.value, str(e))
            return None

    def get_crypto_performance(self) -> Optional[object]:
        """
        Get crypto performance data from FinViz.

        Returns:
            Pandas DataFrame with crypto performance data or None if failed
        """
        try:
            # Import here to avoid circular imports
            from data_feeds.finviz_scraper import fetch_finviz_crypto

            # Check rate limiting if available
            if self.rate_limiter and not self.rate_limiter.wait_if_needed(self.source):
                logger.warning("Rate limit exceeded for FinViz crypto")
                return None

            start_time = datetime.now()

            # Fetch crypto data
            crypto_data = fetch_finviz_crypto()

            # Track performance if available
            if self.performance_tracker:
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                quality_score = (
                    75.0 if crypto_data is not None and not crypto_data.empty else 0.0
                )
                self.performance_tracker.record_success(
                    self.source.value, response_time, quality_score
                )

            return crypto_data

        except Exception as e:
            logger.error(f"FinViz crypto fetch failed: {e}")
            if self.performance_tracker:
                self.performance_tracker.record_error(self.source.value, str(e))
            return None
