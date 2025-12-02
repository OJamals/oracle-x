"""
Unified Oracle Data Interface
Replaces all individual data feed calls in oracle_engine with quality-validated data from DataFeedOrchestrator.
This eliminates placeholder data and provides a single interface for all market intelligence.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from data_feeds.data_feed_orchestrator import (DataSource, MarketData, Quote,
                                               SentimentData, get_market_data,
                                               get_orchestrator, get_quote,
                                               get_sentiment_data,
                                               get_system_health)

logger = logging.getLogger(__name__)
from data_feeds.config_loader import load_config


@dataclass
class MarketIntelligence:
    """Consolidated market intelligence data structure"""

    timestamp: datetime
    tickers: List[str]
    market_data: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    quality_score: float
    data_sources_used: List[str]
    warnings: List[str]


class OracleDataProvider:
    """
    Unified data provider that replaces all individual scrapers in oracle_engine.
    Provides high-quality, validated data with fallback mechanisms.
    """

    def __init__(self):
        self.orchestrator = get_orchestrator()
        self.min_quality_threshold = 60.0
self.config = load_config()
        logger.info("OracleDataProvider initialized with quality validation")

    def get_comprehensive_market_intelligence(
        self, tickers: Optional[List[str]] = None
    ) -> MarketIntelligence:
        """
        Get comprehensive market intelligence replacing get_signals_from_scrapers()

        Args:
            tickers: List of ticker symbols to analyze

        Returns:
            MarketIntelligence object with all market data and quality metrics
        """
        if tickers is None:
            tickers = self._get_default_tickers()

        timestamp = datetime.now()
        warnings = []
        data_sources_used = []

        # Get market data for all tickers
        market_data = {}
        for ticker in tickers:
            quote = self.orchestrator.get_quote(ticker)
            if (
                quote
                and quote.quality_score
                and quote.quality_score >= self.min_quality_threshold
            ):
                market_data[ticker] = {
                    "price": float(quote.price),
                    "change": float(quote.change),
                    "change_percent": float(quote.change_percent),
                    "volume": quote.volume,
                    "quality_score": quote.quality_score,
                    "source": quote.source,
                }
                data_sources_used.append(quote.source)
            else:
                warnings.append(f"Poor quality or missing data for {ticker}")

        # Get sentiment data
        sentiment_data = {}
        for ticker in tickers[:5]:  # Limit sentiment analysis to top 5 tickers
            sentiments = self.orchestrator.get_sentiment_data(ticker)
            if sentiments:
                sentiment_data[ticker] = {}
                for source, sentiment in sentiments.items():
                    sentiment_data[ticker][source] = {
                        "sentiment_score": sentiment.sentiment_score,
                        "confidence": sentiment.confidence,
                        "sample_size": sentiment.sample_size,
                    }
                    data_sources_used.append(source)

        # Calculate technical indicators
        technical_indicators = self._calculate_technical_indicators(market_data)

        # Calculate overall quality score
        quality_scores = [data.get("quality_score", 0) for data in market_data.values()]
        overall_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0
        )

        return MarketIntelligence(
            timestamp=timestamp,
            tickers=tickers,
            market_data=market_data,
            sentiment_data=sentiment_data,
            technical_indicators=technical_indicators,
            quality_score=overall_quality,
            data_sources_used=list(set(data_sources_used)),
            warnings=warnings,
        )

    def get_market_internals(self) -> Dict[str, Any]:
        """
        Get market internals data with real data sources
        Replaces fetch_market_internals() placeholder
        """
        try:
            # Get major indices
            spy_quote = self.orchestrator.get_quote("SPY")
            qqq_quote = self.orchestrator.get_quote("QQQ")
            vix_quote = self.orchestrator.get_quote("^VIX")

            internals = {
                "spy_price": float(spy_quote.price) if spy_quote else None,
                "spy_change": float(spy_quote.change_percent) if spy_quote else None,
                "qqq_price": float(qqq_quote.price) if qqq_quote else None,
                "qqq_change": float(qqq_quote.change_percent) if qqq_quote else None,
                "vix": float(vix_quote.price) if vix_quote else None,
                "quality_score": (
                    min(
                        [
                            q.quality_score
                            for q in [spy_quote, qqq_quote, vix_quote]
                            if q and q.quality_score
                        ]
                    )
                    if any([spy_quote, qqq_quote, vix_quote])
                    else 0
                ),
            }

            # Add breadth indicators using real market data
            spy_change = internals.get("spy_change", 0)
            qqq_change = internals.get("qqq_change", 0)
            avg_change = (
                (spy_change + qqq_change) / 2 if spy_change and qqq_change else 0
            )

            if avg_change > 1.0:
                internals["market_breadth"] = "positive"
            elif avg_change < -1.0:
                internals["market_breadth"] = "negative"
            else:
                internals["market_breadth"] = "neutral"

            return internals

        except Exception as e:
            logger.error(f"Error getting market internals: {e}")
            return {"error": str(e), "quality_score": 0}

    def get_earnings_calendar(
        self, tickers: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get real earnings calendar data
        Replaces fetch_earnings_calendar() placeholder
        """
        # Use the updated earnings_calendar.py implementation
        from data_feeds.earnings_calendar import fetch_earnings_calendar

        return fetch_earnings_calendar(tickers)

    def get_options_analysis(
        self, tickers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get options analysis replacing options_flow placeholder
        Uses free data sources for basic options intelligence
        """
        if tickers is None:
            tickers = self._get_default_tickers()[:5]

        options_analysis = {
            "unusual_volume": [],
            "high_iv_stocks": [],
            "quality_score": 50,  # Mark as basic implementation
        }

        # Use the updated options_flow.py implementation
        from data_feeds.options_flow import fetch_options_flow

        return fetch_options_flow(tickers)

    def get_sentiment_analysis(
        self, tickers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis
        Replaces fetch_sentiment_data() with quality-validated data
        """
        if tickers is None:
            tickers = self._get_default_tickers()[:10]

        sentiment_summary = {
            "overall_sentiment": 0.0,
            "confidence": 0.0,
            "ticker_sentiments": {},
            "quality_score": 0.0,
        }

        total_sentiment = 0.0
        total_confidence = 0.0
        valid_sentiments = 0
        quality_scores = []

        for ticker in tickers:
            sentiments = self.orchestrator.get_sentiment_data(ticker)
            if sentiments:
                ticker_sentiment = 0.0
                ticker_confidence = 0.0
                ticker_sources = 0

                for source, sentiment_data in sentiments.items():
                    ticker_sentiment += sentiment_data.sentiment_score
                    ticker_confidence += sentiment_data.confidence
                    ticker_sources += 1

                if ticker_sources > 0:
                    avg_sentiment = ticker_sentiment / ticker_sources
                    avg_confidence = ticker_confidence / ticker_sources

                    sentiment_summary["ticker_sentiments"][ticker] = {
                        "sentiment": avg_sentiment,
                        "confidence": avg_confidence,
                        "sources": ticker_sources,
                    }

                    total_sentiment += avg_sentiment
                    total_confidence += avg_confidence
                    valid_sentiments += 1
                    quality_scores.append(
                        avg_confidence * 100
                    )  # Convert confidence to quality score

        if valid_sentiments > 0:
            sentiment_summary["overall_sentiment"] = total_sentiment / valid_sentiments
            sentiment_summary["confidence"] = total_confidence / valid_sentiments
            sentiment_summary["quality_score"] = sum(quality_scores) / len(
                quality_scores
            )

        return sentiment_summary

    def validate_data_quality(self) -> Dict[str, Any]:
        """Get comprehensive data quality validation"""
        return self.orchestrator.validate_system_health()

    def _get_default_tickers(self) -> List[str]:
        """Get default ticker universe for analysis"""
        return [
            "SPY",
            "QQQ",
            "IWM",  # Indices
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",  # Tech
            "NVDA",
            "META",
            "NFLX",
            "AMD",  # Growth
            "JPM",
            "BAC",
            "WFC",  # Financials
            "XOM",
            "CVX",  # Energy
        ]

    def _calculate_technical_indicators(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate basic technical indicators from market data"""
        indicators = {
            "high_volume_stocks": [],
            "large_movers": [],
            "quality_score": 100,
        }

        for ticker, data in market_data.items():
            # Identify high volume stocks
            if data.get("volume", 0) > 5000000:
                indicators["high_volume_stocks"].append(
                    {"ticker": ticker, "volume": data["volume"]}
                )

            # Identify large price movers
            change_pct = abs(data.get("change_percent", 0))
            if change_pct > 3.0:  # More than 3% move
                indicators["large_movers"].append(
                    {"ticker": ticker, "change_percent": data["change_percent"]}
                )

        return indicators


# ============================================================================
# Backward Compatibility Functions for Oracle Engine
# ============================================================================

# Global provider instance
_oracle_provider = None


def get_oracle_provider() -> OracleDataProvider:
    """Get the global oracle data provider instance"""
    global _oracle_provider
    if _oracle_provider is None:
        _oracle_provider = OracleDataProvider()
    return _oracle_provider


# These functions provide backward compatibility with existing oracle_engine code
def fetch_market_internals() -> Dict[str, Any]:
    """Backward compatible market internals (replaces placeholder)"""
    return get_oracle_provider().get_market_internals()


def fetch_options_flow(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Backward compatible options flow (replaces placeholder)"""
    return get_oracle_provider().get_options_analysis(tickers)


def fetch_dark_pool_data(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Backward compatible dark pool data (returns acknowledgment of limitation)"""
    # Dark pool data is not available through free sources
    return {
        "message": "Dark pool data requires premium subscription",
        "quality_score": 0,
        "alternative": "Using high volume analysis instead",
        "high_volume_stocks": get_oracle_provider()._calculate_technical_indicators(
            get_oracle_provider()
            .get_comprehensive_market_intelligence(tickers)
            .market_data
        )["high_volume_stocks"],
    }


def fetch_sentiment_data(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Backward compatible sentiment data (replaces multiple sentiment calls)"""
    return get_oracle_provider().get_sentiment_analysis(tickers)


def fetch_earnings_calendar(
    tickers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Backward compatible earnings calendar (replaces placeholder)"""
    return get_oracle_provider().get_earnings_calendar(tickers)


def get_signals_from_scrapers_v2(
    prompt_text: str, chart_image_b64: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhanced version of get_signals_from_scrapers with quality validation
    This is the main function that oracle_engine should use
    """
    provider = get_oracle_provider()

    # Get comprehensive market intelligence
    intelligence = provider.get_comprehensive_market_intelligence()

    # Prepare response in format expected by oracle_engine
    signals = {
        "tickers": intelligence.tickers,
        "market_internals": provider.get_market_internals(),
        "options_flow": provider.get_options_analysis(intelligence.tickers),
        "dark_pools": fetch_dark_pool_data(intelligence.tickers),
        "sentiment_web": provider.get_sentiment_analysis(intelligence.tickers),
        "earnings": provider.get_earnings_calendar(intelligence.tickers),
        "technical_indicators": intelligence.technical_indicators,
        # Quality and metadata
        "data_quality": {
            "overall_score": intelligence.quality_score,
            "sources_used": intelligence.data_sources_used,
            "warnings": intelligence.warnings,
            "timestamp": intelligence.timestamp,
        },
        # System health
        "system_health": provider.validate_data_quality(),
    }

    return signals
