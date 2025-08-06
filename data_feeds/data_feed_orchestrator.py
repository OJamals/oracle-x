"""
DataFeedOrchestrator - Unified Financial Data Interface
Consolidates multiple data sources with quality validation, intelligent fallback, and performance tracking.
Replaces all existing data feed implementations with a single authoritative interface.
"""

import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import yfinance as yf
import requests
import json
from functools import wraps
import numpy as np
from collections import defaultdict, deque
from data_feeds.models import MarketBreadth, GroupPerformance  # Added import
from data_feeds.twelvedata_adapter import TwelveDataAdapter  # New import
from data_feeds.finviz_adapter import FinVizAdapter  # New import
from dotenv import load_dotenv
from data_feeds.twitter_feed import TwitterSentimentFeed  # New import
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ============================================================================
# Core Data Models
# ============================================================================

class DataSource(Enum):
    """Available data sources in priority order"""
    YFINANCE = "yfinance"
    FINNHUB = "finnhub"
    IEX_CLOUD = "iex_cloud"
    REDDIT = "reddit"
    TWITTER = "twitter"
    GOOGLE_TRENDS = "google_trends"
    YAHOO_NEWS = "yahoo_news"
    FRED = "fred"
    SEC_EDGAR = "sec_edgar"
    TWELVE_DATA = "twelve_data"
    FINVIZ = "finviz"

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"  # 95-100% quality score
    GOOD = "good"           # 80-94% quality score
    FAIR = "fair"           # 60-79% quality score
    POOR = "poor"           # 40-59% quality score
    UNUSABLE = "unusable"   # <40% quality score

@dataclass
class Quote:
    symbol: str
    price: Decimal
    change: Decimal
    change_percent: Decimal
    volume: int
    market_cap: Optional[int] = None
    pe_ratio: Optional[Decimal] = None
    day_low: Optional[Decimal] = None
    day_high: Optional[Decimal] = None
    year_low: Optional[Decimal] = None
    year_high: Optional[Decimal] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    quality_score: Optional[float] = None

@dataclass
class MarketData:
    symbol: str
    data: pd.DataFrame
    timeframe: str
    source: str
    timestamp: datetime
    quality_score: float

@dataclass
class SentimentData:
    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float      # 0 to 1
    source: str
    timestamp: datetime
    sample_size: Optional[int] = None
    raw_data: Optional[Dict] = None

@dataclass
class DataQualityMetrics:
    source: str
    quality_score: float
    latency_ms: float
    success_rate: float
    last_updated: datetime
    issues: List[str]

# ============================================================================
# Data Quality Framework
# ============================================================================

class DataValidator:
    """Validates data quality and detects anomalies"""
    
    @staticmethod
    def validate_quote(quote: Quote) -> Tuple[float, List[str]]:
        """Validate quote data and return quality score with issues"""
        issues = []
        score = 100.0
        
        # Check required fields
        if not quote.price or quote.price <= 0:
            issues.append("Invalid or missing price")
            score -= 50
        
        if not quote.volume or quote.volume < 0:
            issues.append("Invalid volume")
            score -= 20
        
        # Check timestamp freshness
        if quote.timestamp:
            age_minutes = (datetime.now() - quote.timestamp).total_seconds() / 60
            if age_minutes > 60:  # Data older than 1 hour
                issues.append(f"Stale data: {age_minutes:.1f} minutes old")
                score -= min(30, age_minutes)
        else:
            issues.append("Missing timestamp")
            score -= 10
        
        # Check price reasonableness
        if quote.day_low and quote.day_high:
            if quote.price < quote.day_low or quote.price > quote.day_high:
                issues.append("Price outside day range")
                score -= 30
        
        return max(0, score), issues
    
    @staticmethod
    def validate_market_data(data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Validate market data DataFrame"""
        issues = []
        score = 100.0
        
        if data.empty:
            return 0, ["Empty dataset"]
        
        # Check for missing data
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        if missing_pct > 5:
            issues.append(f"High missing data: {missing_pct:.1f}%")
            score -= missing_pct
        
        # Check for outliers in price columns
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in data.columns:
                # Check for extreme price movements (>50% in one day)
                if len(data) > 1:
                    pct_change = data[col].pct_change().abs()
                    extreme_moves = (pct_change > 0.5).sum()
                    if extreme_moves > 0:
                        issues.append(f"Extreme price movements in {col}: {extreme_moves}")
                        score -= min(20, extreme_moves * 5)
        
        return max(0, score), issues
    
    @staticmethod
    def detect_anomalies(data: pd.Series, threshold: float = 3.0) -> List[int]:
        """Detect anomalies using Z-score method"""
        if len(data) < 10:
            return []
        
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[z_scores > threshold].index.tolist()

class PerformanceTracker:
    """Tracks data source performance and reliability"""
    
    def __init__(self):
        self.metrics = defaultdict(self._create_metrics_dict)
    
    def _create_metrics_dict(self):
        return {
            'response_times': deque(maxlen=100),
            'success_count': 0,
            'error_count': 0,
            'quality_scores': deque(maxlen=50),
            'last_success': None,
            'last_error': None
        }
    
    def record_success(self, source: str, response_time: float, quality_score: float):
        """Record successful data fetch"""
        metrics = self.metrics[source]
        metrics['response_times'].append(response_time)
        metrics['success_count'] += 1
        metrics['quality_scores'].append(quality_score)
        metrics['last_success'] = datetime.now()
    
    def record_error(self, source: str, error: str):
        """Record data fetch error"""
        metrics = self.metrics[source]
        metrics['error_count'] += 1
        metrics['last_error'] = datetime.now()
        logger.warning(f"Data source {source} error: {error}")
    
    def get_source_ranking(self) -> List[Tuple[str, float]]:
        """Get data sources ranked by performance"""
        rankings = []
        
        for source, metrics in self.metrics.items():
            success_count = metrics['success_count']
            error_count = metrics['error_count']
            total_requests = success_count + error_count
            
            if total_requests == 0:
                continue
            
            success_rate = success_count / total_requests
            avg_response_time = float(np.mean(metrics['response_times'])) if metrics['response_times'] else 10.0
            avg_quality = float(np.mean(metrics['quality_scores'])) if metrics['quality_scores'] else 50.0
            
            # Calculate composite score (0-100)
            score = (success_rate * 40) + (avg_quality * 0.4) + (max(0.0, 100.0 - avg_response_time) * 0.2)
            rankings.append((source, score))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)

# ============================================================================
# Enhanced Caching System
# ============================================================================

class SmartCache:
    """Intelligent caching with TTL and quality-based retention"""
    
    def __init__(self):
        self.cache = {}
        self.ttl_settings = {
            'quote': 30,           # 30 seconds for quotes
            'market_data_1d': 3600,  # 1 hour for daily data
            'market_data_1h': 300,   # 5 minutes for hourly data
            'sentiment': 600,        # 10 minutes for sentiment
            'news': 1800,           # 30 minutes for news
            'company_info': 86400,  # 24 hours for company info
            # Safe defaults for new data types
            'market_breadth': 300,   # 5 minutes
            'group_performance': 1800,  # 30 minutes
        }
    
    def get(self, key: str, data_type: str) -> Optional[Any]:
        """Get cached data if still valid"""
        if key in self.cache:
            data, timestamp, quality_score = self.cache[key]
            age = time.time() - timestamp
            ttl = self.ttl_settings.get(data_type, 3600)
            
            # Extend TTL for high-quality data
            if quality_score > 90:
                ttl *= 1.5
            elif quality_score < 60:
                ttl *= 0.5
            
            if age < ttl:
                return data
            else:
                del self.cache[key]
        
        return None
    
    def set(self, key: str, data: Any, data_type: str, quality_score: float = 100.0):
        """Cache data with quality score"""
        self.cache[key] = (data, time.time(), quality_score)
        
        # Periodic cleanup
        if len(self.cache) > 1000:
            self._cleanup()
    
    def _cleanup(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, (data, timestamp, quality_score) in self.cache.items():
            # Determine data type from key
            data_type = key.split('_')[0] if '_' in key else 'default'
            ttl = self.ttl_settings.get(data_type, 3600)
            
            if current_time - timestamp > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]

# ============================================================================
# Rate Limiting System
# ============================================================================

class RateLimiter:
    """Intelligent rate limiting with quota management"""
    
    def __init__(self):
        self.calls = defaultdict(deque)
        self.limits = {
            DataSource.FINNHUB: (60, 60),      # 60 calls per minute
            DataSource.IEX_CLOUD: (100, 60),   # 100 calls per minute (free tier)
            DataSource.REDDIT: (60, 60),       # 60 calls per minute
            DataSource.TWITTER: (100, 900),    # 100 calls per 15 minutes
            DataSource.FRED: (120, 60),        # 120 calls per minute
            DataSource.TWELVE_DATA: (60, 60),  # 60 req per 60s (quick-path default)
            DataSource.FINVIZ: (12, 60),       # 12 req per 60s
        }
        self.daily_quotas = {
            DataSource.FINNHUB: 1000,    # Daily free quota
            DataSource.IEX_CLOUD: 50000, # Daily free quota
        }
        self.daily_usage = defaultdict(int)
        self.last_reset = datetime.now().date()
    
    def wait_if_needed(self, source: DataSource) -> bool:
        """Check rate limits and wait if needed. Returns False if quota exceeded."""
        # Reset daily counters if new day
        if datetime.now().date() > self.last_reset:
            self.daily_usage.clear()
            self.last_reset = datetime.now().date()
        
        # Check daily quota
        if source in self.daily_quotas:
            if self.daily_usage[source] >= self.daily_quotas[source]:
                logger.warning(f"Daily quota exceeded for {source.value}")
                return False
        
        # Check rate limits
        if source in self.limits:
            max_calls, window = self.limits[source]
            now = time.time()
            
            # Clean old calls
            while self.calls[source] and now - self.calls[source][0] > window:
                self.calls[source].popleft()
            
            # Check if we need to wait
            if len(self.calls[source]) >= max_calls:
                wait_time = window - (now - self.calls[source][0]) + 1
                if wait_time > 0:
                    logger.info(f"Rate limiting {source.value}: waiting {wait_time:.1f} seconds")
                    time.sleep(wait_time)
        
        # Record this call
        self.calls[source].append(time.time())
        self.daily_usage[source] += 1
        return True

# ============================================================================
# Data Source Adapters
# ============================================================================

class YFinanceAdapter:
    """Enhanced yfinance adapter with quality validation"""
    
    def __init__(self, cache: SmartCache, rate_limiter: RateLimiter, performance_tracker: PerformanceTracker):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker
        self.source = DataSource.YFINANCE
    
    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get real-time quote with quality validation"""
        cache_key = f"quote_{symbol}"
        cached_data = self.cache.get(cache_key, "quote")
        if cached_data:
            return cached_data
        
        start_time = time.time()
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'currentPrice' not in info:
                return None
            
            quote = Quote(
                symbol=symbol,
                price=Decimal(str(info['currentPrice'])),
                change=Decimal(str(info.get('change', 0))),
                change_percent=Decimal(str(info.get('changePercent', 0))),
                volume=int(info.get('volume', 0)),
                market_cap=info.get('marketCap'),
                pe_ratio=Decimal(str(info.get('trailingPE', 0))) if info.get('trailingPE') else None,
                day_low=Decimal(str(info.get('dayLow', 0))) if info.get('dayLow') else None,
                day_high=Decimal(str(info.get('dayHigh', 0))) if info.get('dayHigh') else None,
                year_low=Decimal(str(info.get('fiftyTwoWeekLow', 0))) if info.get('fiftyTwoWeekLow') else None,
                year_high=Decimal(str(info.get('fiftyTwoWeekHigh', 0))) if info.get('fiftyTwoWeekHigh') else None,
                timestamp=datetime.now(),
                source=self.source.value
            )
            
            # Validate quality
            quality_score, issues = DataValidator.validate_quote(quote)
            quote.quality_score = quality_score
            
            # Track performance
            response_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_success(self.source.value, response_time, quality_score)
            
            # Cache if quality is acceptable
            if quality_score >= 60:
                self.cache.set(cache_key, quote, "quote", quality_score)
            
            return quote
            
        except Exception as e:
            self.performance_tracker.record_error(self.source.value, str(e))
            logger.error(f"YFinance quote error for {symbol}: {e}")
            return None
    
    def get_market_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[MarketData]:
        """Get historical market data with quality validation"""
        cache_key = f"market_data_{symbol}_{period}_{interval}"
        cached_data = self.cache.get(cache_key, f"market_data_{interval}")
        if cached_data:
            return cached_data
        
        start_time = time.time()
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None
            
            # Validate quality
            quality_score, issues = DataValidator.validate_market_data(data)
            
            market_data = MarketData(
                symbol=symbol,
                data=data,
                timeframe=f"{period}_{interval}",
                source=self.source.value,
                timestamp=datetime.now(),
                quality_score=quality_score
            )
            
            # Track performance
            response_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_success(self.source.value, response_time, quality_score)
            
            # Cache if quality is acceptable
            if quality_score >= 60:
                self.cache.set(cache_key, market_data, f"market_data_{interval}", quality_score)
            
            return market_data
            
        except Exception as e:
            self.performance_tracker.record_error(self.source.value, str(e))
            logger.error(f"YFinance market data error for {symbol}: {e}")
            return None

class RedditAdapter:
    """Reddit sentiment data adapter"""
    
    def __init__(self, cache: SmartCache, rate_limiter: RateLimiter, performance_tracker: PerformanceTracker):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker
        self.source = DataSource.REDDIT
        self._all_sentiment_cache = None
        self._cache_timestamp = None
        self._cache_duration = 300  # 5 minutes cache for all sentiment data
    
    def get_sentiment(self, symbol: str, subreddits: Optional[List[str]] = None) -> Optional[SentimentData]:
        """Get Reddit sentiment data with intelligent caching"""
        if subreddits is None:
            subreddits = ["stocks", "investing", "SecurityAnalysis", "ValueInvesting"]
        
        cache_key = f"sentiment_reddit_{symbol}"
        cached_data = self.cache.get(cache_key, "sentiment")
        if cached_data:
            return cached_data
        
        # Check if we need to refresh the all-sentiment cache
        current_time = time.time()
        if (self._all_sentiment_cache is None or 
            self._cache_timestamp is None or 
            current_time - self._cache_timestamp > self._cache_duration):
            
            if not self.rate_limiter.wait_if_needed(self.source):
                return None
            
            try:
                # Import Reddit sentiment function
                from data_feeds.reddit_sentiment import fetch_reddit_sentiment
                
                start_time = time.time()
                self._all_sentiment_cache = fetch_reddit_sentiment()
                self._cache_timestamp = current_time
                
                # Track the batch performance
                response_time = (time.time() - start_time) * 1000
                quality_score = 70 if self._all_sentiment_cache else 0
                self.performance_tracker.record_success(self.source.value, response_time, quality_score)
                
            except Exception as e:
                self.performance_tracker.record_error(self.source.value, str(e))
                logger.error(f"Reddit sentiment batch error: {e}")
                return None
        
        # Now extract specific symbol data from cached results
        if not self._all_sentiment_cache or symbol not in self._all_sentiment_cache:
            return None
        
        start_time = time.time()
        try:
            symbol_data = self._all_sentiment_cache[symbol]
            
            # Check if we have enhanced sentiment data
            if 'enhanced_sentiment' in symbol_data:
                enhanced = symbol_data['enhanced_sentiment']
                sentiment_data = SentimentData(
                    symbol=symbol,
                    sentiment_score=enhanced.get('ensemble_score', 0.0),
                    confidence=enhanced.get('confidence', 0.0),
                    source=self.source.value,
                    timestamp=datetime.now(),
                    sample_size=enhanced.get('sample_size', 0),
                    raw_data=enhanced
                )
                quality_score = enhanced.get('quality_score', 0)
            else:
                # Fallback to basic sentiment data
                sentiment_score = symbol_data.get('sentiment_score', 0.0)
                confidence = symbol_data.get('confidence', 0.5)
                sample_size = symbol_data.get('sample_size', 0)
                
                # Include sample texts in raw_data for ML engine
                raw_data = symbol_data.copy()
                if 'sample_texts' in symbol_data:
                    raw_data['sample_texts'] = symbol_data['sample_texts']
                
                sentiment_data = SentimentData(
                    symbol=symbol,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    source=self.source.value,
                    timestamp=datetime.now(),
                    sample_size=sample_size,
                    raw_data=raw_data
                )
                quality_score = min(100, (sample_size * 2) + (confidence * 50))
            
            # Track performance for individual symbol extraction
            response_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_success(self.source.value, response_time, quality_score)
            
            # Cache the individual sentiment data
            self.cache.set(cache_key, sentiment_data, "sentiment", quality_score)
            
            return sentiment_data
            
        except Exception as e:
            self.performance_tracker.record_error(self.source.value, str(e))
            logger.error(f"Reddit sentiment error for {symbol}: {e}")
            return None

class TwitterAdapter:
    """Twitter sentiment data adapter using TwitterSentimentFeed"""
    
    def __init__(self, cache: SmartCache, rate_limiter: RateLimiter, performance_tracker: PerformanceTracker):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker
        self.source = DataSource.TWITTER
        self.feed = TwitterSentimentFeed()
        
    def get_sentiment(self, symbol: str, limit: int = 50) -> Optional[SentimentData]:
        """Get Twitter sentiment data for a symbol"""
        cache_key = f"twitter_sentiment_{symbol}_{limit}"
        
        # Check cache first (valid for 30 minutes)
        cached_data = self.cache.get(cache_key, "sentiment")
        if cached_data:
            return cached_data
            
        # Check rate limits
        if not self.rate_limiter.wait_if_needed(self.source):
            logger.warning(f"Rate limit exceeded for Twitter sentiment: {symbol}")
            return None
            
        start_time = time.time()
        try:
            # Fetch tweets using TwitterSentimentFeed
            tweets = self.feed.fetch(symbol, limit=limit)
            
            if not tweets:
                logger.warning(f"No Twitter data found for {symbol}")
                return None
                
            # Extract sentiment data
            sentiments = []
            texts = []
            for tweet in tweets:
                if 'sentiment' in tweet and 'text' in tweet:
                    # Twitter feed already provides VADER sentiment
                    sentiment_score = tweet['sentiment'].get('compound', 0.0)
                    sentiments.append(sentiment_score)
                    texts.append(tweet['text'])
            
            if not sentiments:
                logger.warning(f"No sentiment data extracted from Twitter for {symbol}")
                return None
            
            # Calculate aggregate sentiment
            overall_sentiment = sum(sentiments) / len(sentiments)
            
            # Calculate confidence based on sample size and variance
            sentiment_variance = sum((s - overall_sentiment) ** 2 for s in sentiments) / len(sentiments) if sentiments else 0
            confidence = min(0.95, max(0.1, 1.0 - sentiment_variance))
            
            # Calculate quality score based on sample size and confidence
            sample_weight = min(1.0, len(sentiments) / 20.0)  # Better with more samples
            quality_score = (confidence * 0.7 + sample_weight * 0.3) * 100
            
            # Record performance
            execution_time = time.time() - start_time
            self.performance_tracker.record_success(self.source.value, execution_time, quality_score)
            
            # Create sentiment data object
            sentiment_data = SentimentData(
                symbol=symbol,
                sentiment_score=overall_sentiment,
                confidence=confidence,
                source="twitter_advanced",
                timestamp=datetime.now(),
                sample_size=len(sentiments),
                raw_data={
                    'tweets': tweets,
                    'sample_texts': texts,
                    'individual_sentiments': sentiments,
                    'variance': sentiment_variance,
                    'execution_time': execution_time
                }
            )
            
            # Cache the result
            self.cache.set(cache_key, sentiment_data, "sentiment", quality_score)
            
            logger.info(f"Twitter sentiment for {symbol}: {overall_sentiment:.3f} "
                      f"(confidence: {confidence:.3f}, samples: {len(sentiments)})")
            
            return sentiment_data
            
        except Exception as e:
            self.performance_tracker.record_error(self.source.value, str(e))
            logger.error(f"Failed to fetch Twitter sentiment for {symbol}: {e}")
            return None

class AdvancedSentimentAdapter:
    """Advanced multi-model sentiment analysis adapter"""
    
    def __init__(self, cache: SmartCache, rate_limiter: RateLimiter, performance_tracker: PerformanceTracker):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker
        self.source = DataSource.REDDIT  # Uses same source but enhanced processing
        
        # Lazy load the advanced sentiment engine
        self._engine = None
    
    def get_sentiment(self, symbol: str, texts: Optional[List[str]] = None, sources: Optional[List[str]] = None) -> Optional[SentimentData]:
        """Get advanced sentiment analysis using multi-model ensemble"""
        cache_key = f"advanced_sentiment_{symbol}"
        cached_data = self.cache.get(cache_key, "sentiment")
        if cached_data:
            return cached_data
        
        if not texts:
            # No texts provided - can't analyze
            return None
        
        start_time = time.time()
        try:
            # Lazy load the engine
            if self._engine is None:
                from data_feeds.advanced_sentiment import get_sentiment_engine
                self._engine = get_sentiment_engine()
            
            # Analyze sentiment with advanced engine
            summary = self._engine.get_symbol_sentiment_summary(symbol, texts, sources if sources is not None else None)
            
            # Convert to SentimentData format
            sentiment_data = SentimentData(
                symbol=symbol,
                sentiment_score=summary.overall_sentiment,
                confidence=summary.confidence,
                source="advanced_sentiment",
                timestamp=summary.timestamp,
                sample_size=summary.sample_size,
                raw_data={
                    'bullish_mentions': summary.bullish_mentions,
                    'bearish_mentions': summary.bearish_mentions,
                    'neutral_mentions': summary.neutral_mentions,
                    'trending_direction': summary.trending_direction,
                    'quality_score': summary.quality_score,
                    'ensemble_score': summary.overall_sentiment,
                    'confidence': summary.confidence,
                    'sample_size': summary.sample_size
                }
            )
            
            quality_score = summary.quality_score
            
            # Track performance
            response_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_success("advanced_sentiment", response_time, quality_score)
            
            # Cache the result
            self.cache.set(cache_key, sentiment_data, "sentiment", quality_score)
            
            return sentiment_data
            
        except Exception as e:
            self.performance_tracker.record_error("advanced_sentiment", str(e))
            logger.error(f"Advanced sentiment error for {symbol}: {e}")
            return None

# ============================================================================
# Main DataFeedOrchestrator Class
# ============================================================================

class DataFeedOrchestrator:
    """
    Unified data feed orchestrator that replaces all existing feed implementations.
    Provides intelligent data source selection, quality validation, and fallback mechanisms.
    """
    
    def __init__(self):
        # Initialize core components
        self.cache = SmartCache()
        self.rate_limiter = RateLimiter()
        self.performance_tracker = PerformanceTracker()
        self.validator = DataValidator()
        
        # Initialize data source adapters
        self.adapters = {
            DataSource.YFINANCE: YFinanceAdapter(self.cache, self.rate_limiter, self.performance_tracker),
            DataSource.REDDIT: RedditAdapter(self.cache, self.rate_limiter, self.performance_tracker),
            DataSource.TWITTER: TwitterAdapter(self.cache, self.rate_limiter, self.performance_tracker),
            # New adapters
            DataSource.TWELVE_DATA: TwelveDataAdapter(),
            DataSource.FINVIZ: FinVizAdapter(),
        }
        
        # Initialize advanced sentiment adapter
        self.advanced_sentiment_adapter = AdvancedSentimentAdapter(
            self.cache, self.rate_limiter, self.performance_tracker
        )
        
        # Quality thresholds
        self.min_quality_score = 60.0
        self.preferred_quality_score = 80.0
        
        logger.info("DataFeedOrchestrator initialized with quality validation")
    
    def get_quote(self, symbol: str, preferred_sources: Optional[List[DataSource]] = None) -> Optional[Quote]:
        """
        Get real-time quote with intelligent source selection
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            preferred_sources: List of preferred data sources in order
            
        Returns:
            Quote object with quality score, or None if unavailable
        """
        # Quick-path: accept strings like "twelve_data" and convert to enum
        if preferred_sources:
            normalized: List[DataSource] = []
            for s in preferred_sources:
                if isinstance(s, DataSource):
                    normalized.append(s)
                elif isinstance(s, str):
                    try:
                        # allow both enum value and name casing
                        normalized.append(DataSource(s))
                    except Exception:
                        try:
                            normalized.append(DataSource[s.upper()])
                        except Exception:
                            continue
            if normalized:
                preferred_sources = normalized

        if preferred_sources is None or not preferred_sources:
            # Use performance-based ranking (default to YFINANCE if none)
            source_rankings = self.performance_tracker.get_source_ranking()
            preferred_sources = [DataSource(source) for source, _ in source_rankings 
                               if DataSource(source) in [DataSource.YFINANCE]]
            if not preferred_sources:
                preferred_sources = [DataSource.YFINANCE]

        # Minimal honoring: if Twelve Data is requested, try it first
        if any(s == DataSource.TWELVE_DATA for s in preferred_sources):
            ordered = [DataSource.TWELVE_DATA] + [s for s in preferred_sources if s != DataSource.TWELVE_DATA]
        else:
            ordered = preferred_sources

        best_quote = None
        best_quality = 0
        
        for source in ordered:
            if source not in self.adapters:
                continue
            
            adapter = self.adapters[source]
            # Adapters may not all implement get_quote; guard if needed
            if not hasattr(adapter, "get_quote"):
                continue

            quote = adapter.get_quote(symbol)
            
            if quote and (quote.quality_score or 0) > best_quality:
                best_quote = quote
                best_quality = quote.quality_score or 0
                
                # If we get excellent quality data, use it immediately
                if best_quality >= self.preferred_quality_score:
                    break
        
        return best_quote
    
    def get_market_data(self, symbol: str, period: str = "1y", interval: str = "1d", preferred_sources: Optional[List[DataSource]] = None) -> Optional[MarketData]:
        """Get historical market data with quality validation"""
        # Quick-path: normalize preferred_sources to enums and honor order, prioritizing Twelve Data when requested
        if preferred_sources:
            normalized: List[DataSource] = []
            for s in preferred_sources:
                if isinstance(s, DataSource):
                    normalized.append(s)
                elif isinstance(s, str):
                    try:
                        normalized.append(DataSource(s))
                    except Exception:
                        try:
                            normalized.append(DataSource[s.upper()])
                        except Exception:
                            continue
            preferred_sources = normalized

        if preferred_sources and any(s == DataSource.TWELVE_DATA for s in preferred_sources):
            ordered = [DataSource.TWELVE_DATA] + [s for s in preferred_sources if s != DataSource.TWELVE_DATA]
        else:
            ordered = preferred_sources or [DataSource.YFINANCE]

        # Try ordered sources, fall back to YFINANCE if none succeed
        for source in ordered:
            adapter = self.adapters.get(source)
            if not adapter or not hasattr(adapter, "get_market_data"):
                continue
            md = adapter.get_market_data(symbol, period, interval)
            if md and md.data is not None and not md.data.empty:
                return md

        # Final fallback
        adapter = self.adapters[DataSource.YFINANCE]
        return adapter.get_market_data(symbol, period, interval)
    
    def get_sentiment_data(self, symbol: str, sources: Optional[List[DataSource]] = None) -> Dict[str, SentimentData]:
        """
        Get sentiment data from multiple sources
        
        Args:
            symbol: Stock symbol
            sources: List of sentiment sources to query
            
        Returns:
            Dictionary mapping source names to sentiment data
        """
        if sources is None:
            sources = [DataSource.REDDIT, DataSource.TWITTER]
        
        sentiment_data = {}
        
        for source in sources:
            if source not in self.adapters:
                continue
            
            adapter = self.adapters[source]
            if hasattr(adapter, 'get_sentiment'):
                data = adapter.get_sentiment(symbol)
                if data:
                    sentiment_data[source.value] = data
        
        return sentiment_data
    
    def get_advanced_sentiment_data(self, symbol: str, texts: Optional[List[str]] = None, sources: Optional[List[str]] = None) -> Optional[SentimentData]:
        """
        Get advanced multi-model sentiment analysis
        """
        if texts:
            return self.advanced_sentiment_adapter.get_sentiment(symbol, texts, sources if sources is not None else None)
        else:
            reddit_sentiment = self.get_sentiment_data(symbol)
            if reddit_sentiment and reddit_sentiment.get(DataSource.REDDIT.value):
                reddit_data = reddit_sentiment[DataSource.REDDIT.value]
                raw = reddit_data.raw_data or {}
                if isinstance(raw, dict) and 'sample_texts' in raw:
                    sample_texts = raw['sample_texts']
                    return self.advanced_sentiment_adapter.get_sentiment(symbol, sample_texts, None)
            return None
    
    def get_data_quality_report(self) -> Dict[str, DataQualityMetrics]:
        """Get comprehensive data quality report for all sources"""
        rankings = self.performance_tracker.get_source_ranking()
        
        quality_report = {}
        for source_name, score in rankings:
            metrics = self.performance_tracker.metrics[source_name]
            
            quality_report[source_name] = DataQualityMetrics(
                source=source_name,
                quality_score=score,
                latency_ms=float(np.mean(metrics['response_times'])) if metrics['response_times'] else 0,
                success_rate=metrics['success_count'] / max(1, metrics['success_count'] + metrics['error_count']),
                last_updated=metrics['last_success'] or datetime.now(),
                issues=[]  # TODO: Implement issue tracking
            )
        
        return quality_report
    
    def validate_system_health(self) -> Dict[str, Any]:
        """Validate overall system health and data quality"""
        health_report = {
            'status': 'healthy',
            'timestamp': datetime.now(),
            'sources_available': len(self.adapters),
            'cache_size': len(self.cache.cache),
            'quality_issues': []
        }
        
        # Check data source health
        quality_report = self.get_data_quality_report()
        unhealthy_sources = [source for source, metrics in quality_report.items() 
                           if metrics.quality_score < self.min_quality_score]
        
        if unhealthy_sources:
            health_report['status'] = 'degraded'
            health_report['quality_issues'].extend(unhealthy_sources)
        
        return health_report
    
    def get_market_breadth(self) -> Optional[MarketBreadth]:
        """Get market breadth, cached and validated, from FinViz."""
        cache_key = "market_breadth_finviz"
        cached = self.cache.get(cache_key, "market_breadth")
        if cached:
            return cached
        adapter = self.adapters.get(DataSource.FINVIZ)
        if not adapter:
            return None
        start_time = time.time()
        try:
            breadth = adapter.get_market_breadth()
            if not breadth:
                return None
            issues = []
            if breadth.advancers is None or breadth.advancers < 0:
                issues.append("Invalid advancers")
            if breadth.decliners is None or breadth.decliners < 0:
                issues.append("Invalid decliners")
            quality_score = 50.0 if issues else 90.0
            self.performance_tracker.record_success(DataSource.FINVIZ.value, (time.time()-start_time)*1000, quality_score)
            if quality_score >= self.min_quality_score:
                self.cache.set(cache_key, breadth, "market_breadth", quality_score)
            return breadth
        except Exception as e:
            self.performance_tracker.record_error(DataSource.FINVIZ.value, str(e))
            logger.error(f"FinViz market breadth error: {e}")
            return None

    def get_sector_performance(self) -> list[GroupPerformance]:
        """Get sector performance list from FinViz. May be empty if not implemented."""
        cache_key = "group_performance_finviz_sector"
        cached = self.cache.get(cache_key, "group_performance")
        if cached:
            return cached
        adapter = self.adapters.get(DataSource.FINVIZ)
        if not adapter:
            return []
        start_time = time.time()
        try:
            groups = adapter.get_sector_performance() or []
            def in_range(val: Optional[Decimal]) -> bool:
                return val is None or (Decimal("-100") <= val <= Decimal("100"))
            valid = all(
                g.group_type == "sector" and
                all(in_range(val) for val in [g.perf_1d, g.perf_1w, g.perf_1m, g.perf_3m, g.perf_6m, g.perf_1y, g.perf_ytd])
                for g in groups
            )
            quality_score = 85.0 if valid else 55.0
            self.performance_tracker.record_success(DataSource.FINVIZ.value, (time.time()-start_time)*1000, quality_score)
            if quality_score >= self.min_quality_score:
                self.cache.set(cache_key, groups, "group_performance", quality_score)
            return groups
        except Exception as e:
            self.performance_tracker.record_error(DataSource.FINVIZ.value, str(e))
            logger.error(f"FinViz sector performance error: {e}")
            return []

# ============================================================================
# Unified Interface Functions (Backward Compatibility)
# ============================================================================

# Global orchestrator instance
_global_orchestrator = None

def get_orchestrator() -> DataFeedOrchestrator:
    """Get the global data feed orchestrator instance"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = DataFeedOrchestrator()
    return _global_orchestrator

def get_quote(symbol: str) -> Optional[Quote]:
    """Get real-time quote (unified interface)"""
    return get_orchestrator().get_quote(symbol)

def get_market_data(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[MarketData]:
    """Get historical market data (unified interface)"""
    return get_orchestrator().get_market_data(symbol, period, interval)

def get_sentiment_data(symbol: str) -> Dict[str, SentimentData]:
    """Get sentiment data (unified interface)"""
    return get_orchestrator().get_sentiment_data(symbol)

def get_advanced_sentiment(symbol: str, texts: Optional[List[str]] = None, sources: Optional[List[str]] = None) -> Optional[SentimentData]:
    """Get advanced multi-model sentiment analysis for a symbol"""
    return get_orchestrator().get_advanced_sentiment_data(symbol, texts, sources)

def get_system_health() -> Dict[str, Any]:
    """Get system health report (unified interface)"""
    return get_orchestrator().validate_system_health()

def get_market_breadth() -> Optional[MarketBreadth]:
    """Get market breadth (unified interface)"""
    return get_orchestrator().get_market_breadth()

def get_sector_performance() -> List[GroupPerformance]:
    """Get sector performance (unified interface)"""
    return get_orchestrator().get_sector_performance()
