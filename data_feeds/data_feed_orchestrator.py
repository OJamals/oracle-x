"""
DataFeedOrchestrator - Unified Financial Data Interface
Consolidates multiple data sources with quality validation, intelligent fallback, and performance tracking.
Replaces all existing data feed implementations with a single authoritative interface.
"""

import asyncio
import json
import logging
import os
import time
import warnings
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Async I/O utilities import with fallback
AsyncHTTPClient = None
ASYNC_IO_AVAILABLE = False
try:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.async_io_utils import AsyncHTTPClient

    ASYNC_IO_AVAILABLE = True
except ImportError:
    pass

# Optimized HTTP client import with fallback
try:
    from core.http_client import optimized_get
except ImportError:

    def optimized_get(url, **kwargs):
        """Fallback to standard requests if optimized client unavailable"""
        return requests.get(url, **kwargs)


from dotenv import load_dotenv

from data_feeds.cache.redis_cache_manager import (  # Redis cache manager
    RedisCacheManager,
    get_redis_cache_manager,
)

# Delegate models and feed for consolidation parity
from data_feeds.consolidated_data_feed import (  # absolute imports as required (avoid Quote name clash)
    CompanyInfo,
    ConsolidatedDataFeed,
    NewsItem,
)
from data_feeds.data_types import (
    DataQuality,  # Shared dataclasses
    DataQualityMetrics,
    DataSource,
    MarketData,
    Quote,
    SentimentData,
)
from data_feeds.fallback_manager import (  # New import for intelligent fallback
    FallbackConfig,
    FallbackManager,
    FallbackReason,
)
from data_feeds.models import GroupPerformance, MarketBreadth  # Added import
from data_feeds.sources.finviz_adapter import FinVizAdapter  # New import
from data_feeds.sources.news_adapter import NewsAdapter  # Unified news adapter
from data_feeds.twitter_feed import TwitterSentimentFeed  # New import

# Optional cache services; guard imports so missing dependencies (e.g., schedule) do not break orchestrator
try:
    from data_feeds.cache.cache_invalidation_service import (
        get_cache_invalidation_service,
    )  # type: ignore
except Exception:
    get_cache_invalidation_service = None  # type: ignore

try:
    from data_feeds.cache.cache_warming_service import (
        get_cache_warming_service,
    )  # type: ignore
except Exception:
    get_cache_warming_service = None  # type: ignore
from data_feeds.cache.cache_service import CacheService  # SQLite-backed cache

# Optional Investiny compact formatter
try:
    from data_feeds.investiny_adapter import (
        format_daily_oc as investiny_format_daily_oc,
    )
except Exception:
    investiny_format_daily_oc = None  # optional
# Standardized adapter protocol wrappers (additive; not yet used for routing)
try:
    from data_feeds.adapter_protocol import SourceAdapterProtocol  # type: ignore
    from data_feeds.adapter_wrappers import (  # type: ignore
        FinanceDatabaseAdapterWrapper,
        FinnhubAdapterWrapper,
        FMPAdapterWrapper,
        YFinanceAdapterWrapper,
    )
except Exception:
    # Guarded import to avoid any breaking behavior if wrappers are unavailable
    SourceAdapterProtocol = None  # type: ignore
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.ERROR)

from data_feeds.orchestrator.validation.data_validator import DataValidator
from data_feeds.orchestrator.utils.performance_tracker import PerformanceTracker
from data_feeds.orchestrator.utils.helpers import (
    _to_decimal,
    _parse_datetime,
    _batch_to_decimal,
    _safe_float,
    _safe_int,
    _log_error_and_record,
)
from sentiment.sentiment_engine import get_sentiment_engine

logger = logging.getLogger(__name__)

# Reduce noisy third-party warnings and noisy streamlit messages during batch/bare runs.
# Many ML libs (torch/transformers) surface FutureWarning for deprecated args like
# `encoder_attention_mask`. We suppress those here to keep logs actionable.
try:
    # Quiet streamlit runtime warnings that appear when running outside streamlit server
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    try:
        logging.getLogger(
            "streamlit.runtime.scriptrunner_utils.script_run_context"
        ).setLevel(logging.ERROR)
    except Exception:
        pass
except Exception:
    pass

# Suppress specific FutureWarning about encoder_attention_mask and general FutureWarnings
warnings.filterwarnings(
    "ignore", message=r".*encoder_attention_mask.*", category=FutureWarning
)
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress Streamlit 'missing ScriptRunContext' warnings when running in bare mode
warnings.filterwarnings("ignore", message=r".*missing ScriptRunContext.*")

try:
    # If transformers is installed, reduce its verbosity as well
    import transformers

    try:
        # transformers exposes logging in different places across versions; use getattr to appease static analyzers
        t_logging = getattr(transformers, "logging", None)
        if (
            t_logging
            and hasattr(t_logging, "set_verbosity_error")
            and callable(getattr(t_logging, "set_verbosity_error", None))
        ):
            t_logging.set_verbosity_error()
        else:
            t_utils = getattr(transformers, "utils", None)
            if t_utils:
                t_utils_logging = getattr(t_utils, "logging", None)
                if (
                    t_utils_logging
                    and hasattr(t_utils_logging, "set_verbosity_error")
                    and callable(getattr(t_utils_logging, "set_verbosity_error", None))
                ):
                    t_utils_logging.set_verbosity_error()
    except Exception:
        pass
except Exception:
    # transformers not installed or import failed; ignore
    pass
# Also quietly reduce torch logging if available
try:
    import torch

    try:
        logging.getLogger("torch").setLevel(logging.ERROR)
    except Exception:
        pass
except Exception:
    pass
# ---------------------------------------------------------------------------
# Optimized utility normalization helpers (shared)
# ---------------------------------------------------------------------------
# Cached datetime formats for faster parsing
_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
)


def _to_decimal(val: Any) -> Optional[Decimal]:
    """Optimized decimal conversion with early exits"""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val
    if isinstance(val, (int, float)):
        try:
            return Decimal(val)
        except Exception:
            return None
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() in ("none", "null", "nan", ""):
            return None
        try:
            return Decimal(val)
        except Exception:
            return None
    return None


def _parse_datetime(val: Any) -> Optional[datetime]:
    """Optimized datetime parsing with cached formats"""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(float(val))
        except (ValueError, OSError):
            return None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        # Try cached formats in order of frequency
        for fmt in _DATETIME_FORMATS:
            try:
                return datetime.strptime(val, fmt)
            except ValueError:
                continue
    return None


def _batch_to_decimal(values: List[Any]) -> List[Optional[Decimal]]:
    """Vectorized decimal conversion for batch processing"""
    return [_to_decimal(val) for val in values]


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Fast float conversion with fallback"""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() in ("none", "null", "nan", ""):
            return default
        try:
            return float(val)
        except ValueError:
            return default
    return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Fast integer conversion with fallback"""
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() in ("none", "null", "nan", ""):
            return default
        try:
            return int(float(val))  # Handle strings like "123.0"
        except ValueError:
            return default
    return default


def _log_error_and_record(
    perf: "PerformanceTracker", source: str, msg: str, exc: Exception
):  # Forward ref to PerformanceTracker
    emsg = f"{msg}: {exc}"
    logger.error(emsg)
    try:
        perf.record_error(source, emsg)
    except Exception:
        pass


# ============================================================================
# Core Data Models
# ============================================================================

# ============================================================================
# Data Quality Framework
# ============================================================================


class DataValidator:
    """Optimized data validation and anomaly detection"""

    @staticmethod
    def validate_quote(quote: Quote) -> Tuple[float, List[str]]:
        """Fast quote validation with early exits"""
        issues = []
        score = 100.0

        # Price validation - most critical
        if not quote.price or quote.price <= 0:
            issues.append("Missing or invalid price")
            score -= 40
            return max(0, score), issues  # Early exit for critical failure

        # Volume validation
        if not quote.volume or quote.volume < 0:
            issues.append("Missing or invalid volume")
            score -= 20

        # Timestamp validation - optimized check
        if quote.timestamp:
            age_seconds = (datetime.now() - quote.timestamp).total_seconds()
            if age_seconds > 3600:  # 1 hour
                issues.append("Stale quote timestamp")
                score -= 20
        else:
            issues.append("Missing timestamp")
            score -= 10

        # Range validation - only if all values present
        if (
            quote.day_low is not None
            and quote.day_high is not None
            and quote.price is not None
        ):
            if not (quote.day_low <= quote.price <= quote.day_high):
                issues.append("Price outside day range")
                score -= 10

        return max(0, min(score, 100)), issues

    @staticmethod
    def validate_market_data(data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Optimized market data validation using pandas vectorization"""
        issues = []
        score = 100.0

        if data.empty:
            return 0.0, ["Empty DataFrame"]

        # Fast missing data calculation
        total_cells = data.size
        if total_cells > 0:
            missing_cells = data.isnull().sum().sum()
            missing_pct = (missing_cells / total_cells) * 100
            if missing_pct > 5:
                issues.append(f"Missing data: {missing_pct:.2f}%")
                score -= 20

        # Vectorized outlier detection for numeric columns
        numeric_cols = data.select_dtypes(include=[float, int]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                vals = data[col].dropna()
                if len(vals) > 2:
                    # Fast vectorized z-score calculation
                    mean_val = vals.mean()
                    std_val = vals.std()
                    if std_val > 0:
                        z_scores = ((vals - mean_val) / std_val).abs()
                        if (z_scores > 5).any():
                            issues.append(f"Outlier detected in {col}")
                            score -= 10
                            break  # Don't check all columns if one has outliers

        return max(0, min(score, 100)), issues

    @staticmethod
    def detect_anomalies(data: pd.Series, threshold: float = 3.0) -> List[int]:
        """Vectorized anomaly detection"""
        if data.empty:
            return []

        vals = data.dropna()
        if len(vals) < 2:
            return []

        # Vectorized z-score calculation
        mean_val = vals.mean()
        std_val = vals.std()

        if std_val == 0:
            return []

        z_scores = ((vals - mean_val) / std_val).abs()
        anomalies = vals[z_scores > threshold]

        return anomalies.index.tolist()


class PerformanceTracker:
    """Tracks data source performance and reliability with issue registry."""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(self._create_metrics_dict)
        self.issues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

    @staticmethod
    def _create_metrics_dict():
        return {
            "response_times": deque(maxlen=100),
            "success_count": 0,
            "error_count": 0,
            "quality_scores": deque(maxlen=100),
            "last_success": None,
            "last_error": None,
            "issues": deque(maxlen=50),
        }

    def record_success(
        self,
        source: str,
        response_time: float,
        quality_score: float,
        issues: Optional[List[str]] = None,
    ):
        metrics = self.metrics[source]
        metrics["response_times"].append(response_time)
        metrics["success_count"] += 1
        metrics["quality_scores"].append(quality_score)
        metrics["last_success"] = datetime.now()
        if issues:
            for issue in issues:
                self.record_issue(source, issue)

    def record_error(self, source: str, error: str):
        metrics = self.metrics[source]
        metrics["error_count"] += 1
        metrics["last_error"] = datetime.now()
        logger.warning(f"Data source {source} error: {error}")
        self.record_issue(source, f"ERROR: {error}")

    def record_issue(self, source: str, issue: str):
        try:
            self.metrics[source]["issues"].append(issue)
            self.issues[source].append(issue)
        except Exception:
            pass

    def get_source_ranking(self) -> List[Tuple[str, float]]:
        rankings: List[Tuple[str, float]] = []
        for source, metrics in self.metrics.items():
            avg_quality = (
                (sum(metrics["quality_scores"]) / len(metrics["quality_scores"]))
                if metrics["quality_scores"]
                else 0.0
            )
            rankings.append((source, avg_quality))
        return sorted(rankings, key=lambda x: x[1], reverse=True)


# ============================================================================
# Enhanced Caching System
# ============================================================================


class SmartCache:
    """Intelligent caching with TTL and quality-based retention"""

    def __init__(self, ttl_settings: Optional[Dict[str, int]] = None):
        self.cache = {}
        # Defaults preserve existing behavior
        self.ttl_settings = {
            "quote": 30,  # 30 seconds for quotes
            "market_data_1d": 3600,  # 1 hour for daily data
            "market_data_1h": 300,  # 5 minutes for hourly data
            "sentiment": 600,  # 10 minutes for sentiment
            "news": 1800,  # 30 minutes for news
            "company_info": 86400,  # 24 hours for company info
            # Safe defaults for new data types
            "market_breadth": 300,  # 5 minutes
            "group_performance": 1800,  # 30 minutes
            # Newly added FinViz datasets
            "insider_trading": 900,  # 15 minutes
            "earnings": 900,  # 15 minutes
            "forex": 600,  # 10 minutes
            "crypto": 600,  # 10 minutes
        }
        # Overlay provided settings if any
        if isinstance(ttl_settings, dict) and ttl_settings:
            try:
                for k, v in ttl_settings.items():
                    if isinstance(v, int):
                        self.ttl_settings[k] = v
            except Exception:
                # best-effort; keep defaults on any error
                pass

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
            data_type = key.split("_")[0] if "_" in key else "default"
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

    def __init__(
        self,
        limits_config: Optional[Dict[str, Dict[str, int]]] = None,
        quotas_config: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.calls = defaultdict(deque)
        # Defaults as before
        self.limits = {
            DataSource.FINNHUB: (60, 60),  # 60 calls per minute
            DataSource.IEX_CLOUD: (100, 60),  # 100 calls per minute (free tier)
            DataSource.REDDIT: (60, 60),  # 60 calls per minute
            # DataSource.TWITTER: No rate limits - twscrape is rate-limit free
            DataSource.FRED: (120, 60),  # 120 calls per minute
            DataSource.FINVIZ: (12, 60),  # 12 req per 60s
        }
        self.daily_quotas = {
            DataSource.FINNHUB: 1000,  # Daily free quota
            DataSource.IEX_CLOUD: 50000,  # Daily free quota
        }

        # Apply optional config overlays without breaking existing behavior
        def _ds_from_key(key: str) -> Optional[DataSource]:
            try:
                # accept both enum name and value forms
                for ds in DataSource:
                    if key.upper() == ds.name or key.lower() == ds.value:
                        return ds
            except Exception:
                return None
            return None

        if isinstance(limits_config, dict):
            for k, v in limits_config.items():
                ds = _ds_from_key(k)
                if not ds or not isinstance(v, dict):
                    continue
                # Prefer per_minute; fallback to per_15min
                if (
                    "per_minute" in v
                    and isinstance(v["per_minute"], int)
                    and v["per_minute"] > 0
                ):
                    self.limits[ds] = (int(v["per_minute"]), 60)
                elif (
                    "per_15min" in v
                    and isinstance(v["per_15min"], int)
                    and v["per_15min"] > 0
                ):
                    self.limits[ds] = (int(v["per_15min"]), 900)
        if isinstance(quotas_config, dict):
            for k, v in quotas_config.items():
                ds = _ds_from_key(k)
                if not ds or not isinstance(v, dict):
                    continue
                if (
                    "per_day" in v
                    and isinstance(v["per_day"], int)
                    and v["per_day"] > 0
                ):
                    self.daily_quotas[ds] = int(v["per_day"])
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
                    logger.info(
                        f"Rate limiting {source.value}: waiting {wait_time:.1f} seconds"
                    )
                    time.sleep(wait_time)

        # Record this call
        self.calls[source].append(time.time())
        self.daily_usage[source] += 1
        return True


# ============================================================================
# Data Source Adapters
# ============================================================================


class YFinanceAdapter:
    """Enhanced yfinance adapter with performance optimizations"""

    def __init__(
        self,
        cache: SmartCache,
        rate_limiter: RateLimiter,
        performance_tracker: PerformanceTracker,
    ):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker
        self.source = DataSource.YFINANCE
        # Cache ticker objects to avoid re-initialization
        self._ticker_cache: Dict[str, Any] = {}
        self._ticker_cache_time: Dict[str, float] = {}
        self._ticker_ttl = 300  # 5 minutes

    def _get_ticker(self, symbol: str):
        """Get cached ticker object or create new one"""
        current_time = time.time()

        if (
            symbol in self._ticker_cache
            and symbol in self._ticker_cache_time
            and current_time - self._ticker_cache_time[symbol] < self._ticker_ttl
        ):
            return self._ticker_cache[symbol]

        # Create new ticker and cache it
        ticker = yf.Ticker(symbol)
        self._ticker_cache[symbol] = ticker
        self._ticker_cache_time[symbol] = current_time

        # Cleanup old entries
        if len(self._ticker_cache) > 50:  # Limit cache size
            oldest_symbol = min(
                self._ticker_cache_time.keys(), key=lambda s: self._ticker_cache_time[s]
            )
            del self._ticker_cache[oldest_symbol]
            del self._ticker_cache_time[oldest_symbol]

        return ticker

    def get_quote(self, symbol: str) -> Optional[Quote]:
        logger.debug(f"YFinanceAdapter.get_quote called for symbol={symbol}")
        """Get real-time quote with optimized validation"""
        cache_key = f"quote_{symbol}"
        cached_data = self.cache.get(cache_key, "quote")
        if cached_data:
            return cached_data

        start_time = time.time()
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info

            if not info or "currentPrice" not in info:
                return None

            # Optimized field extraction with safe defaults
            current_price = info.get("currentPrice")
            if not current_price:
                return None

            # Fast market cap processing
            mc_val = info.get("marketCap")
            market_cap = _safe_int(mc_val) if mc_val not in (None, "", "None") else None

            quote = Quote(
                symbol=symbol,
                price=_to_decimal(current_price),
                change=_to_decimal(info.get("change", 0)),
                change_percent=_to_decimal(info.get("changePercent", 0)),
                volume=_safe_int(info.get("volume")),
                market_cap=market_cap,
                pe_ratio=_to_decimal(info.get("trailingPE")),
                day_low=_to_decimal(info.get("dayLow")),
                day_high=_to_decimal(info.get("dayHigh")),
                year_low=_to_decimal(info.get("fiftyTwoWeekLow")),
                year_high=_to_decimal(info.get("fiftyTwoWeekHigh")),
                timestamp=_parse_datetime(info.get("regularMarketTime"))
                or datetime.now(),
                source=self.source.value,
            )

            # Fast validation
            quality_score, issues = DataValidator.validate_quote(quote)
            quote.quality_score = quality_score

            # Track performance
            response_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_success(
                self.source.value, response_time, quality_score, issues
            )

            # Cache if quality is acceptable
            if quality_score >= 60:
                self.cache.set(cache_key, quote, "quote", quality_score)

            return quote

        except Exception as e:
            _log_error_and_record(
                self.performance_tracker,
                self.source.value,
                f"YFinance quote error for {symbol}",
                e,
            )
            return None

    def get_market_data(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> Optional[MarketData]:
        logger.debug(
            f"YFinanceAdapter.get_market_data called for symbol={symbol}, period={period}, interval={interval}"
        )
        """Get historical market data with optimized processing"""
        cache_key = f"market_data_{symbol}_{period}_{interval}"
        cached_data = self.cache.get(cache_key, f"market_data_{interval}")
        if cached_data:
            return cached_data

        start_time = time.time()
        try:
            ticker = self._get_ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                return None

            # Fast validation
            quality_score, issues = DataValidator.validate_market_data(data)

            market_data = MarketData(
                symbol=symbol,
                data=data,
                timeframe=f"{period}_{interval}",
                source=self.source.value,
                timestamp=datetime.now(),
                quality_score=quality_score,
            )

            # Track performance
            response_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_success(
                self.source.value, response_time, quality_score, issues
            )

            # Cache if quality is acceptable
            if quality_score >= 60:
                self.cache.set(
                    cache_key, market_data, f"market_data_{interval}", quality_score
                )

            return market_data

        except Exception as e:
            _log_error_and_record(
                self.performance_tracker,
                self.source.value,
                f"YFinance market data error for {symbol}",
                e,
            )
            return None


class RedditAdapter:
    """Reddit sentiment data adapter"""

    def __init__(
        self,
        cache: SmartCache,
        rate_limiter: RateLimiter,
        performance_tracker: PerformanceTracker,
    ):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker
        self.source = DataSource.REDDIT
        self._all_sentiment_cache = None
        self._cache_timestamp = None
        self._cache_duration = 300  # 5 minutes cache for all sentiment data

    def get_sentiment(
        self, symbol: str, subreddits: Optional[List[str]] = None
    ) -> Optional[SentimentData]:
        logger.debug(
            f"RedditAdapter.get_sentiment called for symbol={symbol}, subreddits={subreddits}"
        )
        """Get Reddit sentiment data with intelligent caching"""
        if subreddits is None:
            subreddits = ["stocks", "investing", "SecurityAnalysis", "ValueInvesting"]

        cache_key = f"sentiment_reddit_{symbol}"
        cached_data = self.cache.get(cache_key, "sentiment")
        if cached_data:
            return cached_data

        # Check if we need to refresh the all-sentiment cache
        current_time = time.time()
        debug_bypass = os.environ.get("DEBUG_REDDIT", "0") == "1"
        if (
            self._all_sentiment_cache is None
            or self._cache_timestamp is None
            or current_time - self._cache_timestamp > self._cache_duration
            or debug_bypass
        ):
            if not self.rate_limiter.wait_if_needed(self.source):
                return None

            try:
                # Import Reddit sentiment function
                from data_feeds.reddit_sentiment import fetch_reddit_sentiment

                start_time = time.time()
                # Increase coverage a bit; limit is capped internally at 50
                self._all_sentiment_cache = fetch_reddit_sentiment(limit=100)
                self._cache_timestamp = current_time
                if os.environ.get("DEBUG_REDDIT", "0") == "1":
                    keys = (
                        len(self._all_sentiment_cache)
                        if isinstance(self._all_sentiment_cache, dict)
                        else "N/A"
                    )
                    logger.error(
                        f"[DEBUG][orchestrator] reddit batch fetched keys={keys}"
                    )

                # Track the batch performance
                response_time = (time.time() - start_time) * 1000
                quality_score = 70 if self._all_sentiment_cache else 0
                self.performance_tracker.record_success(
                    self.source.value, response_time, quality_score
                )

            except Exception as e:
                _log_error_and_record(
                    self.performance_tracker,
                    self.source.value,
                    "Reddit sentiment batch error",
                    e,
                )
                return None

        # Now extract specific symbol data from cached results
        if not self._all_sentiment_cache:
            return None
        # Normalize symbol casing to match keys from reddit_sentiment (tickers are uppercase)
        symbol_key = symbol.upper()
        if not (
            isinstance(self._all_sentiment_cache, dict)
            and (
                symbol_key in self._all_sentiment_cache
                or symbol in self._all_sentiment_cache
            )
        ):
            if os.environ.get("DEBUG_REDDIT", "0") == "1" and isinstance(
                self._all_sentiment_cache, dict
            ):
                logger.error(
                    f"[DEBUG][orchestrator] reddit cache has {len(self._all_sentiment_cache)} keys but missing {symbol_key}"
                )
            return None

        start_time = time.time()
        try:
            # Safely pick symbol data and ensure it's a dict
            symbol_data = None
            if isinstance(self._all_sentiment_cache, dict):
                if symbol_key in self._all_sentiment_cache:
                    symbol_data = self._all_sentiment_cache[symbol_key]
                elif symbol in self._all_sentiment_cache:
                    symbol_data = self._all_sentiment_cache[symbol]
            if not isinstance(symbol_data, dict):
                return None

            # Check if we have enhanced sentiment data
            enhanced = (
                symbol_data.get("enhanced_sentiment")
                if isinstance(symbol_data, dict)
                else None
            )
            if isinstance(enhanced, dict):
                sentiment_data = SentimentData(
                    symbol=symbol,
                    sentiment_score=enhanced.get("ensemble_score", 0.0),
                    confidence=enhanced.get("confidence", 0.0),
                    source=self.source.value,
                    timestamp=datetime.now(),
                    sample_size=enhanced.get("sample_size", 0),
                    raw_data=enhanced,
                )
                quality_score = enhanced.get("quality_score", 0)
            else:
                # Fallback to basic sentiment data
                sentiment_score = (
                    float(symbol_data.get("sentiment_score", 0.0))
                    if isinstance(symbol_data, dict)
                    else 0.0
                )
                confidence = (
                    float(symbol_data.get("confidence", 0.5))
                    if isinstance(symbol_data, dict)
                    else 0.5
                )
                sample_size = (
                    int(symbol_data.get("sample_size", 0))
                    if isinstance(symbol_data, dict)
                    else 0
                )

                # Include sample texts in raw_data for ML engine
                raw_data = dict(symbol_data) if isinstance(symbol_data, dict) else {}
                if isinstance(symbol_data, dict) and "sample_texts" in symbol_data:
                    # Truncate sample texts to reduce token consumption
                    sample_texts = symbol_data.get("sample_texts") or []
                    truncated_texts = [
                        (
                            text[:150] + "..."
                            if isinstance(text, str) and len(text) > 150
                            else text
                        )
                        for text in sample_texts[:3]
                    ]
                    raw_data["sample_texts"] = truncated_texts

                sentiment_data = SentimentData(
                    symbol=symbol,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    source=self.source.value,
                    timestamp=datetime.now(),
                    sample_size=sample_size,
                    raw_data=raw_data,
                )
                quality_score = min(100, (sample_size * 2) + (confidence * 50))

            # Track performance for individual symbol extraction
            response_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_success(
                self.source.value, response_time, quality_score
            )

            # Cache the individual sentiment data
            self.cache.set(cache_key, sentiment_data, "sentiment", quality_score)

            return sentiment_data

        except Exception as e:
            _log_error_and_record(
                self.performance_tracker,
                self.source.value,
                f"Reddit sentiment error for {symbol}",
                e,
            )
            return None


class FinVizNewsSentimentAdapter:
    """Generate sentiment from FinViz aggregated news DataFrame (if available)."""

    def __init__(
        self,
        cache: SmartCache,
        rate_limiter: RateLimiter,
        performance_tracker: PerformanceTracker,
    ):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker
        self.source = DataSource.FINVIZ  # reuse source identifier
        self.adapters_ref = None  # will be set by orchestrator post-initialization

    def get_sentiment(self, symbol: str, limit: int = 60) -> Optional[SentimentData]:
        cache_key = f"finviz_news_sentiment_{symbol.upper()}"
        cached = self.cache.get(cache_key, "sentiment")
        if cached:
            return cached
        if not self.rate_limiter.wait_if_needed(self.source):
            return None
        start_time = time.time()
        try:
            finviz_adapter = None
            try:
                if isinstance(self.adapters_ref, dict):
                    finviz_adapter = self.adapters_ref.get(DataSource.FINVIZ)
            except Exception:
                pass
            if finviz_adapter is not None and hasattr(finviz_adapter, "get_news"):
                data = finviz_adapter.get_news()
            else:
                data = None
            if data is None or not isinstance(data, dict):
                return None
            import pandas as pd  # noqa: F401

            headlines_df = data.get("news") or data.get("headlines")
            if headlines_df is None:
                return None
            if not isinstance(headlines_df, pd.DataFrame):
                return None
            if headlines_df.empty:
                return None
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            sym = symbol.upper()
            title_col = None
            for cand in ["Title", "title", "Headline", "headline"]:
                if cand in headlines_df.columns:
                    title_col = cand
                    break
            if not title_col:
                return None
            titles: List[str] = []
            col_values = list(headlines_df[title_col].tolist())  # type: ignore[arg-type]
            for val in col_values:
                if isinstance(val, str) and val.strip():
                    up = val.upper()
                    if f" {sym} " in f" {up} " or f"${sym}" in up:
                        titles.append(val.strip())
            if not titles:
                titles = [val for val in col_values if isinstance(val, str)][:limit]
            scores = []
            indiv = []
            for t in titles[:limit]:
                try:
                    sc = analyzer.polarity_scores(t).get("compound", 0.0)
                    scores.append(sc)
                    indiv.append({"headline": t[:180], "compound": sc})
                except Exception:
                    continue
            if not scores:
                return None
            avg = float(sum(scores) / len(scores))
            var = sum((s - avg) ** 2 for s in scores) / len(scores) if scores else 0.0
            size_factor = min(1.0, len(scores) / 25.0)
            confidence = max(0.1, min(0.9, size_factor * (1.0 - min(0.6, var))))
            quality = (confidence * 0.7 + size_factor * 0.3) * 100
            sd = SentimentData(
                symbol=symbol,
                sentiment_score=avg,
                confidence=confidence,
                source="finviz_news",
                timestamp=datetime.now(),
                sample_size=len(scores),
                raw_data={
                    "sample_texts": [t[:160] for t in titles[:5]],
                    "individual_headline_scores": indiv[:10],
                    "variance": var,
                },
            )
            self.performance_tracker.record_success(
                self.source.value, (time.time() - start_time) * 1000, quality
            )
            self.cache.set(cache_key, sd, "sentiment", quality)
            return sd
        except Exception as e:
            _log_error_and_record(
                self.performance_tracker,
                self.source.value,
                f"FinVizNewsSentimentAdapter failed for {symbol}",
                e,
            )
            return None


class GenericRSSSentimentAdapter:
    """Configurable RSS feed sentiment adapter.

    Environment variables:
      RSS_FEEDS       comma-separated list of feed URLs
      RSS_INCLUDE_ALL if set (1/true/on) include all headlines, not only those mentioning the symbol
    """

    feed_urls: List[str]
    include_all: bool
    feedparser_available: bool

    def __init__(
        self,
        cache: SmartCache,
        rate_limiter: RateLimiter,
        performance_tracker: PerformanceTracker,
    ):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker
        self.source = DataSource.GOOGLE_TRENDS  # placeholder enum for registry
        self.feed_urls = [
            u.strip() for u in os.getenv("RSS_FEEDS", "").split(",") if u.strip()
        ]
        self.include_all = os.getenv("RSS_INCLUDE_ALL", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            import feedparser  # type: ignore  # noqa: F401

            self.feedparser_available = True
        except Exception:
            self.feedparser_available = False

    def get_sentiment(self, symbol: str, limit: int = 80) -> Optional[SentimentData]:
        if not self.feed_urls or not self.feedparser_available:
            return None
        cache_key = f"rss_sentiment_{symbol.upper()}"
        cached = self.cache.get(cache_key, "sentiment")
        if cached:
            return cached
        if not self.rate_limiter.wait_if_needed(self.source):
            return None
        start = time.time()
        try:
            import feedparser  # type: ignore
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            sym = symbol.upper()
            headlines: List[str] = []
            for url in self.feed_urls:
                try:
                    d = feedparser.parse(url)
                    entries = d.get("entries") or []  # type: ignore[index]
                    for entry in entries[:50]:  # type: ignore[index]
                        try:
                            title = (
                                entry.get("title")
                                if isinstance(entry, dict)
                                else getattr(entry, "title", None)
                            )
                        except Exception:
                            title = None
                        if isinstance(title, str) and title.strip():
                            up = title.upper()
                            if self.include_all or sym in up or f"${sym}" in up:
                                headlines.append(title.strip())
                except Exception:
                    continue
            if not headlines:
                return None
            scores: List[float] = []
            for h in headlines[:limit]:
                try:
                    scores.append(analyzer.polarity_scores(h).get("compound", 0.0))
                except Exception:
                    continue
            if not scores:
                return None
            avg = float(sum(scores) / len(scores))
            size_factor = min(1.0, len(scores) / 30.0)
            variance = (
                sum((s - avg) ** 2 for s in scores) / len(scores) if scores else 0.0
            )
            confidence = max(0.1, min(0.9, size_factor * (1.0 - min(0.6, variance))))
            quality = (confidence * 0.65 + size_factor * 0.35) * 100
            sd = SentimentData(
                symbol=symbol,
                sentiment_score=avg,
                confidence=confidence,
                source="rss_news",
                timestamp=datetime.now(),
                sample_size=len(scores),
                raw_data={"variance": variance, "sample_texts": headlines[:5]},
            )
            self.performance_tracker.record_success(
                "rss_news", (time.time() - start) * 1000, quality
            )
            self.cache.set(cache_key, sd, "sentiment", quality)
            return sd
        except Exception as e:
            self.performance_tracker.record_error("rss_news", str(e))
            logger.debug(f"RSS sentiment failed for {symbol}: {e}")
            return None


class TwitterAdapter:
    """Twitter sentiment data adapter using TwitterSentimentFeed"""

    def __init__(
        self,
        cache: SmartCache,
        rate_limiter: RateLimiter,
        performance_tracker: PerformanceTracker,
    ):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker
        self.source = DataSource.TWITTER
        self.feed = TwitterSentimentFeed()

    def get_sentiment(self, symbol: str, limit: int = 50) -> Optional[SentimentData]:
        logger.debug(
            f"TwitterAdapter.get_sentiment called for symbol={symbol}, limit={limit}"
        )
        """Get Twitter sentiment data for a symbol"""
        cache_key = f"twitter_sentiment_{symbol}_{limit}"

        # Check cache first (valid for 30 minutes)
        cached_data = self.cache.get(cache_key, "sentiment")
        if cached_data:
            return cached_data

        # Skip rate limiting for Twitter since twscrape has no inherent rate limits
        # if not self.rate_limiter.wait_if_needed(self.source):
        #     logger.warning(f"Rate limit exceeded for Twitter sentiment: {symbol}")
        #     return None

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
                if "sentiment" in tweet and "text" in tweet:
                    # Twitter feed already provides VADER sentiment
                    sentiment_score = tweet["sentiment"].get("compound", 0.0)
                    sentiments.append(sentiment_score)
                    texts.append(tweet["text"])

            if not sentiments:
                logger.warning(f"No sentiment data extracted from Twitter for {symbol}")
                return None

            # Calculate aggregate sentiment
            overall_sentiment = sum(sentiments) / len(sentiments)

            # Calculate confidence based on sample size and variance
            sentiment_variance = (
                sum((s - overall_sentiment) ** 2 for s in sentiments) / len(sentiments)
                if sentiments
                else 0
            )
            confidence = min(0.95, max(0.1, 1.0 - sentiment_variance))

            # Calculate quality score based on sample size and confidence
            sample_weight = min(1.0, len(sentiments) / 20.0)  # Better with more samples
            quality_score = (confidence * 0.7 + sample_weight * 0.3) * 100

            # Record performance
            execution_time = time.time() - start_time
            self.performance_tracker.record_success(
                self.source.value, execution_time, quality_score
            )

            # Create sentiment data object
            sentiment_data = SentimentData(
                symbol=symbol,
                sentiment_score=overall_sentiment,
                confidence=confidence,
                source="twitter_advanced",
                timestamp=datetime.now(),
                sample_size=len(sentiments),
                raw_data={
                    "tweets": tweets[:10],  # Limit tweets to reduce output
                    "sample_texts": [
                        text[:150] + "..." if len(text) > 150 else text
                        for text in texts[:5]
                    ],
                    "individual_sentiments": sentiments[:10],
                    "variance": sentiment_variance,
                    "execution_time": execution_time,
                },
            )

            # Cache the result
            self.cache.set(cache_key, sentiment_data, "sentiment", quality_score)

            # Reduce logging verbosity
            logger.debug(
                f"Twitter sentiment for {symbol}: {overall_sentiment:.3f} "
                f"(confidence: {confidence:.3f}, samples: {len(sentiments)})"
            )

            return sentiment_data

        except Exception as e:
            _log_error_and_record(
                self.performance_tracker,
                self.source.value,
                f"Failed to fetch Twitter sentiment for {symbol}",
                e,
            )
            return None


# ============================================================================
# Main DataFeedOrchestrator Class
# ============================================================================


class DataFeedOrchestrator:
    """
    Unified data feed orchestrator that replaces all existing feed implementations.
    Provides intelligent data source selection, quality validation, and fallback mechanisms.
    """

    def __init__(self, config: Optional[Any] = None):
        # Load optional external configuration (non-breaking if unavailable)
        loaded_config = None
        if config is None:
            try:
                from data_feeds.config_loader import (
                    load_config as _load_config,
                )  # type: ignore

                loaded_config = _load_config()
            except Exception:
                loaded_config = None
        else:
            loaded_config = config

        # Initialize core components with config-aware defaults
        ttl_settings = (
            getattr(loaded_config, "cache_ttls", None) if loaded_config else None
        )
        limits_cfg = (
            getattr(loaded_config, "rate_limits", None) if loaded_config else None
        )
        quotas_cfg = getattr(loaded_config, "quotas", None) if loaded_config else None

        self.cache = SmartCache(
            ttl_settings if isinstance(ttl_settings, dict) else None
        )
        self.rate_limiter = RateLimiter(
            limits_config=limits_cfg if isinstance(limits_cfg, dict) else None,
            quotas_config=quotas_cfg if isinstance(quotas_cfg, dict) else None,
        )
        self.performance_tracker = PerformanceTracker()
        self.validator = DataValidator()  # Add validator instance

        # Initialize intelligent fallback manager
        fallback_config = FallbackConfig()
        self.fallback_manager = FallbackManager(fallback_config)

        self.cache_warming_service = None
        self.cache_invalidation_service = None

        # Initialize persistent cache (SQLite) for long-lived artifacts
        try:
            self.persistent_cache = CacheService(
                db_path=os.getenv("CACHE_DB_PATH", "./model_monitoring.db")
            )
        except Exception as e:
            logger.error(f"Failed to initialize persistent cache: {e}")
            self.persistent_cache = None

        # Initialize Redis cache manager for enhanced performance
        try:
            self.redis_cache = get_redis_cache_manager()
            if self.redis_cache:
                logger.info("Redis cache manager initialized successfully")
            else:
                logger.info(
                    "Redis cache manager not available (Redis not installed or disabled)"
                )
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache manager: {e}")
            self.redis_cache = None

        # Initialize cache warming service (optional dependency)
        if get_cache_warming_service:
            try:
                self.cache_warming_service = get_cache_warming_service(self)
                if self.cache_warming_service:
                    logger.info("Cache warming service initialized successfully")
                else:
                    logger.info("Cache warming service not enabled")
            except Exception as e:
                logger.error(f"Failed to initialize cache warming service: {e}")
                self.cache_warming_service = None
        else:
            logger.info("Cache warming service not available (dependency missing)")

        # Initialize proactive cache warming for frequently accessed data
        self._init_proactive_cache_warming()

        # Initialize logger for this instance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize data source adapters
        news_adapter = NewsAdapter(
            self.cache, self.rate_limiter, self.performance_tracker
        )
        self.adapters = {
            DataSource.YFINANCE: YFinanceAdapter(
                self.cache, self.rate_limiter, self.performance_tracker
            ),
            DataSource.REDDIT: RedditAdapter(
                self.cache, self.rate_limiter, self.performance_tracker
            ),
            DataSource.TWITTER: TwitterAdapter(
                self.cache, self.rate_limiter, self.performance_tracker
            ),
            DataSource.YAHOO_NEWS: news_adapter,
            DataSource.NEWS: news_adapter,
            DataSource.FINVIZ: FinVizAdapter(),
        }
        # Attempt to add FinVizNewsSentimentAdapter using existing FINVIZ headlines if available
        try:
            finviz_news_adapter = FinVizNewsSentimentAdapter(
                self.cache, self.rate_limiter, self.performance_tracker
            )
            # Reuse FINVIZ enum (distinct FINVIZ_NEWS may not exist). Stored under FINVIZ to avoid enum changes.
            self.adapters[DataSource.FINVIZ] = (
                self.adapters.get(DataSource.FINVIZ) or finviz_news_adapter
            )  # keep base adapter if already required
            # Keep news sentiment adapter separately for direct sentiment extraction
            self.adapters[(DataSource.FINVIZ, "news")] = finviz_news_adapter  # type: ignore
        except Exception as _e:
            logger.debug(f"FinVizNewsSentimentAdapter init skipped: {_e}")
        # Attempt to register generic RSS sentiment adapter if feedparser & RSS_FEEDS provided
        try:
            rss_adapter = GenericRSSSentimentAdapter(
                self.cache, self.rate_limiter, self.performance_tracker
            )
            if getattr(rss_adapter, "feed_urls", []) and getattr(
                rss_adapter, "feedparser_available", False
            ):
                self.adapters[("RSS", "rss_news")] = rss_adapter  # type: ignore
        except Exception as _e:
            logger.debug(f"GenericRSSSentimentAdapter init skipped: {_e}")
        # Provide back-reference where needed
        for _a in self.adapters.values():
            try:
                if hasattr(_a, "adapters_ref"):
                    _a.adapters_ref = self.adapters
            except Exception:
                pass

        # Quality thresholds
        self.min_quality_score = 60.0
        self.preferred_quality_score = 80.0

        logger.info("DataFeedOrchestrator initialized with quality validation")
        # Trends TTL fallback (hours)
        try:
            self.trends_ttl_seconds = int(os.getenv("TRENDS_TTL_H", "24")) * 3600
        except Exception:
            self.trends_ttl_seconds = 24 * 3600

        # Defaults for new endpoints
        try:
            self.DIVSPLITS_TTL_D = int(os.getenv("DIVSPLITS_TTL_D", "7"))
        except Exception:
            self.DIVSPLITS_TTL_D = 7
        try:
            self.EARNINGS_TTL_H = int(os.getenv("EARNINGS_TTL_H", "24"))
        except Exception:
            self.EARNINGS_TTL_H = 24
        try:
            self.OPTIONS_TTL_MIN = int(os.getenv("OPTIONS_TTL_MIN", "15"))
        except Exception:
            self.OPTIONS_TTL_MIN = 15
        self.CONSERVE_FMP_BANDWIDTH = os.getenv(
            "CONSERVE_FMP_BANDWIDTH", "true"
        ).lower() in ("1", "true", "yes")
        self.RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.0"))
        self.CONTRACT_MULTIPLIER = 100.0

        # Lazy consolidated feed for delegation parity with legacy ConsolidatedDataFeed
        self._consolidated_feed: Optional[ConsolidatedDataFeed] = None

        # Feature flags / config for Investiny compact formatter
        self.enable_investiny_compact: bool = True
        self.investiny_default_range: Optional[str] = None
        try:
            # If external config object exposes these, honor them
            self.enable_investiny_compact = bool(
                getattr(loaded_config, "enable_investiny_compact", True)
            )
            self.investiny_default_range = getattr(
                loaded_config, "investiny_date_range", None
            )
        except Exception:
            pass

        # Initialize standardized adapter wrappers for future orchestration use.
        # This is guarded and does not change behavior if initialization fails.
        try:
            self._init_standard_adapters()
        except Exception as _e:
            logger.warning(f"Standard adapter wrapper initialization skipped: {_e}")
        # Initialize options snapshot schema (best-effort)
        try:
            from data_feeds import options_store as _opts_store  # type: ignore

            _opts_store.ensure_schema(
                os.getenv("CACHE_DB_PATH", "./model_monitoring.db")
            )
        except Exception:
            pass

    def _init_proactive_cache_warming(self) -> None:
        """Initialize proactive cache warming for high-demand data patterns"""
        try:
            # Define frequently accessed data patterns for proactive caching
            self.frequent_patterns = {
                # Market data patterns (highest priority)
                "market_quotes": [
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                    "AMZN",
                    "TSLA",
                    "NVDA",
                    "META",
                    "NFLX",
                ],
                # Prefer ETF proxies here so cache warming never spams caret-based symbols
                # that some free data providers reject (e.g., ^GSPC via yfinance).
                "market_indices": ["SPY", "QQQ", "DIA", "VIXY"],
                "sector_etfs": [
                    "XLF",
                    "XLE",
                    "XLK",
                    "XLV",
                    "XLY",
                    "XLI",
                    "XLB",
                    "XLP",
                    "XLU",
                    "XLRE",
                ],
                # Economic indicators
                "economic_data": ["GDP", "CPI", "UNEMPLOYMENT", "FEDFUNDS"],
                # High-frequency sentiment sources
                "sentiment_sources": ["reddit", "twitter", "news", "finviz"],
                # Technical indicators (commonly requested)
                "technical_indicators": ["SMA", "EMA", "RSI", "MACD", "BBANDS"],
            }

            # Start background cache warming if enabled
            if self.persistent_cache and hasattr(
                self.persistent_cache, "configure_warming"
            ):
                warming_config = self.persistent_cache.get_warming_config()
                if warming_config.get("enable_background_warmer", False):
                    logger.info(
                        "Starting proactive cache warming for high-demand data patterns"
                    )
                    self._start_proactive_warming()

        except Exception as e:
            logger.error(f"Failed to initialize proactive cache warming: {e}")

    def _start_proactive_warming(self) -> None:
        """Start background thread for proactive cache warming"""
        import threading

        def proactive_warmer():
            while True:
                try:
                    self._perform_proactive_warming()
                    # Sleep for configured interval
                    warming_config = self.persistent_cache.get_warming_config()
                    interval = warming_config.get("warming_interval_seconds", 300)
                    time.sleep(interval)

                except Exception as e:
                    logger.error(f"Proactive warming error: {e}")
                    time.sleep(60)  # Brief pause on error

        warming_thread = threading.Thread(target=proactive_warmer, daemon=True)
        warming_thread.start()
        logger.info("Proactive cache warming thread started")

    def _perform_proactive_warming(self) -> None:
        """Perform proactive cache warming based on usage patterns and demand"""
        if not self.persistent_cache:
            return

        try:
            # Get current cache statistics
            cache_stats = self.persistent_cache.get_cache_stats()
            overall_hit_rate = cache_stats["overall"].get("overall_hit_rate", 0)

            # Only warm if hit rate is below threshold
            warming_config = self.persistent_cache.get_warming_config()
            min_threshold = warming_config.get("min_hit_rate_threshold", 0.7)

            if overall_hit_rate >= min_threshold:
                logger.debug(
                    f"Cache hit rate {overall_hit_rate:.2%} above threshold {min_threshold:.2%}, skipping warming"
                )
                return

            logger.info(
                f"Cache hit rate {overall_hit_rate:.2%} below threshold, performing proactive warming..."
            )

            # Warm most frequently accessed market data
            warmed_count = 0
            max_items = warming_config.get("max_warming_items", 50)

            for symbol in self.frequent_patterns["market_quotes"][: max_items // 2]:
                try:
                    # Check if we have recent data, if not, fetch and cache
                    cache_key = f"quote_{symbol}"
                    cached_data = self.persistent_cache.get(cache_key)

                    if not cached_data:
                        # Fetch fresh data and cache it
                        quote_data = self.get_quote(symbol)
                        if quote_data:
                            self.persistent_cache.set(
                                key=cache_key,
                                endpoint="quote",
                                symbol=symbol,
                                ttl_seconds=300,  # 5 minutes
                                payload_json=quote_data,
                                source="proactive_warming",
                            )
                            warmed_count += 1

                except Exception as e:
                    logger.debug(f"Failed to warm quote for {symbol}: {e}")

            # Warm market indices
            for index in self.frequent_patterns["market_indices"][: max_items // 4]:
                try:
                    cache_key = f"index_{index}"
                    cached_data = self.persistent_cache.get(cache_key)

                    if not cached_data:
                        # Fetch index data
                        index_data = self.get_quote(index)
                        if index_data:
                            self.persistent_cache.set(
                                key=cache_key,
                                endpoint="index",
                                symbol=index,
                                ttl_seconds=600,  # 10 minutes
                                payload_json=index_data,
                                source="proactive_warming",
                            )
                            warmed_count += 1

                except Exception as e:
                    logger.debug(f"Failed to warm index {index}: {e}")

            if warmed_count > 0:
                logger.info(f"Proactively warmed {warmed_count} cache entries")

        except Exception as e:
            logger.error(f"Error in proactive cache warming: {e}")

    def enable_cache_warming(
        self, enable: bool = True, interval_seconds: int = 300
    ) -> None:
        """Enable or disable proactive cache warming"""
        if not self.persistent_cache or not hasattr(
            self.persistent_cache, "configure_warming"
        ):
            logger.warning(
                "Cache warming not supported by current cache implementation"
            )
            return

        try:
            self.persistent_cache.configure_warming(
                enable_background_warmer=enable,
                warming_interval_seconds=interval_seconds,
                min_hit_rate_threshold=0.7,
                max_warming_items=100,
            )

            if enable:
                logger.info(f"Cache warming enabled with {interval_seconds}s interval")
                self._start_proactive_warming()
            else:
                logger.info("Cache warming disabled")

        except Exception as e:
            logger.error(f"Failed to configure cache warming: {e}")

    def get_cache_warming_status(self) -> Dict[str, Any]:
        """Get current cache warming status and statistics"""
        if not self.persistent_cache:
            return {"status": "cache_unavailable"}

        try:
            config = self.persistent_cache.get_warming_config()
            stats = self.persistent_cache.get_cache_stats()

            return {
                "warming_enabled": config.get("enable_background_warmer", False),
                "warming_interval_seconds": config.get("warming_interval_seconds", 300),
                "min_hit_rate_threshold": config.get("min_hit_rate_threshold", 0.7),
                "max_warming_items": config.get("max_warming_items", 100),
                "current_hit_rate": stats["overall"].get("overall_hit_rate", 0),
                "cache_layers": stats["overall"].get("cache_layers", 1),
                "proactive_patterns": len(getattr(self, "frequent_patterns", {})),
                "last_warming_check": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get cache warming status: {e}")
            return {"status": "error", "error": str(e)}

        # Initialize cache invalidation service (optional dependency)
        if get_cache_invalidation_service:
            try:
                self.cache_invalidation_service = get_cache_invalidation_service(
                    self.redis_cache, self
                )
                if self.cache_invalidation_service:
                    logger.info("Cache invalidation service initialized successfully")
                else:
                    logger.info("Cache invalidation service not enabled")
            except Exception as e:
                logger.error(f"Failed to initialize cache invalidation service: {e}")
                self.cache_invalidation_service = None
        else:
            logger.info("Cache invalidation service not available (dependency missing)")

    async def _init_options_schema_async(self) -> None:
        """Async version of options schema initialization"""
        try:
            from data_feeds import options_store as _opts_store  # type: ignore

            await _opts_store.ensure_schema_async(
                os.getenv("CACHE_DB_PATH", "./model_monitoring.db")
            )
        except Exception:
            pass

    def get_quote(
        self, symbol: str, preferred_sources: Optional[List[DataSource]] = None
    ) -> Optional[Quote]:
        logger.debug(
            f"get_quote called for symbol={symbol}, preferred_sources={preferred_sources}"
        )
        step_start = time.time()
        timings: Dict[str, float] = {}
        step1 = time.time()
        timings["init"] = step1 - step_start

        # Convert preferred sources to strings for fallback manager
        preferred_source_names = None
        if preferred_sources:
            preferred_source_names = [s.value for s in preferred_sources]
            logger.debug(f"Preferred sources provided: {preferred_source_names}")

        # Get optimized source order from fallback manager
        ordered_source_names = self.fallback_manager.get_fallback_order(
            "quote", preferred_source_names
        )

        # Convert back to DataSource enums
        ordered_sources = []
        for source_name in ordered_source_names:
            try:
                source_enum = DataSource(source_name)
                if source_enum in self.adapters:
                    ordered_sources.append(source_enum)
            except ValueError:
                logger.debug(f"Unknown data source: {source_name}")

        step2 = time.time()
        timings["fallback_order"] = step2 - step1

        best_quote: Optional[Quote] = None
        best_quality = 0

        for source in ordered_sources:
            sub_start = time.time()
            source_name = source.value
            logger.debug(f"Fetching quote from source: {source}")

            # Check if source is in fallback mode and if we should retry
            if self.fallback_manager.is_in_fallback(source_name):
                if not self.fallback_manager.can_retry(source_name):
                    logger.debug(
                        f"Skipping {source_name} - in fallback mode, retry not ready"
                    )
                    continue
                logger.info(f"Attempting recovery for {source_name}")

            try:
                adapter = self.adapters.get(source)
                if adapter and hasattr(adapter, "get_quote"):
                    quote = adapter.get_quote(symbol)  # type: ignore[attr-defined]
                    response_time = time.time() - sub_start
                    timings[f"fetch_{source.value}"] = response_time

                    logger.debug(f"Quote from {source}: {quote}")

                    if quote and getattr(quote, "quality_score", 0):
                        # Record success with fallback manager
                        self.fallback_manager.record_success(source_name, response_time)

                        if quote.quality_score > best_quality:
                            best_quote = quote
                            best_quality = quote.quality_score

                        # If we got excellent quality, we can stop here
                        if quote.quality_score >= 95.0:
                            logger.debug(
                                f"Excellent quality quote from {source_name}, stopping search"
                            )
                            break
                    else:
                        logger.debug(f"No valid quote from {source_name}")
                else:
                    logger.debug(f"Skipping source without get_quote: {source}")

            except Exception as e:
                logger.error(f"Error fetching quote from {source}: {e}")
                should_fallback = self.fallback_manager.record_error(
                    source_name, e, "unknown"
                )
                timings[f"error_{source.value}"] = time.time() - sub_start

                # If this was a recovery attempt, record the failure
                if self.fallback_manager.is_in_fallback(source_name):
                    self.fallback_manager.record_retry_attempt(
                        source_name, success=False
                    )

        timings["total"] = time.time() - step_start
        logger.info(f"Profiling timings for get_quote({symbol}): {timings}")

        # Log fallback status if any sources are in fallback mode
        fallback_status = self.fallback_manager.get_fallback_status()
        if fallback_status:
            logger.info(f"Current fallback status: {fallback_status}")

        return best_quote

    async def get_quote_async(
        self,
        symbol: str,
        preferred_sources: Optional[List[DataSource]] = None,
        max_concurrent: int = 3,
    ) -> Optional[Quote]:
        """Async version of get_quote with concurrent processing for improved performance."""
        logger.debug(
            f"get_quote_async called for symbol={symbol}, preferred_sources={preferred_sources}"
        )
        step_start = time.time()
        timings: Dict[str, float] = {}
        step1 = time.time()
        timings["init"] = step1 - step_start

        # Convert preferred sources to strings for fallback manager
        preferred_source_names = None
        if preferred_sources:
            preferred_source_names = [s.value for s in preferred_sources]
            logger.debug(f"Preferred sources provided: {preferred_source_names}")

        # Get optimized source order from fallback manager
        ordered_source_names = self.fallback_manager.get_fallback_order(
            "quote", preferred_source_names
        )

        # Convert back to DataSource enums
        ordered_sources = []
        for source_name in ordered_source_names:
            try:
                source_enum = DataSource(source_name)
                if source_enum in self.adapters:
                    ordered_sources.append(source_enum)
            except ValueError:
                logger.debug(f"Unknown data source: {source_name}")

        step2 = time.time()
        timings["fallback_order"] = step2 - step1

        # Use semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_from_source(
            source: DataSource,
        ) -> Tuple[Optional[Quote], str, float]:
            """Fetch quote from a single source with semaphore control."""
            async with semaphore:
                sub_start = time.time()
                source_name = source.value
                logger.debug(f"Async fetching quote from source: {source}")

                # Check if source is in fallback mode and if we should retry
                if self.fallback_manager.is_in_fallback(source_name):
                    if not self.fallback_manager.can_retry(source_name):
                        logger.debug(
                            f"Skipping {source_name} - in fallback mode, retry not ready"
                        )
                        return None, source_name, time.time() - sub_start
                    logger.info(f"Attempting recovery for {source_name}")

                try:
                    adapter = self.adapters.get(source)
                    if adapter and hasattr(adapter, "get_quote_async"):
                        # Try async method first
                        quote = await adapter.get_quote_async(symbol)  # type: ignore[attr-defined]
                    elif adapter and hasattr(adapter, "get_quote"):
                        # Fallback to sync method in thread pool
                        import concurrent.futures

                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            quote = await loop.run_in_executor(executor, adapter.get_quote, symbol)  # type: ignore[attr-defined]
                    else:
                        logger.debug(f"Skipping source without quote methods: {source}")
                        return None, source_name, time.time() - sub_start

                    response_time = time.time() - sub_start
                    logger.debug(f"Async quote from {source}: {quote}")

                    if quote and getattr(quote, "quality_score", 0):
                        # Record success with fallback manager
                        self.fallback_manager.record_success(source_name, response_time)
                        return quote, source_name, response_time
                    else:
                        logger.debug(f"No valid quote from {source_name}")
                        return None, source_name, response_time

                except Exception as e:
                    logger.error(f"Error fetching quote from {source}: {e}")
                    should_fallback = self.fallback_manager.record_error(
                        source_name, e, "unknown"
                    )
                    response_time = time.time() - sub_start

                    # If this was a recovery attempt, record the failure
                    if self.fallback_manager.is_in_fallback(source_name):
                        self.fallback_manager.record_retry_attempt(
                            source_name, success=False
                        )

                    return None, source_name, response_time

        # Execute concurrent fetches
        step3 = time.time()
        timings["concurrent_setup"] = step3 - step2

        tasks = [fetch_from_source(source) for source in ordered_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        step4 = time.time()
        timings["concurrent_fetch"] = step4 - step3

        # Process results
        best_quote: Optional[Quote] = None
        best_quality = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception in concurrent fetch: {result}")
                continue

            if not isinstance(result, tuple) or len(result) != 3:
                logger.error(f"Invalid result format: {result}")
                continue

            quote, source_name, response_time = result
            timings[f"fetch_{source_name}"] = response_time

            if (
                quote
                and hasattr(quote, "quality_score")
                and quote.quality_score is not None
            ):
                quality_score = float(quote.quality_score)
                if quality_score > best_quality:
                    best_quote = quote
                    best_quality = quality_score

                    # If we got excellent quality, we can return early
                    if quality_score >= 95.0:
                        logger.debug(
                            f"Excellent quality quote from {source_name}, returning early"
                        )
                        break

        timings["total"] = time.time() - step_start
        logger.info(f"Async profiling timings for get_quote_async({symbol}): {timings}")

        # Log fallback status if any sources are in fallback mode
        fallback_status = self.fallback_manager.get_fallback_status()
        if fallback_status:
            logger.info(f"Current fallback status: {fallback_status}")

        return best_quote

    def get_market_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        preferred_sources: Optional[List[DataSource]] = None,
    ) -> Optional[MarketData]:
        logger.debug(
            f"get_market_data called for symbol={symbol}, period={period}, interval={interval}, preferred_sources={preferred_sources}"
        )
        step_start = time.time()
        timings: Dict[str, float] = {}
        step1 = time.time()
        timings["init"] = step1 - step_start

        # Convert preferred sources to strings for fallback manager
        preferred_source_names = None
        if preferred_sources:
            preferred_source_names = [s.value for s in preferred_sources]
            logger.debug(f"Preferred sources provided: {preferred_source_names}")

        # Get optimized source order from fallback manager
        ordered_source_names = self.fallback_manager.get_fallback_order(
            "market_data", preferred_source_names
        )

        # Convert back to DataSource enums
        ordered_sources = []
        for source_name in ordered_source_names:
            try:
                source_enum = DataSource(source_name)
                if source_enum in self.adapters:
                    ordered_sources.append(source_enum)
            except ValueError:
                logger.debug(f"Unknown data source: {source_name}")

        step2 = time.time()
        timings["fallback_order"] = step2 - step1

        best_data: Optional[MarketData] = None
        best_quality = 0

        for source in ordered_sources:
            sub_start = time.time()
            source_name = source.value
            logger.debug(f"Fetching market data from source: {source}")

            # Check if source is in fallback mode and if we should retry
            if self.fallback_manager.is_in_fallback(source_name):
                if not self.fallback_manager.can_retry(source_name):
                    logger.debug(
                        f"Skipping {source_name} - in fallback mode, retry not ready"
                    )
                    continue
                logger.info(f"Attempting recovery for {source_name}")

            try:
                adapter = self.adapters.get(source)
                if adapter and hasattr(adapter, "get_market_data"):
                    data = adapter.get_market_data(symbol, period, interval)  # type: ignore[attr-defined]
                    response_time = time.time() - sub_start
                    timings[f"fetch_{source.value}"] = response_time

                    logger.debug(f"Market data from {source}: {data}")

                    if data and getattr(data, "quality_score", 0):
                        # Record success with fallback manager
                        self.fallback_manager.record_success(source_name, response_time)

                        if data.quality_score > best_quality:
                            best_data = data
                            best_quality = data.quality_score

                        # If we got excellent quality, we can stop here
                        if data.quality_score >= 95.0:
                            logger.debug(
                                f"Excellent quality market data from {source_name}, stopping search"
                            )
                            break
                    else:
                        logger.debug(f"No valid market data from {source_name}")
                else:
                    logger.debug(f"Skipping source without get_market_data: {source}")

            except Exception as e:
                logger.error(f"Error fetching market data from {source}: {e}")
                should_fallback = self.fallback_manager.record_error(
                    source_name, e, "unknown"
                )
                timings[f"error_{source.value}"] = time.time() - sub_start

                # If this was a recovery attempt, record the failure
                if self.fallback_manager.is_in_fallback(source_name):
                    self.fallback_manager.record_retry_attempt(
                        source_name, success=False
                    )

        timings["total"] = time.time() - step_start
        logger.info(f"Profiling timings for get_market_data({symbol}): {timings}")

        # Log fallback status if any sources are in fallback mode
        fallback_status = self.fallback_manager.get_fallback_status()
        if fallback_status:
            logger.info(f"Current fallback status: {fallback_status}")

        return best_data

    async def get_market_data_async(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        preferred_sources: Optional[List[DataSource]] = None,
        max_concurrent: int = 3,
    ) -> Optional[MarketData]:
        """Async version of get_market_data with concurrent processing for improved performance."""
        logger.debug(
            f"get_market_data_async called for symbol={symbol}, period={period}, interval={interval}, preferred_sources={preferred_sources}"
        )
        step_start = time.time()
        timings: Dict[str, float] = {}
        step1 = time.time()
        timings["init"] = step1 - step_start

        # Convert preferred sources to strings for fallback manager
        preferred_source_names = None
        if preferred_sources:
            preferred_source_names = [s.value for s in preferred_sources]
            logger.debug(f"Preferred sources provided: {preferred_source_names}")

        # Get optimized source order from fallback manager
        ordered_source_names = self.fallback_manager.get_fallback_order(
            "market_data", preferred_source_names
        )

        # Convert back to DataSource enums
        ordered_sources = []
        for source_name in ordered_source_names:
            try:
                source_enum = DataSource(source_name)
                if source_enum in self.adapters:
                    ordered_sources.append(source_enum)
            except ValueError:
                logger.debug(f"Unknown data source: {source_name}")

        step2 = time.time()
        timings["fallback_order"] = step2 - step1

        # Use semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_from_source(
            source: DataSource,
        ) -> Tuple[Optional[MarketData], str, float]:
            """Fetch market data from a single source with semaphore control."""
            async with semaphore:
                sub_start = time.time()
                source_name = source.value
                logger.debug(f"Async fetching market data from source: {source}")

                # Check if source is in fallback mode and if we should retry
                if self.fallback_manager.is_in_fallback(source_name):
                    if not self.fallback_manager.can_retry(source_name):
                        logger.debug(
                            f"Skipping {source_name} - in fallback mode, retry not ready"
                        )
                        return None, source_name, time.time() - sub_start
                    logger.info(f"Attempting recovery for {source_name}")

                try:
                    adapter = self.adapters.get(source)
                    if adapter and hasattr(adapter, "get_market_data_async"):
                        # Try async method first
                        data = await adapter.get_market_data_async(symbol, period, interval)  # type: ignore[attr-defined]
                    elif adapter and hasattr(adapter, "get_market_data"):
                        # Fallback to sync method in thread pool
                        import concurrent.futures

                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            data = await loop.run_in_executor(executor, adapter.get_market_data, symbol, period, interval)  # type: ignore[attr-defined]
                    else:
                        logger.debug(
                            f"Skipping source without market data methods: {source}"
                        )
                        return None, source_name, time.time() - sub_start

                    response_time = time.time() - sub_start
                    logger.debug(f"Async market data from {source}: {data}")

                    if (
                        data
                        and hasattr(data, "quality_score")
                        and data.quality_score is not None
                    ):
                        # Record success with fallback manager
                        self.fallback_manager.record_success(source_name, response_time)
                        quality_score = float(data.quality_score)
                        return data, source_name, response_time
                    else:
                        logger.debug(f"No valid market data from {source_name}")
                        return None, source_name, response_time

                except Exception as e:
                    logger.error(f"Error fetching market data from {source}: {e}")
                    should_fallback = self.fallback_manager.record_error(
                        source_name, e, "unknown"
                    )
                    response_time = time.time() - sub_start

                    # If this was a recovery attempt, record the failure
                    if self.fallback_manager.is_in_fallback(source_name):
                        self.fallback_manager.record_retry_attempt(
                            source_name, success=False
                        )

                    return None, source_name, response_time

        # Execute concurrent fetches
        step3 = time.time()
        timings["concurrent_setup"] = step3 - step2

        tasks = [fetch_from_source(source) for source in ordered_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        step4 = time.time()
        timings["concurrent_fetch"] = step4 - step3

        # Process results
        best_data: Optional[MarketData] = None
        best_quality = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception in concurrent fetch: {result}")
                continue

            if not isinstance(result, tuple) or len(result) != 3:
                logger.error(f"Invalid result format: {result}")
                continue

            data, source_name, response_time = result
            timings[f"fetch_{source_name}"] = response_time

            if (
                data
                and hasattr(data, "quality_score")
                and data.quality_score is not None
            ):
                quality_score = float(data.quality_score)
                if quality_score > best_quality:
                    best_data = data
                    best_quality = quality_score

                    # If we got excellent quality, we can return early
                    if quality_score >= 95.0:
                        logger.debug(
                            f"Excellent quality market data from {source_name}, returning early"
                        )
                        break

        timings["total"] = time.time() - step_start
        logger.info(
            f"Async profiling timings for get_market_data_async({symbol}): {timings}"
        )

        # Log fallback status if any sources are in fallback mode
        fallback_status = self.fallback_manager.get_fallback_status()
        if fallback_status:
            logger.info(f"Current fallback status: {fallback_status}")

        return best_data

    # ------------------------------------------------------------------------
    # New methods delegating to ConsolidatedDataFeed (compatibility bridge)
    # ------------------------------------------------------------------------
    def _get_consolidated(self) -> ConsolidatedDataFeed:
        """Lazily create ConsolidatedDataFeed for delegation paths."""
        if self._consolidated_feed is None:
            self._consolidated_feed = ConsolidatedDataFeed()
        return self._consolidated_feed

    # =========================
    # New endpoints
    # =========================
    def get_dividends_and_splits(self, symbol: str) -> Optional[dict]:
        """
        Uses yfinance.Ticker(symbol).dividends and .splits
        Normalizes to dict: {"dividends":[{date,amount}], "splits":[{date,ratio}]}
        Persists via CacheService with TTL days from DIVSPLITS_TTL_D (default 7)
        """
        if not symbol:
            return None
        params = {"symbol": symbol}
        key_hash = None
        entry = None
        if getattr(self, "persistent_cache", None):
            try:
                key_hash = self.persistent_cache.make_key("dividends_splits", params)  # type: ignore
                entry = self.persistent_cache.get(key_hash)  # type: ignore
                if entry and not entry.is_expired() and entry.payload_json:
                    return entry.payload_json  # type: ignore
            except Exception:
                entry = None

        try:
            t = yf.Ticker(symbol)
            div = t.dividends
            spl = t.splits
            out = {"dividends": [], "splits": []}
            if div is not None and hasattr(div, "items"):
                for idx, val in div.items():
                    try:
                        # idx may be a Timestamp / date-like; fall back gracefully
                        try:
                            date_s = str(
                                getattr(idx, "date")() if hasattr(idx, "date") else idx
                            )
                        except Exception:
                            date_s = str(idx)
                    except Exception:
                        date_s = str(idx)
                    try:
                        amt = float(val)
                    except Exception:
                        amt = None
                    if amt is not None:
                        out["dividends"].append({"date": date_s, "amount": amt})
            if spl is not None and hasattr(spl, "items"):
                for idx, val in spl.items():
                    try:
                        try:
                            date_s = str(
                                getattr(idx, "date")() if hasattr(idx, "date") else idx
                            )
                        except Exception:
                            date_s = str(idx)
                    except Exception:
                        date_s = str(idx)
                    try:
                        ratio = float(val)
                    except Exception:
                        ratio = None
                    if ratio is not None:
                        out["splits"].append({"date": date_s, "ratio": ratio})
        except Exception as e:
            logger.error(f"dividends/splits fetch failed for {symbol}: {e}")
            return None

        if getattr(self, "persistent_cache", None):
            try:
                if key_hash is None:
                    key_hash = self.persistent_cache.make_key("dividends_splits", params)  # type: ignore
                self.persistent_cache.set(  # type: ignore
                    key=key_hash,
                    endpoint="dividends_splits",
                    symbol=symbol,
                    ttl_seconds=int(self.DIVSPLITS_TTL_D) * 86400,
                    payload_json=out,
                    source="yfinance",
                    metadata_json={"params": params},
                )
            except Exception as e:
                logger.warning(f"[cache] dividends_splits write failed: {e}")
        return out

    def get_earnings_calendar_detailed(
        self, tickers: Optional[list[str]] = None
    ) -> Optional[list[dict]]:
        """
        Uses FMP earning_calendar for the next 60 days.
        Respects CONSERVE_FMP_BANDWIDTH (default true). TTL EARNINGS_TTL_H (default 24h).
        If FMP fails (e.g., 403), falls back to FinViz earnings and normalizes to the same schema.
        """
        base_params = {"tickers": tickers or []}

        key_hash = None
        entry = None
        if getattr(self, "persistent_cache", None):
            try:
                key_hash = self.persistent_cache.make_key("earnings_calendar", base_params)  # type: ignore
                entry = self.persistent_cache.get(key_hash)  # type: ignore
                if entry and not entry.is_expired() and entry.payload_json:
                    return entry.payload_json  # type: ignore
            except Exception:
                entry = None

        if (
            self.CONSERVE_FMP_BANDWIDTH
            and entry
            and not entry.is_expired()
            and entry.payload_json
        ):
            return entry.payload_json  # type: ignore

        out_list: list[dict] = []
        source_used = "fmp"
        success = False

        # Try FMP first when API key present
        api_key = os.getenv("FINANCIALMODELINGPREP_API_KEY")
        if api_key:
            today = datetime.utcnow().date()
            to_date = today + timedelta(days=60)
            url = (
                "https://financialmodelingprep.com/api/v3/earning_calendar"
                f"?from={today.isoformat()}&to={to_date.isoformat()}&apikey={api_key}"
            )
            try:
                resp = optimized_get(url, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    for item in data:
                        try:
                            sym = str(item.get("symbol") or item.get("ticker") or "")
                            date_s = str(item.get("date") or item.get("epsDate") or "")
                            time_day = item.get("time") or item.get("timeOfDay")
                            if time_day is not None:
                                time_day = str(time_day).lower()
                            eps_est = (
                                item.get("epsEstimated")
                                if "epsEstimated" in item
                                else item.get("epsEstimate")
                            )
                            eps_act = (
                                item.get("eps")
                                if "eps" in item
                                else item.get("epsActual")
                            )
                            rev_est = (
                                item.get("revenueEstimated")
                                if "revenueEstimated" in item
                                else item.get("revenueEstimate")
                            )
                            rev_act = (
                                item.get("revenue")
                                if "revenue" in item
                                else item.get("revenueActual")
                            )
                            out_list.append(
                                {
                                    "symbol": sym,
                                    "date": date_s,
                                    "time": (
                                        time_day
                                        if time_day in ("bmo", "amc")
                                        else (
                                            None
                                            if time_day in (None, "", "none")
                                            else str(time_day)
                                        )
                                    ),
                                    "eps_estimate": (
                                        float(eps_est) if eps_est is not None else None
                                    ),
                                    "eps_actual": (
                                        float(eps_act) if eps_act is not None else None
                                    ),
                                    "revenue_estimate": (
                                        float(rev_est) if rev_est is not None else None
                                    ),
                                    "revenue_actual": (
                                        float(rev_act) if rev_act is not None else None
                                    ),
                                }
                            )
                        except Exception:
                            continue
                    success = True
            except Exception as e:
                logger.error(f"FMP earnings calendar fetch failed: {e}")

        # Fallback to FinViz when FMP failed or no key
        if not success:
            adapter = self.adapters.get(DataSource.FINVIZ)
            try:
                finviz_payload = (
                    adapter.get_earnings()
                    if adapter and hasattr(adapter, "get_earnings")
                    else None
                )
                df = None
                if isinstance(finviz_payload, dict):
                    # Try common keys first
                    for k in ["earnings", "upcoming_earnings", "calendar", "Earnings"]:
                        if k in finviz_payload:
                            df = finviz_payload[k]
                            break
                    # Otherwise pick the first DataFrame-like value
                    if df is None:
                        for v in finviz_payload.values():
                            if hasattr(v, "empty"):
                                df = v
                                break
                if df is not None and hasattr(df, "empty") and not df.empty:
                    cols = {c.lower(): c for c in df.columns}

                    def pick(*names):
                        for n in names:
                            if n in cols:
                                return cols[n]
                        return None

                    sym_c = pick("ticker", "symbol")
                    date_c = pick("date", "earnings date", "earnings_date")
                    time_c = pick("time", "time of day", "time_of_day")
                    eps_est_c = pick("eps estimate", "eps_estimate", "estimate")
                    eps_act_c = pick("eps", "eps actual", "eps_actual")
                    rev_est_c = pick("revenue estimate", "revenue_estimate")
                    rev_act_c = pick("revenue", "revenue_actual")
                    for _, row in df.iterrows():
                        try:
                            sym = str(row[sym_c]) if sym_c in df.columns else ""
                            date_s = str(row[date_c]) if date_c in df.columns else ""
                            time_day = None
                            if time_c in df.columns:
                                td = (
                                    str(row[time_c]).lower()
                                    if row[time_c] is not None
                                    else None
                                )
                                if td in ("bmo", "amc"):
                                    time_day = td
                                elif td in (
                                    "before market open",
                                    "pre-market",
                                    "premarket",
                                ):
                                    time_day = "bmo"
                                elif td in (
                                    "after market close",
                                    "post-market",
                                    "after-hours",
                                    "after hours",
                                ):
                                    time_day = "amc"

                            def fget(c):
                                if (
                                    c in df.columns
                                    and row[c] is not None
                                    and str(row[c]).strip() != ""
                                ):
                                    try:
                                        return float(row[c])
                                    except Exception:
                                        return None
                                return None

                            out_list.append(
                                {
                                    "symbol": sym,
                                    "date": date_s,
                                    "time": time_day,
                                    "eps_estimate": fget(eps_est_c),
                                    "eps_actual": fget(eps_act_c),
                                    "revenue_estimate": fget(rev_est_c),
                                    "revenue_actual": fget(rev_act_c),
                                }
                            )
                        except Exception:
                            continue
                    success = True
                    source_used = "finviz"
            except Exception as e:
                logger.error(f"FinViz earnings fallback failed: {e}")

        # Optional filter by tickers
        if tickers and out_list:
            tset = set(s.upper() for s in tickers)
            out_list = [
                x for x in out_list if (x.get("symbol", "") or "").upper() in tset
            ]

        # Persist if successful
        if success and out_list and getattr(self, "persistent_cache", None):
            try:
                if key_hash is None:
                    key_hash = self.persistent_cache.make_key("earnings_calendar", base_params)  # type: ignore
                # Wrap list payload in a dict for consistent JSON typing
                self.persistent_cache.set(  # type: ignore
                    key=key_hash,
                    endpoint="earnings_calendar",
                    symbol=None,
                    ttl_seconds=int(self.EARNINGS_TTL_H) * 3600,
                    payload_json={"items": out_list},
                    source=source_used,
                    metadata_json={"params": base_params},
                )
            except Exception as e:
                logger.warning(f"[cache] earnings_calendar write failed: {e}")
        # Normalize return type to list[dict] | None
        if success and out_list:
            return out_list
        if entry and entry.payload_json:
            try:
                if (
                    isinstance(entry.payload_json, dict)
                    and "items" in entry.payload_json
                ):
                    items = entry.payload_json.get("items")
                    if isinstance(items, list):
                        return items  # type: ignore
            except Exception:
                pass
        return None

    def get_options_analytics(
        self, symbol: str, include: list[str] | None = None
    ) -> Optional[dict]:
        """
        Uses yfinance to fetch nearest expiry chain, snapshots via options_store, and computes:
        IV, Greeks, GEX, Max Pain. Caches analytics for OPTIONS_TTL_MIN.
        """
        include = include or ["chain", "iv", "greeks", "gex", "max_pain"]
        if not symbol:
            return None

        params = {"symbol": symbol, "include": sorted(include)}
        key_hash = None
        if getattr(self, "persistent_cache", None):
            try:
                key_hash = self.persistent_cache.make_key("options_analytics", params)  # type: ignore
                entry = self.persistent_cache.get(key_hash)  # type: ignore
                if entry and not entry.is_expired() and entry.payload_json:
                    return entry.payload_json  # type: ignore
            except Exception:
                pass

        # Imports
        try:
            from data_feeds import options_math as _opts_math  # type: ignore
            from data_feeds import options_store as _opts_store  # type: ignore
        except Exception as e:
            logger.error(f"Options modules import failed: {e}")
            return None

        # Underlying price and info
        try:
            t = yf.Ticker(symbol)
            info = t.info or {}
            S = None
            for k in ("regularMarketPrice", "currentPrice", "previousClose", "close"):
                if info.get(k) is not None:
                    try:
                        raw_v = info.get(k)
                        if raw_v is None:
                            continue
                        try:
                            S = float(raw_v)
                        except Exception:
                            continue
                        if S and S > 0:
                            break
                    except Exception:
                        continue
            if not S or S <= 0:
                hist = t.history(period="5d", interval="1d")
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    S = float(hist["Close"].iloc[-1])
        except Exception as e:
            logger.error(f"Failed to get underlying price for {symbol}: {e}")
            return None
        if not S or S <= 0:
            return None

        # Dividend yield estimate q from dividendRate / price
        q = 0.0
        try:
            div_rate = info.get("dividendRate")
            if div_rate is not None and S > 0:
                q = max(0.0, float(div_rate) / float(S))
        except Exception:
            q = 0.0
        r = self.RISK_FREE_RATE

        # Nearest expiry
        try:
            expirations = t.options or []
            if not expirations:
                return None
            today = datetime.utcnow().date()
            exps_sorted = sorted(
                expirations,
                key=lambda d: abs(
                    (datetime.strptime(d, "%Y-%m-%d").date() - today).days
                ),
            )
            nearest_exp = exps_sorted[0]
            expiry_date = datetime.strptime(nearest_exp, "%Y-%m-%d").date()
        except Exception as e:
            logger.error(f"Failed to get options expirations for {symbol}: {e}")
            return None

        T_days = max(1, (expiry_date - today).days)

        T = T_days / 365.0

        # Chains
        try:
            oc = t.option_chain(nearest_exp)
            calls = oc.calls

            puts = oc.puts
        except Exception as e:
            logger.error(f"Failed to get option chain for {symbol} {nearest_exp}: {e}")
            return None

        now_ts = int(time.time())
        db_path = os.getenv("CACHE_DB_PATH", "./model_monitoring.db")
        rows = []

        def safe_num(v, cast=float):
            try:
                if v is None:
                    return None
                return cast(v)
            except Exception:
                return None

        def build_rows(df, put_call: str):
            if df is None:
                return
            for _, row in df.iterrows():
                rows.append(
                    {
                        "symbol": symbol,
                        "expiry": nearest_exp,
                        "chain_date": now_ts,
                        "put_call": put_call,
                        "strike": safe_num(row.get("strike"), float),
                        "last": safe_num(row.get("lastPrice"), float),
                        "bid": safe_num(row.get("bid"), float),
                        "ask": safe_num(row.get("ask"), float),
                        "volume": safe_num(row.get("volume"), int),
                        "open_interest": safe_num(row.get("openInterest"), int),
                        "underlying": float(S),
                        "source": "yfinance",
                    }
                )

        build_rows(calls, "call")
        build_rows(puts, "put")

        # Snapshot upsert
        try:
            _opts_store.upsert_snapshot_many(db_path, rows)
        except Exception as e:
            logger.warning(f"Options snapshot upsert failed: {e}")

        result: dict = {}
        strikes = []

        # IV and Greeks
        iv_map = {}
        greeks_map = {}
        if "iv" in include or "greeks" in include or "gex" in include:
            for rrow in rows:
                pc = rrow["put_call"]
                strike = rrow["strike"]
                if strike is None:
                    continue
                strikes.append(strike)
                mid = None
                bid = rrow.get("bid")
                ask = rrow.get("ask")
                last_p = rrow.get("last")
                if (
                    bid is not None
                    and ask is not None
                    and bid <= ask
                    and bid >= 0
                    and ask > 0
                ):
                    mid = 0.5 * (bid + ask)
                elif last_p is not None and last_p > 0:
                    mid = float(last_p)
                if mid is None or mid <= 0:
                    continue
                try:
                    iv = _opts_math.implied_vol(mid, S, strike, r, q, T, put_call=pc)
                except Exception:
                    iv = None
                if iv is not None:
                    iv_map[(pc, strike)] = iv
                    try:
                        greeks = _opts_math.bs_greeks(
                            S, strike, r, q, iv, T, put_call=pc
                        )
                    except Exception:
                        greeks = None
                    if greeks:
                        greeks_map[(pc, strike)] = greeks

        if "iv" in include:
            result["iv"] = [
                {"put_call": pc, "strike": k, "iv": float(v)}
                for (pc, k), v in iv_map.items()
            ]
        if "greeks" in include:
            out_g = []
            for (pc, k), g in greeks_map.items():
                gg = dict(g)
                out_g.append({"put_call": pc, "strike": k, **gg})
            result["greeks"] = out_g

        if "chain" in include:
            result["chain"] = rows

        # GEX heuristic: gamma * contract_multiplier * S^2 * OI
        if "gex" in include:
            total_gex = 0.0
            for rrow in rows:
                pc = rrow["put_call"]
                strike = rrow["strike"]
                oi = rrow.get("open_interest") or 0
                g = greeks_map.get((pc, strike))
                if not g:
                    continue
                gamma = float(g.get("gamma") or 0.0)
                total_gex += gamma * self.CONTRACT_MULTIPLIER * (S**2) * float(oi)
            result["gex"] = {"total_gamma_exposure": float(total_gex)}

        # Max Pain
        if "max_pain" in include and strikes:
            unique_strikes = sorted(set([s for s in strikes if s is not None]))
            oi_map = {}
            for rrow in rows:
                pc = rrow["put_call"]
                strike = rrow["strike"]
                if strike is None:
                    continue
                oi_map.setdefault((pc, strike), 0)
                oi_map[(pc, strike)] += int(rrow.get("open_interest") or 0)

            def pain_at_price(P):
                pain = 0.0
                for (pc, k), oi in oi_map.items():
                    intrinsic = max(0.0, P - k) if pc == "call" else max(0.0, k - P)
                    pain += intrinsic * oi * self.CONTRACT_MULTIPLIER
                return pain

            best_P = None
            best_pain = None
            for k in unique_strikes:
                p = pain_at_price(k)
                if best_pain is None or p < best_pain:
                    best_pain = p
                    best_P = k
            result["max_pain"] = {
                "strike": float(best_P) if best_P is not None else None,
                "pain": float(best_pain) if best_pain is not None else None,
            }

        # Cache analytics
        if getattr(self, "persistent_cache", None):
            try:
                if key_hash is None:
                    key_hash = self.persistent_cache.make_key("options_analytics", params)  # type: ignore
                self.persistent_cache.set(  # type: ignore
                    key=key_hash,
                    endpoint="options_analytics",
                    symbol=symbol,
                    ttl_seconds=int(self.OPTIONS_TTL_MIN) * 60,
                    payload_json=result,
                    source="yfinance",
                    metadata_json={"params": params, "expiry": nearest_exp},
                )
            except Exception as e:
                logger.warning(f"[cache] options_analytics write failed: {e}")

        return result

    async def get_options_analytics_async(
        self, symbol: str, include: list[str] | None = None
    ) -> Optional[dict]:
        """
        Async version of get_options_analytics with async database operations.
        """
        # Initialize async I/O if needed
        await self._init_options_schema_async()

        # Reuse most of the logic from get_options_analytics but with async database calls
        # For now, delegate to sync version since full async conversion would require
        # making the entire method async (which is a larger refactoring)
        return self.get_options_analytics(symbol, include)

    def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        """
        Delegate to ConsolidatedDataFeed.get_company_info for compatibility during consolidation.
        Propagates None if unavailable.
        """
        try:
            feed = self._get_consolidated()
            info = feed.get_company_info(symbol)
            # Attach/ensure source metadata is present; do not fabricate quality_score
            if info and not getattr(info, "source", None):
                # Keep as None if not provided by source
                pass
            return info
        except Exception as e:
            logger.error(f"Orchestrator company_info error for {symbol}: {e}")
            return None

    def get_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        """
        Delegate to ConsolidatedDataFeed.get_news for compatibility during consolidation.
        Returns empty list if unavailable, matching ConsolidatedDataFeed behavior.
        """
        try:
            feed = self._get_consolidated()
            items = feed.get_news(symbol, limit)
            # Ensure source present on each item if provided by adapter; do not fabricate otherwise
            if not items:
                return []
            return items
        except Exception as e:
            logger.error(f"Orchestrator news error for {symbol}: {e}")
            return []

    def get_multiple_quotes(
        self, symbols: List[str]
    ):  # return type simplified to avoid cross-module Quote mismatch
        """
        Delegate to ConsolidatedDataFeed.get_multiple_quotes for compatibility during consolidation.
        Maps each symbol to its quote where available.
        """
        try:
            feed = self._get_consolidated()
            results = feed.get_multiple_quotes(symbols)
            return results or {}
        except Exception as e:
            logger.error(f"Orchestrator multiple quotes error: {e}")
            return {}

    def get_financial_statements(
        self, symbol: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Delegate to ConsolidatedDataFeed.get_financial_statements for compatibility during consolidation.
        Returns empty dict if unavailable.
        """
        try:
            feed = self._get_consolidated()
            financials = feed.get_financial_statements(symbol)
            return financials or {}
        except Exception as e:
            logger.error(f"Orchestrator financial statements error for {symbol}: {e}")
            return {}

    def get_sentiment_data(
        self, symbol: str, sources: Optional[List[DataSource]] = None
    ) -> Dict[str, SentimentData]:
        logger.debug(
            f"get_sentiment_data called for symbol={symbol}, sources={sources}"
        )
        sentiment_data: Dict[str, SentimentData] = {}
        step_start = time.time()
        timings: Dict[str, float] = {}

        # Determine order of sentiment sources (default list if none provided)
        # Exclude FinViz from default sentiment sources as it doesn't implement get_sentiment
        if sources is None:
            logger.debug("No sources provided, using default sentiment sources.")
        ordered = (
            sources
            if sources
            else [DataSource.REDDIT, DataSource.TWITTER, DataSource.NEWS]
        )
        timings["ordered_list"] = time.time() - step_start

        for source in ordered:
            sub_start = time.time()
            logger.debug(f"Fetching sentiment data from source: {source}")
            try:
                adapter = self.adapters.get(source)
                if not adapter:
                    logger.warning(f"No adapter found for source: {source}")
                    timings[f"missing_{getattr(source, 'value', str(source))}"] = (
                        time.time() - sub_start
                    )
                    continue
                # Capability gating: skip adapters that do not implement get_sentiment
                if not hasattr(adapter, "get_sentiment") or not callable(
                    getattr(adapter, "get_sentiment", None)
                ):
                    logger.info(
                        f"Skipping sentiment fetch for source {source} (no get_sentiment implementation)"
                    )
                    timings[f"skipped_{getattr(source, 'value', str(source))}"] = (
                        time.time() - sub_start
                    )
                    continue

                adapter_name = getattr(source, "value", str(source))
                call_start = time.time()
                sentiment = adapter.get_sentiment(symbol)
                call_end = time.time()
                elapsed = call_end - call_start
                timings[f"fetch_{adapter_name}"] = elapsed
                logger.info(
                    f"[TIMING][oracle_agent_pipeline][{adapter_name}] {elapsed:.2f} seconds for get_sentiment({symbol})"
                )
                print(
                    f"[TIMING][oracle_agent_pipeline][{adapter_name}] {elapsed:.2f} seconds for get_sentiment({symbol})"
                )
                try:
                    import sys as _sys

                    _sys.stdout.flush()
                except Exception:
                    pass
                logger.debug(f"Sentiment from {source}: {sentiment}")
                if sentiment:
                    sentiment_data[adapter_name] = sentiment
            except Exception as e:
                logger.error(f"Error fetching sentiment from {source}: {e}")
                timings[f"error_{getattr(source, 'value', str(source))}"] = (
                    time.time() - sub_start
                )

        timings["total"] = time.time() - step_start
        logger.info(
            f"[TIMING][oracle_agent_pipeline][total] {timings['total']:.2f} seconds for get_sentiment_data({symbol})"
        )
        print(
            f"[TIMING][oracle_agent_pipeline][total] {timings['total']:.2f} seconds for get_sentiment_data({symbol})"
        )
        try:
            import sys as _sys

            _sys.stdout.flush()
        except Exception:
            pass
        logger.info(f"Profiling timings for get_sentiment_data({symbol}): {timings}")
        return sentiment_data

    def list_available_sentiment_sources(self) -> List[str]:
        out: List[str] = []
        for key, adapter in self.adapters.items():
            if not hasattr(adapter, "get_sentiment"):
                continue
            if isinstance(key, tuple):
                out.append(":".join([str(k) for k in key]))
            else:
                out.append(str(getattr(key, "value", key)))
        return sorted(set(out))

    def get_advanced_sentiment_data(
        self,
        symbol: str,
        texts: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> Optional[SentimentData]:
        """Advanced sentiment has been deprecated; returns None for compatibility."""
        logger.info("Advanced sentiment analysis is disabled; returning None")
        return None

    def _init_standard_adapters(self) -> None:
        """
        Create standardized adapter wrappers using orchestrator-owned cache, rate limiter,
        and performance tracker. Store for future orchestration unification.
        This method is additive and non-breaking; current orchestrator methods
        continue to use existing adapters and delegation paths.
        """
        self._standard_adapters: Dict[str, Any] = {}
        # Only proceed if wrapper classes imported successfully
        if "YFinanceAdapterWrapper" not in globals():
            return
        try:
            # Build wrappers around consolidated adapters
            self._standard_adapters["yfinance"] = YFinanceAdapterWrapper(self.cache, self.rate_limiter, self.performance_tracker)  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to init YFinanceAdapterWrapper: {e}")
        try:
            self._standard_adapters["fmp"] = FMPAdapterWrapper(self.cache, self.rate_limiter, self.performance_tracker)  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to init FMPAdapterWrapper: {e}")
        try:
            self._standard_adapters["finnhub"] = FinnhubAdapterWrapper(self.cache, self.rate_limiter, self.performance_tracker)  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to init FinnhubAdapterWrapper: {e}")
        try:
            self._standard_adapters["finance_database"] = FinanceDatabaseAdapterWrapper(self.cache, self.rate_limiter, self.performance_tracker)  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to init FinanceDatabaseAdapterWrapper: {e}")

    def get_data_quality_report(self) -> Dict[str, DataQualityMetrics]:
        """Get comprehensive data quality report for all sources"""
        rankings = self.performance_tracker.get_source_ranking()

        quality_report = {}
        for source_name, score in rankings:
            metrics = self.performance_tracker.metrics[source_name]
            # Aggregate latest issues (errors + quality) for this source
            recent_issues: List[str] = []
            try:
                if source_name in self.performance_tracker.issues:
                    # Keep only last 5 issue messages
                    recent_issues = [
                        i.get("message", "")
                        for i in list(self.performance_tracker.issues[source_name])[-5:]
                    ]
            except Exception:
                recent_issues = []

            quality_report[source_name] = DataQualityMetrics(
                source=source_name,
                quality_score=score,
                latency_ms=(
                    float(np.mean(metrics["response_times"]))
                    if metrics["response_times"]
                    else 0
                ),
                success_rate=metrics["success_count"]
                / max(1, metrics["success_count"] + metrics["error_count"]),
                last_updated=metrics["last_success"] or datetime.now(),
                issues=recent_issues,
            )

        return quality_report

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "redis_connected": bool(self.redis_cache),
            "cache_warming_active": bool(self.cache_warming_service),
            "fallback_manager_active": bool(self.fallback_manager),
            "adapters_loaded": len(self.adapters),
            "total_requests": 0,  # Simplified
            "avg_response_time": 0,  # Simplified
        }

    def get_signals_from_scrapers(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Get comprehensive market signals for advanced learning system
        Replaces the missing method that caused the fallback error
        """
        try:
            signals = {}

            # Get market data for each ticker
            for ticker in tickers:
                try:
                    # Get quote data
                    quote = self.get_quote(ticker)
                    if quote:
                        signals[f"{ticker}_quote"] = {
                            "symbol": quote.symbol,
                            "price": float(quote.price),
                            "change": float(quote.change),
                            "change_percent": float(quote.change_percent),
                            "volume": quote.volume,
                            "market_cap": quote.market_cap,
                        }

                    # Get market data with price history
                    market_data = self.get_market_data(
                        ticker, period="30d", interval="1d"
                    )
                    if (
                        market_data
                        and hasattr(market_data, "data")
                        and not market_data.data.empty
                    ):
                        latest = market_data.data.iloc[-1]
                        signals[f"{ticker}_market_data"] = {
                            "close": float(latest.get("Close", 0)),
                            "volume": int(latest.get("Volume", 0)),
                            "high": float(latest.get("High", 0)),
                            "low": float(latest.get("Low", 0)),
                            "volatility": float(
                                market_data.data["Close"].pct_change().std() * 100
                            ),
                        }

                except Exception as e:
                    self.logger.warning(f"Failed to get data for {ticker}: {e}")
                    # Skip ticker if no real data available
                    continue

            # Add market breadth if available
            try:
                breadth = self.get_market_breadth()
                if breadth:
                    signals["market_breadth"] = {
                        "advancing": breadth.advancing,
                        "declining": breadth.declining,
                        "unchanged": breadth.unchanged,
                        "advance_decline_ratio": breadth.advance_decline_ratio,
                    }
            except Exception as e:
                self.logger.warning(f"Failed to get market breadth: {e}")

            # Add timestamp and metadata
            signals["timestamp"] = datetime.now().isoformat()
            signals["data_source"] = "DataFeedOrchestrator"
            signals["tickers_requested"] = tickers
            signals["signals_count"] = len(
                [k for k in signals.keys() if not k.startswith("_")]
            )

            return signals

        except Exception as e:
            self.logger.error(f"Error in get_signals_from_scrapers: {e}")
            # Return empty structure - no synthetic data
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "tickers_requested": tickers,
                "signals_count": 0,
            }

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
            self.performance_tracker.record_success(
                DataSource.FINVIZ.value,
                (time.time() - start_time) * 1000,
                quality_score,
            )
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
                g.group_type == "sector"
                and all(
                    in_range(val)
                    for val in [
                        g.perf_1d,
                        g.perf_1w,
                        g.perf_1m,
                        g.perf_3m,
                        g.perf_6m,
                        g.perf_1y,
                        g.perf_ytd,
                    ]
                )
                for g in groups
            )
            quality_score = 85.0 if valid else 55.0
            self.performance_tracker.record_success(
                DataSource.FINVIZ.value,
                (time.time() - start_time) * 1000,
                quality_score,
            )
            if quality_score >= self.min_quality_score:
                self.cache.set(cache_key, groups, "group_performance", quality_score)
            return groups
        except Exception as e:
            self.performance_tracker.record_error(DataSource.FINVIZ.value, str(e))
            logger.error(f"FinViz sector performance error: {e}")
            return []

    def get_finviz_news(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Get news and blog data from FinViz."""
        cache_key = "finviz_news"
        cached = self.cache.get(cache_key, "news")
        if cached:
            return cached
        adapter = self.adapters.get(DataSource.FINVIZ)
        if not adapter:
            return None
        start_time = time.time()
        try:
            news_data = adapter.get_news()
            if not news_data:
                return None
            quality_score = 90.0  # News data is generally reliable
            self.performance_tracker.record_success(
                DataSource.FINVIZ.value,
                (time.time() - start_time) * 1000,
                quality_score,
            )
            if quality_score >= self.min_quality_score:
                self.cache.set(cache_key, news_data, "news", quality_score)
            return news_data
        except Exception as e:
            self.performance_tracker.record_error(DataSource.FINVIZ.value, str(e))
            logger.error(f"FinViz news error: {e}")
            return None

    def get_finviz_insider_trading(self) -> Optional[pd.DataFrame]:
        """Get insider trading data from FinViz."""
        cache_key = "finviz_insider_trading"
        cached = self.cache.get(cache_key, "insider_trading")
        if cached is not None:
            return cached
        adapter = self.adapters.get(DataSource.FINVIZ)
        if not adapter:
            return None
        start_time = time.time()
        try:
            insider_data = adapter.get_insider_trading()
            if insider_data is None or insider_data.empty:
                return None
            # Validate data quality
            quality_score = 85.0  # Insider data is generally reliable
            self.performance_tracker.record_success(
                DataSource.FINVIZ.value,
                (time.time() - start_time) * 1000,
                quality_score,
            )
            if quality_score >= self.min_quality_score:
                self.cache.set(
                    cache_key, insider_data, "insider_trading", quality_score
                )
            return insider_data
        except Exception as e:
            self.performance_tracker.record_error(DataSource.FINVIZ.value, str(e))
            logger.error(f"FinViz insider trading error: {e}")
            return None

    def get_finviz_earnings(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Get earnings data from FinViz."""
        cache_key = "finviz_earnings"
        cached = self.cache.get(cache_key, "earnings")
        if cached:
            return cached
        adapter = self.adapters.get(DataSource.FINVIZ)
        if not adapter:
            return None
        start_time = time.time()
        try:
            earnings_data = adapter.get_earnings()
            if not earnings_data:
                return None
            # Validate data quality
            quality_score = 90.0  # Earnings data is generally reliable
            self.performance_tracker.record_success(
                DataSource.FINVIZ.value,
                (time.time() - start_time) * 1000,
                quality_score,
            )
            if quality_score >= self.min_quality_score:
                self.cache.set(cache_key, earnings_data, "earnings", quality_score)
            return earnings_data
        except Exception as e:
            self.performance_tracker.record_error(DataSource.FINVIZ.value, str(e))
            logger.error(f"FinViz earnings error: {e}")
            return None

    def get_finviz_forex(self) -> Optional[pd.DataFrame]:
        """Get forex performance data from FinViz."""
        cache_key = "finviz_forex"
        cached = self.cache.get(cache_key, "forex")
        if cached is not None:
            return cached
        adapter = self.adapters.get(DataSource.FINVIZ)
        if not adapter:
            return None
        start_time = time.time()
        try:
            forex_data = adapter.get_forex()
            if forex_data is None or forex_data.empty:
                return None
            # Validate data quality
            quality_score = 85.0  # Forex data is generally reliable
            self.performance_tracker.record_success(
                DataSource.FINVIZ.value,
                (time.time() - start_time) * 1000,
                quality_score,
            )
            if quality_score >= self.min_quality_score:
                self.cache.set(cache_key, forex_data, "forex", quality_score)
            return forex_data
        except Exception as e:
            self.performance_tracker.record_error(DataSource.FINVIZ.value, str(e))
            logger.error(f"FinViz forex error: {e}")
            return None

    def get_finviz_crypto(self) -> Optional[pd.DataFrame]:
        """Get crypto performance data from FinViz."""
        cache_key = "finviz_crypto"
        cached = self.cache.get(cache_key, "crypto")
        if cached is not None:
            return cached
        adapter = self.adapters.get(DataSource.FINVIZ)
        if not adapter:
            return None
        start_time = time.time()
        try:
            crypto_data = adapter.get_crypto()
            if crypto_data is None or crypto_data.empty:
                return None
            # Validate data quality
            quality_score = 85.0  # Crypto data is generally reliable
            self.performance_tracker.record_success(
                DataSource.FINVIZ.value,
                (time.time() - start_time) * 1000,
                quality_score,
            )
            if quality_score >= self.min_quality_score:
                self.cache.set(cache_key, crypto_data, "crypto", quality_score)
            return crypto_data
        except Exception as e:
            self.performance_tracker.record_error(DataSource.FINVIZ.value, str(e))
            logger.error(f"FinViz crypto error: {e}")
            return None

    # ---------------------------------------------------------------------
    # Google Trends (new orchestrator endpoint with persistent caching)
    # ---------------------------------------------------------------------
    def get_google_trends(
        self,
        keywords: Union[str, List[str]],
        timeframe: str = "now 7-d",
        geo: str = "US",
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Fetch Google Trends interest over time for the given keywords using pytrends,
        with SQLite-backed caching to minimize repeated fetches.

        Returns dict: keyword -> {timestamp_str: value}
        """
        # DEPRECATED: Google Trends feed is disabled upstream and this method will return None or empty
        # data depending on stub implementation. Retained only for backward compatibility. New
        # development should avoid calling this and instead rely on other sentiment/interest signals.
        try:
            from data_feeds.google_trends import fetch_google_trends as _fetch_trends
        except Exception as e:
            logger.error(f"Google Trends module import failed: {e}")
            return None

        # Build a stable cache key using persistent cache service if available
        params = {
            "keywords": keywords if isinstance(keywords, list) else [keywords],
            "timeframe": timeframe,
            "geo": geo,
        }
        key_hash = None
        if getattr(self, "persistent_cache", None):
            try:
                key_hash = self.persistent_cache.make_key("google_trends", params)  # type: ignore
                entry = self.persistent_cache.get(key_hash)  # type: ignore
                if entry and not entry.is_expired() and entry.payload_json:
                    return entry.payload_json  # type: ignore
            except Exception as e:
                logger.debug(f"Persistent cache read failed, proceeding to fetch: {e}")

        # Rate limiting nominally via SmartCache/none for Google (pytrends throttled internally)
        start_time = time.time()
        try:
            result = _fetch_trends(params["keywords"], timeframe=timeframe, geo=geo)
            # Normalize to strictly JSON-serializable dict[str, dict[str, int]]
            if isinstance(result, dict):
                normalized: Dict[str, Dict[str, int]] = {}
                for kw, series in result.items():
                    kw_str = str(kw)
                    norm_series: Dict[str, int] = {}
                    if isinstance(series, dict):
                        for ts, val in series.items():
                            # force keys to strings; pandas.Timestamp will stringify here
                            ts_s = str(ts)
                            # coerce values to int, safely
                            v_i = 0
                            try:
                                v_i = int(val) if val is not None else 0
                            except Exception:
                                try:
                                    v_i = int(float(val)) if val is not None else 0
                                except Exception:
                                    v_i = 0
                            norm_series[ts_s] = v_i
                    normalized[kw_str] = norm_series
                result = normalized
            quality_score = (
                90.0 if isinstance(result, dict) and len(result) > 0 else 60.0
            )
            # Record performance under GOOGLE_TRENDS
            self.performance_tracker.record_success(
                DataSource.GOOGLE_TRENDS.value,
                (time.time() - start_time) * 1000,
                quality_score,
            )
        except Exception as e:
            self.performance_tracker.record_error(
                DataSource.GOOGLE_TRENDS.value, str(e)
            )
            logger.error(f"Google Trends fetch failed: {e}")
            return None

        # Persist in SQLite cache
        if getattr(self, "persistent_cache", None) and isinstance(result, dict):
            try:
                if key_hash is None:
                    key_hash = self.persistent_cache.make_key("google_trends", params)  # type: ignore
                self.persistent_cache.set(  # type: ignore
                    key=key_hash,
                    endpoint="google_trends",
                    symbol=None,
                    ttl_seconds=getattr(self, "trends_ttl_seconds", 24 * 3600),
                    payload_json=result,
                    source=DataSource.GOOGLE_TRENDS.value,
                    metadata_json={"params": params},
                )
                logger.info(
                    f"[cache] google_trends persisted key={key_hash} ttl={getattr(self, 'trends_ttl_seconds', 24*3600)}s"
                )
            except Exception as e:
                # Emit first 120 chars of error for visibility
                logger.warning(f"[cache] google_trends write failed: {str(e)[:120]}")
        return result

    # ============================================================================
    # Enhanced Redis-Backed Methods for Improved Performance
    # ============================================================================

    def get_enhanced_quote(
        self, symbol: str, use_redis_cache: bool = True
    ) -> Optional[Quote]:
        """
        Get enhanced quote with Redis caching for improved performance.
        Falls back to regular quote if Redis is unavailable.
        """
        if not symbol:
            return None

        cache_key = f"enhanced_quote:{symbol}"

        # Try to get from Redis cache first
        if use_redis_cache and self.redis_cache:
            try:
                cached_data = self.redis_cache.get(cache_key, "quote")
                if cached_data:
                    logger.debug(f"Cache hit for enhanced quote: {symbol}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Redis cache read failed for quote {symbol}: {e}")

        # Get quote using regular method
        quote = self.get_quote(symbol)

        # Cache in Redis if available and successful
        if quote and use_redis_cache and self.redis_cache:
            try:
                # Cache for 5 minutes (300 seconds) for quotes
                self.redis_cache.set(
                    "quotes", symbol, quote, "quote", quality_score=90.0
                )
                logger.debug(f"Cached enhanced quote for {symbol}")
            except Exception as e:
                logger.warning(f"Redis cache write failed for quote {symbol}: {e}")

        return quote

    def get_enhanced_market_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        use_redis_cache: bool = True,
    ) -> Optional[MarketData]:
        """
        Get enhanced market data with Redis caching for improved performance.
        Falls back to regular market data if Redis is unavailable.
        """
        if not symbol:
            return None

        cache_key = f"enhanced_market_data:{symbol}:{period}:{interval}"

        # Try to get from Redis cache first
        if use_redis_cache and self.redis_cache:
            try:
                cached_data = self.redis_cache.get(cache_key, "market_data")
                if cached_data:
                    logger.debug(f"Cache hit for enhanced market data: {symbol}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Redis cache read failed for market data {symbol}: {e}")

        # Get market data using regular method
        market_data = self.get_market_data(symbol, period, interval)

        # Cache in Redis if available and successful
        if market_data and use_redis_cache and self.redis_cache:
            try:
                # Cache for 1 hour (3600 seconds) for market data
                ttl = (
                    3600 if interval in ["1d", "1wk", "1mo"] else 300
                )  # Longer TTL for daily data
                self.redis_cache.set(
                    "market_data",
                    f"{symbol}_{period}_{interval}",
                    market_data,
                    "market_data",
                    quality_score=85.0,
                )
                logger.debug(f"Cached enhanced market data for {symbol}")
            except Exception as e:
                logger.warning(
                    f"Redis cache write failed for market data {symbol}: {e}"
                )

        return market_data

    def get_enhanced_sentiment_data(
        self, symbol: str, use_redis_cache: bool = True
    ) -> Dict[str, SentimentData]:
        """
        Get enhanced sentiment data with Redis caching for improved performance.
        Falls back to regular sentiment data if Redis is unavailable.
        """
        if not symbol:
            return {}

        cache_key = f"enhanced_sentiment:{symbol}"

        # Try to get from Redis cache first
        if use_redis_cache and self.redis_cache:
            try:
                cached_data = self.redis_cache.get(cache_key, "sentiment")
                if cached_data:
                    logger.debug(f"Cache hit for enhanced sentiment: {symbol}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Redis cache read failed for sentiment {symbol}: {e}")

        # Get sentiment data using regular method
        sentiment_data = self.get_sentiment_data(symbol)

        # Cache in Redis if available and successful
        if sentiment_data and use_redis_cache and self.redis_cache:
            try:
                # Cache for 15 minutes (900 seconds) for sentiment data
                self.redis_cache.set(
                    "sentiment", symbol, sentiment_data, "sentiment", quality_score=80.0
                )
                logger.debug(f"Cached enhanced sentiment for {symbol}")
            except Exception as e:
                logger.warning(f"Redis cache write failed for sentiment {symbol}: {e}")

        return sentiment_data

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics including Redis, SmartCache, and performance metrics.
        """
        stats = {
            "timestamp": datetime.now().isoformat(),
            "smart_cache": {
                "size": len(self.cache.cache),
                "hit_rate": 0.0,
                "miss_rate": 0.0,
                "evictions": 0,
            },
            "redis_cache": {
                "available": self.redis_cache is not None,
                "connected": False,
                "size": 0,
                "memory_usage": 0,
            },
            "performance_tracker": {
                "sources_tracked": len(self.performance_tracker.metrics),
                "total_requests": 0,
                "total_errors": 0,
            },
        }

        # Add SmartCache statistics
        try:
            # SmartCache doesn't track hits/misses internally, so we provide basic info
            stats["smart_cache"]["size"] = len(self.cache.cache)
            stats["smart_cache"]["hit_rate"] = 0.0  # Not tracked by SmartCache
            stats["smart_cache"]["miss_rate"] = 0.0  # Not tracked by SmartCache
        except Exception as e:
            logger.warning(f"Failed to get SmartCache statistics: {e}")

        # Add Redis cache statistics
        if self.redis_cache:
            try:
                redis_stats = self.redis_cache.get_cache_stats()
                if redis_stats and "analytics" in redis_stats:
                    analytics = redis_stats["analytics"]
                    stats["redis_cache"].update(
                        {
                            "connected": True,
                            "hit_rate": analytics.get("hit_rate", 0.0),
                            "total_requests": analytics.get("total_requests", 0),
                            "hits": analytics.get("hits", 0),
                            "misses": analytics.get("misses", 0),
                            "memory_cache_items": redis_stats.get(
                                "memory_cache_items", 0
                            ),
                            "compression_savings": analytics.get(
                                "compression_savings_bytes", 0
                            ),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to get Redis cache statistics: {e}")

        # Add performance tracker statistics
        try:
            total_requests = 0
            total_errors = 0
            for source_metrics in self.performance_tracker.metrics.values():
                total_requests += source_metrics.get(
                    "success_count", 0
                ) + source_metrics.get("error_count", 0)
                total_errors += source_metrics.get("error_count", 0)

            stats["performance_tracker"]["total_requests"] = total_requests
            stats["performance_tracker"]["total_errors"] = total_errors
            if total_requests > 0:
                stats["performance_tracker"]["error_rate"] = (
                    total_errors / total_requests
                )
        except Exception as e:
            logger.warning(f"Failed to get performance tracker statistics: {e}")

        return stats

    def warm_up_popular_tickers(self, symbols: Optional[List[str]] = None):
        """
        Warm up cache with popular ticker data for improved performance.
        If no symbols provided, uses a default list of popular tickers.
        """
        if symbols is None:
            # Default popular tickers
            symbols = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "NFLX",
                "AMD",
                "INTC",
                "QCOM",
                "CRM",
                "ADBE",
                "PYPL",
                "ZM",
                "SPOT",
                "SPY",
                "QQQ",
                "IWM",
                "VTI",
                "BTC-USD",
                "ETH-USD",
            ]

        if not symbols:
            logger.warning("No symbols provided for cache warming")
            return

        logger.info(f"Starting cache warm-up for {len(symbols)} symbols")

        # Use cache warming service if available
        if self.cache_warming_service:
            try:
                # Use the private _warm_up_at_market_open method as it's the most appropriate
                self.cache_warming_service._warm_up_at_market_open()
                logger.info("Cache warming completed via cache warming service")
                return
            except Exception as e:
                logger.warning(
                    f"Cache warming service failed, falling back to manual warming: {e}"
                )

        # Manual cache warming
        warmed_count = 0
        for symbol in symbols:
            try:
                # Warm up quote data
                quote = self.get_enhanced_quote(symbol, use_redis_cache=True)
                if quote:
                    warmed_count += 1

                # Warm up basic market data
                market_data = self.get_enhanced_market_data(
                    symbol, period="1y", interval="1d", use_redis_cache=True
                )

                # Warm up sentiment data (but don't count failures heavily)
                try:
                    self.get_enhanced_sentiment_data(symbol, use_redis_cache=True)
                except Exception:
                    pass  # Sentiment data might not be available for all symbols

            except Exception as e:
                logger.warning(f"Failed to warm cache for {symbol}: {e}")

        logger.info(
            f"Cache warm-up completed: {warmed_count}/{len(symbols)} symbols processed"
        )

    def clear_enhanced_cache(self):
        """
        Clear enhanced cache layers including Redis and SmartCache.
        """
        cleared_items = 0

        # Clear Redis cache
        if self.redis_cache:
            try:
                self.redis_cache.clear_cache()
                logger.info("Cleared Redis cache")
                cleared_items += 1  # Count successful clear operation
            except Exception as e:
                logger.error(f"Failed to clear Redis cache: {e}")

        # Clear SmartCache
        try:
            self.cache.cache.clear()  # Clear the internal cache dictionary
            logger.info("Cleared SmartCache")
        except Exception as e:
            logger.error(f"Failed to clear SmartCache: {e}")

        # Clear persistent cache if available
        if self.persistent_cache:
            try:
                # Note: This would clear the entire persistent cache database
                # In practice, you might want to be more selective
                logger.info(
                    "Persistent cache clear would require database-specific operations"
                )
            except Exception as e:
                logger.error(f"Failed to clear persistent cache: {e}")

        logger.info(
            f"Enhanced cache clear completed. Total items cleared: {cleared_items}"
        )


# ----------------------------------------------------------------------------
# New delegation methods to ConsolidatedDataFeed for compatibility during consolidation
# ----------------------------------------------------------------------------
def get_company_info(symbol: str) -> Optional[CompanyInfo]:
    """Get company info via orchestrator delegating to ConsolidatedDataFeed."""
    return get_orchestrator().get_company_info(symbol)


def get_news(symbol: str, limit: int = 10) -> List[NewsItem]:
    """Get news via orchestrator delegating to ConsolidatedDataFeed."""
    return get_orchestrator().get_news(symbol, limit)


def get_multiple_quotes(symbols: List[str]):
    """Get multiple quotes via orchestrator delegating to ConsolidatedDataFeed (loosely typed)."""
    return get_orchestrator().get_multiple_quotes(symbols)


def get_financial_statements(symbol: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Get financial statements via orchestrator delegating to ConsolidatedDataFeed."""
    return get_orchestrator().get_financial_statements(symbol)


# ----------------------------------------------------------------------------
# Enhanced unified interface functions (Backward Compatibility)
# ----------------------------------------------------------------------------

# Global orchestrator instance
_global_orchestrator = None


def get_orchestrator() -> DataFeedOrchestrator:
    """Get the global data feed orchestrator instance"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = DataFeedOrchestrator()
    return _global_orchestrator


# ----------------------------------------------------------------------------
# Investiny compact helpers (unified interface)
# ----------------------------------------------------------------------------
def get_investiny_daily_oc(
    symbols: List[str], date_range_or_start: str, end_date: Optional[str] = None
) -> Dict[str, str]:
    """
    Return compact daily open/close strings for each symbol using Investiny:
      {'AAPL': 'YYYY-MM-DD o:x, c:y YYYY-MM-DD o:x, c:y ...', ...}

    Accepts either 'YYYY-MM-DD-YYYY-MM-DD' or ('YYYY-MM-DD', 'YYYY-MM-DD')
    """
    out: Dict[str, str] = {}
    if investiny_format_daily_oc is None:
        return out
    for s in symbols or []:
        try:
            out[s] = investiny_format_daily_oc(s, date_range_or_start, end_date)  # type: ignore
        except Exception as e:
            out[s] = f"error:{e}"
    return out


def get_quote(symbol: str) -> Optional[Quote]:
    """Get real-time quote (unified interface)"""
    return get_orchestrator().get_quote(symbol)


def get_market_data(
    symbol: str, period: str = "1y", interval: str = "1d"
) -> Optional[MarketData]:
    """Get historical market data (unified interface)"""
    return get_orchestrator().get_market_data(symbol, period, interval)


def get_sentiment_data(symbol: str) -> Dict[str, SentimentData]:
    """Get sentiment data (unified interface)"""
    return get_orchestrator().get_sentiment_data(symbol)


def get_system_health() -> Dict[str, Any]:
    """Get system health report (unified interface)"""
    return get_orchestrator().validate_system_health()


def get_market_breadth() -> Optional[MarketBreadth]:
    """Get market breadth (unified interface)"""
    return get_orchestrator().get_market_breadth()


def get_sector_performance() -> List[GroupPerformance]:
    """Get sector performance (unified interface)"""
    return get_orchestrator().get_sector_performance()


def get_finviz_news() -> Optional[Dict[str, pd.DataFrame]]:
    """Get FinViz news data (unified interface)"""
    return get_orchestrator().get_finviz_news()


def get_finviz_insider_trading() -> Optional[pd.DataFrame]:
    """Get FinViz insider trading data (unified interface)"""
    return get_orchestrator().get_finviz_insider_trading()


# ----------------------------------------------------------------------------
# Enhanced unified interface functions (Backward Compatibility)
# ----------------------------------------------------------------------------


def get_finviz_earnings() -> Optional[Dict[str, pd.DataFrame]]:
    """Get FinViz earnings data (unified interface)"""
    return get_orchestrator().get_finviz_earnings()


def get_finviz_forex() -> Optional[pd.DataFrame]:
    """Get FinViz forex data (unified interface)"""
    return get_orchestrator().get_finviz_forex()


def get_finviz_crypto() -> Optional[pd.DataFrame]:
    """Get FinViz crypto data (unified interface)"""
    return get_orchestrator().get_finviz_crypto()


def get_enhanced_quote(symbol: str, use_redis_cache: bool = True) -> Optional[Quote]:
    """Get enhanced quote with Redis caching (unified interface)"""
    return get_orchestrator().get_enhanced_quote(symbol, use_redis_cache)


def get_enhanced_market_data(
    symbol: str, period: str = "1y", interval: str = "1d", use_redis_cache: bool = True
) -> Optional[MarketData]:
    """Get enhanced market data with Redis caching (unified interface)"""
    return get_orchestrator().get_enhanced_market_data(
        symbol, period, interval, use_redis_cache
    )


def get_enhanced_sentiment_data(
    symbol: str, use_redis_cache: bool = True
) -> Dict[str, SentimentData]:
    """Get enhanced sentiment data with Redis caching (unified interface)"""
    return get_orchestrator().get_enhanced_sentiment_data(symbol, use_redis_cache)


def get_cache_statistics() -> Dict[str, Any]:
    """Get comprehensive cache statistics (unified interface)"""
    return get_orchestrator().get_cache_statistics()


def warm_up_popular_tickers(symbols: Optional[List[str]] = None):
    """Warm up cache with popular ticker data (unified interface)"""
    return get_orchestrator().warm_up_popular_tickers(symbols)


def clear_enhanced_cache():
    """Clear enhanced cache layers (unified interface)"""
    return get_orchestrator().clear_enhanced_cache()


def get_advanced_sentiment(self, symbol: str) -> Optional[SentimentData]:
    """Legacy compatibility wrapper for get_advanced_sentiment_data."""
    return self.get_advanced_sentiment_data(symbol)
