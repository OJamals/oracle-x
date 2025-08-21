from __future__ import annotations
import os
import requests
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, Any, List, Tuple, Union
import time
import logging
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import random

# from data_feeds.data_feed_orchestrator import Quote, MarketData  # Reuse orchestrator models


class TwelveDataError(Exception):
    pass


class TwelveDataNotFound(TwelveDataError):
    pass


class TwelveDataThrottled(TwelveDataError):
    pass


class TwelveDataTimeout(TwelveDataError):
    pass


class TwelveDataConnectionError(TwelveDataError):
    pass


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None or value == "" or value == "None":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "" or value == "None":
            return None
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _to_utc(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    # Twelve Data returns ISO strings
    try:
        # e.g. "2024-05-31 16:00:00"
        s = str(ts).replace("T", " ")
        # Try common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    except Exception:
        pass
    return None


@dataclass
class HealthMetrics:
    """Health metrics for API endpoint monitoring"""
    endpoint: str
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    last_success_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.success_count if self.success_count > 0 else 0.0

    @property
    def health_score(self) -> float:
        """Calculate health score based on success rate and latency"""
        # Base score from success rate
        score = self.success_rate * 100

        # Penalty for high latency (over 2000ms)
        if self.avg_latency_ms > 2000:
            penalty = min(50, (self.avg_latency_ms - 2000) / 100)
            score -= penalty

        # Penalty for consecutive failures
        if self.consecutive_failures > 0:
            penalty = min(30, self.consecutive_failures * 5)
            score -= penalty

        return max(0, score)


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 30.0  # Maximum delay in seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add random jitter to prevent thundering herd


@dataclass
class TimeoutConfig:
    """Timeout configurations for different operations"""
    quote_timeout: Tuple[float, float] = (3.0, 10.0)  # connect, read for quotes
    market_data_timeout: Tuple[float, float] = (5.0, 15.0)  # connect, read for market data
    batch_timeout: Tuple[float, float] = (8.0, 25.0)  # connect, read for batch operations


class TwelveDataAdapterEnhanced:
    """
    Enhanced TwelveData adapter with intelligent retry logic, health monitoring,
    and optimized performance.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.twelvedata.com",
        session: Optional[requests.Session] = None,
        retry_config: Optional[RetryConfig] = None,
        timeout_config: Optional[TimeoutConfig] = None
    ):
        self.api_key = api_key or os.getenv("TWELVEDATA_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

        # Configuration
        self.retry_config = retry_config or RetryConfig()
        self.timeout_config = timeout_config or TimeoutConfig()

        # Health monitoring
        self.health_metrics: Dict[str, HealthMetrics] = defaultdict(
            lambda: HealthMetrics(endpoint="default")
        )
        self.health_lock = threading.RLock()

        # Request batching
        self.batch_size = 10  # Maximum symbols per batch request
        self.batch_delay = 0.1  # Delay between batch requests

        # Logger
        self.logger = logging.getLogger(__name__)

        # Initialize endpoint-specific health tracking
        self._init_health_tracking()

    def _init_health_tracking(self):
        """Initialize health tracking for different endpoints"""
        endpoints = ["quote", "time_series", "batch_quote"]
        for endpoint in endpoints:
            self.health_metrics[endpoint] = HealthMetrics(endpoint=endpoint)

    def _record_success(self, endpoint: str, latency_ms: float):
        """Record successful API call"""
        with self.health_lock:
            metrics = self.health_metrics[endpoint]
            metrics.success_count += 1
            metrics.total_latency_ms += latency_ms
            metrics.last_success_time = datetime.now(timezone.utc)
            metrics.consecutive_successes += 1
            metrics.consecutive_failures = 0

    def _record_error(self, endpoint: str, error_type: str):
        """Record failed API call"""
        with self.health_lock:
            metrics = self.health_metrics[endpoint]
            metrics.error_count += 1
            metrics.last_error_time = datetime.now(timezone.utc)
            metrics.consecutive_failures += 1
            metrics.consecutive_successes = 0

    def _get_dynamic_timeout(self, endpoint: str) -> Tuple[float, float]:
        """Get dynamic timeout based on endpoint health"""
        with self.health_lock:
            metrics = self.health_metrics[endpoint]

            # Base timeout
            if endpoint == "quote":
                base_timeout = self.timeout_config.quote_timeout
            elif endpoint == "time_series":
                base_timeout = self.timeout_config.market_data_timeout
            else:
                base_timeout = self.timeout_config.batch_timeout

            # Adjust based on health
            if metrics.consecutive_failures > 2:
                # Increase timeout for unhealthy endpoints
                return (base_timeout[0] * 1.5, base_timeout[1] * 1.5)
            elif metrics.avg_latency_ms > 5000:  # Over 5 seconds average
                # Increase timeout for slow endpoints
                return (base_timeout[0] * 1.2, base_timeout[1] * 1.2)

            return base_timeout

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if request should be retried"""
        if attempt >= self.retry_config.max_retries:
            return False

        # Don't retry authentication errors
        if "authentication" in str(exception).lower() or "401" in str(exception):
            return False

        # Don't retry not found errors
        if isinstance(exception, TwelveDataNotFound):
            return False

        # Retry rate limits, timeouts, and connection errors
        if isinstance(exception, (TwelveDataThrottled, TwelveDataTimeout, TwelveDataConnectionError)):
            return True

        # Retry for other API errors
        if isinstance(exception, TwelveDataError):
            return True

        return False

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt)

        # Apply maximum delay
        delay = min(delay, self.retry_config.max_delay)

        # Add jitter
        if self.retry_config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    def _request_with_retry(self, path: str, params: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Execute request with intelligent retry logic"""
        last_exception = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                start_time = time.time()
                timeout = self._get_dynamic_timeout(endpoint)

                result = self._request(path, params, timeout)
                latency_ms = (time.time() - start_time) * 1000

                # Record success
                self._record_success(endpoint, latency_ms)

                return result

            except Exception as e:
                last_exception = e

                # Record error
                self._record_error(endpoint, type(e).__name__)

                # Check if we should retry
                if not self._should_retry(e, attempt):
                    raise e

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                self.logger.warning(
                    f"TwelveData {endpoint} request failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )

                time.sleep(delay)

        # If we get here, all retries failed
        raise last_exception

    def _request(self, path: str, params: Dict[str, Any], timeout: Tuple[float, float]) -> Dict[str, Any]:
        """Core request method with enhanced error handling"""
        url = f"{self.base_url}/{path.lstrip('/')}"
        qp = dict(params or {})
        if self.api_key:
            qp["apikey"] = self.api_key

        try:
            resp = self.session.get(url, params=qp, timeout=timeout)
        except requests.exceptions.Timeout:
            raise TwelveDataTimeout("Request timeout")
        except requests.exceptions.ConnectionError:
            raise TwelveDataConnectionError("Connection error")
        except requests.exceptions.RequestException as e:
            raise TwelveDataError(f"Request failed: {e}")

        if resp.status_code == 429:
            raise TwelveDataThrottled("Twelve Data rate limit exceeded (429)")
        if resp.status_code == 404:
            raise TwelveDataNotFound("Resource not found (404)")

        try:
            data = resp.json()
        except Exception as e:
            raise TwelveDataError(f"Invalid JSON response: {e}")

        # Twelve Data error format: {"status":"error","message":"..."}
        if isinstance(data, dict) and data.get("status") == "error":
            msg = data.get("message", "Unknown provider error")
            if "not found" in msg.lower():
                raise TwelveDataNotFound(msg)
            if "limit" in msg.lower() or "rate" in msg.lower() or "quota" in msg.lower():
                raise TwelveDataThrottled(msg)
            if "timeout" in msg.lower():
                raise TwelveDataTimeout(msg)
            raise TwelveDataError(msg)

        return data

    def _models(self):
        from data_feeds.data_feed_orchestrator import Quote, MarketData
        return Quote, MarketData

    def get_quote(self, symbol: str) -> Optional[Any]:
        """Get real-time quote with enhanced error handling and retry logic"""
        try:
            data = self._request_with_retry("/quote", {"symbol": symbol}, "quote")

            if not isinstance(data, dict):
                return None

            price = _to_decimal(data.get("close")) or _to_decimal(data.get("price"))
            prev_close = _to_decimal(data.get("previous_close"))
            change = _to_decimal(data.get("change"))
            change_pct = _to_decimal(data.get("percent_change"))
            if change is None and price is not None and prev_close is not None:
                change = price - prev_close
            if change_pct is None and change is not None and prev_close and prev_close != 0:
                try:
                    change_pct = (change / prev_close) * Decimal("100")
                except Exception:
                    change_pct = None

            ts = _to_utc(data.get("timestamp")) or datetime.now(timezone.utc)

            Quote, _MarketData = self._models()
            q = Quote(
                symbol=symbol,
                price=price or Decimal("0"),
                change=change or Decimal("0"),
                change_percent=change_pct or Decimal("0"),
                volume=_to_int(data.get("volume")) or 0,
                market_cap=_to_int(data.get("market_cap")),
                pe_ratio=_to_decimal(data.get("pe")),
                day_low=_to_decimal(data.get("low")),
                day_high=_to_decimal(data.get("high")),
                year_low=_to_decimal(data.get("fifty_two_week", {}).get("low") if isinstance(data.get("fifty_two_week"), dict) else None),
                year_high=_to_decimal(data.get("fifty_two_week", {}).get("high") if isinstance(data.get("fifty_two_week"), dict) else None),
                timestamp=ts,
                source="twelve_data",
                quality_score=None,
            )

            # Validate quality and set quality score
            try:
                from data_feeds.data_feed_orchestrator import DataValidator
                quality_score, _ = DataValidator.validate_quote(q)
                q.quality_score = quality_score
            except Exception:
                # If validation fails, set a reasonable default quality score
                q.quality_score = 80.0  # Assume good quality data from TwelveData

            return q

        except Exception as e:
            self.logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def get_market_data(self, symbol: str, period: str = "1y", interval: str = "1d", outputsize: Optional[int] = None, start: Optional[datetime] = None, end: Optional[datetime] = None) -> Optional[Any]:
        """Get historical market data with enhanced error handling and retry logic"""
        # Canonicalize interval: accept common orchestrator inputs and map to Twelve Data
        key = (interval or "").lower().strip()
        # Normalize synonyms first
        synonyms = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "60m": "1h",
            "1hr": "1h",
            "1w": "1wk",
            "1week": "1wk",
            "1month": "1mo",
            "daily": "1d",
            "hourly": "1h",
        }
        key = synonyms.get(key, key)
        interval_map = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "1h": "1h",
            "1d": "1day",
            "1day": "1day",
            "1wk": "1week",
            "1mo": "1month",
        }
        td_interval = interval_map.get(key)
        if not td_interval:
            supported = sorted(set(interval_map.keys()) | set(synonyms.keys()))
            raise TwelveDataError(f"Unsupported interval '{interval}'. Supported aliases: {supported}")

        # Derive outputsize from period if not explicitly set
        # Twelve Data supports outputsize up to 5000 for time_series
        if outputsize is None:
            period_key = (period or "").lower().strip()
            period_to_bars = {
                "1d": 1,
                "5d": 5,
                "1w": 7,
                "1mo": 31,
                "3mo": 93,
                "6mo": 186,
                "1y": 372,
                "2y": 744,
                "5y": 1860,
                "10y": 3720,
                "ytd": 260,  # approx trading days
                "max": 5000,
            }
            # Scale by interval granularity
            base = period_to_bars.get(period_key, 5000)
            # If intraday intervals, increase bar count within reasonable bounds
            if td_interval in ("1min", "5min", "15min", "30min", "1h"):
                # Approximate 252 trading days/year * 6.5 hours/day = 390 minutes/day
                mult = {
                    "1min": 390,
                    "5min": 78,
                    "15min": 26,
                    "30min": 13,
                    "1h": 7,
                }[td_interval]
                base = min(5000, base * mult)
            outputsize = min(5000, max(1, int(base)))

        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": td_interval,
            "outputsize": outputsize,
            "format": "JSON",
            "timezone": "UTC",
        }
        if start:
            params["start_date"] = start.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if end:
            params["end_date"] = end.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        try:
            data = self._request_with_retry("/time_series", params, "time_series")

            # Expected structure: {"values":[{"datetime":"...","open":"...","high":"...","low":"...","close":"...","volume":"..."}], ...}
            values: List[Dict[str, Any]] = []
            if isinstance(data, dict):
                values = data.get("values") or []
            if not values:
                return None

            # Build pandas DataFrame like orchestrator expects
            import pandas as pd

            rows = []
            for v in reversed(values):  # reverse to chronological increasing
                dt = _to_utc(v.get("datetime"))
                if not dt:
                    continue
                rows.append(
                    {
                        "Datetime": dt,
                        "Open": _to_decimal(v.get("open")),
                        "High": _to_decimal(v.get("high")),
                        "Low": _to_decimal(v.get("low")),
                        "Close": _to_decimal(v.get("close")),
                        "Volume": _to_int(v.get("volume")),
                    }
                )
            if not rows:
                return None

            df = pd.DataFrame(rows)
            df.set_index("Datetime", inplace=True)

            _Quote, MarketData = self._models()
            md = MarketData(
                symbol=symbol,
                data=df,
                timeframe=td_interval,
                source="twelve_data",
                timestamp=df.index[-1].to_pydatetime() if len(df.index) else datetime.now(timezone.utc),
                quality_score=100.0,  # Validator in orchestrator will adjust if needed
            )
            return md

        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return None

    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Optional[Any]]:
        """Get batch quotes with intelligent batching and retry logic"""
        if not symbols:
            return {}

        results = {}

        # Split into batches
        for i in range(0, len(symbols), self.batch_size):
            batch_symbols = symbols[i:i + self.batch_size]

            try:
                # Add small delay between batches to avoid overwhelming the API
                if i > 0:
                    time.sleep(self.batch_delay)

                # Use batch endpoint if available, otherwise fall back to individual calls
                batch_result = self._get_batch_quotes_batch(batch_symbols)
                results.update(batch_result)

            except Exception as e:
                self.logger.warning(f"Batch quote request failed for {batch_symbols}: {e}")

                # Fall back to individual calls for failed batch
                for symbol in batch_symbols:
                    try:
                        quote = self.get_quote(symbol)
                        results[symbol] = quote
                    except Exception as individual_e:
                        self.logger.error(f"Individual quote failed for {symbol}: {individual_e}")
                        results[symbol] = None

        return results

    def _get_batch_quotes_batch(self, symbols: List[str]) -> Dict[str, Optional[Any]]:
        """Get batch quotes using TwelveData batch endpoint"""
        # TwelveData doesn't have a true batch quote endpoint, so we'll simulate it
        # by making multiple requests in parallel or sequential with individual calls
        results = {}

        # For now, implement as sequential calls (could be enhanced with concurrent calls)
        for symbol in symbols:
            try:
                quote = self.get_quote(symbol)
                results[symbol] = quote
            except Exception as e:
                self.logger.warning(f"Quote failed for {symbol} in batch: {e}")
                results[symbol] = None

        return results

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the adapter"""
        with self.health_lock:
            status = {
                "overall_health_score": self._calculate_overall_health(),
                "endpoints": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "max_retries": self.retry_config.max_retries,
                    "batch_size": self.batch_size,
                    "base_url": self.base_url,
                    "has_api_key": bool(self.api_key)
                }
            }

            for endpoint, metrics in self.health_metrics.items():
                status["endpoints"][endpoint] = {
                    "success_count": metrics.success_count,
                    "error_count": metrics.error_count,
                    "success_rate": round(metrics.success_rate * 100, 2),
                    "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                    "health_score": round(metrics.health_score, 2),
                    "consecutive_failures": metrics.consecutive_failures,
                    "consecutive_successes": metrics.consecutive_successes,
                    "last_success_time": metrics.last_success_time.isoformat() if metrics.last_success_time else None,
                    "last_error_time": metrics.last_error_time.isoformat() if metrics.last_error_time else None,
                }

            return status

    def _calculate_overall_health(self) -> float:
        """Calculate overall health score across all endpoints"""
        if not self.health_metrics:
            return 100.0

        total_weight = 0
        weighted_score = 0

        # Weight different endpoints differently
        weights = {
            "quote": 0.5,      # Quotes are most critical
            "time_series": 0.4, # Market data is also important
            "batch_quote": 0.1  # Batch operations are less critical
        }

        for endpoint, metrics in self.health_metrics.items():
            weight = weights.get(endpoint, 0.1)
            total_weight += weight
            weighted_score += metrics.health_score * weight

        return weighted_score / total_weight if total_weight > 0 else 100.0

    def reset_health_metrics(self):
        """Reset health metrics (useful for testing or recovery)"""
        with self.health_lock:
            self.health_metrics.clear()
            self._init_health_tracking()
            self.logger.info("Health metrics reset")

    def get_recommended_timeout(self, endpoint: str) -> Tuple[float, float]:
        """Get recommended timeout based on historical performance"""
        with self.health_lock:
            metrics = self.health_metrics.get(endpoint, HealthMetrics(endpoint))

            if metrics.success_count == 0:
                return self._get_dynamic_timeout(endpoint)

            # Base timeout on 95th percentile latency
            avg_latency = metrics.avg_latency_ms

            # Add buffer based on variance (estimated)
            buffer_factor = 1.5 if metrics.consecutive_failures > 0 else 1.2

            connect_timeout = max(2.0, avg_latency / 1000 * 0.3)  # 30% of avg response time
            read_timeout = max(5.0, avg_latency / 1000 * buffer_factor)

            return (connect_timeout, read_timeout)


# Backward compatibility: alias the enhanced adapter
TwelveDataAdapter = TwelveDataAdapterEnhanced