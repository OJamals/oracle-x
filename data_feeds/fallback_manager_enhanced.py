"""
Enhanced Fallback Manager for Data Sources
Handles intelligent fallback with improved TwelveData prioritization and performance tracking.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class FallbackReason(Enum):
    """Reasons why a data source might be in fallback mode"""
    RATE_LIMITED = "rate_limited"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    AUTHENTICATION_ERROR = "authentication_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class FallbackState:
    """Tracks the fallback state of a data source"""
    source: str
    reason: FallbackReason
    start_time: datetime
    retry_count: int = 0
    last_retry_time: Optional[datetime] = None
    backoff_seconds: float = 60.0  # Start with 1 minute
    max_backoff_seconds: float = 3600.0  # Max 1 hour
    recovery_check_interval: float = 300.0  # Check recovery every 5 minutes

    def should_retry(self) -> bool:
        """Check if we should attempt a retry"""
        if self.last_retry_time is None:
            return True
        elapsed = (datetime.now() - self.last_retry_time).total_seconds()
        return elapsed >= self.backoff_seconds

    def should_check_recovery(self) -> bool:
        """Check if we should test for recovery"""
        if self.last_retry_time is None:
            return True
        elapsed = (datetime.now() - self.last_retry_time).total_seconds()
        return elapsed >= self.recovery_check_interval

    def record_retry_attempt(self, success: bool = False):
        """Record a retry attempt and update backoff"""
        self.last_retry_time = datetime.now()
        if success:
            # Reset on success
            self.retry_count = 0
            self.backoff_seconds = 60.0
        else:
            self.retry_count += 1
            # Exponential backoff with jitter
            self.backoff_seconds = min(
                self.max_backoff_seconds,
                self.backoff_seconds * (2 ** min(self.retry_count, 6)) * (0.5 + 0.5 * (self.retry_count % 3) / 2)
            )

@dataclass
class FallbackConfig:
    """Enhanced configuration for fallback behavior with TwelveData optimization"""
    max_retries: int = 10
    initial_backoff_seconds: float = 60.0
    max_backoff_seconds: float = 3600.0
    recovery_check_interval: float = 300.0
    rate_limit_detection_window: int = 300  # 5 minutes window for rate limit detection
    rate_limit_threshold: int = 5  # Number of rate limit errors before fallback
    permanent_fallback_duration: float = 1800.0  # 30 minutes before recovery attempt

    # Enhanced priority orders with TwelveData prioritized for reliability
    quote_fallback_order: List[str] = field(default_factory=lambda: [
        "twelve_data", "yfinance", "finviz", "iex_cloud", "finnhub"
    ])
    market_data_fallback_order: List[str] = field(default_factory=lambda: [
        "twelve_data", "yfinance", "iex_cloud", "finnhub"
    ])
    news_fallback_order: List[str] = field(default_factory=lambda: [
        "yahoo_news", "finviz", "reddit", "twitter"
    ])

    # TwelveData-specific optimizations
    twelve_data_retry_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 5,  # More retries for paid service
        "base_delay": 2.0,  # Shorter initial delay
        "max_delay": 60.0   # Shorter max delay
    })

class FallbackManagerEnhanced:
    """Enhanced fallback manager with TwelveData optimization"""

    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.fallback_states: Dict[str, FallbackState] = {}
        self.rate_limit_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.RLock()

        # Enhanced performance tracking for source ranking
        self.source_success_streak: Dict[str, int] = defaultdict(int)
        self.source_response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

    def record_error(self, source: str, error: Exception, error_type: str = "unknown") -> bool:
        """
        Record an error from a data source and determine if fallback is needed.
        Returns True if source should be put in fallback mode.
        """
        with self.lock:
            current_time = datetime.now()

            # Determine fallback reason
            reason = self._classify_error(error, error_type)

            # Record in rate limit history if applicable
            if reason == FallbackReason.RATE_LIMITED:
                self.rate_limit_history[source].append(current_time)

                # Check if we should enter fallback mode
                recent_errors = [
                    t for t in self.rate_limit_history[source]
                    if (current_time - t).total_seconds() <= self.config.rate_limit_detection_window
                ]

                if len(recent_errors) >= self.config.rate_limit_threshold:
                    logger.warning(f"Rate limit threshold exceeded for {source}. Entering fallback mode.")
                    self._enter_fallback_mode(source, reason)
                    return True
            else:
                # For non-rate-limit errors, enter fallback mode immediately for severe errors
                if reason in [FallbackReason.AUTHENTICATION_ERROR, FallbackReason.SERVICE_UNAVAILABLE]:
                    logger.warning(f"Critical error for {source}: {reason}. Entering fallback mode.")
                    self._enter_fallback_mode(source, reason)
                    return True

                # For TwelveData, be more lenient with retries
                if source == "twelve_data" and reason in [FallbackReason.API_ERROR, FallbackReason.TIMEOUT]:
                    # Allow more tolerance for TwelveData
                    return False

            return False

    def record_success(self, source: str, response_time: float = 0.0):
        """Record a successful response from a data source"""
        with self.lock:
            # Record performance
            self.performance_history[source].append({
                'timestamp': datetime.now(),
                'response_time': response_time,
                'success': True
            })

            # Record response time for ranking
            self.source_response_times[source].append(response_time)
            self.source_success_streak[source] += 1

            # If source was in fallback mode, check for recovery
            if source in self.fallback_states:
                self.fallback_states[source].record_retry_attempt(success=True)
                logger.info(f"Data source {source} recovered successfully. Removing from fallback mode.")
                del self.fallback_states[source]

    def get_source_ranking(self) -> List[Tuple[str, float]]:
        """Get sources ranked by performance and reliability"""
        with self.lock:
            rankings = []

            for source in set(list(self.performance_history.keys()) + list(self.source_response_times.keys())):
                # Calculate success rate
                history = self.performance_history.get(source, [])
                if not history:
                    success_rate = 1.0 if source == "twelve_data" else 0.5  # Bias towards TwelveData
                else:
                    success_count = sum(1 for h in history if h['success'])
                    success_rate = success_count / len(history)

                # Calculate average response time
                response_times = list(self.source_response_times.get(source, []))
                if not response_times:
                    avg_response_time = 1000 if source == "twelve_data" else 2000  # Bias towards TwelveData
                else:
                    avg_response_time = sum(response_times) / len(response_times)

                # Calculate streak bonus
                streak_bonus = min(self.source_success_streak.get(source, 0) * 0.01, 0.2)

                # Calculate final score (lower is better)
                # Success rate (70%), response time (20%), streak bonus (10%)
                score = (1 - success_rate) * 0.7 + (avg_response_time / 1000) * 0.2 - streak_bonus

                rankings.append((source, score))

            return sorted(rankings, key=lambda x: x[1])

    def get_optimized_fallback_order(self, data_type: str = "quote", preferred_sources: Optional[List[str]] = None) -> List[str]:
        """
        Get optimized fallback order based on performance and current fallback states.
        """
        with self.lock:
            # Get base order
            if data_type == "market_data":
                base_order = self.config.market_data_fallback_order.copy()
            elif data_type == "news":
                base_order = self.config.news_fallback_order.copy()
            else:
                base_order = self.config.quote_fallback_order.copy()

            # Use preferred sources if provided, but still apply fallback logic
            if preferred_sources:
                # Start with preferred sources, then add others
                order = preferred_sources.copy()
                for source in base_order:
                    if source not in order:
                        order.append(source)
            else:
                order = base_order.copy()

            # Get performance-based ranking
            ranking = self.get_source_ranking()
            source_scores = {source: score for source, score in ranking}

            # Sort order by performance scores, with fallback penalties
            def sort_key(source):
                base_score = source_scores.get(source, 1.0)  # Default score

                # Penalty for sources in fallback mode
                if self.is_in_fallback(source):
                    if self.can_retry(source):
                        base_score += 0.5  # Moderate penalty
                    else:
                        base_score += 2.0  # Heavy penalty

                # Bonus for TwelveData (paid service)
                if source == "twelve_data":
                    base_score -= 0.2

                return base_score

            # Sort sources by performance
            available_sources = sorted(order, key=sort_key)

            # Separate fallback sources
            fallback_sources = []
            final_sources = []

            for source in available_sources:
                if self.is_in_fallback(source):
                    if self.can_retry(source):
                        fallback_sources.append(source)
                else:
                    final_sources.append(source)

            # Return available sources first, then retryable fallback sources
            final_order = final_sources + fallback_sources

            logger.debug(f"Optimized fallback order for {data_type}: {final_order}")
            return final_order

    def is_in_fallback(self, source: str) -> bool:
        """Check if a data source is currently in fallback mode"""
        with self.lock:
            return source in self.fallback_states

    def can_retry(self, source: str) -> bool:
        """Check if a source in fallback mode can be retried"""
        with self.lock:
            if source not in self.fallback_states:
                return True
            return self.fallback_states[source].should_retry()

    def should_check_recovery(self, source: str) -> bool:
        """Check if we should test a source for recovery"""
        with self.lock:
            if source not in self.fallback_states:
                return False
            return self.fallback_states[source].should_check_recovery()

    def get_fallback_order(self, data_type: str = "quote", preferred_sources: Optional[List[str]] = None) -> List[str]:
        """
        Get the fallback order for a specific data type, considering current fallback states.
        Uses optimized ordering by default.
        """
        return self.get_optimized_fallback_order(data_type, preferred_sources)

    def record_retry_attempt(self, source: str, success: bool):
        """Record a retry attempt for a source in fallback mode"""
        with self.lock:
            if source in self.fallback_states:
                self.fallback_states[source].record_retry_attempt(success)

                if success:
                    logger.info(f"Recovery attempt successful for {source}")
                    del self.fallback_states[source]
                else:
                    logger.debug(f"Recovery attempt failed for {source}. Next retry in {self.fallback_states[source].backoff_seconds:.1f}s")

    def get_fallback_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current fallback status for all sources"""
        with self.lock:
            status = {}
            for source, state in self.fallback_states.items():
                status[source] = {
                    'reason': state.reason.value,
                    'start_time': state.start_time.isoformat(),
                    'retry_count': state.retry_count,
                    'backoff_seconds': state.backoff_seconds,
                    'can_retry': state.should_retry(),
                    'should_check_recovery': state.should_check_recovery()
                }
            return status

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report for optimization"""
        with self.lock:
            report = {
                "timestamp": datetime.now().isoformat(),
                "source_ranking": self.get_source_ranking(),
                "fallback_status": self.get_fallback_status(),
                "performance_metrics": {}
            }

            for source in set(list(self.performance_history.keys()) + list(self.source_response_times.keys())):
                history = self.performance_history.get(source, [])
                response_times = list(self.source_response_times.get(source, []))

                if history:
                    success_count = sum(1 for h in history if h['success'])
                    report["performance_metrics"][source] = {
                        "total_requests": len(history),
                        "success_rate": success_count / len(history),
                        "success_streak": self.source_success_streak.get(source, 0),
                        "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else None,
                        "in_fallback": source in self.fallback_states
                    }

            return report

    def _classify_error(self, error: Exception, error_type: str) -> FallbackReason:
        """Classify an error to determine fallback reason"""
        error_str = str(error).lower()
        error_type_lower = error_type.lower()

        # Check for specific error types first
        if "twelvedatathrottled" in str(type(error).__name__).lower():
            return FallbackReason.RATE_LIMITED

        if any(keyword in error_str for keyword in ["rate limit", "throttled", "429", "quota exceeded"]):
            return FallbackReason.RATE_LIMITED
        elif any(keyword in error_str for keyword in ["timeout", "timed out"]):
            return FallbackReason.TIMEOUT
        elif any(keyword in error_str for keyword in ["authentication", "unauthorized", "401", "403"]):
            return FallbackReason.AUTHENTICATION_ERROR
        elif any(keyword in error_str for keyword in ["service unavailable", "503", "502", "500"]):
            return FallbackReason.SERVICE_UNAVAILABLE
        elif "api" in error_type_lower:
            return FallbackReason.API_ERROR
        else:
            return FallbackReason.UNKNOWN_ERROR

    def _enter_fallback_mode(self, source: str, reason: FallbackReason):
        """Put a data source into fallback mode"""
        current_time = datetime.now()

        if source in self.fallback_states:
            # Update existing state
            self.fallback_states[source].reason = reason
            self.fallback_states[source].retry_count += 1
        else:
            # Create new fallback state
            self.fallback_states[source] = FallbackState(
                source=source,
                reason=reason,
                start_time=current_time,
                backoff_seconds=self.config.initial_backoff_seconds,
                max_backoff_seconds=self.config.max_backoff_seconds,
                recovery_check_interval=self.config.recovery_check_interval
            )

        logger.warning(f"Data source {source} entered fallback mode due to {reason.value}")

    def cleanup_old_states(self, max_age_hours: float = 24.0):
        """Clean up old fallback states and history"""
        with self.lock:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=max_age_hours)

            # Clean up old fallback states
            to_remove = [
                source for source, state in self.fallback_states.items()
                if state.start_time < cutoff_time
            ]
            for source in to_remove:
                logger.info(f"Removing old fallback state for {source}")
                del self.fallback_states[source]

            # Clean up old rate limit history
            for source, history in self.rate_limit_history.items():
                while history and (current_time - history[0]).total_seconds() > max_age_hours * 3600:
                    history.popleft()

            # Clean up old performance history
            for source, history in self.performance_history.items():
                while history and (current_time - history[0]['timestamp']).total_seconds() > max_age_hours * 3600:
                    history.popleft()