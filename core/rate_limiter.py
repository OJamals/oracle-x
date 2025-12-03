"""
Unified rate limiting system for Oracle-X financial trading system.
Provides comprehensive rate limiting across all data sources with circuit breaker patterns.
"""

import time
import threading
from datetime import datetime
from typing import Dict, Optional
from enum import Enum, auto
import logging
from dataclasses import dataclass
from functools import wraps

from core.types_internal import DataSource
from core.cache.redis_manager import get_cache_manager
from core.cache.sqlite_manager import get_sqlite_cache_manager

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = auto()
    SLIDING_WINDOW = auto()
    TOKEN_BUCKET = auto()
    LEAKY_BUCKET = auto()


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Circuit is open, no requests allowed
    HALF_OPEN = auto()  # Testing if service has recovered


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    max_requests: int
    time_window: int  # seconds
    strategy: RateLimitStrategy = RateLimitStrategy.FIXED_WINDOW
    burst_capacity: Optional[int] = None  # For token bucket
    recovery_rate: Optional[float] = None  # For token bucket/leaky bucket


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures to open circuit
    reset_timeout: int = 60  # Seconds before attempting recovery
    success_threshold: int = 3  # Number of successes to close circuit
    half_open_max_requests: int = 1  # Max requests in half-open state


@dataclass
class RateLimitStats:
    """Statistics for rate limiting."""

    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    current_rps: float = 0.0
    circuit_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    last_reset: Optional[datetime] = None


class RateLimiter:
    """
    Unified rate limiter with circuit breaker pattern for data sources.
    """

    def __init__(self, use_redis: bool = True):
        """
        Initialize rate limiter.

        Args:
            use_redis: Whether to use Redis for distributed rate limiting
        """
        self.use_redis = use_redis
        self.cache = get_cache_manager() if use_redis else get_sqlite_cache_manager()
        self._lock = threading.RLock()

        # Default rate limits from DataSource enum
        self.rate_limits: Dict[DataSource, RateLimitConfig] = {}
        self.circuit_breakers: Dict[DataSource, CircuitBreakerConfig] = {}
        self.stats: Dict[DataSource, RateLimitStats] = {}

        self._initialize_default_limits()

    def _initialize_default_limits(self) -> None:
        """Initialize default rate limits from DataSource enum."""
        for source in DataSource:
            rps = source.requests_per_second
            if rps is None:
                # Unlimited requests
                config = RateLimitConfig(
                    max_requests=1000000,  # Very high limit
                    time_window=1,
                    strategy=RateLimitStrategy.FIXED_WINDOW,
                )
            else:
                # Calculate requests per time window
                max_requests = max(1, int(rps))
                config = RateLimitConfig(
                    max_requests=max_requests,
                    time_window=1,
                    strategy=RateLimitStrategy.FIXED_WINDOW,
                )

            self.rate_limits[source] = config
            self.circuit_breakers[source] = CircuitBreakerConfig()
            self.stats[source] = RateLimitStats()

    def set_rate_limit(self, source: DataSource, config: RateLimitConfig) -> None:
        """
        Set custom rate limit configuration for a data source.

        Args:
            source: Data source
            config: Rate limit configuration
        """
        with self._lock:
            self.rate_limits[source] = config

    def set_circuit_breaker(
        self, source: DataSource, config: CircuitBreakerConfig
    ) -> None:
        """
        Set custom circuit breaker configuration for a data source.

        Args:
            source: Data source
            config: Circuit breaker configuration
        """
        with self._lock:
            self.circuit_breakers[source] = config

    def rate_limit_call(self, source: DataSource, call_type: str = "default") -> bool:
        """
        Check if a call is allowed based on rate limits and circuit breaker state.

        Args:
            source: Data source
            call_type: Type of call (for detailed tracking)

        Returns:
            True if call is allowed, False if rate limited or circuit open
        """
        with self._lock:
            stats = self.stats[source]
            stats.total_requests += 1

            # Check circuit breaker first
            circuit_state = self._get_circuit_state(source)
            if circuit_state == CircuitBreakerState.OPEN:
                stats.rejected_requests += 1
                logger.warning(f"Circuit open for {source.name}, call rejected")
                return False
            elif circuit_state == CircuitBreakerState.HALF_OPEN:
                half_open_config = self.circuit_breakers[source]
                if stats.allowed_requests >= half_open_config.half_open_max_requests:
                    stats.rejected_requests += 1
                    logger.warning(f"Half-open circuit limit reached for {source.name}")
                    return False

            # Check rate limits
            if not self._check_rate_limit(source, call_type):
                stats.rejected_requests += 1
                logger.warning(f"Rate limit exceeded for {source.name}")
                return False

            stats.allowed_requests += 1
            return True

    def _check_rate_limit(self, source: DataSource, call_type: str) -> bool:
        """
        Check rate limit using the configured strategy.

        Args:
            source: Data source
            call_type: Type of call

        Returns:
            True if within rate limit, False otherwise
        """
        config = self.rate_limits[source]
        key = f"ratelimit:{source.name}:{call_type}"

        if config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._fixed_window_strategy(key, config)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._sliding_window_strategy(key, config)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._token_bucket_strategy(key, config)
        elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return self._leaky_bucket_strategy(key, config)
        else:
            return True  # Unknown strategy, allow call

    def _fixed_window_strategy(self, key: str, config: RateLimitConfig) -> bool:
        """Fixed window rate limiting strategy."""
        current_window = int(time.time() // config.time_window)
        window_key = f"{key}:{current_window}"

        # Get current count for this window
        current_count = self.cache.increment_counter(window_key, 0) or 0

        if current_count >= config.max_requests:
            return False

        # Increment counter and set TTL for the window
        new_count = self.cache.increment_counter(window_key, 1, config.time_window * 2)
        return new_count is not None and new_count <= config.max_requests

    def _sliding_window_strategy(self, key: str, config: RateLimitConfig) -> bool:
        """Sliding window rate limiting strategy."""
        current_time = time.time()
        window_start = current_time - config.time_window

        # Use Redis sorted set for sliding window
        member = f"{current_time}:{threading.get_ident()}"
        pipe = self.cache.redis_client.pipeline() if self.use_redis else None

        if self.use_redis:
            pipe.zadd(key, {member: current_time})
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.expire(key, config.time_window * 2)
            _, _, count, _ = pipe.execute()
        else:
            # For SQLite, we'd need a different approach
            # This is simplified - in production, you'd use a proper sliding window
            return self._fixed_window_strategy(key, config)

        return count <= config.max_requests

    def _token_bucket_strategy(self, key: str, config: RateLimitConfig) -> bool:
        """Token bucket rate limiting strategy."""
        burst = config.burst_capacity or config.max_requests
        now = time.time()

        # Get current token state
        state_key = f"{key}:state"
        state = self.cache.get_hash_all(state_key) or {}

        tokens = float(state.get("tokens", burst))
        last_update = float(state.get("last_update", now))

        # Add tokens based on time passed
        time_passed = now - last_update
        new_tokens = time_passed * (config.max_requests / config.time_window)
        tokens = min(burst, tokens + new_tokens)

        if tokens < 1:
            return False

        # Consume one token
        tokens -= 1

        # Update state
        self.cache.set_hash_field(state_key, "tokens", tokens)
        self.cache.set_hash_field(state_key, "last_update", now)
        (
            self.cache.redis_client.expire(state_key, config.time_window * 2)
            if self.use_redis
            else None
        )

        return True

    def _leaky_bucket_strategy(self, key: str, config: RateLimitConfig) -> bool:
        """Leaky bucket rate limiting strategy."""
        # Similar to token bucket but different semantics
        # Implementation would be similar with different token handling
        return self._token_bucket_strategy(key, config)

    def _get_circuit_state(self, source: DataSource) -> CircuitBreakerState:
        """
        Get current circuit breaker state for a data source.

        Args:
            source: Data source

        Returns:
            Current circuit state
        """
        cb_config = self.circuit_breakers[source]
        key = f"circuit:{source.name}"

        state_data = self.cache.get_hash_all(key) or {}
        current_state = state_data.get("state", CircuitBreakerState.CLOSED.name)
        last_failure = float(state_data.get("last_failure", 0))

        current_state = CircuitBreakerState[current_state]

        # Check if we should transition from OPEN to HALF_OPEN
        if current_state == CircuitBreakerState.OPEN:
            time_since_failure = time.time() - last_failure
            if time_since_failure >= cb_config.reset_timeout:
                return CircuitBreakerState.HALF_OPEN

        return current_state

    def record_success(self, source: DataSource) -> None:
        """
        Record a successful call for circuit breaker.

        Args:
            source: Data source
        """
        with self._lock:
            key = f"circuit:{source.name}"
            state_data = self.cache.get_hash_all(key) or {}
            current_state = state_data.get("state", CircuitBreakerState.CLOSED.name)
            current_state = CircuitBreakerState[current_state]
            success_count = int(state_data.get("success_count", 0))

            if current_state == CircuitBreakerState.HALF_OPEN:
                success_count += 1
                if success_count >= self.circuit_breakers[source].success_threshold:
                    # Close the circuit
                    self.cache.set_hash_field(
                        key, "state", CircuitBreakerState.CLOSED.name
                    )
                    self.cache.set_hash_field(key, "failure_count", 0)
                    self.cache.set_hash_field(key, "success_count", 0)
                    self.cache.set_hash_field(key, "last_failure", 0)
                    logger.info(f"Circuit closed for {source.name}")
                else:
                    self.cache.set_hash_field(key, "success_count", success_count)

    def record_failure(self, source: DataSource) -> None:
        """
        Record a failed call for circuit breaker.

        Args:
            source: Data source
        """
        with self._lock:
            key = f"circuit:{source.name}"
            state_data = self.cache.get_hash_all(key) or {}
            current_state = state_data.get("state", CircuitBreakerState.CLOSED.name)
            current_state = CircuitBreakerState[current_state]
            failure_count = int(state_data.get("failure_count", 0))

            if current_state == CircuitBreakerState.HALF_OPEN:
                # Immediate trip to OPEN state
                self.cache.set_hash_field(key, "state", CircuitBreakerState.OPEN.name)
                self.cache.set_hash_field(key, "last_failure", time.time())
                logger.warning(
                    f"Circuit re-opened for {source.name} after failure in half-open state"
                )
                return

            failure_count += 1
            if failure_count >= self.circuit_breakers[source].failure_threshold:
                # Trip the circuit
                self.cache.set_hash_field(key, "state", CircuitBreakerState.OPEN.name)
                self.cache.set_hash_field(key, "last_failure", time.time())
                logger.warning(
                    f"Circuit opened for {source.name} after {failure_count} failures"
                )
            else:
                self.cache.set_hash_field(key, "failure_count", failure_count)
                self.cache.set_hash_field(key, "last_failure", time.time())

    def get_stats(self, source: DataSource) -> RateLimitStats:
        """
        Get rate limiting statistics for a data source.

        Args:
            source: Data source

        Returns:
            Rate limiting statistics
        """
        with self._lock:
            stats = self.stats[source]
            # Update current RPS
            if stats.total_requests > 0:
                stats.current_rps = stats.allowed_requests / max(
                    1, (time.time() - (stats.last_reset or time.time()).timestamp())
                )
            stats.circuit_state = self._get_circuit_state(source)
            return stats

    def reset_stats(self, source: DataSource) -> None:
        """
        Reset statistics for a data source.

        Args:
            source: Data source
        """
        with self._lock:
            self.stats[source] = RateLimitStats(last_reset=datetime.now())

    def reset_circuit(self, source: DataSource) -> None:
        """
        Reset circuit breaker for a data source.

        Args:
            source: Data source
        """
        with self._lock:
            key = f"circuit:{source.name}"
            self.cache.set_hash_field(key, "state", CircuitBreakerState.CLOSED.name)
            self.cache.set_hash_field(key, "failure_count", 0)
            self.cache.set_hash_field(key, "success_count", 0)
            self.cache.set_hash_field(key, "last_failure", 0)
            logger.info(f"Circuit reset for {source.name}")


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(use_redis: bool = True) -> RateLimiter:
    """
    Get or create the global rate limiter instance.

    Args:
        use_redis: Whether to use Redis for distributed rate limiting

    Returns:
        RateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(use_redis=use_redis)
    return _rate_limiter


def rate_limit_decorator(source: DataSource, call_type: str = "default"):
    """
    Decorator for rate limiting function calls.

    Args:
        source: Data source to rate limit against
        call_type: Type of call for detailed tracking

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()

            if not limiter.rate_limit_call(source, call_type):
                raise RateLimitExceededError(f"Rate limit exceeded for {source.name}")

            try:
                result = func(*args, **kwargs)
                limiter.record_success(source)
                return result
            except Exception:
                limiter.record_failure(source)
                raise

        return wrapper

    return decorator


class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded."""

    pass


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass
