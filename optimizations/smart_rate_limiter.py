"""
Smart Rate Limiter for API Calls

Implements intelligent rate limiting with:
- Request distribution across time windows
- Burst protection
- Per-source quota management
- Automatic backoff
"""

import asyncio
import time
from collections import deque
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate-limited API source"""

    requests_per_minute: int
    requests_per_hour: Optional[int] = None
    daily_quota: Optional[int] = None
    burst_size: int = None  # Max requests in burst
    backoff_factor: float = 1.5  # Exponential backoff multiplier

    def __post_init__(self):
        if self.burst_size is None:
            # Default burst size is 50% of per-minute limit
            self.burst_size = max(1, int(self.requests_per_minute * 0.5))


class SmartRateLimiter:
    """
    Intelligent rate limiter with predictive throttling.

    Features:
    - Distributes requests evenly across time window
    - Prevents rate limit errors before they happen
    - Tracks multiple time windows (minute, hour, day)
    - Automatic backoff on rate limit detection
    - Per-source configuration
    """

    def __init__(self):
        # Rate limit configurations per source
        self.configs: Dict[str, RateLimitConfig] = {
            "twelve_data": RateLimitConfig(
                requests_per_minute=8,
                requests_per_hour=450,
                daily_quota=800,
                burst_size=4,
            ),
            "finnhub": RateLimitConfig(
                requests_per_minute=60, daily_quota=1000, burst_size=20
            ),
            "iex_cloud": RateLimitConfig(
                requests_per_minute=100, daily_quota=50000, burst_size=30
            ),
            "reddit": RateLimitConfig(requests_per_minute=60, burst_size=20),
            "finviz": RateLimitConfig(requests_per_minute=12, burst_size=5),
        }

        # Request history per source
        self.request_history: Dict[str, deque] = {
            source: deque(maxlen=1000) for source in self.configs.keys()
        }

        # Backoff state
        self.backoff_until: Dict[str, float] = {}
        self.backoff_count: Dict[str, int] = {}

        # Locks for thread safety
        self.locks: Dict[str, asyncio.Lock] = {
            source: asyncio.Lock() for source in self.configs.keys()
        }

    def _get_request_count(self, source: str, window_seconds: int) -> int:
        """Count requests in time window"""
        if source not in self.request_history:
            return 0

        cutoff = time.time() - window_seconds
        history = self.request_history[source]

        # Count requests after cutoff
        return sum(1 for timestamp in history if timestamp > cutoff)

    def _calculate_wait_time(self, source: str, config: RateLimitConfig) -> float:
        """
        Calculate optimal wait time to avoid rate limits.
        Uses intelligent distribution instead of naive blocking.
        """
        now = time.time()

        # Check if in backoff period
        if source in self.backoff_until and now < self.backoff_until[source]:
            return self.backoff_until[source] - now

        # Check minute window
        minute_requests = self._get_request_count(source, 60)

        if minute_requests >= config.requests_per_minute:
            # Find oldest request in minute window
            history = self.request_history[source]
            minute_ago = now - 60

            # Find when the oldest request will expire
            for timestamp in history:
                if timestamp > minute_ago:
                    wait_time = 60 - (now - timestamp) + 0.5  # Add 0.5s buffer
                    logger.debug(
                        f"{source}: Rate limit approaching, waiting {wait_time:.1f}s"
                    )
                    return wait_time

        # Check hour window (if configured)
        if config.requests_per_hour:
            hour_requests = self._get_request_count(source, 3600)
            if hour_requests >= config.requests_per_hour:
                # Find oldest request in hour window
                history = self.request_history[source]
                hour_ago = now - 3600

                for timestamp in history:
                    if timestamp > hour_ago:
                        wait_time = 3600 - (now - timestamp) + 1.0
                        logger.info(
                            f"{source}: Hourly limit reached, waiting {wait_time:.1f}s"
                        )
                        return wait_time

        # Check burst protection
        recent_requests = self._get_request_count(source, 10)  # Last 10 seconds
        if recent_requests >= config.burst_size:
            wait_time = 2.0  # Small delay to prevent burst
            logger.debug(f"{source}: Burst limit reached, waiting {wait_time:.1f}s")
            return wait_time

        # Intelligent spacing: distribute requests evenly
        if minute_requests > 0:
            # Calculate optimal spacing
            target_spacing = 60.0 / config.requests_per_minute

            # Time since last request
            history = self.request_history[source]
            if history:
                time_since_last = now - history[-1]

                # If requests are coming too fast, add small delay
                if time_since_last < target_spacing * 0.7:  # 70% of optimal
                    wait_time = (target_spacing * 0.7) - time_since_last
                    return max(0, wait_time)

        return 0.0  # No wait needed

    async def acquire(self, source: str) -> None:
        """
        Acquire permission to make API request.
        Blocks until rate limit allows the request.

        Args:
            source: API source name (e.g., 'twelve_data', 'finnhub')
        """
        if source not in self.configs:
            logger.warning(f"No rate limit config for {source}, allowing request")
            return

        config = self.configs[source]
        lock = self.locks[source]

        async with lock:
            # Calculate wait time
            wait_time = self._calculate_wait_time(source, config)

            # Wait if needed
            if wait_time > 0:
                logger.debug(f"{source}: Throttling for {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

            # Record request
            self.request_history[source].append(time.time())

    def record_rate_limit_error(self, source: str):
        """
        Record that a rate limit error occurred.
        Implements exponential backoff.
        """
        if source not in self.configs:
            return

        # Increment backoff counter
        if source not in self.backoff_count:
            self.backoff_count[source] = 0
        self.backoff_count[source] += 1

        config = self.configs[source]

        # Calculate backoff time (exponential)
        backoff_time = min(
            300,  # Max 5 minutes
            60 * (config.backoff_factor ** self.backoff_count[source]),
        )

        self.backoff_until[source] = time.time() + backoff_time

        logger.warning(
            f"{source}: Rate limit hit! Backing off for {backoff_time:.1f}s "
            f"(attempt {self.backoff_count[source]})"
        )

    def record_success(self, source: str):
        """Record successful request (resets backoff)"""
        if source in self.backoff_count:
            self.backoff_count[source] = 0
        if source in self.backoff_until:
            del self.backoff_until[source]

    def get_stats(self, source: str) -> dict:
        """Get rate limit statistics for a source"""
        if source not in self.configs:
            return {}

        config = self.configs[source]

        return {
            "source": source,
            "requests_last_minute": self._get_request_count(source, 60),
            "requests_last_hour": self._get_request_count(source, 3600),
            "limit_per_minute": config.requests_per_minute,
            "limit_per_hour": config.requests_per_hour,
            "backoff_count": self.backoff_count.get(source, 0),
            "in_backoff": source in self.backoff_until
            and time.time() < self.backoff_until[source],
            "utilization_percent": (
                self._get_request_count(source, 60) / config.requests_per_minute * 100
            ),
        }

    def get_all_stats(self) -> Dict[str, dict]:
        """Get stats for all sources"""
        return {source: self.get_stats(source) for source in self.configs.keys()}


# Global instance
_rate_limiter = None


def get_rate_limiter() -> SmartRateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = SmartRateLimiter()
    return _rate_limiter


# Decorator for rate-limited functions
def rate_limited(source: str):
    """
    Decorator to automatically rate limit async functions.

    Usage:
        @rate_limited('twelve_data')
        async def fetch_data(symbol):
            ...
    """

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            await limiter.acquire(source)

            try:
                result = await func(*args, **kwargs)
                limiter.record_success(source)
                return result
            except Exception as e:
                # Check if rate limit error
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "too many requests" in error_msg:
                    limiter.record_rate_limit_error(source)
                raise

        return wrapper

    return decorator


# Example usage
async def test_rate_limiter():
    """Test the rate limiter"""
    limiter = SmartRateLimiter()

    # Simulate many requests to twelve_data
    print("Testing TwelveData rate limiting (8 req/min)...")

    for i in range(15):
        start = time.time()
        await limiter.acquire("twelve_data")
        elapsed = time.time() - start

        print(f"Request {i+1}: waited {elapsed:.2f}s")

        # Show stats every 5 requests
        if (i + 1) % 5 == 0:
            stats = limiter.get_stats("twelve_data")
            print(
                f"  Stats: {stats['requests_last_minute']}/{stats['limit_per_minute']} "
                f"requests, {stats['utilization_percent']:.1f}% utilized"
            )

    print("\nFinal stats:")
    print(limiter.get_all_stats())


if __name__ == "__main__":
    # Run test
    asyncio.run(test_rate_limiter())
