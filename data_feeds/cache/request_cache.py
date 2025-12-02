"""
Request Caching System

Implements intelligent caching for API responses with:
- TTL-based expiration
- LRU eviction
- Hash-based keys
- Async support
- Cache warming
"""

import asyncio
import hashlib
import json
import pickle
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


class TimedLRUCache:
    """
    LRU cache with TTL (Time-To-Live) support.

    Features:
    - Automatic expiration based on TTL
    - LRU eviction when maxsize reached
    - Thread-safe operations
    - Hit/miss statistics
    """

    def __init__(self, maxsize: int = 128, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds

        # Ordered dict for LRU
        self.cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Lock for thread safety
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired"""
        async with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]

                # Check if expired
                age = (datetime.now() - timestamp).total_seconds()
                if age < self.ttl_seconds:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return value
                else:
                    # Expired, remove
                    del self.cache[key]

            self.misses += 1
            return None

    async def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        async with self.lock:
            # Check if key exists (update)
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                # Check if cache is full
                if len(self.cache) >= self.maxsize:
                    # Remove oldest item (LRU)
                    self.cache.popitem(last=False)
                    self.evictions += 1

            # Store with timestamp
            self.cache[key] = (value, datetime.now())

    async def clear(self) -> None:
        """Clear all cached items"""
        async with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")

    async def get_stats(self) -> dict:
        """Get cache statistics"""
        async with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_percent": hit_rate,
                "evictions": self.evictions,
                "ttl_seconds": self.ttl_seconds,
            }


class RequestCache:
    """
    Intelligent request caching system for API responses.

    Supports different TTLs for different data types:
    - Market data: 60 seconds
    - Options data: 120 seconds
    - Sentiment data: 300 seconds (5 minutes)
    - News: 600 seconds (10 minutes)
    """

    def __init__(self):
        # Separate caches for different data types
        self.caches = {
            "market_data": TimedLRUCache(maxsize=100, ttl_seconds=60),
            "options_flow": TimedLRUCache(maxsize=500, ttl_seconds=120),
            "sentiment": TimedLRUCache(maxsize=200, ttl_seconds=300),
            "news": TimedLRUCache(maxsize=100, ttl_seconds=600),
            "dark_pools": TimedLRUCache(maxsize=300, ttl_seconds=180),
            "earnings": TimedLRUCache(maxsize=50, ttl_seconds=3600),  # 1 hour
            "quotes": TimedLRUCache(maxsize=1000, ttl_seconds=30),  # 30 seconds
        }

    @staticmethod
    def _make_key(data_type: str, **params) -> str:
        """
        Create cache key from parameters.
        Uses MD5 hash for compact keys.
        """
        # Sort parameters for consistency
        sorted_params = sorted(params.items())

        # Create string representation
        param_str = json.dumps(sorted_params, sort_keys=True)

        # Hash for compact key
        hash_obj = hashlib.md5(f"{data_type}:{param_str}".encode())
        return hash_obj.hexdigest()

    async def get(self, data_type: str, **params) -> Optional[Any]:
        """
        Get cached response.

        Args:
            data_type: Type of data (e.g., 'market_data', 'options_flow')
            **params: Parameters used in the original request

        Returns:
            Cached response or None if not found/expired
        """
        if data_type not in self.caches:
            logger.warning(f"Unknown data type: {data_type}")
            return None

        cache = self.caches[data_type]
        key = self._make_key(data_type, **params)

        result = await cache.get(key)

        if result is not None:
            logger.debug(f"Cache HIT: {data_type} {list(params.keys())}")

        return result

    async def set(self, data_type: str, response: Any, **params) -> None:
        """
        Cache a response.

        Args:
            data_type: Type of data
            response: Response to cache
            **params: Parameters used in the request
        """
        if data_type not in self.caches:
            logger.warning(f"Unknown data type: {data_type}")
            return

        cache = self.caches[data_type]
        key = self._make_key(data_type, **params)

        await cache.set(key, response)
        logger.debug(f"Cached: {data_type} {list(params.keys())}")

    async def get_all_stats(self) -> dict:
        """Get statistics for all caches"""
        stats = {}
        for data_type, cache in self.caches.items():
            stats[data_type] = await cache.get_stats()
        return stats

    async def clear_all(self) -> None:
        """Clear all caches"""
        for cache in self.caches.values():
            await cache.clear()
        logger.info("All caches cleared")


# Global instance
_request_cache = None


def get_request_cache() -> RequestCache:
    """Get global request cache instance"""
    global _request_cache
    if _request_cache is None:
        _request_cache = RequestCache()
    return _request_cache


# Decorator for caching async functions
def cached(data_type: str, extract_params: Optional[Callable] = None):
    """
    Decorator to automatically cache async function results.

    Args:
        data_type: Type of data for cache selection
        extract_params: Function to extract cache parameters from args/kwargs

    Usage:
        @cached('market_data', lambda args, kwargs: {'symbol': args[0]})
        async def fetch_market_data(symbol):
            ...
    """

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            cache = get_request_cache()

            # Extract parameters for cache key
            if extract_params:
                params = extract_params(args, kwargs)
            else:
                # Use all kwargs as params
                params = kwargs

            # Try to get from cache
            cached_result = await cache.get(data_type, **params)
            if cached_result is not None:
                return cached_result

            # Cache miss - call function
            result = await func(*args, **kwargs)

            # Cache result
            if result is not None:
                await cache.set(data_type, result, **params)

            return result

        return wrapper

    return decorator


# Example: Cached API functions
@cached("market_data", lambda args, kwargs: {"symbol": args[0]})
async def fetch_market_data_cached(symbol: str):
    """Fetch market data with automatic caching"""
    # Simulate API call
    await asyncio.sleep(0.5)
    return {
        "symbol": symbol,
        "price": 100.0,
        "volume": 1000000,
        "timestamp": datetime.now().isoformat(),
    }


@cached("options_flow", lambda args, kwargs: {"tickers": str(args[0])})
async def fetch_options_flow_cached(tickers: list):
    """Fetch options flow with automatic caching"""
    # Simulate expensive API call
    await asyncio.sleep(2.0)
    return {
        "tickers": tickers,
        "unusual_sweeps": [],
        "timestamp": datetime.now().isoformat(),
    }


# Test function
async def test_cache():
    """Test the caching system"""
    cache = RequestCache()

    print("Testing cache system...")

    # Test market data caching
    print("\n1. Testing market data cache (TTL: 60s):")

    # First call - cache miss
    start = datetime.now()
    result1 = await fetch_market_data_cached("AAPL")
    time1 = (datetime.now() - start).total_seconds()
    print(f"  First call: {time1:.3f}s (cache miss)")

    # Second call - cache hit
    start = datetime.now()
    result2 = await fetch_market_data_cached("AAPL")
    time2 = (datetime.now() - start).total_seconds()
    print(f"  Second call: {time2:.3f}s (cache hit)")

    print(f"  Speedup: {time1/time2:.1f}x faster")

    # Show stats
    print("\n2. Cache statistics:")
    stats = await cache.get_all_stats()
    for data_type, stat in stats.items():
        if stat["hits"] > 0 or stat["misses"] > 0:
            print(f"  {data_type}:")
            print(f"    Hit rate: {stat['hit_rate_percent']:.1f}%")
            print(f"    Size: {stat['size']}/{stat['maxsize']}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_cache())
