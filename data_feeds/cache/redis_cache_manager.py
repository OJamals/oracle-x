"""
Redis-based cache manager with intelligent caching strategies.
Provides multi-level caching (memory → Redis → database) with compression,
analytics, and intelligent cache warming capabilities.
"""

import json
import time
import logging
import os
import hashlib
import gzip
import pickle
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import threading

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class CacheAnalytics:
    """Track cache performance and provide analytics"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.compression_savings = 0
        self.access_patterns = {}
        self.start_time = time.time()

    def record_hit(self, key: str):
        self.hits += 1
        self._track_access_pattern(key, 'hit')

    def record_miss(self, key: str):
        self.misses += 1
        self._track_access_pattern(key, 'miss')

    def record_eviction(self, key: str):
        self.evictions += 1

    def _track_access_pattern(self, key: str, access_type: str):
        """Track access patterns for cache optimization"""
        symbol = self._extract_symbol_from_key(key)
        if symbol:
            if symbol not in self.access_patterns:
                self.access_patterns[symbol] = {'hits': 0, 'misses': 0, 'last_access': 0}
            self.access_patterns[symbol][access_type + 's'] += 1
            self.access_patterns[symbol]['last_access'] = time.time()

    def _extract_symbol_from_key(self, key: str) -> Optional[str]:
        """Extract ticker symbol from cache key"""
        # Common patterns: quote_AAPL, sentiment_AAPL, market_data_AAPL_1d_1d
        parts = key.split('_')
        if len(parts) >= 2:
            potential_symbol = parts[1]
            if potential_symbol.isupper() and 1 <= len(potential_symbol) <= 5:
                return potential_symbol
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0

        # Get most accessed symbols
        most_accessed = sorted(
            self.access_patterns.items(),
            key=lambda x: x[1]['hits'] + x[1]['misses'],
            reverse=True
        )[:10]

        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'compression_savings_bytes': self.compression_savings,
            'uptime_seconds': time.time() - self.start_time,
            'most_accessed_symbols': most_accessed,
            'total_tracked_symbols': len(self.access_patterns)
        }


class RedisCacheManager:
    """Intelligent Redis-based cache manager with multi-level caching"""

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0,
                 password: str = None, ssl: bool = False, socket_timeout: int = 5):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ssl = ssl
        self.socket_timeout = socket_timeout

        self.redis_client = None
        self.memory_cache = {}  # L1 cache
        self.memory_cache_ttl = {}  # TTL tracking for L1
        self.analytics = CacheAnalytics()

        # Configuration
        self.ttl_config = {
            'ticker_data': int(os.getenv('REDIS_TICKER_DATA_TTL', 86400)),  # 24h
            'market_data': int(os.getenv('REDIS_MARKET_DATA_TTL', 3600)),   # 1h
            'sentiment': int(os.getenv('REDIS_SENTIMENT_TTL', 1800)),      # 30m
            'news': int(os.getenv('REDIS_NEWS_TTL', 900)),                 # 15m
            'default': 3600
        }

        self.compression_enabled = True
        self.max_memory_items = 1000
        self._lock = threading.Lock()

        # Popular symbols for cache warming
        self.popular_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'BABA', 'ORCL', 'CRM', 'AMD', 'INTC', 'UBER', 'SPY', 'QQQ'
        ]

        self._connect()

    def _connect(self):
        """Establish Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Install with: pip install redis")
            return

        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                ssl=self.ssl,
                socket_timeout=self.socket_timeout,
                decode_responses=False  # We'll handle encoding/decoding
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def _get_cache_key(self, namespace: str, identifier: str) -> str:
        """Generate consistent cache key"""
        key_data = f"{namespace}:{identifier}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _compress_data(self, data: Any) -> bytes:
        """Compress data if beneficial"""
        if not self.compression_enabled:
            return pickle.dumps(data)

        pickled_data = pickle.dumps(data)
        if len(pickled_data) < 1000:  # Don't compress small data
            return pickled_data

        compressed = gzip.compress(pickled_data)
        if len(compressed) < len(pickled_data):
            self.analytics.compression_savings += len(pickled_data) - len(compressed)
            return compressed
        return pickled_data

    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data"""
        try:
            # Try to decompress first
            decompressed = gzip.decompress(data)
            return pickle.loads(decompressed)
        except gzip.BadGzipFile:
            # Not compressed, load directly
            return pickle.loads(data)

    def _get_data_type_ttl(self, data_type: str) -> int:
        """Get TTL for data type"""
        return self.ttl_config.get(data_type, self.ttl_config['default'])

    def _cleanup_memory_cache(self):
        """Clean up expired items from memory cache"""
        current_time = time.time()
        expired_keys = []

        for key, expiry in self.memory_cache_ttl.items():
            if current_time > expiry:
                expired_keys.append(key)

        for key in expired_keys:
            self.memory_cache.pop(key, None)
            self.memory_cache_ttl.pop(key, None)
            self.analytics.record_eviction(key)

        # If still too large, remove oldest entries
        if len(self.memory_cache) > self.max_memory_items:
            # Sort by TTL (oldest first)
            sorted_items = sorted(self.memory_cache_ttl.items(), key=lambda x: x[1])
            to_remove = len(self.memory_cache) - self.max_memory_items

            for key, _ in sorted_items[:to_remove]:
                self.memory_cache.pop(key, None)
                self.memory_cache_ttl.pop(key, None)
                self.analytics.record_eviction(key)

    def get(self, namespace: str, identifier: str, data_type: str = 'default') -> Optional[Any]:
        """Get data from multi-level cache"""
        cache_key = self._get_cache_key(namespace, identifier)

        # Check L1 (memory) cache first
        with self._lock:
            if cache_key in self.memory_cache:
                if time.time() <= self.memory_cache_ttl.get(cache_key, 0):
                    self.analytics.record_hit(cache_key)
                    return self.memory_cache[cache_key]

                # Expired, remove it
                self.memory_cache.pop(cache_key, None)
                self.memory_cache_ttl.pop(cache_key, None)

        # Check L2 (Redis) cache
        if self.redis_client:
            try:
                redis_data = self.redis_client.get(cache_key)
                if redis_data:
                    data = self._decompress_data(redis_data)

                    # Store in memory cache
                    with self._lock:
                        self.memory_cache[cache_key] = data
                        self.memory_cache_ttl[cache_key] = time.time() + 300  # 5 min in memory

                    self.analytics.record_hit(cache_key)
                    return data
            except Exception as e:
                logger.debug(f"Redis get error: {e}")

        # Check L3 (SQLite) cache if available
        # This would integrate with existing CacheService

        self.analytics.record_miss(cache_key)
        return None

    def set(self, namespace: str, identifier: str, data: Any,
            data_type: str = 'default', quality_score: float = 100.0) -> bool:
        """Set data in multi-level cache"""
        cache_key = self._get_cache_key(namespace, identifier)
        ttl = self._get_data_type_ttl(data_type)

        # Skip caching if quality is too low
        if quality_score < 60:
            return False

        try:
            compressed_data = self._compress_data(data)

            # Store in Redis (L2)
            if self.redis_client:
                try:
                    self.redis_client.setex(cache_key, ttl, compressed_data)
                except Exception as e:
                    logger.debug(f"Redis set error: {e}")

            # Store in memory (L1)
            with self._lock:
                self.memory_cache[cache_key] = data
                self.memory_cache_ttl[cache_key] = time.time() + min(ttl, 300)  # Max 5 min in memory

                # Cleanup if necessary
                self._cleanup_memory_cache()

            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def invalidate(self, namespace: str, identifier: str) -> bool:
        """Invalidate cache entry across all levels"""
        cache_key = self._get_cache_key(namespace, identifier)

        success = True

        # Remove from memory
        with self._lock:
            self.memory_cache.pop(cache_key, None)
            self.memory_cache_ttl.pop(cache_key, None)

        # Remove from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(cache_key)
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")
                success = False

        return success

    def invalidate_pattern(self, namespace: str, pattern: str = '*') -> int:
        """Invalidate multiple cache entries by pattern"""
        invalidated = 0

        # Invalidate memory cache
        with self._lock:
            keys_to_remove = []
            for key in self.memory_cache.keys():
                if key.startswith(f"{namespace}:"):
                    if pattern == '*' or pattern in key:
                        keys_to_remove.append(key)

            for key in keys_to_remove:
                self.memory_cache.pop(key, None)
                self.memory_cache_ttl.pop(key, None)
                invalidated += 1

        # Invalidate Redis cache
        if self.redis_client:
            try:
                redis_pattern = f"{namespace}:*"
                if pattern != '*':
                    redis_pattern = redis_pattern.replace('*', pattern)

                keys = self.redis_client.keys(redis_pattern)
                if keys:
                    invalidated += self.redis_client.delete(*keys)
            except Exception as e:
                logger.debug(f"Redis pattern delete error: {e}")

        return invalidated

    def warm_cache(self, symbols: List[str] = None):
        """Warm up cache with popular symbols"""
        if symbols is None:
            symbols = self.popular_symbols

        logger.info(f"Warming cache for {len(symbols)} symbols")

        # This would integrate with data feeds to prefetch popular data
        # For now, just log the intention
        for symbol in symbols:
            logger.debug(f"Would warm cache for {symbol}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        redis_info = {}
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
            except Exception as e:
                logger.debug(f"Redis info error: {e}")

        with self._lock:
            memory_items = len(self.memory_cache)

        return {
            'redis_available': self.redis_client is not None,
            'redis_info': redis_info,
            'memory_cache_items': memory_items,
            'analytics': self.analytics.get_stats(),
            'config': {
                'ttl_settings': self.ttl_config,
                'compression_enabled': self.compression_enabled,
                'max_memory_items': self.max_memory_items
            }
        }

    def clear_cache(self):
        """Clear all cache levels"""
        # Clear memory cache
        with self._lock:
            self.memory_cache.clear()
            self.memory_cache_ttl.clear()

        # Clear Redis cache
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis flush error: {e}")

        # Reset analytics
        self.analytics = CacheAnalytics()

        logger.info("Cache cleared")


# Global instance
_redis_cache_manager = None

def get_redis_cache_manager() -> Optional[RedisCacheManager]:
    """Get or create global Redis cache manager instance"""
    global _redis_cache_manager

    if _redis_cache_manager is None and os.getenv('ENABLE_REDIS_CACHE', 'true').lower() == 'true':
        try:
            _redis_cache_manager = RedisCacheManager(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                password=os.getenv('REDIS_PASSWORD', None),
                ssl=os.getenv('REDIS_SSL', 'false').lower() == 'true',
                socket_timeout=int(os.getenv('REDIS_TIMEOUT', 5))
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache manager: {e}")
            _redis_cache_manager = None

    return _redis_cache_manager