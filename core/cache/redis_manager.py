"""
Redis cache manager for Oracle-X financial trading system.
Provides optimized Redis caching with connection pooling and comprehensive cache management.
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Optional, Any, Dict, List, TypeVar
import redis
from redis.connection import ConnectionPool
import logging
from functools import wraps

from core.types import MarketData, OptionContract, DataSource

# Type variable for generic caching
T = TypeVar('T')

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """
    Redis cache manager with connection pooling and optimized caching strategies.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, password: Optional[str] = None,
                 max_connections: int = 10, **kwargs):
        """
        Initialize Redis cache manager.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if any)
            max_connections: Maximum number of connections in pool
            **kwargs: Additional Redis connection parameters
        """
        self.pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=True,
            **kwargs
        )
        self.redis_client = redis.Redis(connection_pool=self.pool)
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test Redis connection and log connection status."""
        try:
            self.redis_client.ping()
            logger.info(f"Redis cache manager connected successfully to {self.pool.connection_kwargs['host']}:{self.pool.connection_kwargs['port']}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def get_cached_data(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """
        Get cached data with optional TTL extension.
        
        Args:
            key: Cache key
            ttl: Optional TTL to extend if data exists
            
        Returns:
            Cached data or None if not found/expired
        """
        try:
            data = self.redis_client.get(key)
            if data is not None and ttl is not None:
                # Extend TTL if data exists and new TTL provided
                self.redis_client.expire(key, ttl)
            
            if data:
                try:
                    return json.loads(data, parse_float=Decimal)
                except json.JSONDecodeError:
                    return data
            return None
        except redis.RedisError as e:
            logger.error(f"Error getting cached data for key {key}: {e}")
            return None
    
    def set_cached_data(self, key: str, data: Any, ttl: int = 300) -> bool:
        """
        Set data in cache with TTL.
        
        Args:
            key: Cache key
            data: Data to cache (will be JSON serialized)
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if isinstance(data, (MarketData, OptionContract)):
                # Use the model's dict method for proper serialization
                serialized_data = json.dumps(data.dict(), default=self._json_serializer)
            else:
                serialized_data = json.dumps(data, default=self._json_serializer)
            
            result = self.redis_client.setex(key, ttl, serialized_data)
            return bool(result)
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing data for cache key {key}: {e}")
            return False
        except redis.RedisError as e:
            logger.error(f"Error setting cached data for key {key}: {e}")
            return False
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex types."""
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, (DataSource,)):
            return obj.name
        elif hasattr(obj, 'dict'):
            return obj.dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def delete_key(self, key: str) -> bool:
        """
        Delete a key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False otherwise
        """
        try:
            result = self.redis_client.delete(key)
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def get_keys(self, pattern: str = '*') -> List[str]:
        """
        Get keys matching pattern.
        
        Args:
            pattern: Redis key pattern
            
        Returns:
            List of matching keys
        """
        try:
            return self.redis_client.keys(pattern)
        except redis.RedisError as e:
            logger.error(f"Error getting keys with pattern {pattern}: {e}")
            return []
    
    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds, or None if key doesn't exist or has no TTL
        """
        try:
            ttl = self.redis_client.ttl(key)
            return ttl if ttl >= 0 else None
        except redis.RedisError as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return None
    
    def increment_counter(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """
        Increment a counter value.
        
        Args:
            key: Counter key
            amount: Amount to increment
            ttl: Optional TTL to set (only if key doesn't exist)
            
        Returns:
            New counter value or None on error
        """
        try:
            pipeline = self.redis_client.pipeline()
            pipeline.incrby(key, amount)
            if ttl is not None:
                pipeline.expire(key, ttl)
            results = pipeline.execute()
            return results[0]
        except redis.RedisError as e:
            logger.error(f"Error incrementing counter {key}: {e}")
            return None
    
    def get_hash_field(self, key: str, field: str) -> Optional[Any]:
        """
        Get a field from a hash.
        
        Args:
            key: Hash key
            field: Field name
            
        Returns:
            Field value or None if not found
        """
        try:
            value = self.redis_client.hget(key, field)
            if value:
                try:
                    return json.loads(value, parse_float=Decimal)
                except json.JSONDecodeError:
                    return value
            return None
        except redis.RedisError as e:
            logger.error(f"Error getting hash field {field} from {key}: {e}")
            return None
    
    def set_hash_field(self, key: str, field: str, value: Any) -> bool:
        """
        Set a field in a hash.
        
        Args:
            key: Hash key
            field: Field name
            value: Field value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            serialized_value = json.dumps(value, default=self._json_serializer)
            result = self.redis_client.hset(key, field, serialized_value)
            return bool(result)
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing hash field value: {e}")
            return False
        except redis.RedisError as e:
            logger.error(f"Error setting hash field {field} in {key}: {e}")
            return False
    
    def get_hash_all(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get all fields from a hash.
        
        Args:
            key: Hash key
            
        Returns:
            Dictionary of all fields or None if error
        """
        try:
            data = self.redis_client.hgetall(key)
            if data:
                result = {}
                for field, value in data.items():
                    try:
                        result[field] = json.loads(value, parse_float=Decimal)
                    except json.JSONDecodeError:
                        result[field] = value
                return result
            return {}
        except redis.RedisError as e:
            logger.error(f"Error getting all hash fields from {key}: {e}")
            return None
    
    def clear_cache(self, pattern: str = '*') -> int:
        """
        Clear cache keys matching pattern.
        
        Args:
            pattern: Key pattern to match
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.get_keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.error(f"Error clearing cache with pattern {pattern}: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            info = self.redis_client.info()
            stats = {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))
            }
            return stats
        except redis.RedisError as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


# Global cache manager instance
_cache_manager: Optional[RedisCacheManager] = None


def get_cache_manager() -> RedisCacheManager:
    """
    Get or create the global Redis cache manager instance.
    
    Returns:
        RedisCacheManager instance
        
    Raises:
        RuntimeError: If Redis is not available
    """
    global _cache_manager
    if _cache_manager is None:
        # Default configuration - should be overridden by application config
        _cache_manager = RedisCacheManager()
    
    # Test connection
    try:
        _cache_manager.redis_client.ping()
        return _cache_manager
    except redis.ConnectionError:
        logger.warning("Redis not available, falling back to memory cache")
        # In a real implementation, you might fall back to a memory cache
        raise RuntimeError("Redis cache is not available")


def cache_decorator(ttl: int = 300, key_prefix: str = "cache"):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Cache TTL in seconds
        key_prefix: Prefix for cache keys
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Generate cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_result = cache_manager.get_cached_data(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache_manager.set_cached_data(cache_key, result, ttl)
            logger.debug(f"Cache miss for {cache_key}, cached result")
            
            return result
        return wrapper
    return decorator
