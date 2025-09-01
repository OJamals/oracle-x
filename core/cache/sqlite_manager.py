"""
SQLite cache manager for Oracle-X financial trading system.
Provides local SQLite caching for offline operation and fallback scenarios.
"""

import sqlite3
import json
import time
from datetime import datetime
from decimal import Decimal
from typing import Optional, Any, Dict, List, Union
import logging
from pathlib import Path
import threading

from core.types import MarketData, OptionContract, DataSource

logger = logging.getLogger(__name__)


class SQLiteCacheManager:
    """
    SQLite cache manager with thread-safe operations and automatic cleanup.
    """
    
    def __init__(self, db_path: Optional[str] = None, max_size_mb: int = 100):
        """
        Initialize SQLite cache manager.
        
        Args:
            db_path: Path to SQLite database file (default: in-memory)
            max_size_mb: Maximum database size in MB (for cleanup)
        """
        self.db_path = db_path or ":memory:"
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()
        # For in-memory databases, we need to maintain a single connection
        self._connection = None
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the database schema."""
        with self._lock:
            conn = self._get_connection()
            try:
                # Check if table exists first
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='cache'
                """)
                table_exists = cursor.fetchone() is not None
                
                if not table_exists:
                    logger.info("Creating cache table...")
                    conn.execute("""
                        CREATE TABLE cache (
                            key TEXT PRIMARY KEY,
                            value TEXT NOT NULL,
                            ttl INTEGER NOT NULL,
                            created_at REAL NOT NULL,
                            last_accessed REAL NOT NULL
                        )
                    """)
                    
                    conn.execute("""
                        CREATE INDEX idx_cache_ttl 
                        ON cache(ttl)
                    """)
                    
                    conn.execute("""
                        CREATE INDEX idx_cache_last_accessed 
                        ON cache(last_accessed)
                    """)
                    
                    conn.commit()
                    logger.info("Cache table created successfully")
                else:
                    logger.info("Cache table already exists")
                
                logger.info(f"SQLite cache initialized at {self.db_path}")
            except Exception as e:
                logger.error(f"Error initializing database: {e}")
                raise
            finally:
                # Only close the connection for file-based databases
                if self.db_path != ":memory:":
                    conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper configuration."""
        # For in-memory databases, we need to maintain a single connection
        if self.db_path == ":memory:":
            if self._connection is None:
                self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=NORMAL")
                self._connection.execute("PRAGMA foreign_keys=ON")
            return self._connection
        else:
            # For file-based databases, create a new connection each time
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn
    
    def _cleanup_old_entries(self) -> None:
        """Clean up expired entries and enforce size limits."""
        with self._lock:
            conn = self._get_connection()
            try:
                # Remove expired entries
                current_time = time.time()
                conn.execute(
                    "DELETE FROM cache WHERE ttl > 0 AND created_at + ttl <= ?",
                    (current_time,)
                )
                
                # Check database size and remove oldest entries if needed
                if self.db_path != ":memory:":
                    db_size_mb = Path(self.db_path).stat().st_size / (1024 * 1024)
                    if db_size_mb > self.max_size_mb:
                        # Remove 10% of oldest entries
                        cutoff_time = current_time - (86400 * 7)  # 1 week ago
                        conn.execute(
                            "DELETE FROM cache WHERE last_accessed < ? "
                            "ORDER BY last_accessed ASC LIMIT (SELECT COUNT(*) * 0.1 FROM cache)",
                            (cutoff_time,)
                        )
                
                conn.commit()
            finally:
                # Only close the connection for file-based databases
                if self.db_path != ":memory:":
                    conn.close()
    
    def get_cached_data(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """
        Get cached data with optional TTL extension.
        
        Args:
            key: Cache key
            ttl: Optional TTL to extend if data exists
            
        Returns:
            Cached data or None if not found/expired
        """
        with self._lock:
            conn = self._get_connection()
            try:
                current_time = time.time()
                
                # Get data and check if expired
                cursor = conn.execute(
                    "SELECT value, ttl, created_at FROM cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                value_str, original_ttl, created_at = row
                
                # Check if expired
                if original_ttl > 0 and created_at + original_ttl <= current_time:
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                    return None
                
                # Update last accessed time
                conn.execute(
                    "UPDATE cache SET last_accessed = ? WHERE key = ?",
                    (current_time, key)
                )
                
                # Extend TTL if requested
                if ttl is not None:
                    conn.execute(
                        "UPDATE cache SET ttl = ? WHERE key = ?",
                        (ttl, key)
                    )
                
                conn.commit()
                
                # Deserialize the value
                try:
                    return json.loads(value_str, parse_float=Decimal)
                except json.JSONDecodeError:
                    return value_str
                    
            except Exception as e:
                logger.error(f"Error getting cached data for key {key}: {e}")
                return None
            finally:
                # Only close the connection for file-based databases
                if self.db_path != ":memory:":
                    conn.close()
    
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
        with self._lock:
            conn = self._get_connection()
            try:
                current_time = time.time()
                
                # Serialize the data
                if isinstance(data, (MarketData, OptionContract)):
                    serialized_data = json.dumps(data.dict(), default=self._json_serializer)
                else:
                    serialized_data = json.dumps(data, default=self._json_serializer)
                
                # Insert or replace the cache entry
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, ttl, created_at, last_accessed) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, serialized_data, ttl, current_time, current_time)
                )
                
                conn.commit()
                
                # Perform periodic cleanup
                if current_time % 10 == 0:  # Cleanup every 10th call
                    self._cleanup_old_entries()
                
                return True
                
            except (TypeError, ValueError) as e:
                logger.error(f"Error serializing data for cache key {key}: {e}")
                return False
            except Exception as e:
                logger.error(f"Error setting cached data for key {key}: {e}")
                return False
            finally:
                # Only close the connection for file-based databases
                if self.db_path != ":memory:":
                    conn.close()
    
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
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return conn.total_changes > 0
            except Exception as e:
                logger.error(f"Error deleting cache key {key}: {e}")
                return False
            finally:
                # Only close the connection for file-based databases
                if self.db_path != ":memory:":
                    conn.close()
    
    def get_keys(self, pattern: str = '%') -> List[str]:
        """
        Get keys matching pattern (SQL LIKE pattern).
        
        Args:
            pattern: SQL LIKE pattern
            
        Returns:
            List of matching keys
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT key FROM cache WHERE key LIKE ?",
                    (pattern,)
                )
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Error getting keys with pattern {pattern}: {e}")
                return []
            finally:
                # Only close the connection for file-based databases
                if self.db_path != ":memory:":
                    conn.close()
    
    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds, or None if key doesn't exist
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT ttl, created_at FROM cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                ttl, created_at = row
                if ttl == 0:  # No expiration
                    return None
                
                remaining = created_at + ttl - time.time()
                return max(0, int(remaining))
                
            except Exception as e:
                logger.error(f"Error getting TTL for key {key}: {e}")
                return None
            finally:
                # Only close the connection for file-based databases
                if self.db_path != ":memory:":
                    conn.close()
    
    def clear_cache(self, pattern: str = '%') -> int:
        """
        Clear cache keys matching pattern.
        
        Args:
            pattern: SQL LIKE pattern to match
            
        Returns:
            Number of keys deleted
        """
        with self._lock:
            conn = self._get_connection()
            try:
                if pattern == '%':  # Clear all
                    conn.execute("DELETE FROM cache")
                else:
                    conn.execute("DELETE FROM cache WHERE key LIKE ?", (pattern,))
                
                conn.commit()
                return conn.total_changes
            except Exception as e:
                logger.error(f"Error clearing cache with pattern {pattern}: {e}")
                return 0
            finally:
                conn.close()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            conn = self._get_connection()
            try:
                stats = {}
                
                # Total entries
                cursor = conn.execute("SELECT COUNT(*) FROM cache")
                stats['total_entries'] = cursor.fetchone()[0]
                
                # Expired entries
                current_time = time.time()
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM cache WHERE ttl > 0 AND created_at + ttl <= ?",
                    (current_time,)
                )
                stats['expired_entries'] = cursor.fetchone()[0]
                
                # Memory usage (approximate)
                if self.db_path != ":memory:":
                    stats['db_size_mb'] = Path(self.db_path).stat().st_size / (1024 * 1024)
                else:
                    # Estimate in-memory size
                    cursor = conn.execute(
                        "SELECT SUM(LENGTH(key) + LENGTH(value) + 16) FROM cache"
                    )
                    stats['estimated_size_mb'] = cursor.fetchone()[0] / (1024 * 1024) if cursor.fetchone()[0] else 0
                
                # Hit rate (would need tracking)
                stats['hit_rate'] = 0.0  # Not tracked in this simple implementation
                
                return stats
                
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")
                return {}
            finally:
                conn.close()
    
    def vacuum(self) -> None:
        """Perform database vacuum to optimize storage."""
        if self.db_path != ":memory:":
            with self._lock:
                conn = self._get_connection()
                try:
                    conn.execute("VACUUM")
                    conn.commit()
                    logger.info("SQLite cache vacuum completed")
                except Exception as e:
                    logger.error(f"Error during vacuum: {e}")
                finally:
                    conn.close()


# Global SQLite cache manager instance
_sqlite_cache_manager: Optional[SQLiteCacheManager] = None


def get_sqlite_cache_manager(db_path: Optional[str] = None) -> SQLiteCacheManager:
    """
    Get or create the global SQLite cache manager instance.
    
    Args:
        db_path: Optional path to SQLite database file
        
    Returns:
        SQLiteCacheManager instance
    """
    global _sqlite_cache_manager
    if _sqlite_cache_manager is None:
        _sqlite_cache_manager = SQLiteCacheManager(db_path)
    return _sqlite_cache_manager


def get_unified_cache_manager(prefer_redis: bool = True, **kwargs) -> Union[SQLiteCacheManager, Any]:
    """
    Get a cache manager, preferring Redis if available, falling back to SQLite.
    
    Args:
        prefer_redis: Whether to prefer Redis over SQLite
        **kwargs: Additional arguments for cache managers
        
    Returns:
        Cache manager instance
    """
    if prefer_redis:
        try:
            from core.cache.redis_manager import get_cache_manager as get_redis_manager
            return get_redis_manager(**kwargs)
        except (ImportError, RuntimeError):
            logger.warning("Redis not available, falling back to SQLite cache")
            return get_sqlite_cache_manager(**kwargs)
    else:
        return get_sqlite_cache_manager(**kwargs)
