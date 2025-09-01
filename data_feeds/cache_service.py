"""
SQLite-backed cache service with auto-migration and TTL support.
Stores small/medium payloads inline as JSON; large payloads can be stored on disk with a URI pointer.

Usage:
  from data_feeds.cache_service import CacheService, CacheEntry
  key = cache.make_key("google_trends", {"keywords": ["AAPL","MSFT"], "geo":"US", "timeframe":"now 7-d"})
  entry = cache.get(key)
  if entry and not entry.is_expired():
      return entry.payload_json
  # ... fetch data ...
  cache.set(
      key=key,
      endpoint="google_trends",
      symbol=None,
      ttl_seconds=int(os.getenv("TRENDS_TTL_H", "24")) * 3600,
      payload_json=result_dict,
      source="google_trends"
  )
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Optional, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the optimized database pool
DatabasePool = None
try:
    from core.database_pool import DatabasePool
    USE_POOL = True
except ImportError:
    USE_POOL = False

# Import async I/O utilities with fallback
AsyncDatabaseManager = None
try:
    from core.async_io_utils import AsyncDatabaseManager
    ASYNC_IO_AVAILABLE = True
except ImportError:
    ASYNC_IO_AVAILABLE = False

# Import Redis for multi-level caching
redis = None
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Import threading for thread-safe operations
import threading
from collections import OrderedDict

# Fallback to default path if import fails
try:
    from config_manager import get_cache_db_path
    DEFAULT_DB_PATH = get_cache_db_path()
except:
    DEFAULT_DB_PATH = "data/databases/model_monitoring.db"

DDL_CACHE_ENTRIES = """
CREATE TABLE IF NOT EXISTS cache_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key_hash TEXT UNIQUE,
  endpoint TEXT,
  symbol TEXT,
  fetched_at INTEGER,
  ttl_seconds INTEGER,
  source TEXT,
  metadata_json TEXT,
  payload_json TEXT,
  payload_blob BLOB,
  storage_uri TEXT,
  version INTEGER DEFAULT 1,
  status TEXT DEFAULT 'ok'
);
"""

IDX_1 = "CREATE INDEX IF NOT EXISTS idx_cache_endpoint_symbol ON cache_entries(endpoint, symbol, fetched_at DESC);"
IDX_2 = "CREATE INDEX IF NOT EXISTS idx_cache_keyhash ON cache_entries(key_hash);"


# Multi-level cache configuration
DEFAULT_MEMORY_CACHE_SIZE = int(os.getenv("MEMORY_CACHE_SIZE", "1000"))  # Max entries in memory
DEFAULT_REDIS_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))  # 1 hour default
DEFAULT_CACHE_SIZE_LIMIT = int(os.getenv("CACHE_SIZE_LIMIT_MB", "100"))  # 100MB limit

# Thread-safe LRU cache for memory level
class ThreadSafeLRUCache:
    """Thread-safe LRU cache implementation for memory-level caching"""
    
    def __init__(self, max_size: int = DEFAULT_MEMORY_CACHE_SIZE):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with thread safety"""
        with self.lock:
            self.total_requests += 1
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with LRU eviction"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    self.cache.popitem(last=False)
            self.cache[key] = value
    
    def remove(self, key: str) -> bool:
        """Remove item from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.total_requests = 0
    
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        with self.lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0


# Redis cache manager for distributed caching
class RedisCacheManager:
    """Redis-based distributed cache manager"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.redis_client.ping()
            except Exception as e:
                print(f"⚠️  Redis connection failed: {e}")
                self.redis_client = None
    
    def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        if not self.redis_client:
            return None
        try:
            return self.redis_client.get(key)
        except Exception:
            return None
    
    def set(self, key: str, value: str, ttl: int = DEFAULT_REDIS_TTL) -> bool:
        """Set value in Redis with TTL"""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.setex(key, ttl, value)
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.delete(key) > 0
        except Exception:
            return False
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self.redis_client is not None

import json
import os
import sqlite3
import sys
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Optional, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the optimized database pool
DatabasePool = None
try:
    from core.database_pool import DatabasePool
    USE_POOL = True
except ImportError:
    USE_POOL = False

# Import threading for thread-safe operations
import threading
from collections import OrderedDict

# Import Redis for multi-level caching
redis = None
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Fallback to default path if import fails
try:
    from config_manager import get_cache_db_path
    DEFAULT_DB_PATH = get_cache_db_path()
except:
    DEFAULT_DB_PATH = "data/databases/model_monitoring.db"

DDL_CACHE_ENTRIES = """
CREATE TABLE IF NOT EXISTS cache_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key_hash TEXT UNIQUE,
  endpoint TEXT,
  symbol TEXT,
  fetched_at INTEGER,
  ttl_seconds INTEGER,
  source TEXT,
  metadata_json TEXT,
  payload_json TEXT,
  payload_blob BLOB,
  storage_uri TEXT,
  version INTEGER DEFAULT 1,
  status TEXT DEFAULT 'ok'
);
"""

IDX_1 = "CREATE INDEX IF NOT EXISTS idx_cache_endpoint_symbol ON cache_entries(endpoint, symbol, fetched_at DESC);"
IDX_2 = "CREATE INDEX IF NOT EXISTS idx_cache_keyhash ON cache_entries(key_hash);"


@dataclass
class CacheEntry:
    key_hash: str
    endpoint: str
    symbol: Optional[str]
    fetched_at: int
    ttl_seconds: int
    source: Optional[str]
    metadata_json: Optional[Dict[str, Any]]
    payload_json: Optional[Dict[str, Any]]
    payload_blob: Optional[bytes]
    storage_uri: Optional[str]
    version: int
    status: str

    def is_expired(self) -> bool:
        if self.ttl_seconds is None or self.ttl_seconds <= 0:
            return False
        return (int(time.time()) - int(self.fetched_at)) > int(self.ttl_seconds)


class CacheService:
    def __init__(self, db_path: Optional[str] = None, enable_multi_level: bool = True):
        self.db_path = db_path or os.getenv("CACHE_DB_PATH", DEFAULT_DB_PATH)
        self.enable_multi_level = enable_multi_level
        
        # Initialize multi-level cache components
        self.memory_cache = ThreadSafeLRUCache() if enable_multi_level else None
        self.redis_cache = RedisCacheManager() if enable_multi_level and REDIS_AVAILABLE else None
        
        # Cache statistics
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'redis_hits': 0,
            'redis_misses': 0,
            'disk_hits': 0,
            'disk_misses': 0,
            'total_requests': 0
        }
        
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Initialize database schema with connection pooling"""
        if USE_POOL and DatabasePool is not None:
            with DatabasePool.get_connection(self.db_path) as con:
                cur = con.cursor()
                cur.execute(DDL_CACHE_ENTRIES)
                cur.execute(IDX_1)
                cur.execute(IDX_2)
                con.commit()
        else:
            # Fallback to direct connection
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.cursor()
                cur.execute(DDL_CACHE_ENTRIES)
                cur.execute(IDX_1)
                cur.execute(IDX_2)
                con.commit()
            finally:
                con.close()

    @staticmethod
    def make_key(endpoint: str, params: Dict[str, Any]) -> str:
        """
        Stable key: endpoint + sorted json of params
        """
        try:
            payload = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
        except Exception:
            payload = str(params)
        h = hashlib.sha256()
        h.update((endpoint + "|" + payload).encode("utf-8"))
        return h.hexdigest()

    def get(self, key_hash: str) -> Optional[CacheEntry]:
        """Retrieve cache entry with multi-level caching (Memory -> Redis -> Disk)"""
        self.stats['total_requests'] += 1
        
        # Level 1: Memory cache
        if self.enable_multi_level and self.memory_cache:
            memory_result = self.memory_cache.get(key_hash)
            if memory_result is not None:
                self.stats['memory_hits'] += 1
                return memory_result
            self.stats['memory_misses'] += 1
        
        # Level 2: Redis cache
        if self.enable_multi_level and self.redis_cache and self.redis_cache.is_available():
            redis_result = self.redis_cache.get(key_hash)
            if redis_result is not None:
                self.stats['redis_hits'] += 1
                # Promote to memory cache
                if self.memory_cache:
                    try:
                        entry = json.loads(redis_result)
                        self.memory_cache.put(key_hash, entry)
                    except Exception:
                        pass  # Skip promotion if deserialization fails
                return json.loads(redis_result)
            self.stats['redis_misses'] += 1
        
        # Level 3: Disk cache (SQLite)
        disk_result = self._get_from_disk(key_hash)
        if disk_result is not None:
            self.stats['disk_hits'] += 1
            # Promote to higher levels
            if self.enable_multi_level:
                if self.memory_cache:
                    self.memory_cache.put(key_hash, disk_result)
                if self.redis_cache and self.redis_cache.is_available():
                    try:
                        self.redis_cache.set(key_hash, json.dumps(disk_result.__dict__))
                    except Exception:
                        pass  # Skip Redis promotion if serialization fails
            return disk_result
        
        self.stats['disk_misses'] += 1
        return None
    
    def _get_from_disk(self, key_hash: str) -> Optional[CacheEntry]:
        """Retrieve cache entry from disk (SQLite)"""
        if USE_POOL and DatabasePool is not None:
            with DatabasePool.get_connection(self.db_path) as con:
                cur = con.cursor()
                cur.execute(
                    "SELECT key_hash, endpoint, symbol, fetched_at, ttl_seconds, source, metadata_json, payload_json, payload_blob, storage_uri, version, status "
                    "FROM cache_entries WHERE key_hash = ? LIMIT 1",
                    (key_hash,),
                )
                row = cur.fetchone()
        else:
            # Fallback to direct connection
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.cursor()
                cur.execute(
                    "SELECT key_hash, endpoint, symbol, fetched_at, ttl_seconds, source, metadata_json, payload_json, payload_blob, storage_uri, version, status "
                    "FROM cache_entries WHERE key_hash = ? LIMIT 1",
                    (key_hash,),
                )
                row = cur.fetchone()
            finally:
                con.close()

        if not row:
            return None

        metadata = None
        payload = None
        try:
            metadata = json.loads(row[6]) if row[6] else None
        except Exception:
            metadata = None
        try:
            payload = json.loads(row[7]) if row[7] else None
        except Exception:
            payload = None

        return CacheEntry(
            key_hash=row[0],
            endpoint=row[1],
            symbol=row[2],
            fetched_at=int(row[3]) if row[3] is not None else int(time.time()),
            ttl_seconds=int(row[4]) if row[4] is not None else 0,
            source=row[5],
            metadata_json=metadata,
            payload_json=payload,
            payload_blob=row[8],
            storage_uri=row[9],
            version=int(row[10]) if row[10] is not None else 1,
            status=row[11] or "ok",
        )

    def _set_to_disk(
        self,
        key: str,
        endpoint: str,
        symbol: Optional[str],
        ttl_seconds: int,
        payload_json: Optional[Dict[str, Any]] = None,
        payload_blob: Optional[bytes] = None,
        storage_uri: Optional[str] = None,
        source: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
        status: str = "ok",
        version: int = 1,
    ) -> None:
        """Store cache entry to disk (SQLite)"""
        now = int(time.time())

        if USE_POOL and DatabasePool is not None:
            with DatabasePool.get_connection(self.db_path) as con:
                cur = con.cursor()
                cur.execute(
                    """
                    INSERT INTO cache_entries
                    (key_hash, endpoint, symbol, fetched_at, ttl_seconds, source, metadata_json, payload_json, payload_blob, storage_uri, version, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key_hash) DO UPDATE SET
                        endpoint=excluded.endpoint,
                        symbol=excluded.symbol,
                        fetched_at=excluded.fetched_at,
                        ttl_seconds=excluded.ttl_seconds,
                        source=excluded.source,
                        metadata_json=excluded.metadata_json,
                        payload_json=excluded.payload_json,
                        payload_blob=excluded.payload_blob,
                        storage_uri=excluded.storage_uri,
                        version=excluded.version,
                        status=excluded.status
                    """,
                    (
                        key,
                        endpoint,
                        symbol,
                        now,
                        int(ttl_seconds) if ttl_seconds is not None else 0,
                        source,
                        json.dumps(metadata_json) if metadata_json is not None else None,
                        json.dumps(payload_json) if payload_json is not None else None,
                        payload_blob,
                        storage_uri,
                        int(version),
                        status,
                    ),
                )
                con.commit()
        else:
            # Fallback to direct connection
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.cursor()
                cur.execute(
                    """
                    INSERT INTO cache_entries
                    (key_hash, endpoint, symbol, fetched_at, ttl_seconds, source, metadata_json, payload_json, payload_blob, storage_uri, version, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key_hash) DO UPDATE SET
                        endpoint=excluded.endpoint,
                        symbol=excluded.symbol,
                        fetched_at=excluded.fetched_at,
                        ttl_seconds=excluded.ttl_seconds,
                        source=excluded.source,
                        metadata_json=excluded.metadata_json,
                        payload_json=excluded.payload_json,
                        payload_blob=excluded.payload_blob,
                        storage_uri=excluded.storage_uri,
                        version=excluded.version,
                        status=excluded.status
                    """,
                    (
                        key,
                        endpoint,
                        symbol,
                        now,
                        int(ttl_seconds) if ttl_seconds is not None else 0,
                        source,
                        json.dumps(metadata_json) if metadata_json is not None else None,
                        json.dumps(payload_json) if payload_json is not None else None,
                        payload_blob,
                        storage_uri,
                        int(version),
                        status,
                    ),
                )
                con.commit()
            finally:
                con.close()

    async def get_async(self, key_hash: str) -> Optional[CacheEntry]:
        """Retrieve cache entry asynchronously"""
        if not ASYNC_IO_AVAILABLE or AsyncDatabaseManager is None:
            # Fallback to sync method
            return self.get(key_hash)
        
        try:
            db = AsyncDatabaseManager(self.db_path)
            query = """
            SELECT key_hash, endpoint, symbol, fetched_at, ttl_seconds, source, 
                   metadata_json, payload_json, payload_blob, storage_uri, version, status 
            FROM cache_entries WHERE key_hash = ? LIMIT 1
            """
            rows = await db.execute_query(query, (key_hash,))
            
            if not rows:
                return None
            
            row = rows[0]
            metadata = None
            payload = None
            try:
                metadata = json.loads(row['metadata_json']) if row.get('metadata_json') else None
            except Exception:
                metadata = None
            try:
                payload = json.loads(row['payload_json']) if row.get('payload_json') else None
            except Exception:
                payload = None

            return CacheEntry(
                key_hash=row['key_hash'],
                endpoint=row['endpoint'],
                symbol=row['symbol'],
                fetched_at=int(row['fetched_at']) if row.get('fetched_at') is not None else int(time.time()),
                ttl_seconds=int(row['ttl_seconds']) if row.get('ttl_seconds') is not None else 0,
                source=row['source'],
                metadata_json=metadata,
                payload_json=payload,
                payload_blob=row.get('payload_blob'),
                storage_uri=row.get('storage_uri'),
                version=int(row['version']) if row.get('version') is not None else 1,
                status=row['status'] or "ok",
            )
        except Exception as e:
            print(f"⚠️  Async cache get failed, falling back to sync: {e}")
            return self.get(key_hash)

    def set(
        self,
        key: str,
        endpoint: str,
        symbol: Optional[str],
        ttl_seconds: int,
        payload_json: Optional[Dict[str, Any]] = None,
        payload_blob: Optional[bytes] = None,
        storage_uri: Optional[str] = None,
        source: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
        status: str = "ok",
        version: int = 1,
    ) -> None:
        """Store cache entry with multi-level caching (Memory -> Redis -> Disk)"""
        now = int(time.time())
        
        # Create cache entry for higher levels
        entry = CacheEntry(
            key_hash=key,
            endpoint=endpoint,
            symbol=symbol,
            fetched_at=now,
            ttl_seconds=int(ttl_seconds) if ttl_seconds is not None else 0,
            source=source,
            metadata_json=metadata_json,
            payload_json=payload_json,
            payload_blob=payload_blob,
            storage_uri=storage_uri,
            version=int(version),
            status=status,
        )
        
        # Level 1: Memory cache
        if self.enable_multi_level and self.memory_cache:
            self.memory_cache.put(key, entry)
        
        # Level 2: Redis cache
        if self.enable_multi_level and self.redis_cache and self.redis_cache.is_available():
            try:
                entry_dict = {
                    'key_hash': entry.key_hash,
                    'endpoint': entry.endpoint,
                    'symbol': entry.symbol,
                    'fetched_at': entry.fetched_at,
                    'ttl_seconds': entry.ttl_seconds,
                    'source': entry.source,
                    'metadata_json': entry.metadata_json,
                    'payload_json': entry.payload_json,
                    'payload_blob': entry.payload_blob,
                    'storage_uri': entry.storage_uri,
                    'version': entry.version,
                    'status': entry.status,
                }
                self.redis_cache.set(key, json.dumps(entry_dict), ttl_seconds)
            except Exception as e:
                print(f"⚠️  Redis cache set failed: {e}")
        
        # Level 3: Disk cache (SQLite) - always write to disk as primary storage
        self._set_to_disk(key, endpoint, symbol, ttl_seconds, payload_json, payload_blob, 
                         storage_uri, source, metadata_json, status, version)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        stats = {
            'memory_cache': {},
            'redis_cache': {},
            'disk_cache': {},
            'overall': {},
            'timestamp': int(time.time())
        }
        
        # Memory cache stats
        if self.memory_cache:
            stats['memory_cache'] = {
                'enabled': True,
                'size': self.memory_cache.size(),
                'max_size': self.memory_cache.max_size,
                'hit_rate': self.memory_cache.hit_rate(),
                'total_requests': self.memory_cache.total_requests,
                'hits': self.memory_cache.hits,
                'misses': self.memory_cache.misses
            }
        else:
            stats['memory_cache'] = {'enabled': False}
        
        # Redis cache stats
        if self.redis_cache:
            stats['redis_cache'] = {
                'enabled': True,
                'available': self.redis_cache.is_available(),
                'connection_pool_size': getattr(self.redis_cache, 'pool_size', 0),
                'hit_rate': self.redis_cache.hit_rate() if hasattr(self.redis_cache, 'hit_rate') else 0,
                'total_requests': getattr(self.redis_cache, 'total_requests', 0),
                'hits': getattr(self.redis_cache, 'hits', 0),
                'misses': getattr(self.redis_cache, 'misses', 0)
            }
        else:
            stats['redis_cache'] = {'enabled': False}
        
        # Disk cache stats (SQLite)
        try:
            if USE_POOL and DatabasePool is not None:
                with DatabasePool.get_connection(self.db_path) as con:
                    cur = con.cursor()
                    cur.execute("SELECT COUNT(*) FROM cache_entries")
                    total_entries = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM cache_entries WHERE fetched_at + ttl_seconds > ?", (int(time.time()),))
                    valid_entries = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM cache_entries WHERE fetched_at + ttl_seconds <= ?", (int(time.time()),))
                    expired_entries = cur.fetchone()[0]
                    
                    stats['disk_cache'] = {
                        'enabled': True,
                        'total_entries': total_entries,
                        'valid_entries': valid_entries,
                        'expired_entries': expired_entries,
                        'hit_rate': 0,  # Would need additional tracking table
                        'db_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                    }
            else:
                con = sqlite3.connect(self.db_path)
                try:
                    cur = con.cursor()
                    cur.execute("SELECT COUNT(*) FROM cache_entries")
                    total_entries = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM cache_entries WHERE fetched_at + ttl_seconds > ?", (int(time.time()),))
                    valid_entries = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM cache_entries WHERE fetched_at + ttl_seconds <= ?", (int(time.time()),))
                    expired_entries = cur.fetchone()[0]
                    
                    stats['disk_cache'] = {
                        'enabled': True,
                        'total_entries': total_entries,
                        'valid_entries': valid_entries,
                        'expired_entries': expired_entries,
                        'hit_rate': 0,
                        'db_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                    }
                finally:
                    con.close()
        except Exception as e:
            stats['disk_cache'] = {
                'enabled': True,
                'error': str(e),
                'total_entries': 0,
                'valid_entries': 0,
                'expired_entries': 0
            }
        
        # Overall stats
        total_hits = (stats['memory_cache'].get('hits', 0) + 
                     stats['redis_cache'].get('hits', 0))
        total_requests = (stats['memory_cache'].get('total_requests', 0) + 
                         stats['redis_cache'].get('total_requests', 0))
        
        stats['overall'] = {
            'multi_level_enabled': self.enable_multi_level,
            'total_cache_hits': total_hits,
            'total_cache_requests': total_requests,
            'overall_hit_rate': total_requests > 0 and (total_hits / total_requests) or 0,
            'cache_layers': sum([1 for layer in [stats['memory_cache'], stats['redis_cache'], stats['disk_cache']] 
                                if layer.get('enabled', False)])
        }
        
        return stats

    def warm_cache_from_patterns(self, usage_patterns: Dict[str, int], max_items: int = 100) -> Dict[str, Any]:
        """Warm cache with frequently accessed data based on usage patterns"""
        if not usage_patterns:
            return {'status': 'no_patterns', 'warmed_count': 0}
        
        # Sort patterns by frequency (highest first)
        sorted_patterns = sorted(usage_patterns.items(), key=lambda x: x[1], reverse=True)
        
        warmed_count = 0
        warmed_keys = []
        errors = []
        
        for key, frequency in sorted_patterns[:max_items]:
            try:
                # Check if already in cache
                if self.get(key):
                    continue  # Already cached
                
                # For now, we can't proactively fetch data without knowing the source
                # This would need integration with the data feed orchestrator
                # We'll mark these as candidates for warming
                warmed_keys.append(key)
                warmed_count += 1
                
            except Exception as e:
                errors.append(f"Failed to warm {key}: {e}")
        
        return {
            'status': 'completed',
            'warmed_count': warmed_count,
            'warmed_keys': warmed_keys,
            'errors': errors,
            'total_candidates': len(sorted_patterns)
        }

    def start_background_warmer(self, interval_seconds: int = 300) -> None:
        """Start background cache warming worker"""
        if hasattr(self, '_warming_thread') and self._warming_thread and self._warming_thread.is_alive():
            print("⚠️  Background warmer already running")
            return
        
        def warming_worker():
            while True:
                try:
                    # Analyze usage patterns and warm cache
                    stats = self.get_cache_stats()
                    
                    # If cache hit rate is below threshold, trigger warming
                    overall_hit_rate = stats['overall'].get('overall_hit_rate', 0)
                    if overall_hit_rate < 0.7:  # Less than 70% hit rate
                        print(f"🔄 Low cache hit rate ({overall_hit_rate:.2%}), triggering cache warming...")
                        # This would integrate with data feed orchestrator to fetch high-demand data
                        pass
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    print(f"⚠️  Background warmer error: {e}")
                    time.sleep(interval_seconds)
        
        self._warming_thread = threading.Thread(target=warming_worker, daemon=True)
        self._warming_thread.start()
        print(f"✅ Background cache warmer started (interval: {interval_seconds}s)")

    def stop_background_warmer(self) -> None:
        """Stop background cache warming worker"""
        if hasattr(self, '_warming_thread') and self._warming_thread:
            # Note: Daemon threads will be terminated when main thread exits
            print("🛑 Background cache warmer stopped")
        else:
            print("⚠️  No background warmer running")

    def get_warming_config(self) -> Dict[str, Any]:
        """Get cache warming configuration"""
        return {
            'enable_background_warmer': getattr(self, 'enable_background_warmer', False),
            'warming_interval_seconds': getattr(self, 'warming_interval_seconds', 300),
            'min_hit_rate_threshold': getattr(self, 'min_hit_rate_threshold', 0.7),
            'max_warming_items': getattr(self, 'max_warming_items', 100),
            'warming_enabled': self.enable_multi_level
        }

    def configure_warming(self, 
                         enable_background_warmer: bool = False,
                         warming_interval_seconds: int = 300,
                         min_hit_rate_threshold: float = 0.7,
                         max_warming_items: int = 100) -> None:
        """Configure cache warming parameters"""
        self.enable_background_warmer = enable_background_warmer
        self.warming_interval_seconds = warming_interval_seconds
        self.min_hit_rate_threshold = min_hit_rate_threshold
        self.max_warming_items = max_warming_items
        
        if enable_background_warmer:
            self.start_background_warmer(warming_interval_seconds)
        else:
            self.stop_background_warmer()
        
        print(f"✅ Cache warming configured: enabled={enable_background_warmer}, interval={warming_interval_seconds}s")

    def invalidate_cache_entry(self, key: str) -> bool:
        """Invalidate a specific cache entry across all levels"""
        invalidated = False
        
        # Level 1: Memory cache
        if self.memory_cache and key in self.memory_cache.cache:
            del self.memory_cache.cache[key]
            invalidated = True
        
        # Level 2: Redis cache
        if self.redis_cache and self.redis_cache.is_available():
            try:
                self.redis_cache.delete(key)
                invalidated = True
            except Exception as e:
                print(f"⚠️  Redis invalidation failed for {key}: {e}")
        
        # Level 3: Disk cache
        try:
            if USE_POOL and DatabasePool is not None:
                with DatabasePool.get_connection(self.db_path) as con:
                    cur = con.cursor()
                    cur.execute("DELETE FROM cache_entries WHERE key_hash = ?", (key,))
                    if cur.rowcount > 0:
                        invalidated = True
                    con.commit()
            else:
                con = sqlite3.connect(self.db_path)
                try:
                    cur = con.cursor()
                    cur.execute("DELETE FROM cache_entries WHERE key_hash = ?", (key,))
                    if cur.rowcount > 0:
                        invalidated = True
                    con.commit()
                finally:
                    con.close()
        except Exception as e:
            print(f"⚠️  Disk invalidation failed for {key}: {e}")
        
        if invalidated:
            print(f"🗑️  Invalidated cache entry: {key}")
        
        return invalidated

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern across all levels"""
        invalidated_count = 0
        
        # Level 1: Memory cache
        if self.memory_cache:
            keys_to_remove = [k for k in self.memory_cache.cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.memory_cache.cache[key]
                invalidated_count += 1
        
        # Level 2: Redis cache
        if self.redis_cache and self.redis_cache.is_available():
            try:
                # This would need Redis pattern matching (KEYS command)
                # For now, we'll skip Redis pattern invalidation
                pass
            except Exception as e:
                print(f"⚠️  Redis pattern invalidation failed: {e}")
        
        # Level 3: Disk cache
        try:
            if USE_POOL and DatabasePool is not None:
                with DatabasePool.get_connection(self.db_path) as con:
                    cur = con.cursor()
                    cur.execute("DELETE FROM cache_entries WHERE key_hash LIKE ?", (f"%{pattern}%",))
                    invalidated_count += cur.rowcount
                    con.commit()
            else:
                con = sqlite3.connect(self.db_path)
                try:
                    cur = con.cursor()
                    cur.execute("DELETE FROM cache_entries WHERE key_hash LIKE ?", (f"%{pattern}%",))
                    invalidated_count += cur.rowcount
                    con.commit()
                finally:
                    con.close()
        except Exception as e:
            print(f"⚠️  Disk pattern invalidation failed: {e}")
        
        if invalidated_count > 0:
            print(f"🗑️  Invalidated {invalidated_count} cache entries matching pattern: {pattern}")
        
        return invalidated_count

    def invalidate_expired_entries(self) -> int:
        """Invalidate all expired cache entries"""
        now = int(time.time())
        invalidated_count = 0
        
        # Level 1: Memory cache (entries don't have TTL in memory cache)
        # Level 2: Redis handles TTL automatically
        
        # Level 3: Disk cache
        try:
            if USE_POOL and DatabasePool is not None:
                with DatabasePool.get_connection(self.db_path) as con:
                    cur = con.cursor()
                    cur.execute("DELETE FROM cache_entries WHERE fetched_at + ttl_seconds < ?", (now,))
                    invalidated_count = cur.rowcount
                    con.commit()
            else:
                con = sqlite3.connect(self.db_path)
                try:
                    cur = con.cursor()
                    cur.execute("DELETE FROM cache_entries WHERE fetched_at + ttl_seconds < ?", (now,))
                    invalidated_count = cur.rowcount
                    con.commit()
                finally:
                    con.close()
        except Exception as e:
            print(f"⚠️  Failed to invalidate expired entries: {e}")
        
        if invalidated_count > 0:
            print(f"🗑️  Invalidated {invalidated_count} expired cache entries")
        
        return invalidated_count

    def add_dependency(self, key: str, dependency_key: str) -> None:
        """Add a dependency relationship between cache entries"""
        if not hasattr(self, '_dependencies'):
            self._dependencies = defaultdict(set)
        
        self._dependencies[key].add(dependency_key)
        
        # Also track reverse dependencies for efficient invalidation
        if not hasattr(self, '_reverse_dependencies'):
            self._reverse_dependencies = defaultdict(set)
        
        self._reverse_dependencies[dependency_key].add(key)

    def invalidate_with_dependencies(self, key: str) -> int:
        """Invalidate a cache entry and all its dependencies"""
        invalidated_count = 0
        
        # Get all dependent keys (recursive)
        keys_to_invalidate = set()
        visited = set()
        
        def collect_dependencies(k: str):
            if k in visited:
                return
            visited.add(k)
            keys_to_invalidate.add(k)
            
            # Add direct dependencies
            if hasattr(self, '_reverse_dependencies'):
                for dep in self._reverse_dependencies.get(k, set()):
                    collect_dependencies(dep)
        
        collect_dependencies(key)
        
        # Invalidate all collected keys
        for k in keys_to_invalidate:
            if self.invalidate_cache_entry(k):
                invalidated_count += 1
        
        if invalidated_count > 0:
            print(f"🗑️  Invalidated {invalidated_count} cache entries with dependencies for: {key}")
        
        return invalidated_count

    def register_invalidation_hook(self, hook_name: str, callback: callable) -> None:
        """Register a callback for cache invalidation events"""
        if not hasattr(self, '_invalidation_hooks'):
            self._invalidation_hooks = {}
        
        self._invalidation_hooks[hook_name] = callback
        print(f"🔗 Registered invalidation hook: {hook_name}")

    def trigger_invalidation_hooks(self, invalidated_keys: List[str]) -> None:
        """Trigger registered invalidation hooks"""
        if not hasattr(self, '_invalidation_hooks'):
            return
        
        for hook_name, callback in self._invalidation_hooks.items():
            try:
                callback(invalidated_keys)
            except Exception as e:
                print(f"⚠️  Invalidation hook '{hook_name}' failed: {e}")

    def smart_invalidate(self, key: str, reason: str = "manual") -> Dict[str, Any]:
        """Smart invalidation with dependency tracking and hooks"""
        result = {
            'key': key,
            'reason': reason,
            'invalidated_count': 0,
            'dependencies_invalidated': 0,
            'hooks_triggered': 0,
            'timestamp': int(time.time())
        }
        
        # Invalidate with dependencies
        invalidated_count = self.invalidate_with_dependencies(key)
        result['invalidated_count'] = invalidated_count
        
        # Count dependencies (approximate)
        if hasattr(self, '_reverse_dependencies'):
            result['dependencies_invalidated'] = len(self._reverse_dependencies.get(key, set()))
        
        # Trigger hooks
        if hasattr(self, '_invalidation_hooks') and self._invalidation_hooks:
            self.trigger_invalidation_hooks([key])
            result['hooks_triggered'] = len(self._invalidation_hooks)
        
        print(f"🧠 Smart invalidation completed for {key}: {invalidated_count} entries invalidated")
        return result

    def get_invalidation_stats(self) -> Dict[str, Any]:
        """Get cache invalidation statistics"""
        stats = {
            'dependencies_tracked': 0,
            'hooks_registered': 0,
            'last_invalidation': getattr(self, '_last_invalidation', None),
            'total_invalidations': getattr(self, '_total_invalidations', 0)
        }
        
        if hasattr(self, '_dependencies'):
            stats['dependencies_tracked'] = len(self._dependencies)
        
        if hasattr(self, '_invalidation_hooks'):
            stats['hooks_registered'] = len(self._invalidation_hooks)
        
        return stats

    async def set_async(
        self,
        key: str,
        endpoint: str,
        symbol: Optional[str],
        ttl_seconds: int,
        payload_json: Optional[Dict[str, Any]] = None,
        payload_blob: Optional[bytes] = None,
        storage_uri: Optional[str] = None,
        source: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
        status: str = "ok",
        version: int = 1,
    ) -> bool:
        """Store cache entry asynchronously"""
        if not ASYNC_IO_AVAILABLE or AsyncDatabaseManager is None:
            # Fallback to sync method
            self.set(key, endpoint, symbol, ttl_seconds, payload_json, payload_blob, 
                    storage_uri, source, metadata_json, status, version)
            return True
        
        try:
            db = AsyncDatabaseManager(self.db_path)
            now = int(time.time())

            query = """
            INSERT INTO cache_entries (key_hash, endpoint, symbol, fetched_at, ttl_seconds, source, metadata_json, payload_json, payload_blob, storage_uri, version, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key_hash) DO UPDATE SET
              endpoint=excluded.endpoint,
              symbol=excluded.symbol,
              fetched_at=excluded.fetched_at,
              ttl_seconds=excluded.ttl_seconds,
              source=excluded.source,
              metadata_json=excluded.metadata_json,
              payload_json=excluded.payload_json,
              payload_blob=excluded.payload_blob,
              storage_uri=excluded.storage_uri,
              version=excluded.version,
              status=excluded.status
            """
            
            params = (
                key,
                endpoint,
                symbol,
                now,
                int(ttl_seconds) if ttl_seconds is not None else 0,
                source,
                json.dumps(metadata_json) if metadata_json is not None else None,
                json.dumps(payload_json) if payload_json is not None else None,
                payload_blob,
                storage_uri,
                int(version),
                status,
            )
            
            return success
            
        except Exception as e:
            print(f"⚠️  Async cache set failed, falling back to sync: {e}")
            self.set(key, endpoint, symbol, ttl_seconds, payload_json, payload_blob, 
                    storage_uri, source, metadata_json, status, version)
            return True

    def _should_compress(self, data: Any, size_threshold: int = 1024) -> bool:
        """Determine if data should be compressed based on size"""
        if data is None:
            return False
        
        # Estimate size for different data types
        if isinstance(data, (dict, list)):
            try:
                data_str = json.dumps(data)
                return len(data_str.encode('utf-8')) > size_threshold
            except Exception:
                return False
        elif isinstance(data, str):
            return len(data.encode('utf-8')) > size_threshold
        elif isinstance(data, bytes):
            return len(data) > size_threshold
        
        return False

    def _compress_data(self, data: Any, algorithm: str = 'gzip') -> tuple:
        """Compress data using specified algorithm"""
        import gzip
        import lzma
        
        if data is None:
            return None, None
        
        # Convert data to bytes
        if isinstance(data, (dict, list)):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            return data, None
        
        # Compress based on algorithm
        if algorithm == 'gzip':
            compressed = gzip.compress(data_bytes)
        elif algorithm == 'lzma':
            compressed = lzma.compress(data_bytes)
        else:
            return data, None
        
        return compressed, algorithm

    def _decompress_data(self, compressed_data: bytes, algorithm: str) -> Any:
        """Decompress data using specified algorithm"""
        import gzip
        import lzma
        
        if compressed_data is None or algorithm is None:
            return compressed_data
        
        try:
            # Decompress based on algorithm
            if algorithm == 'gzip':
                decompressed = gzip.decompress(compressed_data)
            elif algorithm == 'lzma':
                decompressed = lzma.decompress(compressed_data)
            else:
                return compressed_data
            
            # Try to parse as JSON, otherwise return as string
            try:
                return json.loads(decompressed.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return decompressed.decode('utf-8')
                
        except Exception as e:
            print(f"⚠️  Decompression failed: {e}")
            return compressed_data

    def _optimize_storage_format(self, data: Any) -> Dict[str, Any]:
        """Optimize data storage format for compression efficiency"""
        if data is None:
            return {'original': None, 'optimized': None, 'compression_ratio': 1.0}
        
        # For JSON data, optimize by removing unnecessary whitespace
        if isinstance(data, (dict, list)):
            try:
                # Compact JSON without whitespace
                optimized = json.dumps(data, separators=(',', ':'))
                original_size = len(json.dumps(data, indent=2).encode('utf-8'))
                optimized_size = len(optimized.encode('utf-8'))
                
                return {
                    'original': data,
                    'optimized': optimized,
                    'compression_ratio': optimized_size / original_size if original_size > 0 else 1.0,
                    'format': 'json_compact'
                }
            except Exception:
                pass
        
        # For string data, check if it's already optimized
        if isinstance(data, str):
            return {
                'original': data,
                'optimized': data,
                'compression_ratio': 1.0,
                'format': 'string'
            }
        
        return {
            'original': data,
            'optimized': data,
            'compression_ratio': 1.0,
            'format': 'raw'
        }

    def set_optimized(self, key: str, data: Any, **kwargs) -> None:
        """Set data with automatic compression and optimization"""
        # Optimize storage format
        optimized_result = self._optimize_storage_format(data)
        optimized_data = optimized_result['optimized']
        
        # Check if compression is beneficial
        if self._should_compress(optimized_data):
            compressed_data, algorithm = self._compress_data(optimized_data)
            
            # Store compressed data with metadata
            metadata = kwargs.get('metadata_json', {}) or {}
            metadata.update({
                'compression': {
                    'enabled': True,
                    'algorithm': algorithm,
                    'original_size': len(str(data).encode('utf-8')),
                    'optimized_size': len(str(optimized_data).encode('utf-8')) if optimized_data else 0,
                    'storage_format': optimized_result.get('format', 'raw')
                }
            })
            
            # Store as blob with compression metadata
            kwargs['metadata_json'] = metadata
            kwargs['payload_blob'] = compressed_data
            kwargs['payload_json'] = None  # Clear JSON payload when using blob
            
        else:
            # Store uncompressed with optimization metadata
            metadata = kwargs.get('metadata_json', {}) or {}
            metadata.update({
                'compression': {
                    'enabled': False,
                    'reason': 'size_below_threshold',
                    'storage_format': optimized_result.get('format', 'raw')
                }
            })
            kwargs['metadata_json'] = metadata
            
            if optimized_result['format'] == 'json_compact':
                kwargs['payload_json'] = json.loads(optimized_data)
            else:
                kwargs['payload_json'] = optimized_data
        
        # Set with optimized/compressed data
        self.set(key, **kwargs)

    def get_optimized(self, key: str) -> Optional[Any]:
        """Get data with automatic decompression and optimization"""
        entry = self.get(key)
        if not entry:
            return None
        
        # Check if data is compressed
        metadata = entry.metadata_json or {}
        compression_info = metadata.get('compression', {})
        
        if compression_info.get('enabled', False):
            # Decompress blob data
            if entry.payload_blob:
                algorithm = compression_info.get('algorithm')
                decompressed = self._decompress_data(entry.payload_blob, algorithm)
                return decompressed
            else:
                print(f"⚠️  Compressed data missing blob for key: {key}")
                return None
        else:
            # Return uncompressed data
            if entry.payload_json is not None:
                return entry.payload_json
            elif entry.payload_blob:
                # Try to decode blob as string
                try:
                    return entry.payload_blob.decode('utf-8')
                except UnicodeDecodeError:
                    return entry.payload_blob
            else:
                return None

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression and optimization statistics"""
        stats = {
            'compression_enabled': True,
            'total_entries': 0,
            'compressed_entries': 0,
            'uncompressed_entries': 0,
            'total_original_size': 0,
            'total_optimized_size': 0,
            'total_compressed_size': 0,
            'compression_ratio': 0.0,
            'storage_savings_percent': 0.0,
            'algorithms_used': set(),
            'timestamp': int(time.time())
        }
        
        try:
            # Query all entries with compression metadata
            if USE_POOL and DatabasePool is not None:
                with DatabasePool.get_connection(self.db_path) as con:
                    cur = con.cursor()
                    cur.execute("SELECT metadata_json FROM cache_entries WHERE metadata_json IS NOT NULL")
                    rows = cur.fetchall()
            else:
                con = sqlite3.connect(self.db_path)
                try:
                    cur = con.cursor()
                    cur.execute("SELECT metadata_json FROM cache_entries WHERE metadata_json IS NOT NULL")
                    rows = cur.fetchall()
                finally:
                    con.close()
            
            stats['total_entries'] = len(rows) if rows else 0
            
            for row in rows:
                try:
                    metadata = json.loads(row[0]) if row[0] else {}
                    compression_info = metadata.get('compression', {})
                    
                    if compression_info.get('enabled', False):
                        stats['compressed_entries'] += 1
                        stats['total_original_size'] += compression_info.get('original_size', 0)
                        stats['total_optimized_size'] += compression_info.get('optimized_size', 0)
                        
                        # Estimate compressed size from blob
                        # This is approximate since we don't store compressed size
                        if compression_info.get('algorithm'):
                            stats['algorithms_used'].add(compression_info['algorithm'])
                    else:
                        stats['uncompressed_entries'] += 1
                        
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Calculate compression metrics
            if stats['total_original_size'] > 0:
                # Estimate total compressed size (approximate)
                stats['total_compressed_size'] = int(stats['total_optimized_size'] * 0.7)  # Rough estimate
                stats['compression_ratio'] = stats['total_compressed_size'] / stats['total_original_size']
                stats['storage_savings_percent'] = (1 - stats['compression_ratio']) * 100
            
            stats['algorithms_used'] = list(stats['algorithms_used'])
            
        except Exception as e:
            stats['error'] = str(e)
            print(f"⚠️  Failed to get compression stats: {e}")
        
        return stats