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

# Fallback to default path if import fails
try:
    from env_config import get_cache_db_path
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
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("CACHE_DB_PATH", DEFAULT_DB_PATH)
        self._ensure_db()

    def _ensure_db(self) -> None:
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
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT key_hash, endpoint, symbol, fetched_at, ttl_seconds, source, metadata_json, payload_json, payload_blob, storage_uri, version, status "
                "FROM cache_entries WHERE key_hash = ? LIMIT 1",
                (key_hash,),
            )
            row = cur.fetchone()
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
        finally:
            con.close()

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
        now = int(time.time())
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute(
                """
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