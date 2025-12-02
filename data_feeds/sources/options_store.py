"""
SQLite snapshot helpers for options chains.

Schema:
CREATE TABLE IF NOT EXISTS options_chain_snapshots(
  symbol TEXT,
  expiry DATE,
  chain_date INTEGER,
  put_call TEXT,
  strike REAL,
  last REAL,
  bid REAL,
  ask REAL,
  volume INTEGER,
  open_interest INTEGER,
  underlying REAL,
  source TEXT,
  PRIMARY KEY(symbol, expiry, chain_date, put_call, strike)
);

Provides:
- ensure_schema(db_path)
- upsert_snapshot_row(db_path, row: dict)
- upsert_snapshot_many(db_path, rows: list[dict])
- load_latest_snapshot(db_path, symbol: str, expiry: str) -> list[dict]
- compute_oi_delta(db_path, symbol: str, expiry: str) -> list[dict]

Reuses the same DB path as CacheService for consistency:
DEFAULT_DB_PATH = os.getenv("CACHE_DB_PATH", "./model_monitoring.db")
"""

from __future__ import annotations

import os
import sqlite3
from typing import List, Dict, Optional, Any

DEFAULT_DB_PATH = os.getenv("CACHE_DB_PATH", "./model_monitoring.db")

# Import the optimized database pool
DatabasePool = None
USE_POOL = False
try:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.database_pool import DatabasePool

    USE_POOL = True
except ImportError:
    pass

# Import async I/O utilities with fallback
AsyncDatabaseManager = None
ASYNC_IO_AVAILABLE = False
try:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.async_io_utils import AsyncDatabaseManager

    ASYNC_IO_AVAILABLE = True
except ImportError:
    pass

DDL = """
CREATE TABLE IF NOT EXISTS options_chain_snapshots(
  symbol TEXT,
  expiry DATE,
  chain_date INTEGER,
  put_call TEXT,
  strike REAL,
  last REAL,
  bid REAL,
  ask REAL,
  volume INTEGER,
  open_interest INTEGER,
  underlying REAL,
  source TEXT,
  PRIMARY KEY(symbol, expiry, chain_date, put_call, strike)
);
"""

IDX_1 = "CREATE INDEX IF NOT EXISTS idx_opts_sym_exp_date ON options_chain_snapshots(symbol, expiry, chain_date);"
IDX_2 = "CREATE INDEX IF NOT EXISTS idx_opts_sym_exp_pc ON options_chain_snapshots(symbol, expiry, put_call, strike);"


def ensure_schema(db_path: Optional[str] = None) -> None:
    """
    Ensure options_chain_snapshots exists with indexes.
    """
    db_file = db_path or DEFAULT_DB_PATH

    if USE_POOL and DatabasePool is not None:
        with DatabasePool.get_connection(db_file) as con:
            cur = con.cursor()
            cur.execute(DDL)
            cur.execute(IDX_1)
            cur.execute(IDX_2)
            con.commit()
    else:
        # Fallback to direct connection
        con = sqlite3.connect(db_file)
        try:
            cur = con.cursor()
            cur.execute(DDL)
            cur.execute(IDX_1)
            cur.execute(IDX_2)
            con.commit()
        finally:
            con.close()


def upsert_snapshot_row(db_path: Optional[str], row: Dict[str, Any]) -> None:
    """
    Upsert a single contract row snapshot.
    """
    ensure_schema(db_path)
    db_file = db_path or DEFAULT_DB_PATH

    if USE_POOL and DatabasePool is not None:
        with DatabasePool.get_connection(db_file) as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO options_chain_snapshots
                (symbol, expiry, chain_date, put_call, strike, last, bid, ask, volume, open_interest, underlying, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, expiry, chain_date, put_call, strike) DO UPDATE SET
                  last=excluded.last,
                  bid=excluded.bid,
                  ask=excluded.ask,
                  volume=excluded.volume,
                  open_interest=excluded.open_interest,
                  underlying=excluded.underlying,
                  source=excluded.source
                """,
                (
                    row.get("symbol"),
                    row.get("expiry"),
                    int(row["chain_date"]) if row.get("chain_date") is not None else 0,
                    row.get("put_call"),
                    float(row["strike"]) if row.get("strike") is not None else 0.0,
                    float(row["last"]) if row.get("last") is not None else None,
                    float(row["bid"]) if row.get("bid") is not None else None,
                    float(row["ask"]) if row.get("ask") is not None else None,
                    int(row["volume"]) if row.get("volume") is not None else None,
                    (
                        int(row["open_interest"])
                        if row.get("open_interest") is not None
                        else None
                    ),
                    (
                        float(row["underlying"])
                        if row.get("underlying") is not None
                        else None
                    ),
                    row.get("source"),
                ),
            )
            con.commit()
    else:
        # Fallback to direct connection
        con = sqlite3.connect(db_file)
        try:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO options_chain_snapshots
                (symbol, expiry, chain_date, put_call, strike, last, bid, ask, volume, open_interest, underlying, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, expiry, chain_date, put_call, strike) DO UPDATE SET
                  last=excluded.last,
                  bid=excluded.bid,
                  ask=excluded.ask,
                  volume=excluded.volume,
                  open_interest=excluded.open_interest,
                  underlying=excluded.underlying,
                  source=excluded.source
                """,
                (
                    row.get("symbol"),
                    row.get("expiry"),
                    int(row["chain_date"]) if row.get("chain_date") is not None else 0,
                    row.get("put_call"),
                    float(row["strike"]) if row.get("strike") is not None else 0.0,
                    float(row["last"]) if row.get("last") is not None else None,
                    float(row["bid"]) if row.get("bid") is not None else None,
                    float(row["ask"]) if row.get("ask") is not None else None,
                    int(row["volume"]) if row.get("volume") is not None else None,
                    (
                        int(row["open_interest"])
                        if row.get("open_interest") is not None
                        else None
                    ),
                    (
                        float(row["underlying"])
                        if row.get("underlying") is not None
                        else None
                    ),
                    row.get("source"),
                ),
            )
            con.commit()
        finally:
            con.close()


def upsert_snapshot_many(db_path: Optional[str], rows: List[Dict[str, Any]]) -> None:
    """
    Upsert multiple contract row snapshots in a single transaction.
    """
    ensure_schema(db_path)
    db_file = db_path or DEFAULT_DB_PATH

    if USE_POOL and DatabasePool is not None:
        with DatabasePool.get_connection(db_file) as con:
            cur = con.cursor()
            data = []
            for row in rows:
                data.append(
                    (
                        row.get("symbol"),
                        row.get("expiry"),
                        (
                            int(row["chain_date"])
                            if row.get("chain_date") is not None
                            else 0
                        ),
                        row.get("put_call"),
                        float(row["strike"]) if row.get("strike") is not None else 0.0,
                        float(row["last"]) if row.get("last") is not None else None,
                        float(row["bid"]) if row.get("bid") is not None else None,
                        float(row["ask"]) if row.get("ask") is not None else None,
                        int(row["volume"]) if row.get("volume") is not None else None,
                        (
                            int(row["open_interest"])
                            if row.get("open_interest") is not None
                            else None
                        ),
                        (
                            float(row["underlying"])
                            if row.get("underlying") is not None
                            else None
                        ),
                        row.get("source"),
                    )
                )

            cur.executemany(
                """
                INSERT INTO options_chain_snapshots
                (symbol, expiry, chain_date, put_call, strike, last, bid, ask, volume, open_interest, underlying, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, expiry, chain_date, put_call, strike) DO UPDATE SET
                  last=excluded.last,
                  bid=excluded.bid,
                  ask=excluded.ask,
                  volume=excluded.volume,
                  open_interest=excluded.open_interest,
                  underlying=excluded.underlying,
                  source=excluded.source
                """,
                data,
            )
            con.commit()
    else:
        # Fallback to direct connection
        con = sqlite3.connect(db_file)
        try:
            cur = con.cursor()
            data = []
            for row in rows:
                data.append(
                    (
                        row.get("symbol"),
                        row.get("expiry"),
                        (
                            int(row["chain_date"])
                            if row.get("chain_date") is not None
                            else 0
                        ),
                        row.get("put_call"),
                        float(row["strike"]) if row.get("strike") is not None else 0.0,
                        float(row["last"]) if row.get("last") is not None else None,
                        float(row["bid"]) if row.get("bid") is not None else None,
                        float(row["ask"]) if row.get("ask") is not None else None,
                        int(row["volume"]) if row.get("volume") is not None else None,
                        (
                            int(row["open_interest"])
                            if row.get("open_interest") is not None
                            else None
                        ),
                        (
                            float(row["underlying"])
                            if row.get("underlying") is not None
                            else None
                        ),
                        row.get("source"),
                    )
                )

            cur.executemany(
                """
                INSERT INTO options_chain_snapshots
                (symbol, expiry, chain_date, put_call, strike, last, bid, ask, volume, open_interest, underlying, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, expiry, chain_date, put_call, strike) DO UPDATE SET
                  last=excluded.last,
                  bid=excluded.bid,
                  ask=excluded.ask,
                  volume=excluded.volume,
                  open_interest=excluded.open_interest,
                  underlying=excluded.underlying,
                  source=excluded.source
                """,
                data,
            )
            con.commit()
        finally:
            con.close()


def load_latest_snapshot(
    db_path: Optional[str], symbol: str, expiry: str
) -> List[Dict[str, Any]]:
    """
    Load the latest snapshot for a given symbol and expiry.
    """
    ensure_schema(db_path)
    db_file = db_path or DEFAULT_DB_PATH

    if USE_POOL and DatabasePool is not None:
        with DatabasePool.get_connection(db_file) as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT * FROM options_chain_snapshots
                WHERE symbol = ? AND expiry = ?
                ORDER BY chain_date DESC
                LIMIT 1
                """,
                (symbol, expiry),
            )
            row = cur.fetchone()
            if row:
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row))]
            return []
    else:
        # Fallback to direct connection
        con = sqlite3.connect(db_file)
        try:
            cur = con.cursor()
            cur.execute(
                """
                SELECT * FROM options_chain_snapshots
                WHERE symbol = ? AND expiry = ?
                ORDER BY chain_date DESC
                LIMIT 1
                """,
                (symbol, expiry),
            )
            row = cur.fetchone()
            if row:
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row))]
            return []
        finally:
            con.close()


def compute_oi_delta(
    db_path: Optional[str], symbol: str, expiry: str
) -> List[Dict[str, Any]]:
    """
    Compute per-contract OI delta between the two most recent snapshot times for symbol/expiry.
    """
    ensure_schema(db_path)
    db_file = db_path or DEFAULT_DB_PATH

    if USE_POOL and DatabasePool is not None:
        with DatabasePool.get_connection(db_file) as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT DISTINCT chain_date FROM options_chain_snapshots
                WHERE symbol = ? AND expiry = ?
                ORDER BY chain_date DESC
                LIMIT 2
                """,
                (symbol, expiry),
            )
            dates = [row[0] for row in cur.fetchall()]
            if len(dates) < 2:
                return []
            latest, prev = dates[0], dates[1]

            cur.execute(
                """
                SELECT put_call, strike, open_interest FROM options_chain_snapshots
                WHERE symbol = ? AND expiry = ? AND chain_date = ?
                """,
                (symbol, expiry, latest),
            )
            latest_map = {
                (row[0], float(row[1])): int(row[2]) if row[2] is not None else 0
                for row in cur.fetchall()
            }

            cur.execute(
                """
                SELECT put_call, strike, open_interest FROM options_chain_snapshots
                WHERE symbol = ? AND expiry = ? AND chain_date = ?
                """,
                (symbol, expiry, prev),
            )
            prev_map = {
                (row[0], float(row[1])): int(row[2]) if row[2] is not None else 0
                for row in cur.fetchall()
            }

            keys = set(latest_map.keys()) | set(prev_map.keys())
            out: List[Dict[str, Any]] = []
            for k in sorted(keys):
                pc, strike = k
                oi_new = latest_map.get(k, 0)
                oi_old = prev_map.get(k, 0)
                out.append(
                    {"put_call": pc, "strike": strike, "oi_delta": oi_new - oi_old}
                )
            return out
    else:
        # Fallback to direct connection
        con = sqlite3.connect(db_file)
        try:
            cur = con.cursor()
            cur.execute(
                """
                SELECT DISTINCT chain_date FROM options_chain_snapshots
                WHERE symbol = ? AND expiry = ?
                ORDER BY chain_date DESC
                LIMIT 2
                """,
                (symbol, expiry),
            )
            dates = [row[0] for row in cur.fetchall()]
            if len(dates) < 2:
                return []
            latest, prev = dates[0], dates[1]

            cur.execute(
                """
                SELECT put_call, strike, open_interest FROM options_chain_snapshots
                WHERE symbol = ? AND expiry = ? AND chain_date = ?
                """,
                (symbol, expiry, latest),
            )
            latest_map = {
                (row[0], float(row[1])): int(row[2]) if row[2] is not None else 0
                for row in cur.fetchall()
            }

            cur.execute(
                """
                SELECT put_call, strike, open_interest FROM options_chain_snapshots
                WHERE symbol = ? AND expiry = ? AND chain_date = ?
                """,
                (symbol, expiry, prev),
            )
            prev_map = {
                (row[0], float(row[1])): int(row[2]) if row[2] is not None else 0
                for row in cur.fetchall()
            }

            keys = set(latest_map.keys()) | set(prev_map.keys())
            out: List[Dict[str, Any]] = []
            for k in sorted(keys):
                pc, strike = k
                oi_new = latest_map.get(k, 0)
                oi_old = prev_map.get(k, 0)
                out.append(
                    {"put_call": pc, "strike": strike, "oi_delta": oi_new - oi_old}
                )
            return out
        finally:
            con.close()


# Async versions of database operations
async def ensure_schema_async(db_path: Optional[str] = None) -> None:
    """
    Ensure options_chain_snapshots exists with indexes (async version).
    """
    if not ASYNC_IO_AVAILABLE or AsyncDatabaseManager is None:
        # Fallback to sync version
        ensure_schema(db_path)
        return

    db_file = db_path or DEFAULT_DB_PATH
    manager = AsyncDatabaseManager(db_file)

    try:
        await manager.execute_write(DDL)
        await manager.execute_write(IDX_1)
        await manager.execute_write(IDX_2)
    except Exception as e:
        # Fallback to sync version on error
        ensure_schema(db_path)


async def upsert_snapshot_row_async(
    db_path: Optional[str], row: Dict[str, Any]
) -> None:
    """
    Upsert a single contract row snapshot (async version).
    """
    if not ASYNC_IO_AVAILABLE or AsyncDatabaseManager is None:
        # Fallback to sync version
        upsert_snapshot_row(db_path, row)
        return

    await ensure_schema_async(db_path)
    db_file = db_path or DEFAULT_DB_PATH
    manager = AsyncDatabaseManager(db_file)

    try:
        query = """
        INSERT INTO options_chain_snapshots
        (symbol, expiry, chain_date, put_call, strike, last, bid, ask, volume, open_interest, underlying, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, expiry, chain_date, put_call, strike) DO UPDATE SET
          last=excluded.last,
          bid=excluded.bid,
          ask=excluded.ask,
          volume=excluded.volume,
          open_interest=excluded.open_interest,
          underlying=excluded.underlying,
          source=excluded.source
        """

        params = (
            row.get("symbol"),
            row.get("expiry"),
            int(row["chain_date"]) if row.get("chain_date") is not None else 0,
            row.get("put_call"),
            float(row["strike"]) if row.get("strike") is not None else 0.0,
            float(row["last"]) if row.get("last") is not None else None,
            float(row["bid"]) if row.get("bid") is not None else None,
            float(row["ask"]) if row.get("ask") is not None else None,
            int(row["volume"]) if row.get("volume") is not None else None,
            int(row["open_interest"]) if row.get("open_interest") is not None else None,
            float(row["underlying"]) if row.get("underlying") is not None else None,
            row.get("source"),
        )

        await manager.execute_write(query, params)
    except Exception as e:
        # Fallback to sync version on error
        upsert_snapshot_row(db_path, row)


async def upsert_snapshot_many_async(
    db_path: Optional[str], rows: List[Dict[str, Any]]
) -> None:
    """
    Upsert multiple contract row snapshots in a single transaction (async version).
    """
    if not ASYNC_IO_AVAILABLE or AsyncDatabaseManager is None:
        # Fallback to sync version
        upsert_snapshot_many(db_path, rows)
        return

    await ensure_schema_async(db_path)
    db_file = db_path or DEFAULT_DB_PATH
    manager = AsyncDatabaseManager(db_file)

    try:
        query = """
        INSERT INTO options_chain_snapshots
        (symbol, expiry, chain_date, put_call, strike, last, bid, ask, volume, open_interest, underlying, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, expiry, chain_date, put_call, strike) DO UPDATE SET
          last=excluded.last,
          bid=excluded.bid,
          ask=excluded.ask,
          volume=excluded.volume,
          open_interest=excluded.open_interest,
          underlying=excluded.underlying,
          source=excluded.source
        """

        data = []
        for row in rows:
            data.append(
                (
                    row.get("symbol"),
                    row.get("expiry"),
                    int(row["chain_date"]) if row.get("chain_date") is not None else 0,
                    row.get("put_call"),
                    float(row["strike"]) if row.get("strike") is not None else 0.0,
                    float(row["last"]) if row.get("last") is not None else None,
                    float(row["bid"]) if row.get("bid") is not None else None,
                    float(row["ask"]) if row.get("ask") is not None else None,
                    int(row["volume"]) if row.get("volume") is not None else None,
                    (
                        int(row["open_interest"])
                        if row.get("open_interest") is not None
                        else None
                    ),
                    (
                        float(row["underlying"])
                        if row.get("underlying") is not None
                        else None
                    ),
                    row.get("source"),
                )
            )

        await manager.execute_many(query, data)
    except Exception as e:
        # Fallback to sync version on error
        upsert_snapshot_many(db_path, rows)


async def load_latest_snapshot_async(
    db_path: Optional[str], symbol: str, expiry: str
) -> List[Dict[str, Any]]:
    """
    Load the latest snapshot for a given symbol and expiry (async version).
    """
    if not ASYNC_IO_AVAILABLE or AsyncDatabaseManager is None:
        # Fallback to sync version
        return load_latest_snapshot(db_path, symbol, expiry)

    await ensure_schema_async(db_path)
    db_file = db_path or DEFAULT_DB_PATH
    manager = AsyncDatabaseManager(db_file)

    try:
        query = """
        SELECT * FROM options_chain_snapshots
        WHERE symbol = ? AND expiry = ?
        ORDER BY chain_date DESC
        LIMIT 1
        """

        results = await manager.execute_query(query, (symbol, expiry))
        if results:
            return results
        return []
    except Exception as e:
        # Fallback to sync version on error
        return load_latest_snapshot(db_path, symbol, expiry)


async def compute_oi_delta_async(
    db_path: Optional[str], symbol: str, expiry: str
) -> List[Dict[str, Any]]:
    """
    Compute per-contract OI delta between the two most recent snapshot times for symbol/expiry (async version).
    """
    if not ASYNC_IO_AVAILABLE or AsyncDatabaseManager is None:
        # Fallback to sync version
        return compute_oi_delta(db_path, symbol, expiry)

    await ensure_schema_async(db_path)
    db_file = db_path or DEFAULT_DB_PATH
    manager = AsyncDatabaseManager(db_file)

    try:
        # Get the two most recent dates
        date_query = """
        SELECT DISTINCT chain_date FROM options_chain_snapshots
        WHERE symbol = ? AND expiry = ?
        ORDER BY chain_date DESC
        LIMIT 2
        """

        date_results = await manager.execute_query(date_query, (symbol, expiry))
        dates = [row["chain_date"] for row in date_results]

        if len(dates) < 2:
            return []

        latest, prev = dates[0], dates[1]

        # Get latest data
        latest_query = """
        SELECT put_call, strike, open_interest FROM options_chain_snapshots
        WHERE symbol = ? AND expiry = ? AND chain_date = ?
        """
        latest_results = await manager.execute_query(
            latest_query, (symbol, expiry, latest)
        )
        latest_map = {
            (row["put_call"], float(row["strike"])): (
                int(row["open_interest"]) if row["open_interest"] is not None else 0
            )
            for row in latest_results
        }

        # Get previous data
        prev_results = await manager.execute_query(latest_query, (symbol, expiry, prev))
        prev_map = {
            (row["put_call"], float(row["strike"])): (
                int(row["open_interest"]) if row["open_interest"] is not None else 0
            )
            for row in prev_results
        }

        # Compute deltas
        keys = set(latest_map.keys()) | set(prev_map.keys())
        out: List[Dict[str, Any]] = []
        for k in sorted(keys):
            pc, strike = k
            oi_new = latest_map.get(k, 0)
            oi_old = prev_map.get(k, 0)
            out.append({"put_call": pc, "strike": strike, "oi_delta": oi_new - oi_old})

        return out
    except Exception as e:
        # Fallback to sync version on error
        return compute_oi_delta(db_path, symbol, expiry)
