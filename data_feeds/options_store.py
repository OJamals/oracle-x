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
    con = sqlite3.connect(db_path or DEFAULT_DB_PATH)
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
    con = sqlite3.connect(db_path or DEFAULT_DB_PATH)
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
                int(row.get("chain_date")),
                row.get("put_call"),
                float(row.get("strike")),
                float(row.get("last")) if row.get("last") is not None else None,
                float(row.get("bid")) if row.get("bid") is not None else None,
                float(row.get("ask")) if row.get("ask") is not None else None,
                int(row.get("volume")) if row.get("volume") is not None else None,
                int(row.get("open_interest")) if row.get("open_interest") is not None else None,
                float(row.get("underlying")) if row.get("underlying") is not None else None,
                row.get("source"),
            ),
        )
        con.commit()
    finally:
        con.close()


def upsert_snapshot_many(db_path: Optional[str], rows: List[Dict[str, Any]]) -> None:
    """
    Upsert many contract rows in a single transaction.
    """
    if not rows:
        return
    ensure_schema(db_path)
    con = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    try:
        cur = con.cursor()
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
            [
                (
                    r.get("symbol"),
                    r.get("expiry"),
                    int(r.get("chain_date")),
                    r.get("put_call"),
                    float(r.get("strike")),
                    float(r.get("last")) if r.get("last") is not None else None,
                    float(r.get("bid")) if r.get("bid") is not None else None,
                    float(r.get("ask")) if r.get("ask") is not None else None,
                    int(r.get("volume")) if r.get("volume") is not None else None,
                    int(r.get("open_interest")) if r.get("open_interest") is not None else None,
                    float(r.get("underlying")) if r.get("underlying") is not None else None,
                    r.get("source"),
                )
                for r in rows
            ],
        )
        con.commit()
    finally:
        con.close()


def load_latest_snapshot(db_path: Optional[str], symbol: str, expiry: str) -> List[Dict[str, Any]]:
    """
    Load the latest snapshot (by max chain_date) for a symbol/expiry, returning all contracts at that snapshot time.
    """
    ensure_schema(db_path)
    con = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT symbol, expiry, chain_date, put_call, strike, last, bid, ask, volume, open_interest, underlying, source
            FROM options_chain_snapshots
            WHERE symbol = ? AND expiry = ?
            ORDER BY chain_date DESC
            """,
            (symbol, expiry),
        )
        rows = cur.fetchall()
        if not rows:
            return []
        latest_ts = rows[0][2]
        out: List[Dict[str, Any]] = []
        for r in rows:
            if r[2] != latest_ts:
                continue
            out.append(
                {
                    "symbol": r[0],
                    "expiry": r[1],
                    "chain_date": r[2],
                    "put_call": r[3],
                    "strike": r[4],
                    "last": r[5],
                    "bid": r[6],
                    "ask": r[7],
                    "volume": r[8],
                    "open_interest": r[9],
                    "underlying": r[10],
                    "source": r[11],
                }
            )
        return out
    finally:
        con.close()


def compute_oi_delta(db_path: Optional[str], symbol: str, expiry: str) -> List[Dict[str, Any]]:
    """
    Compute per-contract OI delta between the two most recent snapshot times for symbol/expiry.
    """
    ensure_schema(db_path)
    con = sqlite3.connect(db_path or DEFAULT_DB_PATH)
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
        latest_map = {(row[0], float(row[1])): int(row[2]) if row[2] is not None else 0 for row in cur.fetchall()}

        cur.execute(
            """
            SELECT put_call, strike, open_interest FROM options_chain_snapshots
            WHERE symbol = ? AND expiry = ? AND chain_date = ?
            """,
            (symbol, expiry, prev),
        )
        prev_map = {(row[0], float(row[1])): int(row[2]) if row[2] is not None else 0 for row in cur.fetchall()}

        keys = set(latest_map.keys()) | set(prev_map.keys())
        out: List[Dict[str, Any]] = []
        for k in sorted(keys):
            pc, strike = k
            oi_new = latest_map.get(k, 0)
            oi_old = prev_map.get(k, 0)
            out.append({"put_call": pc, "strike": strike, "oi_delta": oi_new - oi_old})
        return out
    finally:
        con.close()