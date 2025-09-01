"""
Investiny adapter

Simple data feed that:
1) Searches Investing.com to resolve investing_id for a given symbol/name
2) Fetches historical OHLCV by investing_id
3) Normalizes results into a pandas.DataFrame

Usage:
    from data_feeds.investiny_adapter import get_history
    df = get_history("AAPL", start_date="2020-01-01", end_date="2020-12-31", interval="daily")
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Tuple
import os

import pandas as pd

# Async I/O utilities import with fallback
AsyncHTTPClient = None
ASYNC_IO_AVAILABLE = False
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.async_io_utils import AsyncHTTPClient
    ASYNC_IO_AVAILABLE = True
except ImportError:
    pass

# Mute noisy loggers by default (user can override at app level)
import logging as _logging
for _name in ("investiny", "httpx", "urllib3", "httpcore", "anyio"):
    try:
        _logging.getLogger(_name).setLevel(_logging.ERROR)
    except Exception:
        pass

try:
    # Official public API per docs:
    # - search_assets(query, limit=..., type=..., exchange=...)
    # - historical_data(investing_id=..., from_date="MM/DD/YYYY", to_date="MM/DD/YYYY", interval=...)
    from investiny import search_assets, historical_data
except Exception as e:  # pragma: no cover
    raise ImportError(
        "The 'investiny' package is required. Install with: pip install investiny"
    ) from e


# ---- Public API ----

def search_investing_id(
    query: str,
    *,
    country: Optional[str] = None,
    instrument_type: Optional[str] = None,
    max_results: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Search Investing.com instruments and return the best match with investing_id.

    Parameters:
        query: Symbol or company name, e.g., "AAPL" or "Apple".
        country: Optional country filter string understood by Investing.com (e.g., "United States").
        instrument_type: Optional type filter (e.g., "stocks"). If provided, will prefer matches of this type.
        max_results: Limit result count to scan for best match.

    Returns:
        A dict with keys like:
            {
              "investing_id": 6408,
              "symbol": "AAPL",
              "name": "Apple Inc",
              "type": "stocks",
              "exchange": "...",
              "country": "...",
            }
        or None if nothing matched.
    """
    # Investiny search API per docs: search_assets(query=..., limit=..., type=..., exchange=...)
    # type examples: "Stock", "ETF", "Index", etc. Exchange example: "NASDAQ".
    # Normalize input and default to U.S. stocks on NASDAQ/NYSE to improve reliability.
    q = (query or "").strip().upper()
    kwargs: Dict[str, Any] = {"query": q, "limit": max_results}
    # Per docs: 'type' expects "Stock" etc. Ensure proper capitalization.
    if instrument_type:
        kwargs["type"] = instrument_type if instrument_type[0].isupper() else instrument_type.capitalize()
    else:
        kwargs["type"] = "Stock"
    # Prefer NASDAQ first pass; if it fails we will retry with NYSE and then without exchange.
    exchanges_try = []
    if country:
        exchanges_try = [country]
    else:
        exchanges_try = ["NASDAQ", "NYSE", ""]  # final empty string means no exchange filter

    results: List[Dict[str, Any]] = []
    for ex in exchanges_try:
        attempt_kwargs = dict(kwargs)
        if ex != "":
            attempt_kwargs["exchange"] = ex
        else:
            attempt_kwargs.pop("exchange", None)
        try:
            results = search_assets(**attempt_kwargs) or []
        except Exception:
            results = []
        # Stop on first non-empty result set
        if results:
            break

    if not results:
        return None

    # Basic scoring: prioritize exact symbol match, then name contains, then any.
    def score(item: Dict[str, Any]) -> Tuple[int, int]:
        s = item.get("symbol") or ""
        n = item.get("name") or ""
        t = (item.get("type") or "").lower()

        score_symbol_exact = 1 if s.upper() == query.upper() else 0
        score_name_contains = 1 if query.lower() in n.lower() else 0
        score_type = 1 if (instrument_type and t == instrument_type.lower()) else 0
        # Higher is better
        return (score_symbol_exact + score_type, score_name_contains)

    filtered = results[: max_results or 10]
    # Some APIs may return description as company name; enrich for scoring
    for it in filtered:
        if not it.get("name") and it.get("description"):
            it["name"] = it.get("description")
    filtered.sort(key=score, reverse=True)

    best = filtered[0]
    # Per docs the field containing the numeric ID is "ticker" (string). Convert to int.
    investing_id = best.get("investing_id") or best.get("id") or best.get("pairId")
    if investing_id is None and "ticker" in best:
        investing_id = best.get("ticker")
    if investing_id is None:
        return None

    # Normalize keys
    return {
        "investing_id": int(investing_id),
        "symbol": best.get("symbol"),
        "name": best.get("name"),
        "type": (best.get("type") or "").lower() if best.get("type") else None,
        "exchange": best.get("exchange"),
        "country": best.get("country"),
        "raw": best,
    }


def fetch_historical_by_id(
    investing_id: int,
    start_date: str,
    end_date: str,
    interval: str = "daily",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV for an investing_id.

    Parameters:
        investing_id: Investing.com instrument ID
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        interval: one of {'daily','weekly','monthly'}

    Returns:
        pandas.DataFrame with columns: ['date','open','high','low','close','volume']
        date is timezone-naive pandas.Timestamp (UTC date)
    """
    tf = _map_interval(interval)

    # Per docs, historical_data expects dates as "MM/DD/YYYY"
    sd = _parse_date(start_date)
    ed = _parse_date(end_date)

    try:
        data: List[Dict[str, Any]] = historical_data(
            investing_id=int(investing_id),
            from_date=sd.strftime("%m/%d/%Y"),
            to_date=ed.strftime("%m/%d/%Y"),
            interval=tf,  # Expecting 'D','W','M' or intraday like '1','5','15','30','60'
        ) or []
    except Exception as e:  # pragma: no cover
        data = []
        last_error: Optional[Exception] = e

    if not data:
        # Return empty DataFrame with expected columns if nothing retrieved
        return _empty_history_df()

    # Normalize into DataFrame
    # Some endpoints return {'s':'ok','t':[...],'o':[...],'h':[...],'l':[...],'c':[...],'v':[...]}
    if isinstance(data, dict) and set(data.keys()) & {"t","o","h","l","c"}:
        t = data.get("t") or []
        o = data.get("o") or []
        h = data.get("h") or []
        l = data.get("l") or []
        c = data.get("c") or []
        v = data.get("v") or [None] * len(t)
        rows: List[Dict[str, Any]] = []
        n = min(len(t), len(o), len(h), len(l), len(c), len(v))
        for i in range(n):
            rows.append(
                {
                    "date": _coerce_to_timestamp(t[i]),
                    "open": o[i],
                    "high": h[i],
                    "low": l[i],
                    "close": c[i],
                    "volume": v[i],
                }
            )
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(data)

    # Investiny/TV history commonly returns either:
    # 1) Column arrays in a single row with keys t,o,h,l,c,v
    # 2) Row-wise dicts with keys 't','o','h','l','c','v'
    # 3) Already normalized with 'date', 'open', ...
    if not df.empty and set(df.columns) >= {"t", "o", "h", "l", "c"}:
        # Case 1: single row with arrays
        if df.shape[0] == 1 and isinstance(df.iloc[0].get("t"), (list, tuple)):
            recs: List[Dict[str, Any]] = []
            t = df.iloc[0].get("t") or []
            o = df.iloc[0].get("o") or []
            h = df.iloc[0].get("h") or []
            l = df.iloc[0].get("l") or []
            c = df.iloc[0].get("c") or []
            v = df.iloc[0].get("v") or [None] * len(t)
            for i in range(len(t)):
                recs.append(
                    {
                        "date": _coerce_to_timestamp(t[i]),
                        "open": o[i] if i < len(o) else None,
                        "high": h[i] if i < len(h) else None,
                        "low": l[i] if i < len(l) else None,
                        "close": c[i] if i < len(c) else None,
                        "volume": v[i] if i < len(v) else None,
                    }
                )
            df = pd.DataFrame(recs)
        else:
            # Case 2: row-wise
            df = df.rename(columns={"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df["date"] = df["date"].apply(_coerce_to_timestamp)
    else:
        # Case 3 or fallback
        if "date" in df.columns:
            df["date"] = df["date"].apply(_coerce_to_timestamp)
        else:
            for alt in ("datetime", "time", "timestamp"):
                if alt in df.columns:
                    df["date"] = df[alt].apply(_coerce_to_timestamp)
                    break

    # Some variants may use 'vol' or 'turnover'
    if "vol" in df.columns and "volume" not in df.columns:
        df["volume"] = df["vol"]

    expected_cols = ["date", "open", "high", "low", "close", "volume"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Reorder/select expected columns
    df = df[expected_cols].copy()

    # Clean and de-duplicate:
    # 1) Drop rows missing any OHLC or date
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    # 2) Ensure numeric types
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 3) Round prices to reduce float noise
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = df[c].round(6)
    # 4) Drop duplicate dates (keep last)
    df = df.drop_duplicates(subset=["date"], keep="last")
    # 5) Sort by date and reset index
    df = df.sort_values("date").reset_index(drop=True)
    # 6) Final NA guard on date
    df = df[df["date"].notna()]

    return df


def get_history(
    symbol_or_name: str,
    start_date: str,
    end_date: str,
    interval: str = "daily",
    *,
    country: Optional[str] = None,
    instrument_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper:
    - Searches for the investing_id by symbol/name
    - Fetches historical data for the resolved ID

    Returns empty DataFrame if no match is found.
    """
    meta = search_investing_id(
        symbol_or_name, country=country, instrument_type=instrument_type
    )
    if not meta:
        return _empty_history_df()
    return fetch_historical_by_id(
        investing_id=meta["investing_id"],
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )


def format_daily_oc(
    symbol_or_name: str,
    date_range_or_start: str,
    end_date: Optional[str] = None,
) -> str:
    """
    Return compact daily 'YYYY-MM-DD o:OPEN, c:CLOSE' strings over a date range.
    Accepted:
      - format_daily_oc('TSLA', '2025-01-01-2025-01-05')
      - format_daily_oc('TSLA', '2025-01-01', '2025-01-05')
    """
    start, end = _coerce_range(date_range_or_start, end_date)
    df = get_history(symbol_or_name, start_date=start, end_date=end, interval="daily")
    if df is None or getattr(df, "empty", True):
        return f"{symbol_or_name.upper()}: no data in range {start}..{end}"
    try:
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    except Exception:
        pass
    parts: List[str] = []
    for _, r in df.iterrows():
        d = str(r.get("date"))[:10]
        try:
            o = float(r.get("open")) if pd.notna(r.get("open")) else None
            c = float(r.get("close")) if pd.notna(r.get("close")) else None
        except Exception:
            o, c = None, None
        parts.append(f"{d} o:{o}, c:{c}")
    return " ".join(parts)


# ---- Helpers ----

def _map_interval(interval: str) -> str:
    """
    Map user-friendly intervals to investiny/TradingView resolution codes.
    Docs warning indicates valid daily resolution is 'D' (uppercase), not '1d'.
    """
    iv = (interval or "").lower().strip()
    # Intraday
    if iv in ("1", "1min", "1m"):
        return "1"
    if iv in ("5", "5min", "5m"):
        return "5"
    if iv in ("15", "15min", "15m"):
        return "15"
    if iv in ("30", "30min", "30m"):
        return "30"
    if iv in ("60", "60min", "1h"):
        return "60"
    # Daily/Weekly/Monthly
    if iv in ("d", "1d", "day", "daily"):
        return "D"
    if iv in ("w", "1w", "week", "weekly"):
        return "W"
    if iv in ("m", "1mo", "mon", "month", "monthly"):
        return "M"
    # default to daily
    return "D"


def _parse_date(d: str) -> _dt.date:
    if isinstance(d, _dt.date) and not isinstance(d, _dt.datetime):
        return d
    return _dt.datetime.strptime(str(d), "%Y-%m-%d").date()


def _coerce_range(date_range_or_start: str, end_date: Optional[str]) -> Tuple[str, str]:
    """
    Accept 'YYYY-MM-DD-YYYY-MM-DD' or ('YYYY-MM-DD','YYYY-MM-DD') and return (start,end)
    Robustly split on the middle separator between two dates.
    """
    s = (date_range_or_start or "").strip()
    if end_date:
        return s[:10], end_date.strip()[:10]
    # Try explicit pattern YYYY-MM-DD-YYYY-MM-DD
    if len(s) >= 21 and s[4] == "-" and s[7] == "-" and s[10] == "-" and s[15] == "-" and s[18] == "-":
        return s[:10], s[11:21]
    # Generic split: look for a separator after first 10 chars
    for sep in ["-", " to ", "–", "—"]:
        idx = s.find(sep, 10)
        if idx != -1:
            return s[:10].strip(), s[idx + len(sep):idx + len(sep) + 10].strip()
    # Fallback: mirror same date
    return s[:10], s[:10]


def _coerce_to_timestamp(x: Any) -> pd.Timestamp:
    # Investiny returns no explicit date in JSON according to docs summary snippet.
    # However, many forks return 'date' or epoch; keep robust parsing.
    try:
        xi = int(x)
        if xi > 10_000_000_000:  # likely ms
            return pd.to_datetime(xi, unit="ms", utc=True).tz_convert(None)
        return pd.to_datetime(xi, unit="s", utc=True).tz_convert(None)
    except Exception:
        try:
            return pd.to_datetime(x, utc=True).tz_convert(None)
        except Exception:
            return pd.NaT


def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])