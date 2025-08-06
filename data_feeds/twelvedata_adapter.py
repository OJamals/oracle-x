from __future__ import annotations
import os
import requests
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, Any, List

# from data_feeds.data_feed_orchestrator import Quote, MarketData  # Reuse orchestrator models


class TwelveDataError(Exception):
    pass


class TwelveDataNotFound(TwelveDataError):
    pass


class TwelveDataThrottled(TwelveDataError):
    pass


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None or value == "" or value == "None":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "" or value == "None":
            return None
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _to_utc(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    # Twelve Data returns ISO strings
    try:
        # e.g. "2024-05-31 16:00:00"
        s = str(ts).replace("T", " ")
        # Try common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    except Exception:
        pass
    return None


class TwelveDataAdapter:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.twelvedata.com", session: Optional[requests.Session] = None):
        self.api_key = api_key or os.getenv("TWELVEDATA_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = (5, 15)  # connect, read

    def _request(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        qp = dict(params or {})
        if self.api_key:
            qp["apikey"] = self.api_key
        resp = self.session.get(url, params=qp, timeout=self.timeout)
        if resp.status_code == 429:
            raise TwelveDataThrottled("Twelve Data rate limit exceeded (429)")
        if resp.status_code == 404:
            raise TwelveDataNotFound("Resource not found (404)")
        try:
            data = resp.json()
        except Exception as e:
            raise TwelveDataError(f"Invalid JSON response: {e}")

        # Twelve Data error format: {"status":"error","message":"..."}
        if isinstance(data, dict) and data.get("status") == "error":
            msg = data.get("message", "Unknown provider error")
            if "not found" in msg.lower():
                raise TwelveDataNotFound(msg)
            if "limit" in msg.lower() or "rate" in msg.lower():
                raise TwelveDataThrottled(msg)
            raise TwelveDataError(msg)
        return data

    def _models(self):
        from data_feeds.data_feed_orchestrator import Quote, MarketData
        return Quote, MarketData

    def get_quote(self, symbol: str) -> Optional[Any]:
        data = self._request("/quote", {"symbol": symbol})
        if not isinstance(data, dict):
            return None

        price = _to_decimal(data.get("price"))
        prev_close = _to_decimal(data.get("previous_close"))
        change = _to_decimal(data.get("change"))
        change_pct = _to_decimal(data.get("percent_change"))
        if change is None and price is not None and prev_close is not None:
            change = price - prev_close
        if change_pct is None and change is not None and prev_close and prev_close != 0:
            try:
                change_pct = (change / prev_close) * Decimal("100")
            except Exception:
                change_pct = None

        ts = _to_utc(data.get("timestamp")) or datetime.now(timezone.utc)

        Quote, _MarketData = self._models()
        q = Quote(
            symbol=symbol,
            price=price or Decimal("0"),
            change=change or Decimal("0"),
            change_percent=change_pct or Decimal("0"),
            volume=_to_int(data.get("volume")) or 0,
            market_cap=_to_int(data.get("market_cap")),
            pe_ratio=_to_decimal(data.get("pe")),
            day_low=_to_decimal(data.get("low")),
            day_high=_to_decimal(data.get("high")),
            year_low=_to_decimal(data.get("fifty_two_week", {}).get("low") if isinstance(data.get("fifty_two_week"), dict) else None),
            year_high=_to_decimal(data.get("fifty_two_week", {}).get("high") if isinstance(data.get("fifty_two_week"), dict) else None),
            timestamp=ts,
            source="twelve_data",
            quality_score=None,
        )
        return q

    def get_market_data(self, symbol: str, period: str = "1y", interval: str = "1d", outputsize: Optional[int] = None, start: Optional[datetime] = None, end: Optional[datetime] = None) -> Optional[Any]:
        # Canonicalize interval: accept common orchestrator inputs and map to Twelve Data
        key = (interval or "").lower().strip()
        # Normalize synonyms first
        synonyms = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "60m": "1h",
            "1hr": "1h",
            "1w": "1wk",
            "1week": "1wk",
            "1month": "1mo",
            "daily": "1d",
            "hourly": "1h",
        }
        key = synonyms.get(key, key)
        interval_map = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "1h": "1h",
            "1d": "1day",
            "1day": "1day",
            "1wk": "1week",
            "1mo": "1month",
        }
        td_interval = interval_map.get(key)
        if not td_interval:
            supported = sorted(set(interval_map.keys()) | set(synonyms.keys()))
            raise TwelveDataError(f"Unsupported interval '{interval}'. Supported aliases: {supported}")
 
        # Derive outputsize from period if not explicitly set
        # Twelve Data supports outputsize up to 5000 for time_series
        if outputsize is None:
            period_key = (period or "").lower().strip()
            period_to_bars = {
                "1d": 1,
                "5d": 5,
                "1w": 7,
                "1mo": 31,
                "3mo": 93,
                "6mo": 186,
                "1y": 372,
                "2y": 744,
                "5y": 1860,
                "10y": 3720,
                "ytd": 260,  # approx trading days
                "max": 5000,
            }
            # Scale by interval granularity
            base = period_to_bars.get(period_key, 5000)
            # If intraday intervals, increase bar count within reasonable bounds
            if td_interval in ("1min", "5min", "15min", "30min", "1h"):
                # Approximate 252 trading days/year * 6.5 hours/day = 390 minutes/day
                mult = {
                    "1min": 390,
                    "5min": 78,
                    "15min": 26,
                    "30min": 13,
                    "1h": 7,
                }[td_interval]
                base = min(5000, base * mult)
            outputsize = min(5000, max(1, int(base)))
 
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": td_interval,
            "outputsize": outputsize,
            "format": "JSON",
            "timezone": "UTC",
        }
        if start:
            params["start_date"] = start.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if end:
            params["end_date"] = end.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
 
        # Minimal debug log to validate mapping during smoke test
        try:
            print(f"[TwelveDataAdapter] time_series: period='{period}' interval_in='{interval}' mapped='{td_interval}' symbol='{symbol}' outputsize={outputsize}")
        except Exception:
            pass
 
        data = self._request("/time_series", params)
        # Expected structure: {"values":[{"datetime":"...","open":"...","high":"...","low":"...","close":"...","volume":"..."}], ...}
        values: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            values = data.get("values") or []
        if not values:
            return None

        # Build pandas DataFrame like orchestrator expects
        import pandas as pd

        rows = []
        for v in reversed(values):  # reverse to chronological increasing
            dt = _to_utc(v.get("datetime"))
            if not dt:
                continue
            rows.append(
                {
                    "Datetime": dt,
                    "Open": _to_decimal(v.get("open")),
                    "High": _to_decimal(v.get("high")),
                    "Low": _to_decimal(v.get("low")),
                    "Close": _to_decimal(v.get("close")),
                    "Volume": _to_int(v.get("volume")),
                }
            )
        if not rows:
            return None

        df = pd.DataFrame(rows)
        df.set_index("Datetime", inplace=True)

        _Quote, MarketData = self._models()
        md = MarketData(
            symbol=symbol,
            data=df,
            timeframe=td_interval,
            source="twelve_data",
            timestamp=df.index[-1].to_pydatetime() if len(df.index) else datetime.now(timezone.utc),
            quality_score=100.0,  # Validator in orchestrator will adjust if needed
        )
        return md