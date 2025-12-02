#!/usr/bin/env python3
"""
Investiny demo script (compact): fetch last 2 years of daily OHLCV
for TSLA, AAPL, MSFT with minimal, deduped output.

Run:
    PYTHONPATH=. python scripts/investiny_demo.py
"""

import json
import sys
import datetime as dt
import os

try:
    from data_feeds.investiny_adapter import search_investing_id, get_history
except Exception as e:
    print("import_error", str(e))
    sys.exit(1)


def main() -> int:
    # Silence HTTP client logs and third-party info
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    try:
        import logging

        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("investiny").setLevel(logging.ERROR)
        # Some environments route through httpcore/anyio
        logging.getLogger("httpcore").setLevel(logging.ERROR)
        logging.getLogger("anyio").setLevel(logging.ERROR)
    except Exception:
        pass

    # Window: ~2y
    try:
        start = (dt.date.today().replace(day=1) - dt.timedelta(days=730)).strftime(
            "%Y-%m-%d"
        )
    except Exception:
        start = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
    end = dt.date.today().strftime("%Y-%m-%d")

    symbols = ["TSLA", "AAPL", "MSFT"]
    out = {}

    for sym in symbols:
        # Fetch history and clean
        df = get_history(sym, start_date=start, end_date=end, interval="daily")

        # If None or empty after cleaning, print None placeholders using actual computed first/last dates
        if df is None or getattr(df, "empty", True):
            out[sym] = f"{start} o:None, c:None - {end} o:None, c:None"
            continue

        # Chronological order and unique dates
        try:
            df = (
                df.sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .reset_index(drop=True)
            )
        except Exception:
            pass

        # Compute first and last available trading dates as strings
        first_dt = str(df.loc[df.index[0], "date"])[:10]
        last_dt = str(df.loc[df.index[-1], "date"])[:10]

        # Extract starts/ends
        def _num(x):
            try:
                return float(x)
            except Exception:
                return None

        start_open = _num(df.loc[df.index[0], "open"])
        start_close = _num(df.loc[df.index[0], "close"])
        end_open = _num(df.loc[df.index[-1], "open"])
        end_close = _num(df.loc[df.index[-1], "close"])

        out[sym] = (
            f"{first_dt} o:{start_open}, c:{start_close} - {last_dt} o:{end_open}, c:{end_close}"
        )

    # Output: { "SYMBOL": "$startdate o:$open, c:$close - $enddate o:$open, c:$close", ... }
    print(json.dumps(out, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
