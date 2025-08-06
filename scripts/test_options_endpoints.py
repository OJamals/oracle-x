import os
import sqlite3
from datetime import datetime

# Ensure DB path is set for CacheService compatibility
os.environ.setdefault("CACHE_DB_PATH", "./model_monitoring.db")


def test_options_math():
    from data_feeds.options_math import bs_price, bs_greeks, implied_vol

    S, K, r, q, sigma, T = 100.0, 100.0, 0.02, 0.01, 0.20, 30 / 365.0
    call = bs_price(S, K, r, q, sigma, T, "call")
    iv_back = implied_vol(call, S, K, r, q, T, "call")
    greeks = bs_greeks(S, K, r, q, sigma, T, "call")

    print("--- options_math test ---")
    print("call_price=", round(call, 6))
    print("iv_backsolve=", None if iv_back is None else round(iv_back, 6))
    print(
        "greeks_subset=",
        {k: round(greeks[k], 6) for k in ["delta", "gamma", "vega", "theta", "rho"]},
    )
    valid = (iv_back is not None) and (abs(iv_back - sigma) <= 5e-3)
    print("options_math_valid=", valid)
    return {"call_price": call, "iv_backsolve": iv_back, "valid": valid}


def test_options_store():
    from data_feeds import options_store as store

    print("\n--- options_store schema test ---")
    store.ensure_schema(os.getenv("CACHE_DB_PATH"))
    con = sqlite3.connect(os.getenv("CACHE_DB_PATH"))
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='options_chain_snapshots'"
        )
        exists = cur.fetchone() is not None
        print("options_chain_snapshots_exists=", exists)
        return {"options_chain_snapshots_exists": exists}
    finally:
        con.close()


def test_earnings_cache():
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator as O

    print("\n--- earnings cache test ---")
    o = O()
    r1 = o.get_earnings_calendar_detailed()
    print("earnings_first_type=", type(r1).__name__)
    print("earnings_first_len=", None if r1 is None else len(r1))
    r2 = o.get_earnings_calendar_detailed()
    print("earnings_second_type=", type(r2).__name__)
    print("earnings_second_len=", None if r2 is None else len(r2))
    if isinstance(r1, list) and r1:
        sample = r1[0]
        print(
            "earnings_sample_row=",
            {
                "symbol": sample.get("symbol"),
                "date": sample.get("date"),
                "time": sample.get("time"),
                "eps_estimate": sample.get("eps_estimate"),
                "eps_actual": sample.get("eps_actual"),
            },
        )
    con = sqlite3.connect(os.getenv("CACHE_DB_PATH"))
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM cache_entries WHERE endpoint='earnings_calendar'"
        )
        cache_cnt = cur.fetchone()[0]
        print("earnings_cache_rows=", cache_cnt)
        cur.execute(
            "SELECT source, ttl_seconds FROM cache_entries WHERE endpoint='earnings_calendar' ORDER BY fetched_at DESC LIMIT 1"
        )
        print("earnings_cache_meta=", cur.fetchone())
        return {"cache_rows": cache_cnt}
    finally:
        con.close()


def main():
    print("RUN @", datetime.utcnow().isoformat(), "Z")
    m = test_options_math()
    s = test_options_store()
    e = test_earnings_cache()
    print("\n--- VALIDATED DATA POINT ---")
    # Provide a single validated datum that can be easily asserted
    print(
        {
            "call_price": None if m["call_price"] is None else round(m["call_price"], 6),
            "iv_backsolve": None
            if m["iv_backsolve"] is None
            else round(m["iv_backsolve"], 6),
            "options_math_valid": m["valid"],
            "options_store_table": s["options_chain_snapshots_exists"],
            "earnings_cache_rows": e["cache_rows"],
        }
    )
    print("DONE @", datetime.utcnow().isoformat(), "Z")


if __name__ == "__main__":
    main()