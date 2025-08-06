import os
import json
import traceback
from datetime import datetime
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure DB path for CacheService and set conservative defaults
os.environ.setdefault("CACHE_DB_PATH", "./model_monitoring.db")
os.environ.setdefault("CONSERVE_FMP_BANDWIDTH", "true")

def _safe_run(name, fn):
    print(f"\n=== {name} ===")
    try:
        out = fn()
        # Compact preview helper
        def compact(obj, max_len=2000):
            try:
                s = json.dumps(obj, default=str)[:max_len]
                return s
            except Exception:
                return str(obj)[:max_len]
        print(compact(out))
        return {"name": name, "ok": True, "result": out}
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        traceback.print_exc()
        return {"name": name, "ok": False, "error": str(e)}

def main():
    from datetime import timezone as _tz
    print("RUN @", datetime.now(_tz.utc).isoformat(), "Z")

    # Lazy imports to avoid hard failures if optional deps unavailable
    from data_feeds.data_feed_orchestrator import (
        get_orchestrator,
        get_quote as orch_get_quote,
        get_market_data as orch_get_market_data,
        get_sentiment_data as orch_get_sentiment_data,
        get_advanced_sentiment as orch_get_advanced_sentiment,
        get_market_breadth as orch_get_market_breadth,
        get_sector_performance as orch_get_sector_performance,
        get_finviz_news as orch_get_finviz_news,
        get_finviz_insider_trading as orch_get_finviz_insider_trading,
        get_finviz_earnings as orch_get_finviz_earnings,
        get_finviz_forex as orch_get_finviz_forex,
        get_finviz_crypto as orch_get_finviz_crypto,
        get_system_health as orch_get_system_health,
    )

    o = get_orchestrator()
    tickers = ["AAPL", "TSLA", "NVDA"]

    results = []

    # Quotes
    def test_quotes():
        out = {}
        for t in tickers:
            q = orch_get_quote(t)
            if q:
                out[t] = {
                    "symbol": q.symbol,
                    "price": float(q.price) if q.price is not None else None,
                    "change_pct": float(q.change_percent) if q.change_percent is not None else None,
                    "volume": q.volume,
                    "quality": q.quality_score,
                    "source": q.source,
                }
            else:
                out[t] = None
        return out
    results.append(_safe_run("quotes", test_quotes))

    # Market data daily and hourly for AAPL
    def test_market_data_daily():
        md = orch_get_market_data("AAPL", period="1y", interval="1d")
        if not md:
            return None
        df = md.data
        return {
            "symbol": md.symbol,
            "rows": None if df is None else int(len(df)),
            "cols": None if df is None else list(df.columns),
            "source": md.source,
            "quality": md.quality_score,
            "timeframe": md.timeframe,
        }
    results.append(_safe_run("market_data_daily", test_market_data_daily))

    def test_market_data_hourly():
        md = orch_get_market_data("AAPL", period="1mo", interval="1h")
        if not md:
            return None
        df = md.data
        return {
            "symbol": md.symbol,
            "rows": None if df is None else int(len(df)),
            "cols": None if df is None else list(df.columns),
            "source": md.source,
            "quality": md.quality_score,
            "timeframe": md.timeframe,
        }
    results.append(_safe_run("market_data_hourly", test_market_data_hourly))

    # Sentiment (Reddit + Twitter where available)
    def test_sentiment_basic():
        out = {}
        for t in tickers:
            m = orch_get_sentiment_data(t)
            if not m:
                out[t] = {}
                continue
            # summarize
            ssum = {}
            for src, d in m.items():
                ssum[src] = {
                    "score": d.sentiment_score,
                    "confidence": d.confidence,
                    "samples": d.sample_size,
                }
            out[t] = ssum
        return out
    results.append(_safe_run("sentiment_basic", test_sentiment_basic))

    # Advanced sentiment (may return None if insufficient texts)
    def test_advanced_sentiment():
        r = orch_get_advanced_sentiment("AAPL")
        if not r:
            return None
        return {
            "symbol": r.symbol,
            "score": r.sentiment_score,
            "confidence": r.confidence,
            "samples": r.sample_size,
            "source": r.source,
        }
    results.append(_safe_run("advanced_sentiment", test_advanced_sentiment))

    # FinViz endpoints
    results.append(_safe_run("finviz_market_breadth", orch_get_market_breadth))
    results.append(_safe_run("finviz_sector_performance", orch_get_sector_performance))
    results.append(_safe_run("finviz_news", orch_get_finviz_news))
    results.append(_safe_run("finviz_insider_trading", orch_get_finviz_insider_trading))
    results.append(_safe_run("finviz_earnings", orch_get_finviz_earnings))
    results.append(_safe_run("finviz_forex", orch_get_finviz_forex))
    results.append(_safe_run("finviz_crypto", orch_get_finviz_crypto))

    # Google Trends (optional)
    def test_google_trends():
        # Call instance method since top-level helper isn't exported
        return o.get_google_trends(["AAPL", "TSLA", "NVDA"], timeframe="now 7-d", geo="US")
    results.append(_safe_run("google_trends", test_google_trends))

    # Options analytics (yfinance based)
    def test_options_analytics():
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        return DataFeedOrchestrator().get_options_analytics("AAPL", include=["chain", "iv", "greeks", "gex", "max_pain"])
    results.append(_safe_run("options_analytics", test_options_analytics))

    # System health
    results.append(_safe_run("system_health", orch_get_system_health))

    # Final summary
    from datetime import timezone as _tz2
    summary = {
        "run_at_utc": datetime.now(_tz2.utc).isoformat(),
        "total": len(results),
        "passed": sum(1 for r in results if r["ok"]),
        "failed": [r["name"] for r in results if not r["ok"]],
    }
    print("\n--- SUMMARY ---")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()