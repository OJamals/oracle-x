#!/usr/bin/env python3
"""
CLI Validation Utility for Oracle-X

Purpose:
  Query orchestrated data via data_feeds.data_feed_orchestrator and print normalized JSON
  for manual validation in docs/DATA_VALIDATION_CHECKLIST.md. Also includes a generic
  compare helper for % difference checks.

Dependencies:
  - Standard library only (argparse, json, datetime, typing, decimal). Pandas optional handling via isinstance checks.

Behavior:
  - Uses module-level forwards from data_feeds.data_feed_orchestrator
  - Prints a single JSON object to stdout
  - Exit code 0 on success with data, non-zero if fetch fails or invalid input
  - Gracefully handles missing API keys and prints "skipped (missing key)" when applicable
  - Respects orchestrator's .env loading via config_loader inside orchestrator
  - Supports --json flag for structured output without status messages
  - Supports --exit-zero-even-on-error flag for test continuity

Examples:
  python cli_validate.py quote --symbol AAPL
  python cli_validate.py market_data --symbol AAPL --period 1y --interval 1d --preferred_sources twelve_data
  python cli_validate.py company_info --symbol MSFT
  python cli_validate.py news --symbol AAPL --limit 5
  python cli_validate.py multiple_quotes --symbols AAPL,MSFT,SPY
  python cli_validate.py financial_statements --symbol AAPL
  python cli_validate.py sentiment --symbol TSLA
  python cli_validate.py advanced_sentiment --symbol TSLA
  python cli_validate.py market_breadth
  python cli_validate.py sector_performance
  python cli_validate.py compare --value 195.23 --ref_value 196.5 --tolerance_pct 2.0
  python cli_validate.py quote --symbol AAPL --json --exit-zero-even-on-error
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from decimal import Decimal

# Global flags for structured output mode
_STRUCTURED_OUTPUT = False
_EXIT_ZERO_EVEN_ON_ERROR = False

# Lazy import of orchestrator to ensure flags are parsed first
_orch = None
def _get_orch():
    global _orch
    if _orch is not None:
        return _orch
    try:
        from data_feeds import data_feed_orchestrator as orch
        _orch = orch
        return _orch
    except Exception as e:
        # Always emit structured JSON respecting flags
        out({"ok": False, "error": f"Failed to import orchestrator: {e}"}, 2)

def _dt_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _decimal_to_float(x: Any) -> Any:
    if isinstance(x, Decimal):
        try:
            return float(x)
        except Exception:
            return str(x)
    return x


def _json_default(o: Any) -> Any:
    # Safe fallback for dataclasses used in orchestrator
    if hasattr(o, "__dict__"):
        d = {}
        for k, v in o.__dict__.items():
            if isinstance(v, datetime):
                d[k] = _dt_iso(v)
            elif isinstance(v, Decimal):
                d[k] = _decimal_to_float(v)
            else:
                d[k] = v
        return d
    if isinstance(o, datetime):
        return _dt_iso(o)
    if isinstance(o, Decimal):
        return _decimal_to_float(o)
    return str(o)


def _df_last_row_summary(df) -> Optional[Dict[str, Any]]:
    try:
        import pandas as pd  # only used when present
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        last = df.iloc[-1]
        def get(col):
            return _decimal_to_float(last[col]) if col in df.columns else None
        # Common OHLCV columns
        return {
            "Date": str(last["Date"]) if "Date" in df.columns else None,
            "Open": get("Open"),
            "High": get("High"),
            "Low": get("Low"),
            "Close": get("Close"),
            "AdjClose": get("Adj Close") if "Adj Close" in df.columns else get("AdjClose"),
            "Volume": get("Volume"),
        }
    except Exception:
        return None


def out(obj: Dict[str, Any], code: int) -> None:
    print(json.dumps(obj, default=_json_default, ensure_ascii=False))
    sys.exit(code if not _EXIT_ZERO_EVEN_ON_ERROR else 0)


def cmd_quote(args: argparse.Namespace) -> None:
    sym = args.symbol.upper().strip()
    preferred = None
    if args.preferred_sources:
        preferred = [s.strip() for s in args.preferred_sources.split(",") if s.strip()]
    try:
        # Prefer direct orchestrator method to honor preferred sources if provided
        if preferred:
            # Orchestrator get_quote can accept strings per its normalization
            quote = _get_orch().get_orchestrator().get_quote(sym, preferred_sources=preferred)
        else:
            quote = _get_orch().get_quote(sym)
        if not quote:
            out({"ok": False, "cmd": "quote", "symbol": sym, "error": "no data"}, 1)
        payload = {
            "ok": True,
            "cmd": "quote",
            "symbol": quote.symbol,
            "source": getattr(quote, "source", None),
            "timestamp": _dt_iso(getattr(quote, "timestamp", None)),
            "price": _decimal_to_float(getattr(quote, "price", None)),
            "change_percent": _decimal_to_float(getattr(quote, "change_percent", None)),
            "volume": getattr(quote, "volume", None),
            "market_cap": getattr(quote, "market_cap", None),
        }
        out(payload, 0)
    except Exception as e:
        out({"ok": False, "cmd": "quote", "symbol": sym, "error": str(e)}, 2)


def cmd_market_data(args: argparse.Namespace) -> None:
    sym = args.symbol.upper().strip()
    period = args.period
    interval = args.interval
    preferred = None
    if args.preferred_sources:
        preferred = [s.strip() for s in args.preferred_sources.split(",") if s.strip()]
    try:
        if preferred:
            md = _get_orch().get_orchestrator().get_market_data(sym, period=period, interval=interval, preferred_sources=preferred)
        else:
            md = _get_orch().get_market_data(sym, period=period, interval=interval)
        if not md or getattr(md, "data", None) is None:
            out({"ok": False, "cmd": "market_data", "symbol": sym, "error": "no data"}, 1)
        last_summary = _df_last_row_summary(md.data)
        payload = {
            "ok": True,
            "cmd": "market_data",
            "symbol": sym,
            "source": getattr(md, "source", None),
            "timestamp": _dt_iso(getattr(md, "timestamp", None)),
            "timeframe": getattr(md, "timeframe", f"{period}_{interval}"),
            "last_row": last_summary,
            "rows": int(getattr(md.data, "shape", [0, 0])[0]) if getattr(md, "data", None) is not None else 0,
        }
        out(payload, 0)
    except Exception as e:
        out({"ok": False, "cmd": "market_data", "symbol": sym, "error": str(e)}, 2)


def cmd_company_info(args: argparse.Namespace) -> None:
    sym = args.symbol.upper().strip()
    try:
        info = _get_orch().get_company_info(sym)
        if not info:
            out({"ok": False, "cmd": "company_info", "symbol": sym, "error": "no data"}, 1)
        payload = {
            "ok": True,
            "cmd": "company_info",
            "symbol": getattr(info, "symbol", sym),
            "name": getattr(info, "name", None),
            "sector": getattr(info, "sector", None),
            "market_cap": getattr(info, "market_cap", None),
            "source": getattr(info, "source", None),
        }
        out(payload, 0)
    except Exception as e:
        out({"ok": False, "cmd": "company_info", "symbol": sym, "error": str(e)}, 2)


def cmd_news(args: argparse.Namespace) -> None:
    sym = args.symbol.upper().strip()
    limit = args.limit
    try:
        items = _get_orch().get_news(sym, limit=limit) or []
        titles = [{"title": getattr(n, "title", ""), "published": _dt_iso(getattr(n, "published", None))} for n in items]
        payload = {
            "ok": True,
            "cmd": "news",
            "symbol": sym,
            "count": len(items),
            "first_n": titles,
        }
        out(payload, 0)
    except Exception as e:
        out({"ok": False, "cmd": "news", "symbol": sym, "error": str(e)}, 2)


def cmd_multiple_quotes(args: argparse.Namespace) -> None:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    try:
        results = _get_orch().get_multiple_quotes(symbols) or {}
        norm = {}
        for sym, q in results.items():
            norm[sym] = {
                "symbol": getattr(q, "symbol", sym),
                "source": getattr(q, "source", None),
                "timestamp": _dt_iso(getattr(q, "timestamp", None)),
                "price": _decimal_to_float(getattr(q, "price", None)),
                "change_percent": _decimal_to_float(getattr(q, "change_percent", None)),
            }
        payload = {
            "ok": True,
            "cmd": "multiple_quotes",
            "count": len(norm),
            "quotes": norm,
        }
        out(payload, 0)
    except Exception as e:
        out({"ok": False, "cmd": "multiple_quotes", "symbols": symbols, "error": str(e)}, 2)


def _df_shape(df) -> Optional[List[int]]:
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            return list(df.shape)
        return None
    except Exception:
        return None


def cmd_financial_statements(args: argparse.Namespace) -> None:
    sym = args.symbol.upper().strip()
    try:
        fin = _get_orch().get_financial_statements(sym) or {}
        summary = {}
        for key, df in fin.items():
            summary[key] = {
                "shape": _df_shape(df),
            }
        payload = {
            "ok": True,
            "cmd": "financial_statements",
            "symbol": sym,
            "summary": summary,
        }
        out(payload, 0)
    except Exception as e:
        out({"ok": False, "cmd": "financial_statements", "symbol": sym, "error": str(e)}, 2)


def cmd_sentiment(args: argparse.Namespace) -> None:
    sym = args.symbol.upper().strip()
    try:
        data = _get_orch().get_sentiment_data(sym) or {}
        # Normalize per-source score & confidence
        norm = {}
        for src, sd in data.items():
            norm[src] = {
                "symbol": getattr(sd, "symbol", sym),
                "score": getattr(sd, "sentiment_score", None),
                "confidence": getattr(sd, "confidence", None),
                "timestamp": _dt_iso(getattr(sd, "timestamp", None)),
                "sample_size": getattr(sd, "sample_size", None),
            }
        # If no data returned and keys missing, likely due to missing credentials or blocked endpoints
        if not norm:
            msg = "skipped (missing key or no sentiment available)"
            out({"ok": True, "cmd": "sentiment", "symbol": sym, "note": msg, "data": {}}, 0)
        out({"ok": True, "cmd": "sentiment", "symbol": sym, "data": norm}, 0)
    except Exception as e:
        out({"ok": False, "cmd": "sentiment", "symbol": sym, "error": str(e)}, 2)


def cmd_advanced_sentiment(args: argparse.Namespace) -> None:
    sym = args.symbol.upper().strip()
    try:
        # Attempt via orchestrator which will derive texts from reddit raw_data if available
        sd = _get_orch().get_advanced_sentiment(sym, texts=None, sources=None)
        if not sd:
            out({"ok": True, "cmd": "advanced_sentiment", "symbol": sym, "note": "skipped (missing key or no texts available)"}, 0)
        payload = {
            "ok": True,
            "cmd": "advanced_sentiment",
            "symbol": sym,
            "score": getattr(sd, "sentiment_score", None),
            "confidence": getattr(sd, "confidence", None),
            "timestamp": _dt_iso(getattr(sd, "timestamp", None)),
            "sample_size": getattr(sd, "sample_size", None),
            "source": getattr(sd, "source", None),
        }
        out(payload, 0)
    except Exception as e:
        out({"ok": False, "cmd": "advanced_sentiment", "symbol": sym, "error": str(e)}, 2)


def cmd_market_breadth(_: argparse.Namespace) -> None:
    try:
        breadth = _get_orch().get_market_breadth()
        if not breadth:
            out({"ok": True, "cmd": "market_breadth", "note": "skipped (unavailable adapter or missing key)"}, 0)
        payload = {
            "ok": True,
            "cmd": "market_breadth",
            "advancers": getattr(breadth, "advancers", None),
            "decliners": getattr(breadth, "decliners", None),
            "unchanged": getattr(breadth, "unchanged", None),
            "put_call_ratio": getattr(breadth, "put_call_ratio", None),
            "timestamp": _dt_iso(getattr(breadth, "timestamp", None)) if hasattr(breadth, "timestamp") else None,
        }
        out(payload, 0)
    except Exception as e:
        out({"ok": False, "cmd": "market_breadth", "error": str(e)}, 2)


def cmd_sector_performance(_: argparse.Namespace) -> None:
    try:
        groups = _get_orch().get_sector_performance() or []
        # Summarize first few rows
        rows = []
        for g in groups[:11]:
            rows.append({
                "group": getattr(g, "name", getattr(g, "group_name", None)),
                "type": getattr(g, "group_type", None),
                "perf_1d": _decimal_to_float(getattr(g, "perf_1d", None)),
                "perf_1w": _decimal_to_float(getattr(g, "perf_1w", None)),
                "perf_1m": _decimal_to_float(getattr(g, "perf_1m", None)),
                "perf_ytd": _decimal_to_float(getattr(g, "perf_ytd", None)),
            })
        if not rows:
            out({"ok": True, "cmd": "sector_performance", "note": "skipped (unavailable adapter or missing key)", "rows": []}, 0)
        out({"ok": True, "cmd": "sector_performance", "rows": rows, "count": len(groups)}, 0)
    except Exception as e:
        out({"ok": False, "cmd": "sector_performance", "error": str(e)}, 2)


def cmd_compare(args: argparse.Namespace) -> None:
    try:
        value = float(args.value)
        ref_value = float(args.ref_value)
        tol = float(args.tolerance_pct)
        if ref_value == 0:
            pct_diff = float("inf")
        else:
            pct_diff = abs((value - ref_value) / ref_value) * 100.0
        passed = pct_diff <= tol
        out({
            "ok": True,
            "cmd": "compare",
            "value": value,
            "ref_value": ref_value,
            "tolerance_pct": tol,
            "pct_diff": pct_diff,
            "result": "pass" if passed else "fail"
        }, 0 if passed else 1)
    except Exception as e:
        out({"ok": False, "cmd": "compare", "error": str(e)}, 2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Oracle-X CLI Validation")
    # Add global flags to main parser
    p.add_argument("--json", action="store_true", help="Output full JSON without status messages")
    p.add_argument("--exit-zero-even-on-error", action="store_true", help="Exit with code 0 even on error for test continuity")
    
    sub = p.add_subparsers(dest="command", required=True)

    # quote
    sp = sub.add_parser("quote", help="Get a real-time quote")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--preferred_sources", required=False, help="Comma-separated list, e.g., yfinance,twelve_data")
    sp.set_defaults(func=cmd_quote)
    
    # market_data
    sp = sub.add_parser("market_data", help="Get historical market data")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--period", default="1y")
    sp.add_argument("--interval", default="1d")
    sp.add_argument("--preferred_sources", required=False, help="Comma-separated list, e.g., yfinance,twelve_data")
    sp.set_defaults(func=cmd_market_data)
    
    # company_info
    sp = sub.add_parser("company_info", help="Get company info")
    sp.add_argument("--symbol", required=True)
    sp.set_defaults(func=cmd_company_info)
    
    # news
    sp = sub.add_parser("news", help="Get company news")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--limit", type=int, default=5)
    sp.set_defaults(func=cmd_news)
    
    # multiple_quotes
    sp = sub.add_parser("multiple_quotes", help="Get multiple quotes")
    sp.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g., AAPL,MSFT,SPY")
    sp.set_defaults(func=cmd_multiple_quotes)
    
    # financial_statements
    sp = sub.add_parser("financial_statements", help="Get financial statements if available")
    sp.add_argument("--symbol", required=True)
    sp.set_defaults(func=cmd_financial_statements)
    
    # sentiment
    sp = sub.add_parser("sentiment", help="Get sentiment by source")
    sp.add_argument("--symbol", required=True)
    sp.set_defaults(func=cmd_sentiment)
    
    # advanced_sentiment
    sp = sub.add_parser("advanced_sentiment", help="Get advanced sentiment if texts available")
    sp.add_argument("--symbol", required=True)
    sp.set_defaults(func=cmd_advanced_sentiment)
    
    # market_breadth
    sp = sub.add_parser("market_breadth", help="Get market breadth summary")
    sp.set_defaults(func=cmd_market_breadth)
    
    # sector_performance
    sp = sub.add_parser("sector_performance", help="Get sector performance summary")
    sp.set_defaults(func=cmd_sector_performance)
    
    # compare
    sp = sub.add_parser("compare", help="Compare values and evaluate tolerance")
    sp.add_argument("--value", required=True)
    sp.add_argument("--ref_value", required=True)
    sp.add_argument("--tolerance_pct", required=True)
    sp.set_defaults(func=cmd_compare)
    
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Set global flags
    global _STRUCTURED_OUTPUT, _EXIT_ZERO_EVEN_ON_ERROR
    _STRUCTURED_OUTPUT = args.json
    _EXIT_ZERO_EVEN_ON_ERROR = args.exit_zero_even_on_error

    try:
        args.func(args)
    except SystemExit as se:
        # Allow explicit sys.exit from out()
        raise
    except Exception as e:
        out({"ok": False, "cmd": getattr(args, "command", None), "error": str(e)}, 2)


if __name__ == "__main__":
    main()