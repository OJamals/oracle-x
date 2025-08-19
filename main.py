import warnings
warnings.filterwarnings(
    "ignore",
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated*",
    category=FutureWarning
)
import base64
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, cast
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # yfinance returns DataFrames
from oracle_engine.agent import oracle_agent_pipeline
from vector_db.qdrant_store import ensure_collection, store_trade_vector, embed_text
import re
from oracle_engine.prompt_chain import extract_scenario_tree
import yfinance as yf
from oracle_engine.model_attempt_logger import pop_attempts, get_attempts_snapshot
from typing import TYPE_CHECKING, Any
# Orchestrator unified data feeds (guarded import)
try:
    from agent_bundle.data_feed_orchestrator import (
        get_orchestrator,
        DataFeedOrchestrator,  # type: ignore
    )
except Exception:  # pragma: no cover - orchestrator optional
    get_orchestrator = None  # type: ignore
    class DataFeedOrchestrator:  # type: ignore
        ...

# Ensure root logger writes INFO+ messages to the original stdout so TIMING
# printouts emitted via logger.info appear in CLI captures (pipes / tee).
import logging as _logging
import sys as _sys
try:
    _root_logger = _logging.getLogger()
    # Only add a StreamHandler to original stdout if none exists that writes to a stream
    if not any(isinstance(h, _logging.StreamHandler) for h in list(_root_logger.handlers)):
        target_stream = _sys.__stdout__ if hasattr(_sys, "__stdout__") and _sys.__stdout__ is not None else _sys.stdout
        _sh = _logging.StreamHandler(target_stream)
        _sh.setLevel(_logging.INFO)
        _sh.setFormatter(_logging.Formatter("%(message)s"))
        _root_logger.addHandler(_sh)
    _root_logger.setLevel(_logging.INFO)
except Exception:
    # Best-effort; don't fail startup if logging setup can't be modified
    pass

def fetch_price_history(ticker: str, days: int = 60, orchestrator: Optional[Any] = None) -> Optional[pd.DataFrame]:
    """Fetch historical price data prioritizing orchestrator feeds, fallback to yfinance direct.

    Preference order:
      1. Orchestrator market data (cached, quality scored)
      2. Direct yfinance download
    """
    # Try orchestrator (daily interval) if provided
    if orchestrator is not None:
        try:
            md = orchestrator.get_market_data(ticker, period=f"{days}d", interval="1d")
            if md and isinstance(md.data, pd.DataFrame) and not md.data.empty:
                return md.data
        except Exception as e:  # Non-fatal
            print(f"[WARN] Orchestrator market data failed for {ticker}: {e}")
    # Fallback: direct yfinance
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(
            ticker,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            progress=False,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception as e:
        print(f"[WARN] Failed to fetch price history for {ticker}: {e}")
    return None

def plot_price_chart(ticker: str, image_path: str, days: int = 60, orchestrator: Optional[Any] = None):
    df = fetch_price_history(ticker, days, orchestrator)
    if df is None or df.empty:
        print(f"[WARN] No price data for {ticker}, skipping chart.")
        return None
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label=f"{ticker} Close", color='royalblue')
    plt.title(f"{ticker} Price Chart (Last {days} Days)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    return image_path

def plot_scenario_tree_chart(scenario_tree: Dict[str, Any], image_path: str):
    if not isinstance(scenario_tree, dict):
        return None
    labels = []
    values = []
    for k, v in scenario_tree.items():
        labels.append(k.replace('_', ' ').title())
        pct = re.search(r'(\d+)%', v)
        values.append(int(pct[1]) if pct else 0)
    if not values or sum(values) == 0:
        return None
    plt.figure(figsize=(5, 5))
    plt.pie(values, labels=labels, autopct='%1.0f%%', startangle=90, colors=['#4e79a7','#f28e2b','#e15759'])
    plt.title("Scenario Probabilities")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    return image_path

def load_chart_image_base64(image_path: str) -> str:
    """Load a chart image file and encode it as base64 string."""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded

def ensure_playbooks_dir():
    if not os.path.exists("playbooks"):
        os.makedirs("playbooks")

def get_prompt_text(args) -> str:
    prompt_text = args.prompt
    if not prompt_text:
        try:
            from auto_prompt import get_auto_prompt
            prompt_text = get_auto_prompt()
            print("[INFO] Auto-generated prompt text:", prompt_text)
        except Exception as e:
            print(f"[WARN] Auto-generation failed: {e}")
            prompt_text = (
                "TSLA, NVDA, SPY trending bullish after earnings beats. "
                "Options sweeps unusually high. Reddit sentiment surging."
            )
            print("[INFO] Using default prompt text.")
    return prompt_text

def _sanitize_llm_json(raw: str) -> str:
    """Attempt to sanitize common LLM JSON issues before json.loads.
    - Strip markdown fences
    - Remove leading/trailing text outside outermost braces
    - Fix common trailing commas
    - Balance braces crudely if off by one
    """
    if not isinstance(raw, str):
        return raw
    import re
    txt = raw.strip()
    # Remove markdown fences
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z0-9]*\n", "", txt)
        if txt.endswith("```"):
            txt = txt[:-3]
    # Extract first top-level JSON object heuristically
    first_brace = txt.find('{')
    last_brace = txt.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        txt = txt[first_brace:last_brace+1]
    # Remove trailing commas before } or ]
    txt = re.sub(r",\s*(}\s*)", r"\1", txt)
    txt = re.sub(r",\s*(]\s*)", r"\1", txt)
    # Balance braces (very naive)
    if txt.count('{') > txt.count('}'):
        txt += '}' * (txt.count('{') - txt.count('}'))
    return txt

def parse_and_save_playbook(playbook_json_str: str, today: str) -> Dict[str, Any]:
    raw_original = playbook_json_str
    sanitized = _sanitize_llm_json(playbook_json_str)
    attempts_snapshot = get_attempts_snapshot()
    try:
        parsed = json.loads(sanitized)
        if isinstance(parsed, dict):
            playbook: Dict[str, Any] = parsed
        else:
            raise ValueError("Playbook output is not a dictionary.")
    except Exception as e:
        print(f"\n[ERROR] LLM output was not valid JSON after sanitation: {e}. Saving raw output instead.")
        playbook = {"raw_output": raw_original, "sanitized": sanitized}  # type: ignore[assignment]
    if not isinstance(playbook, dict):  # final guard
        playbook = {"raw_output": raw_original}  # type: ignore[assignment]
    # Attach model attempt metadata
    meta = playbook.get("_meta") if isinstance(playbook, dict) else None
    if not isinstance(meta, dict):
        playbook["_meta"] = {"model_attempts": attempts_snapshot}
    else:
        meta["model_attempts"] = attempts_snapshot

    print("\n=== ORACLE-X FINAL PLAYBOOK ===\n")
    print(json.dumps(playbook, indent=2))

    ensure_playbooks_dir()
    playbook_file = f"playbooks/{today}.json"
    with open(playbook_file, "w") as f:
        json.dump(playbook, f, indent=2)
    print(f"\n\u001a Playbook saved to: {playbook_file}")
    # Clear attempts after persistence
    pop_attempts()
    return playbook

def store_trades_to_qdrant(playbook: Dict[str, Any], playbook_json_str: str, today: str):
    """Attempt to store trade vectors; accurately report successes/failures.

    Performs a lightweight embedding health check first to avoid repeated connection errors.
    """
    ensure_collection()
    trades = playbook.get("trades") if isinstance(playbook, dict) else None
    if not trades or not isinstance(trades, list):
        print("⚠️ No valid trades found to store to Qdrant.\n")
        return
    # Health check: try a tiny embed call once
    health_ok = False
    test_phrase = "health check"
    try:
        probe_vec = embed_text(test_phrase)
        if isinstance(probe_vec, list) and probe_vec and len(probe_vec) in (512, 768, 1024, 1536):
            health_ok = True
        else:
            print("[WARN] Embedding service returned empty or unexpected dimension vector on health check – will skip vector storage.")
    except Exception as e:
        print(f"[WARN] Embedding health check failed: {e} – skipping vector storage.")
    success = 0
    attempted = 0
    for idx, trade in enumerate(trades):
        if not isinstance(trade, dict):
            print(f"[ERROR] Trade at index {idx} is not a dictionary. Skipping.")
            continue
        trade["date"] = today
        if "thesis" not in trade or not trade["thesis"]:
            trade["thesis"] = trade.get("reasoning") or trade.get("notes") or "No thesis provided."
        scenario_tree = trade.get("scenario_tree")
        if not isinstance(scenario_tree, dict):
            extracted_tree = extract_scenario_tree(playbook_json_str)
            if isinstance(extracted_tree, dict):
                trade["scenario_tree"] = extracted_tree
                print(f"[INFO] scenario_tree extracted for {trade.get('ticker', 'UNKNOWN')} using robust extractor.")
            elif isinstance(scenario_tree, str):
                print(f"[WARN] scenario_tree is a string, attempting to parse: {scenario_tree[:60]}...")
                base = re.search(r"base case.*?([0-9]+%.*?)\n", scenario_tree, re.IGNORECASE)
                bull = re.search(r"bull case.*?([0-9]+%.*?)\n", scenario_tree, re.IGNORECASE)
                bear = re.search(r"bear case.*?([0-9]+%.*?)\n", scenario_tree, re.IGNORECASE)
                if base and bull and bear:
                    trade["scenario_tree"] = {
                        "base_case": base[1].strip(),
                        "bull_case": bull[1].strip(),
                        "bear_case": bear[1].strip(),
                    }
                    print(f"[INFO] Parsed scenario_tree for {trade.get('ticker', 'UNKNOWN')}")
                else:
                    print(f"[ERROR] Could not parse scenario_tree for {trade.get('ticker', 'UNKNOWN')}, using fallback.")
                    trade["scenario_tree"] = None
            else:
                print(f"[ERROR] scenario_tree missing or not a dict for {trade.get('ticker', 'UNKNOWN')}, using fallback.")
                trade["scenario_tree"] = None
        if not isinstance(trade["scenario_tree"], dict):
            trade["scenario_tree"] = {
                "base_case": "70% - No scenario_tree provided.",
                "bull_case": "20% - No scenario_tree provided.",
                "bear_case": "10% - No scenario_tree provided."
            }
        if not health_ok:
            continue  # Skip storage attempts entirely
        attempted += 1
        try:
            if store_trade_vector(trade):
                success += 1
        except Exception as e:
            print(f"[ERROR] Failed to store trade vector for {trade.get('ticker', 'UNKNOWN')}: {e}")
    if not health_ok:
        print("⚠️ Skipped vector storage for all trades due to embedding service unavailable.\n")
    else:
        print(f"✅ Stored {success} / {attempted} attempted trade vectors to Qdrant.\n")

def enrich_trades_with_data_feeds(trades: List[Dict[str, Any]], orchestrator: Optional[Any]):
    """Augment each trade dict in-place with unified orchestrator data (quote, sentiment, advanced sentiment).

    Safe, best-effort enrichment. Skips silently if orchestrator unavailable.
    """
    if orchestrator is None:
        return
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        ticker = trade.get("ticker")
        if not isinstance(ticker, str) or not ticker:
            continue
        # Quote
        try:
            q = orchestrator.get_quote(ticker)
            if q:
                trade.setdefault("quote", {
                    "price": float(q.price),
                    "change": float(q.change),
                    "change_percent": float(q.change_percent),
                    "volume": q.volume,
                    "source": q.source,
                    # Some quote objects may not yet have a quality_score attribute depending on adapter path
                    "quality_score": getattr(q, 'quality_score', None),
                    "timestamp": q.timestamp.isoformat() if q.timestamp else None,
                })
        except Exception as e:
            print(f"[WARN] Quote enrichment failed for {ticker}: {e}")
        # Basic sentiment (reddit/twitter)
        try:
            sent_map = orchestrator.get_sentiment_data(ticker)
            if sent_map:
                trade.setdefault("sentiment", {
                    k: {
                        "score": v.sentiment_score,
                        "confidence": v.confidence,
                        "sample_size": v.sample_size,
                        "source": v.source,
                        "timestamp": v.timestamp.isoformat(),
                    } for k, v in sent_map.items()
                })
        except Exception as e:
            print(f"[WARN] Sentiment enrichment failed for {ticker}: {e}")
        # Advanced sentiment aggregation
        try:
            adv = orchestrator.get_advanced_sentiment_data(ticker)
            if adv:
                trade.setdefault("advanced_sentiment", {
                    "score": adv.sentiment_score,
                    "confidence": adv.confidence,
                    "sample_size": adv.sample_size,
                    "source": adv.source,
                    "timestamp": adv.timestamp.isoformat(),
                })
        except Exception as e:
            print(f"[WARN] Advanced sentiment enrichment failed for {ticker}: {e}")

def enrich_playbook_top_level(playbook: Dict[str, Any], orchestrator: Optional[Any]):
    """Add market breadth, sector performance summary, and system health to top-level playbook keys.
    Avoid overwriting existing keys if already present."""
    if orchestrator is None:
        return
    try:
        if "market_breadth" not in playbook:
            breadth = orchestrator.get_market_breadth()
            if breadth:
                playbook["market_breadth"] = {
                    "advancers": breadth.advancers,
                    "decliners": breadth.decliners,
                    "unchanged": breadth.unchanged,
                }
    except Exception as e:
        print(f"[WARN] Market breadth enrichment failed: {e}")
    try:
        if "sector_performance" not in playbook:
            sectors = orchestrator.get_sector_performance()
            if sectors:
                # Summarize limited subset to keep token size manageable
                summarized = []
                for g in sectors[:10]:
                    try:
                        summarized.append({
                            "group": getattr(g, "name", getattr(g, "group", "unknown")),
                            "perf_1d": float(getattr(g, "perf_1d")) if getattr(g, "perf_1d", None) is not None else None,
                            "perf_ytd": float(getattr(g, "perf_ytd")) if getattr(g, "perf_ytd", None) is not None else None,
                        })
                    except Exception:
                        continue
                if summarized:
                    playbook["sector_performance"] = summarized
    except Exception as e:
        print(f"[WARN] Sector performance enrichment failed: {e}")
    try:
        if "data_feed_health" not in playbook:
            playbook["data_feed_health"] = orchestrator.validate_system_health()
    except Exception as e:
        print(f"[WARN] System health enrichment failed: {e}")

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize objects for JSON serialization (datetime -> isoformat)."""
    from datetime import datetime as _dt
    import numpy as _np
    from decimal import Decimal as _Dec
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_json(v) for v in obj)
    if isinstance(obj, _dt):
        return obj.isoformat()
    # Normalize numpy scalar types
    if isinstance(obj, (_np.floating, _np.integer)):
        try:
            return obj.item()
        except Exception:
            return float(obj) if isinstance(obj, _np.floating) else int(obj)
    # Decimal to float (fallback) for JSON
    if isinstance(obj, _Dec):
        return float(obj)
    return obj

def run_oracle_pipeline(prompt_text: str, today: Optional[str] = None) -> Dict[str, Any]:
    import io
    import sys
    import time
    from contextlib import redirect_stdout
    t0 = time.time()
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")
    timings = {}
    logs = io.StringIO()
    chart_paths = []
    scenario_chart_paths = []
    with redirect_stdout(logs):
        print("\n[DEBUG] Prompt text sent to agent pipeline:\n", prompt_text)
        t3 = time.time()
        playbook_json_str = oracle_agent_pipeline(prompt_text, None)
        timings['oracle_agent_pipeline'] = time.time() - t3
        print("\n[DEBUG] Raw LLM output:\n", playbook_json_str)
        t4 = time.time()
        playbook = parse_and_save_playbook(playbook_json_str, today)
        timings['parse_and_save_playbook'] = time.time() - t4
        # Enrich with orchestrator data prior to vector storage
        orch = None
        if get_orchestrator is not None:
            try:
                orch = get_orchestrator()
            except Exception as e:
                print(f"[WARN] Could not initialize orchestrator: {e}")
        trades: List[Any] = playbook.get('trades', []) if isinstance(playbook, dict) else []
        if trades and isinstance(trades, list):
            enrich_trades_with_data_feeds(cast(List[Dict[str, Any]], trades), orch)  # in-place
        enrich_playbook_top_level(playbook, orch)
        # Persist updated playbook file with enrichment (overwrite same date file)
        try:
            ensure_playbooks_dir()
            sanitized = _sanitize_for_json(playbook)
            with open(f"playbooks/{today}.json", "w") as f:
                json.dump(sanitized, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to persist enriched playbook: {e}")
        t5 = time.time()
        store_trades_to_qdrant(playbook, playbook_json_str, today)
        timings['store_trades_to_qdrant'] = time.time() - t5
        # --- Generate real charts for each trade ---
        for trade in trades:
            if not isinstance(trade, dict):
                continue  # Skip invalid entries
            ticker = trade.get('ticker')
            if ticker:
                price_chart_path = f"charts/{today}_{ticker}_price.png"
                os.makedirs("charts", exist_ok=True)
                out_path = plot_price_chart(ticker, price_chart_path, orchestrator=orch)
                if out_path:
                    chart_paths.append(out_path)
                    trade['price_chart'] = out_path
            scenario_tree = trade.get('scenario_tree')
            if isinstance(scenario_tree, dict):
                scenario_chart_path = f"charts/{today}_{trade.get('ticker','trade')}_scenario.png"
                out_path = plot_scenario_tree_chart(scenario_tree, scenario_chart_path)
                if out_path:
                    scenario_chart_paths.append(out_path)
                    trade['scenario_chart'] = out_path
    timings['total'] = time.time() - t0
    # Also write the captured internal print() buffer to the original stdout
    # so that external captures (pipes, tee, CI logs) include adapter-level
    # timing prints which were emitted while redirect_stdout was active.
    try:
        import sys as _sys
        # sys.__stdout__ points to the original stdout prior to any redirection
        if hasattr(_sys, "__stdout__") and _sys.__stdout__ is not None:
            try:
                _sys.__stdout__.write(logs.getvalue())
                _sys.__stdout__.flush()
            except Exception:
                # Best-effort; ignore if write/flush fails in constrained environments
                pass
    except Exception:
        pass

    print("[TIMING] Pipeline step timings (seconds):", timings)
    return {
        "playbook": playbook,
        "chart_paths": chart_paths,
        "scenario_chart_paths": scenario_chart_paths,
        "logs": f"{logs.getvalue()}\n[TIMING] {timings}",
        "date": today,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Oracle-X pipeline from CLI.")
    parser.add_argument('--prompt', type=str, default=None, help='Market summary or prompt text')
    parser.add_argument('--date', type=str, default=None, help='Date for playbook (YYYY-MM-DD)')
    args = parser.parse_args()
    # Use local functions directly (avoid self-import)
    prompt_text = get_prompt_text(args)
    print(f"[CLI] Running pipeline with prompt: {prompt_text}")
    result = run_oracle_pipeline(prompt_text, today=args.date)

    playbook = result["playbook"]
    print("\n================ ORACLE-X PLAYBOOK ================\n")
    trades = playbook.get("trades")
    if trades and isinstance(trades, list):
        print(f"Found {len(trades)} trade(s):\n")
        for i, trade in enumerate(trades, 1):
            print(f"--- Trade #{i} ---")
            print(f"Ticker:        {trade.get('ticker', 'N/A')}")
            print(f"Direction:     {trade.get('direction', 'N/A')}")
            print(f"Instrument:    {trade.get('instrument', 'N/A')}")
            print(f"Entry Range:   {trade.get('entry_range', 'N/A')}")
            print(f"Profit Target: {trade.get('profit_target', 'N/A')}")
            print(f"Stop Loss:     {trade.get('stop_loss', 'N/A')}")
            print(f"Counter Signal:{trade.get('counter_signal', 'N/A')}")
            print(f"Thesis:        {trade.get('thesis', 'N/A')}")
            scenario_tree = trade.get('scenario_tree')
            if isinstance(scenario_tree, dict):
                print("Scenario Tree:")
                for k, v in scenario_tree.items():
                    print(f"  {k.replace('_', ' ').title()}: {v}")
            if trade.get('price_chart'):
                print(f"[Chart] Price chart: {trade['price_chart']}")
            if trade.get('scenario_chart'):
                print(f"[Chart] Scenario chart: {trade['scenario_chart']}")
            print()
    else:
        print("No trades found in playbook.\n")
    if tape := playbook.get("tomorrows_tape"):
        print("--- Tomorrow's Tape ---")
        print(tape)
        print()
    for k, v in playbook.items():
        if k not in ("trades", "tomorrows_tape"):
            print(f"{k.title()}: {v}\n")
    print("================ RAW JSON OUTPUT ================\n")
    try:
        print(json.dumps(_sanitize_for_json(playbook), indent=2))
    except Exception as e:
        print(f"[ERROR] Failed to serialize playbook JSON: {e}")
    print("\n================ PIPELINE LOGS ================\n")
    print(result["logs"])
    print("\n================ TIMINGS ================\n")
    print(result["logs"].split("[TIMING]")[-1].strip())