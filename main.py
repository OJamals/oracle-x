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
import numpy as np
import matplotlib.pyplot as plt
from oracle_engine.agent import oracle_agent_pipeline
from vector_db.qdrant_store import ensure_collection, store_trade_vector
import re
from oracle_engine.prompt_chain import extract_scenario_tree
import yfinance as yf

def fetch_price_history(ticker, days=60):
    """Fetch historical price data for a ticker using yfinance."""
    end = datetime.now()
    start = end - timedelta(days=days)
    return yf.download(
        ticker,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        progress=False,
    )

def plot_price_chart(ticker, image_path, days=60):
    df = fetch_price_history(ticker, days)
    if df.empty:
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

def plot_scenario_tree_chart(scenario_tree, image_path):
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

def load_chart_image_base64(image_path):
    """Load a chart image file and encode it as base64 string."""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded

def ensure_playbooks_dir():
    if not os.path.exists("playbooks"):
        os.makedirs("playbooks")

def get_prompt_text(args):
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

def parse_and_save_playbook(playbook_json_str, today):
    try:
        playbook = json.loads(playbook_json_str)
        if not isinstance(playbook, dict):
            raise ValueError("Playbook output is not a dictionary.")
    except Exception as e:
        print(f"\n[ERROR] LLM output was not valid JSON or dict: {e}. Saving raw output instead.")
        playbook = {"raw_output": playbook_json_str}

    print("\n=== ORACLE-X FINAL PLAYBOOK ===\n")
    print(json.dumps(playbook, indent=2))

    ensure_playbooks_dir()
    playbook_file = f"playbooks/{today}.json"
    with open(playbook_file, "w") as f:
        json.dump(playbook, f, indent=2)
    print(f"\n\u001a Playbook saved to: {playbook_file}")
    return playbook

def store_trades_to_qdrant(playbook, playbook_json_str, today):
    # Ensure Qdrant collection exists before storing trades
    ensure_collection()
    if "trades" not in playbook or not isinstance(playbook["trades"], list):
        print("⚠️ No valid trades found to store to Qdrant.\n")
        return
    count = 0
    for idx, trade in enumerate(playbook["trades"]):
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
        try:
            store_trade_vector(trade)
            count += 1
        except Exception as e:
            print(f"[ERROR] Failed to store trade vector for {trade.get('ticker', 'UNKNOWN')}: {e}")
    print(f"✅ Stored {count} trades to Qdrant.\n")

def run_oracle_pipeline(prompt_text, today=None):
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
        t5 = time.time()
        store_trades_to_qdrant(playbook, playbook_json_str, today)
        timings['store_trades_to_qdrant'] = time.time() - t5
        # --- Generate real charts for each trade ---
        trades = playbook.get('trades', [])
        for trade in trades:
            if ticker := trade.get('ticker'):
                price_chart_path = f"charts/{today}_{ticker}_price.png"
                os.makedirs("charts", exist_ok=True)
                if out_path := plot_price_chart(ticker, price_chart_path):
                    chart_paths.append(out_path)
                    trade['price_chart'] = out_path
            if scenario_tree := trade.get('scenario_tree'):
                scenario_chart_path = f"charts/{today}_{trade.get('ticker','trade')}_scenario.png"
                if out_path := plot_scenario_tree_chart(
                    scenario_tree, scenario_chart_path
                ):
                    scenario_chart_paths.append(out_path)
                    trade['scenario_chart'] = out_path
    timings['total'] = time.time() - t0
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
    from main import get_prompt_text, run_oracle_pipeline
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
    print(json.dumps(playbook, indent=2))
    print("\n================ PIPELINE LOGS ================\n")
    print(result["logs"])
    print("\n================ TIMINGS ================\n")
    print(result["logs"].split("[TIMING]")[-1].strip())