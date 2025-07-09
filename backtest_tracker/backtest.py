import json
import os
from datetime import datetime


try:
    import yfinance as yf  # Simple for price data!
except ImportError:
    yf = None

PLAYBOOKS_DIR = "playbooks/"
BACKTEST_RESULTS = "backtest_results.json"

def load_playbook(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def backtest_trade(trade):
    ticker = trade["ticker"]
    direction = trade["direction"].lower()
    entry_price = float(trade["entry"].replace("$", "").split("-")[0])
    take_profit = float(trade["take_profit"].replace("$", ""))
    stop_loss = float(trade["stop_loss"].replace("$", ""))
    today = datetime.now().strftime("%Y-%m-%d")

    if yf is None:
        print("[ERROR] yfinance is not installed. Cannot fetch price data.")
        return {"ticker": ticker, "result": "No data"}

    # Fetch next-day price data
    try:
        df = yf.download(ticker, period="5d", interval="1d")
        if df.empty:
            return {"ticker": ticker, "result": "No data"}
        close_price = df["Close"][-1]
    except Exception as e:
        print(f"[ERROR] yfinance download failed: {e}")
        return {"ticker": ticker, "result": "No data"}

    # Evaluate hit/miss
    if direction == "long":
        if close_price >= take_profit:
            result = "Hit TP"
        elif close_price <= stop_loss:
            result = "Hit SL"
        else:
            result = "Miss"
    elif close_price <= take_profit:
        result = "Hit TP"
    elif close_price >= stop_loss:
        result = "Hit SL"
    else:
        result = "Miss"

    return {
        "ticker": ticker,
        "direction": direction,
        "entry": entry_price,
        "close": round(close_price, 2),
        "result": result,
        "date": today
    }

def run_backtest():
    results = []
    files = [f for f in os.listdir(PLAYBOOKS_DIR) if f.endswith(".json")]

    for file in files:
        playbook = load_playbook(os.path.join(PLAYBOOKS_DIR, file))
        for trade in playbook.get("trades", []):
            trade_result = backtest_trade(trade)
            results.append(trade_result)

    # Save results
    with open(BACKTEST_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Backtest results saved to: {BACKTEST_RESULTS}")

if __name__ == "__main__":
    run_backtest()
