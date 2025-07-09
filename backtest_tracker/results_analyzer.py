import json
from collections import Counter

def analyze_results():
    """
    Analyze backtest results and print a summary.
    """
    try:
        with open("backtest_results.json", "r") as f:
            results = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load backtest_results.json: {e}")
        return

    outcomes = [r["result"] for r in results]
    summary = Counter(outcomes)

    total = len(outcomes)
    hit_tp = summary.get("Hit TP", 0)
    hit_sl = summary.get("Hit SL", 0)
    miss = summary.get("Miss", 0)

    print("\n=== BACKTEST SUMMARY ===")
    print(f"Total Trades: {total}")
    print(f"Hit Take Profit: {hit_tp} ({hit_tp/total:.1%})")
    print(f"Hit Stop Loss: {hit_sl} ({hit_sl/total:.1%})")
    print(f"Missed: {miss} ({miss/total:.1%})")
    print("=========================\n")

if __name__ == "__main__":
    analyze_results()
