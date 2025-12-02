#!/usr/bin/env python3

import json
from datetime import datetime


def check_new_pipeline_prices():
    """Check the latest pipeline output for price accuracy"""

    # Read the latest pipeline output
    try:
        with open(
            "/Users/omar/Documents/Projects/oracle-x/playbooks/standard_playbook_20250902_123620.json",
            "r",
        ) as f:
            data = json.load(f)

        playbook_str = data.get("playbook", "{}")
        if isinstance(playbook_str, str):
            playbook = json.loads(playbook_str)
        else:
            playbook = playbook_str

        trades = playbook.get("trades", [])

        print("=== NEW PIPELINE OUTPUT PRICES ===")

        # Current market prices for comparison
        current_prices = {
            "SPY": 635.86,
            "QQQ": 561.50,  # Approximate
            "AMD": 159.61,
            "TSLA": 328.98,
        }

        for trade in trades:
            ticker = trade.get("ticker", "N/A")
            entry_range = trade.get("entry_range", "N/A")

            if ticker in current_prices:
                current = current_prices[ticker]

                # Parse entry range
                try:
                    if "-" in entry_range:
                        low, high = entry_range.split("-")
                        entry_mid = (float(low) + float(high)) / 2
                    else:
                        entry_mid = float(entry_range)

                    diff = current - entry_mid
                    diff_pct = (diff / entry_mid) * 100

                    print(f"\n{ticker}:")
                    print(f"  Current Price: ${current:.2f}")
                    print(f"  Pipeline Entry: {entry_range} (mid: ${entry_mid:.2f})")
                    print(f"  Difference: ${diff:.2f} ({diff_pct:+.1f}%)")

                    if abs(diff_pct) < 5:
                        print(f"  ✅ ACCURATE (within 5%)")
                    else:
                        print(f"  ❌ INACCURATE (>5% difference)")

                except Exception as e:
                    print(f"  Error parsing entry range: {e}")
            else:
                print(f"\n{ticker}: No current price available for comparison")

    except Exception as e:
        print(f"Error reading pipeline output: {e}")


if __name__ == "__main__":
    check_new_pipeline_prices()
