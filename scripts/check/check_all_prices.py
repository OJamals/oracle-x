#!/usr/bin/env python3

import yfinance as yf
import json
from datetime import datetime


def check_current_prices():
    """Check current prices for all tickers in the recent playbook"""
    tickers = [
        "TSLA",
        "SPY",
        "AMD",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "NFLX",
    ]

    print("=== CURRENT MARKET PRICES ===")
    current_prices = {}

    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                price = data["Close"].iloc[-1]
                current_prices[ticker] = price
                print(f"{ticker}: ${price:.2f}")
            else:
                print(f"{ticker}: No data available")
        except Exception as e:
            print(f"{ticker}: Error - {e}")

    print("\n=== PIPELINE OUTPUT COMPARISON ===")
    pipeline_prices = {
        "TSLA": {"entry": "330-333", "target": "315", "stop": "340"},
        "SPY": {"entry": "438-440", "target": "430", "stop": "443"},
        "AMD": {"entry": "105-107", "target": "112", "stop": "102"},
    }

    for ticker in tickers:
        if ticker in current_prices:
            current = current_prices[ticker]
            pipeline = pipeline_prices[ticker]
            entry_mid = (
                float(pipeline["entry"].split("-")[0])
                + (
                    float(pipeline["entry"].split("-")[1])
                    - float(pipeline["entry"].split("-")[0])
                )
                / 2
            )

            print(f"\n{ticker}:")
            print(f"  Current Price: ${current:.2f}")
            print(f"  Pipeline Entry: {pipeline['entry']} (mid: ${entry_mid:.2f})")
            print(
                f"  Difference: ${current - entry_mid:.2f} ({((current - entry_mid)/entry_mid)*100:.1f}%)"
            )

            if abs(current - entry_mid) > entry_mid * 0.1:  # More than 10% difference
                print(f"  ⚠️  WARNING: Price difference > 10%!")


if __name__ == "__main__":
    check_current_prices()
