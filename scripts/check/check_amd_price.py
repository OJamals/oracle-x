#!/usr/bin/env python3

import yfinance as yf
from datetime import datetime


def check_amd_price():
    """Check current AMD stock price"""
    try:
        ticker = yf.Ticker("AMD")
        data = ticker.history(period="1d")

        if not data.empty:
            current_price = data["Close"].iloc[-1]
            print(f"Current AMD price: ${current_price:.2f}")
            print(f"Date: {data.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")

            # Also get some recent history
            hist_data = ticker.history(period="5d")
            print(f"\nRecent 5-day price range:")
            print(f"High: ${hist_data['High'].max():.2f}")
            print(f"Low: ${hist_data['Low'].min():.2f}")
            print(f"Average: ${hist_data['Close'].mean():.2f}")

            return current_price
        else:
            print("No data available for AMD")
            return None

    except Exception as e:
        print(f"Error fetching AMD price: {e}")
        return None


if __name__ == "__main__":
    check_amd_price()
