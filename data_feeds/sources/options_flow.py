import yfinance as yf
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def fetch_options_flow(tickers=None) -> dict:
    """
    Fetch unusual options flow data using free yfinance options data.
    Analyzes options volume and open interest for unusual activity.
    Returns:
        dict: Unusual options sweeps analysis.
    """
    if tickers is None:
        tickers = ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA"]

    unusual_sweeps = []

    for ticker in tickers[:5]:  # Limit to 5 tickers for performance
        try:
            stock = yf.Ticker(ticker)

            # Get current stock price for context
            hist = stock.history(period="2d")
            if hist.empty:
                continue

            current_price = hist["Close"].iloc[-1]

            # Get options chain for next expiration
            options_dates = stock.options
            if not options_dates:
                continue

            # Use the nearest expiration date
            exp_date = options_dates[0]
            options_chain = stock.option_chain(exp_date)

            # Analyze calls for unusual volume
            calls = options_chain.calls
            if not calls.empty:
                # Find strikes near current price (within 10%)
                price_range = current_price * 0.1
                near_money_calls = calls[
                    (calls["strike"] >= current_price - price_range)
                    & (calls["strike"] <= current_price + price_range)
                ]

                for _, call in near_money_calls.iterrows():
                    volume = call.get("volume", 0)
                    open_interest = call.get("openInterest", 0)

                    # Detect unusual volume (volume > 2x open interest or volume > 1000)
                    if volume > max(open_interest * 2, 1000):
                        unusual_sweeps.append(
                            {
                                "ticker": ticker,
                                "direction": "Call",
                                "strike": float(call["strike"]),
                                "volume": int(volume),
                                "open_interest": int(open_interest),
                                "bid": float(call.get("bid", 0)),
                                "ask": float(call.get("ask", 0)),
                                "expiration": exp_date,
                                "current_price": round(float(current_price), 2),
                                "moneyness": round(
                                    (call["strike"] - current_price)
                                    / current_price
                                    * 100,
                                    2,
                                ),
                                "volume_oi_ratio": round(
                                    volume / max(open_interest, 1), 2
                                ),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

            # Analyze puts for unusual volume
            puts = options_chain.puts
            if not puts.empty:
                # Find strikes near current price (within 10%)
                near_money_puts = puts[
                    (puts["strike"] >= current_price - price_range)
                    & (puts["strike"] <= current_price + price_range)
                ]

                for _, put in near_money_puts.iterrows():
                    volume = put.get("volume", 0)
                    open_interest = put.get("openInterest", 0)

                    # Detect unusual volume
                    if volume > max(open_interest * 2, 1000):
                        unusual_sweeps.append(
                            {
                                "ticker": ticker,
                                "direction": "Put",
                                "strike": float(put["strike"]),
                                "volume": int(volume),
                                "open_interest": int(open_interest),
                                "bid": float(put.get("bid", 0)),
                                "ask": float(put.get("ask", 0)),
                                "expiration": exp_date,
                                "current_price": round(float(current_price), 2),
                                "moneyness": round(
                                    (current_price - put["strike"])
                                    / current_price
                                    * 100,
                                    2,
                                ),
                                "volume_oi_ratio": round(
                                    volume / max(open_interest, 1), 2
                                ),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

        except Exception as e:
            logger.error(f"Error fetching options flow for {ticker}: {e}")
            continue

    # Sort by volume (highest first)
    unusual_sweeps.sort(key=lambda x: x["volume"], reverse=True)

    return {
        "unusual_sweeps": unusual_sweeps,
        "total_sweeps": len(unusual_sweeps),
        "data_source": "yfinance_options",
        "timestamp": datetime.now().isoformat(),
        "analysis_criteria": {
            "volume_threshold": "volume > max(open_interest * 2, 1000)",
            "price_range": "within 10% of current price",
            "expiration": "nearest expiration date",
        },
    }
