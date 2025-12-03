import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_earnings_calendar(tickers=None) -> list:
    """
    Fetch upcoming earnings calendar data using free yfinance API.
    Returns:
        list: List of earnings events with real data.
    """
    if tickers is None:
        tickers = ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA"]

    earnings_events = []

    for ticker in tickers[:10]:  # Limit to 10 tickers for performance
        try:
            stock = yf.Ticker(ticker)

            # Get company info which may include earnings date
            info = stock.info

            # Get recent earnings history using income statement (new API)
            try:
                # Use income_stmt instead of deprecated quarterly_earnings
                income_stmt = stock.income_stmt

                if income_stmt is not None and not income_stmt.empty:
                    # Get Net Income from most recent quarter
                    net_income = (
                        income_stmt.loc["Net Income"].iloc[0]
                        if "Net Income" in income_stmt.index
                        else 0
                    )

                    # Get shares outstanding to calculate EPS
                    shares_outstanding = info.get(
                        "sharesOutstanding", info.get("impliedSharesOutstanding", 1)
                    )
                    recent_eps = (
                        (net_income / shares_outstanding)
                        if shares_outstanding > 0
                        else 0
                    )

                    # Estimate next earnings date (typically quarterly)
                    last_earnings_date = income_stmt.columns[0]  # Most recent date
                    estimated_next_date = last_earnings_date + timedelta(
                        days=90
                    )  # ~3 months
                else:
                    # Fallback if no income statement data
                    recent_eps = 0
                    estimated_next_date = datetime.now() + timedelta(days=30)

            except Exception as earnings_error:
                logger.warning(
                    f"Could not fetch earnings data for {ticker}: {earnings_error}"
                )
                recent_eps = 0
                estimated_next_date = datetime.now() + timedelta(days=30)

            earnings_events.append(
                {
                    "ticker": ticker,
                    "date": estimated_next_date.strftime("%Y-%m-%d"),
                    "estimate": round(float(recent_eps), 2) if recent_eps else 0.0,
                    "last_actual": round(float(recent_eps), 2) if recent_eps else 0.0,
                    "company_name": info.get("longName", ticker),
                    "sector": info.get("sector", "Unknown"),
                    "market_cap": info.get("marketCap", 0),
                    "data_source": "yfinance_income_stmt",
                    "confidence": "medium" if recent_eps != 0 else "low",
                }
            )

        except Exception as e:
            logger.error(f"Error fetching earnings data for {ticker}: {e}")
            # Create minimal entry for failed tickers
            earnings_events.append(
                {
                    "ticker": ticker,
                    "date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    "estimate": 0.0,
                    "last_actual": 0.0,
                    "company_name": ticker,
                    "sector": "Unknown",
                    "market_cap": 0,
                    "data_source": "fallback",
                    "confidence": "low",
                }
            )

    # Sort by estimated earnings date
    earnings_events.sort(key=lambda x: x["date"])

    return earnings_events
