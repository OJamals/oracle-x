import yfinance as yf
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def fetch_dark_pool_data(tickers=None) -> dict:
    """
    Fetch dark pool block trade data using enhanced volume analysis.

    NEW: Uses improved detection algorithm with:
    - Volume spike analysis (>1.5x average)
    - Large trade detection (5-minute interval analysis)
    - Dark pool probability scoring
    - Lower threshold for better sensitivity

    NOTE: For production-grade dark pool data, see:
    - data_feeds/finra_ats_adapter.py (FREE official FINRA data)
    - docs/DARK_POOL_INTEGRATION_PLAN.md (Polygon.io integration guide)

    Returns:
        dict: Dark pool block trades analysis with enhanced signals.
    """
    if tickers is None:
        tickers = ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA"]

    dark_pools = []

    for ticker in tickers[:10]:  # Increased to 10 tickers
        try:
            # Get recent price and volume data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="30d", interval="1d")

            if hist.empty:
                continue

            # Calculate volume statistics
            avg_volume = hist["Volume"].mean()
            recent_volume = hist["Volume"].iloc[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # NEW: Get intraday data for large trade detection
            try:
                intraday = stock.history(period="1d", interval="5m")
                large_trades_count = 0

                if not intraday.empty and "Volume" in intraday.columns:
                    # Detect 5-minute intervals with unusually high volume
                    volume_threshold = (
                        avg_volume * 0.01
                    )  # 1% of daily avg = large trade
                    large_trades_count = (intraday["Volume"] > volume_threshold).sum()
            except:
                large_trades_count = 0  # Fallback if intraday unavailable

            # NEW: Calculate dark pool probability using multiple factors
            dark_pool_probability = min(
                (volume_ratio - 1.0) * 0.5  # Volume spike component
                + (large_trades_count / 78) * 0.3  # Large trade component
                + 0.2,  # Base probability
                1.0,
            )

            # IMPROVED: Lower threshold from 1.5x to catch more signals
            if volume_ratio > 1.3 or dark_pool_probability > 0.4:
                current_price = hist["Close"].iloc[-1]

                # Estimate block size based on volume anomaly
                estimated_block_size = int(
                    (recent_volume - avg_volume) * 0.15
                )  # 15% of excess

                # Determine signal strength
                if dark_pool_probability > 0.7:
                    signal = "high_institutional_activity"
                elif dark_pool_probability > 0.5:
                    signal = "elevated_institutional_activity"
                else:
                    signal = "possible_institutional_activity"

                dark_pools.append(
                    {
                        "ticker": ticker,
                        "block_size": max(
                            estimated_block_size, 5000
                        ),  # Minimum 5k shares
                        "price": round(float(current_price), 2),
                        "volume_ratio": round(volume_ratio, 2),
                        "avg_volume": int(avg_volume),
                        "recent_volume": int(recent_volume),
                        "large_trades_detected": int(large_trades_count),
                        "dark_pool_probability": round(dark_pool_probability, 2),
                        "signal": signal,
                        "confidence": round(dark_pool_probability, 2),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        except Exception as e:
            logger.error(f"Error fetching dark pool data for {ticker}: {e}")
            continue

    return {
        "dark_pools": dark_pools,
        "data_source": "volume_analysis_proxy",
        "timestamp": datetime.now().isoformat(),
        "total_detected": len(dark_pools),
    }
