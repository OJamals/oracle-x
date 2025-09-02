import yfinance as yf
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def fetch_dark_pool_data(tickers=None) -> dict:
    """
    Fetch dark pool block trade data using volume analysis.
    Uses unusual volume patterns as proxy for dark pool activity.
    Returns:
        dict: Dark pool block trades analysis.
    """
    if tickers is None:
        tickers = ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA"]
    
    dark_pools = []
    
    for ticker in tickers[:5]:  # Limit to 5 tickers for performance
        try:
            # Get recent price and volume data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="30d")
            
            if hist.empty:
                continue
                
            # Calculate volume statistics
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].iloc[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Detect unusual volume (potential dark pool activity)
            if volume_ratio > 1.5:  # 50% above average volume
                current_price = hist['Close'].iloc[-1]
                
                # Estimate block size based on volume anomaly
                estimated_block_size = int((recent_volume - avg_volume) * 0.1)  # 10% of excess volume
                
                dark_pools.append({
                    "ticker": ticker,
                    "block_size": max(estimated_block_size, 10000),  # Minimum 10k shares
                    "price": round(float(current_price), 2),
                    "volume_ratio": round(volume_ratio, 2),
                    "avg_volume": int(avg_volume),
                    "recent_volume": int(recent_volume),
                    "timestamp": datetime.now().isoformat(),
                    "confidence": min(volume_ratio / 2.0, 1.0)  # Confidence based on volume anomaly
                })
                
        except Exception as e:
            logger.error(f"Error fetching dark pool data for {ticker}: {e}")
            continue
    
    return {
        "dark_pools": dark_pools,
        "data_source": "volume_analysis_proxy",
        "timestamp": datetime.now().isoformat(),
        "total_detected": len(dark_pools)
    }
