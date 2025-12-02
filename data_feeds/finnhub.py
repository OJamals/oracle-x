import os
import requests
from typing import Dict, Any, Optional

# Async I/O utilities import with fallback
AsyncHTTPClient = None
ASYNC_IO_AVAILABLE = False
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.async_io_utils import AsyncHTTPClient
    ASYNC_IO_AVAILABLE = True
except ImportError:
    pass

from .api_key_validator import execute_with_fallback

# Optimized HTTP client import with fallback
try:
    from core.http_client import optimized_get
except ImportError:
    def optimized_get(url, **kwargs):
        """Fallback to standard requests if optimized client unavailable"""
        return requests.get(url, **kwargs)

# Try to import configuration manager
try:
    from core.config import config
    CONFIG_MANAGER_AVAILABLE = True
    def get_finnhub_api_key():
        return config.data_feeds.finnhub_api_key
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    def get_finnhub_api_key():
        return None

def fetch_finnhub_quote(symbol: str) -> Dict[str, Any]:
    """
    Fetch real-time quote from Finnhub.io with enhanced error handling and fallback support.
    Returns dict with price and meta info, or empty dict on error.
    """
    def _fetch_with_api_key() -> Dict[str, Any]:
        """Primary fetch function using API key"""
        # Get API key from configuration manager if available, otherwise fallback to environment
        if CONFIG_MANAGER_AVAILABLE:
            api_key = get_finnhub_api_key()
        else:
            api_key = os.environ.get("FINNHUB_API_KEY")

        if not api_key:
            raise ValueError("FINNHUB_API_KEY not configured")

        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
        resp = optimized_get(url, timeout=10)  # Increased timeout
        resp.raise_for_status()
        return resp.json()

    def _fetch_fallback() -> Dict[str, Any]:
        """Fallback function with demo/limited data"""
        print(f"[WARNING] Using Finnhub fallback mode for {symbol} - limited functionality")
        # Return empty data structure for fallback
        return {
            "c": 0.0,  # Current price
            "h": 0.0,  # High price
            "l": 0.0,  # Low price
            "o": 0.0,  # Open price
            "pc": 0.0, # Previous close
            "t": 0,    # Timestamp
            "warning": "Using fallback mode - no real-time data available"
        }

    try:
        return execute_with_fallback('finnhub', _fetch_with_api_key, _fetch_fallback)
    except Exception as e:
        print(f"[ERROR] Finnhub fetch failed for {symbol}: {e}")
        return _fetch_fallback()

def fetch_finnhub_news(symbol: str) -> list:
    """
    Fetch latest news for a symbol from Finnhub.io with enhanced error handling.
    Returns a list of news dicts, or empty list on error.
    """
    def _fetch_news_with_api_key() -> list:
        """Primary news fetch function using API key"""
        # Get API key from configuration manager if available, otherwise fallback to environment
        if CONFIG_MANAGER_AVAILABLE:
            api_key = get_finnhub_api_key()
        else:
            api_key = os.environ.get("FINNHUB_API_KEY")

        if not api_key:
            raise ValueError("FINNHUB_API_KEY not configured")

        # Use dynamic date range
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&token={api_key}"
        resp = optimized_get(url, timeout=10)  # Increased timeout
        resp.raise_for_status()
        return resp.json()

    def _fetch_news_fallback() -> list:
        """Fallback function with demo/limited news data"""
        print(f"[WARNING] Using Finnhub news fallback mode for {symbol} - limited functionality")
        return [
            {
                "category": "general",
                "datetime": int(datetime.now().timestamp()),
                "headline": f"News data unavailable for {symbol} - using fallback mode",
                "id": 0,
                "image": "",
                "related": symbol,
                "source": "fallback",
                "summary": "Real-time news data is not available. Please configure FINNHUB_API_KEY for live news feeds.",
                "url": "",
                "warning": "Using fallback mode - no real-time news available"
            }
        ]

    try:
        return execute_with_fallback('finnhub', _fetch_news_with_api_key, _fetch_news_fallback)
    except Exception as e:
        print(f"[ERROR] Finnhub news fetch failed for {symbol}: {e}")
        return _fetch_news_fallback()
