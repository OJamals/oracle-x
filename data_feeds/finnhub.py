import os
import requests

def fetch_finnhub_quote(symbol: str) -> dict:
    """
    Fetch real-time quote from Finnhub.io (free tier, requires API key).
    Returns dict with price and meta info, or empty dict on error.
    """
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        print("[ERROR] FINNHUB_API_KEY not set in environment.")
        return {}
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] Finnhub fetch failed for {symbol}: {e}")
        return {}

def fetch_finnhub_news(symbol: str) -> list:
    """
    Fetch latest news for a symbol from Finnhub.io.
    Returns a list of news dicts, or empty list on error.
    """
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        print("[ERROR] FINNHUB_API_KEY not set in environment.")
        return []
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2025-07-01&to=2025-07-09&token={api_key}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] Finnhub news fetch failed for {symbol}: {e}")
        return []
