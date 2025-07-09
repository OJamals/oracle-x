import os
import requests

def fetch_iex_quote(symbol: str) -> dict:
    """
    Fetch real-time quote from IEX Cloud (free tier, requires API key).
    Returns dict with price and meta info, or empty dict on error.
    """
    api_key = os.environ.get("IEX_CLOUD_API_KEY")
    if not api_key:
        print("[ERROR] IEX_CLOUD_API_KEY not set in environment.")
        return {}
    url = f"https://cloud.iexapis.com/stable/stock/{symbol}/quote?token={api_key}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] IEX Cloud fetch failed for {symbol}: {e}")
        return {}
