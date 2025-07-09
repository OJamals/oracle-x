import os
import requests

def fetch_alpha_vantage_quote(symbol: str) -> dict:
    """
    Fetch real-time quote from Alpha Vantage (free tier, requires API key).
    Returns dict with price and meta info, or empty dict on error.
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("[ERROR] ALPHA_VANTAGE_API_KEY not set in environment.")
        return {}
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json().get("Global Quote", {})
    except Exception as e:
        print(f"[ERROR] Alpha Vantage fetch failed for {symbol}: {e}")
        return {}
