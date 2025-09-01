import contextlib
import requests
from bs4 import BeautifulSoup
import random
import yaml
import os

# Optimized HTTP client import with fallback
try:
    from core.http_client import optimized_get
except ImportError:
    def optimized_get(url, **kwargs):
        """Fallback to standard requests if optimized client unavailable"""
        return requests.get(url, **kwargs)

def fetch_ticker_universe(source="finviz", sample_size=20, static_list=None):
    """
    Fetch a diverse list of active tickers from a public source (default: Finviz Screener).
    Optionally merge with a static list and sample a subset.
    Returns a list of tickers (strings).
    """
    tickers = set()
    if static_list:
        tickers.update(static_list)
    if source == "finviz":
        _extracted_from_fetch_ticker_universe_11(tickers)
    # Fallback: add some well-known tickers if none found
    if not tickers:
        tickers.update(["AAPL", "TSLA", "MSFT", "GOOG", "AMZN", "NVDA", "AMD", "META", "NFLX", "SPY"])
    # Sample a diverse subset
    tickers = list(tickers)
    if len(tickers) > sample_size:
        tickers = random.sample(tickers, sample_size)
    return tickers


# TODO Rename this here and in `fetch_ticker_universe`
def _extracted_from_fetch_ticker_universe_11(tickers):
    url = "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o1000,sh_price_o5&ft=4"
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    ]
    headers = {"User-Agent": random.choice(user_agents)}
    try:
        resp = optimized_get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for row in soup.select("tr.screener-body-table-nw"):  # Finviz table rows
            cells = row.find_all("td")
            if len(cells) > 1:
                ticker = cells[1].get_text(strip=True)
                if ticker.isalpha() and 1 < len(ticker) <= 5:
                    tickers.add(ticker.upper())
    except Exception as e:
        print(f"[ERROR] Failed to fetch tickers from Finviz: {e}")
