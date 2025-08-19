import requests
from bs4 import BeautifulSoup
import random
import time
import yaml
import math
import os

def fetch_headlines_yahoo_finance() -> list:
    """
    Scrape latest headlines from Yahoo Finance (free, open-source).
    Returns a list of headline strings. Adds user-agent spoofing, retry, and proxy support.
    """
    url = "https://finance.yahoo.com/"
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    ]
    # Load proxy from config if present
    proxy = None
    try:
        with open("config/settings.yaml", "r") as f:
            config = yaml.safe_load(f)
            proxy = config.get("proxy")
    except Exception:
        pass
    proxies = {"http": proxy, "https": proxy} if proxy else None

    max_retries = 5
    base_backoff = 1.5  # seconds
    for attempt in range(max_retries):
        headers = {"User-Agent": random.choice(user_agents)}
        try:
            resp = requests.get(url, headers=headers, timeout=7, proxies=proxies)
            if resp.status_code == 429:
                print(
                    "[WARN] Yahoo returned 429 Too Many Requests. Retrying with new user-agent..."
                )
                # Exponential backoff with jitter for 429s
                sleep_for = base_backoff * (2 ** attempt)
                jitter = random.uniform(0, 0.5)
                time.sleep(min(30, sleep_for + jitter))
                continue
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            return [h.get_text() for h in soup.find_all('h3')]
        except Exception as e:
            print(f"[ERROR] Yahoo Finance news scrape attempt {attempt+1} failed: {e}")
            # Backoff with jitter; abort early if network unreachable repeatedly
            sleep_for = base_backoff * (2 ** attempt)
            jitter = random.uniform(0, 0.5)
            time.sleep(min(30, sleep_for + jitter))
    print("[ERROR] Yahoo Finance news scrape failed after retries.")
    return []
