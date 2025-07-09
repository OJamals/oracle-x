import requests
from bs4 import BeautifulSoup
import random
import time
import yaml

def fetch_finviz_breadth() -> dict:
    """
    Scrape market breadth data from Finviz (free, open-source).
    Returns a dict with advancers, decliners, and unchanged counts. Adds user-agent spoofing, retry, and proxy support.
    """
    url = "https://finviz.com/"
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

    max_retries = 4
    backoff = 2
    for attempt in range(max_retries):
        headers = {"User-Agent": random.choice(user_agents)}
        try:
            resp = requests.get(url, headers=headers, timeout=7, proxies=proxies)
            if resp.status_code == 403:
                print("[WARN] Finviz returned 403 Forbidden. Retrying with new user-agent...")
                time.sleep(backoff * (attempt + 1))
                continue
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            stats = {}
            for stat in soup.find_all('td', class_='snapshot-td2-cp'):
                text = stat.get_text()
                if "Advancers" in text:
                    stats["advancers"] = int(text.split()[1].replace(",", ""))
                if "Decliners" in text:
                    stats["decliners"] = int(text.split()[1].replace(",", ""))
            return stats
        except Exception as e:
            print(f"[ERROR] Finviz breadth scrape attempt {attempt+1} failed: {e}")
            time.sleep(backoff * (attempt + 1))
    print("[ERROR] Finviz breadth scrape failed after retries.")
    return {}
