import json
import os
from datetime import datetime

from data_feeds.market_internals import fetch_market_internals
from data_feeds.options_flow import fetch_options_flow
from data_feeds.dark_pools import fetch_dark_pool_data
from data_feeds.sentiment import fetch_sentiment_data
from data_feeds.earnings_calendar import fetch_earnings_calendar

def ensure_signals_dir():
    if not os.path.exists("signals"):
        os.makedirs("signals")

def run_signals_scraper():
    today = datetime.now().strftime("%Y-%m-%d")

    signals = {
        "market_internals": fetch_market_internals(),
        "options_flow": fetch_options_flow(),
        "dark_pools": fetch_dark_pool_data(),
        "sentiment_data": fetch_sentiment_data(),
        "earnings_calendar": fetch_earnings_calendar()
    }

    ensure_signals_dir()
    signals_file = f"signals/{today}.json"

    with open(signals_file, "w") as f:
        json.dump(signals, f, indent=2)

    print(f"\n✅ Signals snapshot saved to: {signals_file}")

if __name__ == "__main__":
    run_signals_scraper()
