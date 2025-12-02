import json
import os
from datetime import datetime
from pathlib import Path

from data_feeds.market_internals import fetch_market_internals
from data_feeds.options_flow import fetch_options_flow
from data_feeds.dark_pools import fetch_dark_pool_data
from data_feeds.sentiment import fetch_sentiment_data
from data_feeds.earnings_calendar import fetch_earnings_calendar
from utils.common import ensure_directory_exists


def run_signals_scraper(output_dir: Path | str = "signals") -> Path:
    """Collect the daily signals snapshot and persist to disk."""
    today = datetime.now().strftime("%Y-%m-%d")
    signals = {
        "market_internals": fetch_market_internals(),
        "options_flow": fetch_options_flow(),
        "dark_pools": fetch_dark_pool_data(),
        "sentiment_data": fetch_sentiment_data(),
        "earnings_calendar": fetch_earnings_calendar()
    }

    output_dir = Path(output_dir)
    ensure_directory_exists(output_dir)
    signals_file = output_dir / f"{today}.json"

    with signals_file.open("w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2)

    print(f"\n✅ Signals snapshot saved to: {signals_file}")
    return signals_file

if __name__ == "__main__":
    run_signals_scraper()
