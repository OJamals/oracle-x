import pandas as pd
from decimal import Decimal
from typing import List, Dict, Optional
import re
import requests
import os
from bs4 import BeautifulSoup

# Import optimized HTTP client
try:
    from core.http_client import optimized_get

    OPTIMIZED_HTTP_AVAILABLE = True
except ImportError:
    OPTIMIZED_HTTP_AVAILABLE = False
    optimized_get = None
    print("[WARNING] Optimized HTTP client not available, falling back to requests")

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

# Import finvizfinance modules
from finvizfinance.group.performance import Performance
from finvizfinance.group.overview import Overview
from finvizfinance.news import News
from finvizfinance.insider import Insider
from finvizfinance.earnings import Earnings
from finvizfinance.forex import Forex
from finvizfinance.crypto import Crypto


def fetch_finviz_breadth() -> dict:
    """
    Get market breadth data from Finviz using finvizfinance and homepage scraping.
    Returns a dict with advancers, decliners, new highs, and new lows counts.
    """
    try:
        # Get advancers/decliners from homepage scraping (this data is not available in finvizfinance)
        url = "https://finviz.com/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        if OPTIMIZED_HTTP_AVAILABLE and optimized_get:
            resp = optimized_get(url, headers=headers, timeout=10)
        else:
            resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        page_text = soup.get_text()

        # Extract advancing/declining data using regex
        advancing_pattern = r"Advancing[\d.]+%\s*\((\d+)\)"
        declining_pattern = r"Declining\((\d+)\)"

        advancing_match = re.search(advancing_pattern, page_text)
        declining_match = re.search(declining_pattern, page_text)

        advancers = int(advancing_match.group(1)) if advancing_match else None
        decliners = int(declining_match.group(1)) if declining_match else None

        # Also look for new highs/lows
        new_high_pattern = r"New High[\d.]+%\s*\((\d+)\)"
        new_low_pattern = r"New Low\((\d+)\)"

        high_match = re.search(new_high_pattern, page_text)
        low_match = re.search(new_low_pattern, page_text)

        new_highs = int(high_match.group(1)) if high_match else None
        new_lows = int(low_match.group(1)) if low_match else None

        return {
            "advancers": advancers,
            "decliners": decliners,
            "new_highs": new_highs,
            "new_lows": new_lows,
        }

    except Exception as e:
        print(f"[ERROR] Finviz breadth fetch failed: {e}")
        return {}


def fetch_finviz_sector_performance() -> List[Dict[str, Optional[Decimal]]]:
    """
    Get sector performance data from Finviz using finvizfinance library.
    Returns a list of dictionaries with sector performance data.
    """
    try:
        # Use finvizfinance to get sector performance data
        fgperformance = Performance()
        df = fgperformance.screener_view(group="Sector")

        sectors = []
        for _, row in df.iterrows():

            def parse_percentage(text):
                if pd.isna(text) or text == "-" or text == "":
                    return None
                try:
                    # Remove % and whitespace, handle negative values
                    clean_text = str(text).strip().replace("%", "").replace(",", "")
                    if clean_text:
                        return Decimal(clean_text)
                    return None
                except:
                    return None

            sector_data = {
                "sector_name": str(row["Name"]),
                "perf_1d": parse_percentage(row.get("Change")),  # Daily change
                "perf_1w": parse_percentage(row.get("Perf Week")),
                "perf_1m": parse_percentage(row.get("Perf Month")),
                "perf_3m": parse_percentage(row.get("Perf Quart")),
                "perf_6m": parse_percentage(row.get("Perf Half")),
                "perf_1y": parse_percentage(row.get("Perf Year")),
                "perf_ytd": parse_percentage(row.get("Perf YTD")),
            }
            sectors.append(sector_data)

        return sectors

    except Exception as e:
        print(f"[ERROR] Finviz sector performance fetch failed: {e}")
        return []


def fetch_finviz_news() -> Dict[str, pd.DataFrame]:
    """
    Get news and blog data from Finviz using finvizfinance library.
    Returns a dictionary with 'news' and 'blogs' DataFrames.
    """
    try:
        fnews = News()
        all_news = fnews.get_news()
        return all_news
    except Exception as e:
        print(f"[ERROR] Finviz news fetch failed: {e}")
        return {}


def fetch_finviz_insider_trading() -> pd.DataFrame:
    """
    Get insider trading data from Finviz using finvizfinance library.
    Returns a DataFrame with insider trading information.
    """
    try:
        finsider = Insider(option="top owner trade")
        return finsider.get_insider()
    except Exception as e:
        print(f"[ERROR] Finviz insider trading fetch failed: {e}")
        return pd.DataFrame()


def fetch_finviz_earnings() -> Dict[str, pd.DataFrame]:
    """
    Get earnings data from Finviz using finvizfinance library.
    Returns a dictionary with earnings data partitioned by day.
    """
    try:
        fEarnings = Earnings()
        df_days = fEarnings.partition_days(mode="financial")
        return df_days
    except Exception as e:
        print(f"[ERROR] Finviz earnings fetch failed: {e}")
        return {}


def fetch_finviz_forex() -> pd.DataFrame:
    """
    Get forex performance data from Finviz using finvizfinance library.
    Returns a DataFrame with forex performance data.
    """
    try:
        fforex = Forex()
        return fforex.performance()
    except Exception as e:
        print(f"[ERROR] Finviz forex fetch failed: {e}")
        return pd.DataFrame()


def fetch_finviz_crypto() -> pd.DataFrame:
    """
    Get crypto performance data from Finviz using finvizfinance library.
    Returns a DataFrame with crypto performance data.
    """
    try:
        fcrypto = Crypto()
        return fcrypto.performance()
    except Exception as e:
        print(f"[ERROR] Finviz crypto fetch failed: {e}")
        return pd.DataFrame()
