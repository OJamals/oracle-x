"""
Consolidated Financial Data Feed
A comprehensive financial data aggregator that unifies multiple data sources
with intelligent fallback, caching, and rate limiting.
"""

import asyncio
import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import finnhub
import pandas as pd
import requests
import requests_cache
import yfinance as yf
from dotenv import load_dotenv
from financedatabase import Currencies, Equities, ETFs, Funds, Indices

# Optimized HTTP client import with fallback
try:
    from core.http_client import optimized_get
except ImportError:

    def optimized_get(url, **kwargs):
        """Fallback to standard requests if optimized client unavailable"""
        return requests.get(url, **kwargs)


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================
from data_feeds.cache.cache_service import CacheEntry, CacheService


class DataSource(Enum):
    YFINANCE = "yfinance"
    FINNHUB = "finnhub"
    FMP = "financial_modeling_prep"
    FINANCE_DATABASE = "finance_database"
    ALPHA_VANTAGE = "alpha_vantage"
    INVESTINY = "investiny"
    QUANTSUMORE = "quantsumore"
    STOCKDEX = "stockdex"


@dataclass
class Quote:
    symbol: str
    price: Decimal
    change: Decimal
    change_percent: Decimal
    volume: int
    market_cap: Optional[int] = None
    pe_ratio: Optional[Decimal] = None
    day_low: Optional[Decimal] = None
    day_high: Optional[Decimal] = None
    year_low: Optional[Decimal] = None
    year_high: Optional[Decimal] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None


@dataclass
class CompanyInfo:
    symbol: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    exchange: Optional[str] = None
    market_cap: Optional[int] = None
    employees: Optional[int] = None
    description: Optional[str] = None
    website: Optional[str] = None
    ceo: Optional[str] = None
    source: Optional[str] = None


@dataclass
class NewsItem:
    title: str
    summary: str
    url: str
    published: datetime
    source: str
    sentiment: Optional[str] = None


# ============================================================================
# Rate Limiting
# ============================================================================


class RateLimiter:
    def __init__(self):
        self.calls = defaultdict(list)
        self.limits = {
            DataSource.FINNHUB: (60, 60),  # 60 calls per minute
            DataSource.FMP: (250, 86400),  # 250 calls per day
            DataSource.ALPHA_VANTAGE: (5, 60),  # 5 calls per minute
        }

    def wait_if_needed(self, source: DataSource):
        if source not in self.limits:
            return

        max_calls, window = self.limits[source]
        now = time.time()

        # Clean old calls
        self.calls[source] = [
            call_time for call_time in self.calls[source] if now - call_time < window
        ]

        # Check if we need to wait
        if len(self.calls[source]) >= max_calls:
            wait_time = window - (now - self.calls[source][0]) + 1
            if wait_time > 0:
                logger.info(
                    f"Rate limiting {source.value}: waiting {wait_time:.1f} seconds"
                )
                time.sleep(wait_time)

        # Record this call
        self.calls[source].append(now)


# ============================================================================
# Caching
# ============================================================================


class DataCache:
    def __init__(self):
        self.cache = {}
        self.ttl = {
            "quote": 30,  # 30 seconds
            "historical_daily": 3600,  # 1 hour
            "historical_intraday": 300,  # 5 minutes
            "financials": 86400,  # 24 hours
            "company_info": 604800,  # 7 days
            "news": 1800,  # 30 minutes
        }

    def get(self, key: str, data_type: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl.get(data_type, 3600):
                return data
            else:
                del self.cache[key]
        return None

    def set(self, key: str, data: Any, data_type: str):
        self.cache[key] = (data, time.time())


# ============================================================================
# Source Adapters
# ============================================================================


class SourceAdapter:
    def __init__(self, cache: DataCache, rate_limiter: RateLimiter):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.source = None

    def _get_cached_or_fetch(self, key: str, data_type: str, fetch_func):
        # Check cache first
        cached = self.cache.get(key, data_type)
        if cached is not None:
            return cached

        # Apply rate limiting
        if self.source:
            self.rate_limiter.wait_if_needed(self.source)

        # Fetch and cache
        try:
            data = fetch_func()
            if data is not None:
                self.cache.set(key, data, data_type)
            return data
        except Exception as e:
            logger.error(f"Error fetching {key} from {self.source}: {e}")
            return None


class YFinanceAdapter(SourceAdapter):
    def __init__(self, cache: DataCache, rate_limiter: RateLimiter):
        super().__init__(cache, rate_limiter)
        self.source = DataSource.YFINANCE

    def get_quote(self, symbol: str) -> Optional[Quote]:
        def fetch():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")

            if info and not hist.empty:
                current_price = hist["Close"].iloc[-1]
                prev_close = info.get("previousClose", current_price)
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100 if prev_close else 0

                return Quote(
                    symbol=symbol,
                    price=Decimal(str(current_price)),
                    change=Decimal(str(change)),
                    change_percent=Decimal(str(change_percent)),
                    volume=int(info.get("volume", 0)),
                    market_cap=info.get("marketCap"),
                    pe_ratio=(
                        Decimal(str(info.get("trailingPE", 0)))
                        if info.get("trailingPE")
                        else None
                    ),
                    day_low=(
                        Decimal(str(info.get("dayLow", 0)))
                        if info.get("dayLow")
                        else None
                    ),
                    day_high=(
                        Decimal(str(info.get("dayHigh", 0)))
                        if info.get("dayHigh")
                        else None
                    ),
                    year_low=(
                        Decimal(str(info.get("fiftyTwoWeekLow", 0)))
                        if info.get("fiftyTwoWeekLow")
                        else None
                    ),
                    year_high=(
                        Decimal(str(info.get("fiftyTwoWeekHigh", 0)))
                        if info.get("fiftyTwoWeekHigh")
                        else None
                    ),
                    timestamp=datetime.now(),
                    source=self.source.value,
                )
            return None

        return self._get_cached_or_fetch(f"quote_{symbol}", "quote", fetch)

    def get_historical(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        def fetch():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if not hist.empty:
                hist.reset_index(inplace=True)
                hist["source"] = self.source.value
                return hist
            return None

        return self._get_cached_or_fetch(
            f"historical_{symbol}_{period}", "historical_daily", fetch
        )

    def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        def fetch():
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if info:
                return CompanyInfo(
                    symbol=symbol,
                    name=info.get("longName", info.get("shortName", "")),
                    sector=info.get("sector"),
                    industry=info.get("industry"),
                    country=info.get("country"),
                    exchange=info.get("exchange"),
                    market_cap=info.get("marketCap"),
                    employees=info.get("fullTimeEmployees"),
                    description=info.get("longBusinessSummary"),
                    website=info.get("website"),
                    source=self.source.value,
                )
            return None

        return self._get_cached_or_fetch(
            f"company_info_{symbol}", "company_info", fetch
        )

    def get_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        def fetch():
            ticker = yf.Ticker(symbol)
            news = ticker.news

            news_items = []
            for item in news[:limit]:
                news_items.append(
                    NewsItem(
                        title=item.get("title", ""),
                        summary=item.get("summary", ""),
                        url=item.get("link", ""),
                        published=datetime.fromtimestamp(
                            item.get("providerPublishTime", 0)
                        ),
                        source=self.source.value,
                    )
                )
            return news_items

        return self._get_cached_or_fetch(f"news_{symbol}_{limit}", "news", fetch) or []


class FinnhubAdapter(SourceAdapter):
    def __init__(self, cache: DataCache, rate_limiter: RateLimiter):
        super().__init__(cache, rate_limiter)
        self.source = DataSource.FINNHUB
        self.client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

    def get_quote(self, symbol: str) -> Optional[Quote]:
        def fetch():
            quote_data = self.client.quote(symbol)
            profile = self.client.company_profile2(symbol=symbol)

            if quote_data and "c" in quote_data:
                return Quote(
                    symbol=symbol,
                    price=Decimal(str(quote_data["c"])),
                    change=Decimal(str(quote_data.get("d", 0))),
                    change_percent=Decimal(str(quote_data.get("dp", 0))),
                    volume=0,  # Not available in basic quote
                    market_cap=(
                        profile.get("marketCapitalization") * 1000000
                        if profile.get("marketCapitalization")
                        else None
                    ),
                    day_low=(
                        Decimal(str(quote_data.get("l", 0)))
                        if quote_data.get("l")
                        else None
                    ),
                    day_high=(
                        Decimal(str(quote_data.get("h", 0)))
                        if quote_data.get("h")
                        else None
                    ),
                    timestamp=datetime.now(),
                    source=self.source.value,
                )
            return None

        return self._get_cached_or_fetch(f"quote_{symbol}", "quote", fetch)

    def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        def fetch():
            profile = self.client.company_profile2(symbol=symbol)

            if profile:
                return CompanyInfo(
                    symbol=symbol,
                    name=profile.get("name", ""),
                    sector=profile.get("finnhubIndustry"),
                    industry=profile.get("finnhubIndustry"),
                    country=profile.get("country"),
                    exchange=profile.get("exchange"),
                    market_cap=(
                        profile.get("marketCapitalization") * 1000000
                        if profile.get("marketCapitalization")
                        else None
                    ),
                    description=profile.get("description"),
                    website=profile.get("weburl"),
                    source=self.source.value,
                )
            return None

        return self._get_cached_or_fetch(
            f"company_info_{symbol}", "company_info", fetch
        )

    def get_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        def fetch():
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            news = self.client.company_news(symbol, _from=from_date, to=to_date)

            news_items = []
            for item in news[:limit]:
                news_items.append(
                    NewsItem(
                        title=item.get("headline", ""),
                        summary=item.get("summary", ""),
                        url=item.get("url", ""),
                        published=datetime.fromtimestamp(item.get("datetime", 0)),
                        source=self.source.value,
                    )
                )
            return news_items

        return self._get_cached_or_fetch(f"news_{symbol}_{limit}", "news", fetch) or []


class FMPAdapter(SourceAdapter):
    def __init__(self, cache: DataCache, rate_limiter: RateLimiter):
        super().__init__(cache, rate_limiter)
        self.source = DataSource.FMP
        self.api_key = os.getenv("FINANCIALMODELINGPREP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def _make_request(self, endpoint: str) -> Optional[Dict]:
        try:
            url = f"{self.base_url}/{endpoint}?apikey={self.api_key}"
            response = optimized_get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"FMP API error: {e}")
        return None

    def get_quote(self, symbol: str) -> Optional[Quote]:
        def fetch():
            data = self._make_request(f"quote/{symbol}")
            if data and len(data) > 0:
                item = data[0]
                return Quote(
                    symbol=symbol,
                    price=Decimal(str(item.get("price", 0))),
                    change=Decimal(str(item.get("change", 0))),
                    change_percent=Decimal(str(item.get("changesPercentage", 0))),
                    volume=int(item.get("volume", 0)),
                    market_cap=item.get("marketCap"),
                    pe_ratio=(
                        Decimal(str(item.get("pe", 0))) if item.get("pe") else None
                    ),
                    day_low=(
                        Decimal(str(item.get("dayLow", 0)))
                        if item.get("dayLow")
                        else None
                    ),
                    day_high=(
                        Decimal(str(item.get("dayHigh", 0)))
                        if item.get("dayHigh")
                        else None
                    ),
                    year_low=(
                        Decimal(str(item.get("yearLow", 0)))
                        if item.get("yearLow")
                        else None
                    ),
                    year_high=(
                        Decimal(str(item.get("yearHigh", 0)))
                        if item.get("yearHigh")
                        else None
                    ),
                    timestamp=datetime.now(),
                    source=self.source.value,
                )
            else:
                logger.warning(
                    f"FMPAdapter.get_quote: No data returned for symbol '{symbol}'. Response: {data}"
                )
            return None

        return self._get_cached_or_fetch(f"quote_{symbol}", "quote", fetch)

    def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        def fetch():
            data = self._make_request(f"profile/{symbol}")
            if data and len(data) > 0:
                item = data[0]
                return CompanyInfo(
                    symbol=symbol,
                    name=item.get("companyName", ""),
                    sector=item.get("sector"),
                    industry=item.get("industry"),
                    country=item.get("country"),
                    exchange=item.get("exchangeShortName"),
                    market_cap=item.get("mktCap"),
                    employees=item.get("fullTimeEmployees"),
                    description=item.get("description"),
                    website=item.get("website"),
                    ceo=item.get("ceo"),
                    source=self.source.value,
                )
            else:
                logger.warning(
                    f"FMPAdapter.get_company_info: No data returned for symbol '{symbol}'. Response: {data}"
                )
            return None

        return self._get_cached_or_fetch(
            f"company_info_{symbol}", "company_info", fetch
        )

    def get_historical(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        def fetch():
            # FMP historical-price-full endpoint gets all available data
            # We'll filter by date after receiving the data
            data = self._make_request(f"historical-price-full/{symbol}")
            if data and "historical" in data and len(data["historical"]) > 0:
                df = pd.DataFrame(data["historical"])
                if not df.empty:
                    df["Date"] = pd.to_datetime(df["date"])
                    df = df.rename(
                        columns={
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "adjClose": "Adj Close",
                            "volume": "Volume",
                        }
                    )
                    df["source"] = self.source.value
                    df = df.sort_values("Date")

                    # Filter by date range if specified
                    if from_date:
                        from_dt = pd.to_datetime(from_date)
                        df = df[df["Date"] >= from_dt]

                    if to_date:
                        to_dt = pd.to_datetime(to_date)
                        df = df[df["Date"] <= to_dt]

                    # If no date filters, return last year of data
                    if not from_date and not to_date:
                        one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
                        df = df[df["Date"] >= one_year_ago]

                    return df
            return None

        cache_key = f"historical_{symbol}_{from_date}_{to_date}"
        return self._get_cached_or_fetch(cache_key, "historical_daily", fetch)


class FinanceDatabaseAdapter(SourceAdapter):
    def __init__(self, cache: DataCache, rate_limiter: RateLimiter):
        super().__init__(cache, rate_limiter)
        self.source = DataSource.FINANCE_DATABASE
        self.equities = Equities()
        self.etfs = ETFs()
        self.funds = Funds()
        self.indices = Indices()
        self.currencies = Currencies()

    def search_equities(self, **kwargs) -> Dict:
        def fetch():
            result = self.equities.select(**kwargs)
            # Convert FinanceFrame to dict to avoid truth value ambiguity
            if hasattr(result, "to_dict"):
                return result.to_dict()
            return result

        cache_key = f"equities_search_{hash(str(kwargs))}"
        result = self._get_cached_or_fetch(cache_key, "company_info", fetch)
        return result if result is not None else {}

    def search_etfs(self, **kwargs) -> Dict:
        def fetch():
            result = self.etfs.select(**kwargs)
            # Convert FinanceFrame to dict to avoid truth value ambiguity
            if hasattr(result, "to_dict"):
                return result.to_dict()
            return result

        cache_key = f"etfs_search_{hash(str(kwargs))}"
        result = self._get_cached_or_fetch(cache_key, "company_info", fetch)
        return result if result is not None else {}


class InvestinyAdapter(SourceAdapter):
    def __init__(self, cache: DataCache, rate_limiter: RateLimiter):
        super().__init__(cache, rate_limiter)
        self.source = DataSource.INVESTINY
        # Import investiny functions
        try:
            from investiny.historical import historical_data
            from investiny.info import info
            from investiny.search import search_assets

            self.historical_data = historical_data
            self.search_assets = search_assets
            self.info = info
            self.available = True
        except ImportError:
            logger.error("Investiny package not available")
            self.available = False

    def _get_asset_id(self, symbol: str) -> Optional[int]:
        """Search for asset and return the ticker ID"""
        if not self.available:
            return None

        try:
            search_results = self.search_assets(symbol, limit=1)
            if search_results and len(search_results) > 0:
                return int(search_results[0]["ticker"])
        except Exception as e:
            logger.error(f"Error searching for {symbol} in Investiny: {e}")
        return None

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get current quote from Investiny"""
        if not self.available:
            return None

        def fetch():
            try:
                asset_id = self._get_asset_id(symbol)
                if not asset_id:
                    return None

                # Get recent historical data (last 2 days to get current and previous)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=2)

                hist_data = self.historical_data(
                    asset_id,
                    start_date.strftime("%m/%d/%Y"),
                    end_date.strftime("%m/%d/%Y"),
                )

                if hist_data and len(hist_data) >= 1:
                    # Get most recent data
                    dates = sorted(hist_data.keys(), reverse=True)
                    current_data = hist_data[dates[0]]

                    # Handle different data structures
                    if isinstance(current_data, dict):
                        current_price = current_data.get(
                            "close", current_data.get("price", 0)
                        )
                        volume = current_data.get("volume", 0)
                        day_low = current_data.get("low")
                        day_high = current_data.get("high")
                    elif isinstance(current_data, list) and len(current_data) > 0:
                        # If it's a list, assume it's [open, high, low, close, volume]
                        current_price = (
                            current_data[3]
                            if len(current_data) > 3
                            else current_data[-1]
                        )
                        volume = current_data[4] if len(current_data) > 4 else 0
                        day_low = current_data[2] if len(current_data) > 2 else None
                        day_high = current_data[1] if len(current_data) > 1 else None
                    else:
                        # Fallback - treat as a single value
                        try:
                            if isinstance(current_data, (int, float)):
                                current_price = float(current_data)
                            elif isinstance(current_data, str):
                                current_price = float(current_data)
                            else:
                                current_price = 0
                        except (TypeError, ValueError):
                            current_price = 0
                        volume = 0
                        day_low = None
                        day_high = None

                    # Calculate change if we have previous day
                    change = 0
                    change_percent = 0
                    if len(dates) > 1:
                        prev_data = hist_data[dates[1]]
                        if isinstance(prev_data, dict):
                            prev_price = prev_data.get(
                                "close", prev_data.get("price", 0)
                            )
                        elif isinstance(prev_data, list) and len(prev_data) > 3:
                            prev_price = prev_data[3]
                        else:
                            try:
                                if isinstance(prev_data, (int, float)):
                                    prev_price = float(prev_data)
                                elif isinstance(prev_data, str):
                                    prev_price = float(prev_data)
                                else:
                                    prev_price = 0
                            except (TypeError, ValueError):
                                prev_price = 0

                        if prev_price:
                            change = current_price - prev_price
                            change_percent = (change / prev_price) * 100

                    return Quote(
                        symbol=symbol,
                        price=Decimal(str(current_price)),
                        change=Decimal(str(change)),
                        change_percent=Decimal(str(change_percent)),
                        volume=int(volume),
                        day_low=Decimal(str(day_low)) if day_low else None,
                        day_high=Decimal(str(day_high)) if day_high else None,
                        timestamp=datetime.now(),
                        source=self.source.value,
                    )
                return None
            except Exception as e:
                logger.error(f"Error getting quote from Investiny: {e}")
                return None

        return self._get_cached_or_fetch(f"quote_{symbol}", "quote", fetch)

    def get_historical(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data from Investiny"""
        if not self.available:
            return None

        def fetch():
            try:
                asset_id = self._get_asset_id(symbol)
                if not asset_id:
                    return None

                # Convert period to days
                days_map = {
                    "1d": 1,
                    "5d": 5,
                    "1mo": 30,
                    "3mo": 90,
                    "6mo": 180,
                    "1y": 365,
                    "2y": 730,
                    "5y": 1825,
                }
                days = days_map.get(period, 365)

                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                hist_data = self.historical_data(
                    asset_id,
                    start_date.strftime("%m/%d/%Y"),
                    end_date.strftime("%m/%d/%Y"),
                )

                if hist_data:
                    # Convert to DataFrame
                    data_list = []
                    for date_str, data in hist_data.items():
                        row = {
                            "Date": pd.to_datetime(date_str),
                            "Open": data.get("open", data.get("price", 0)),
                            "High": data.get("high", data.get("price", 0)),
                            "Low": data.get("low", data.get("price", 0)),
                            "Close": data.get("close", data.get("price", 0)),
                            "Volume": data.get("volume", 0),
                        }
                        data_list.append(row)

                    if data_list:
                        df = pd.DataFrame(data_list)
                        df = df.sort_values("Date").reset_index(drop=True)
                        df["source"] = self.source.value
                        return df

                return None
            except Exception as e:
                logger.error(f"Error getting historical data from Investiny: {e}")
                return None

        return self._get_cached_or_fetch(
            f"historical_{symbol}_{period}", "historical_daily", fetch
        )

    def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        """Get company info from Investiny"""
        if not self.available:
            return None

        def fetch():
            try:
                asset_id = self._get_asset_id(symbol)
                if not asset_id:
                    return None

                info_data = self.info(str(asset_id))
                if info_data:
                    # Parse info data (format may vary)
                    info_str = str(info_data)

                    return CompanyInfo(
                        symbol=symbol,
                        name=symbol,  # Investiny may not provide detailed company info
                        description=info_str[:500] if len(info_str) > 500 else info_str,
                        source=self.source.value,
                    )
                return None
            except Exception as e:
                logger.error(f"Error getting company info from Investiny: {e}")
                return None

        return self._get_cached_or_fetch(
            f"company_info_{symbol}", "company_info", fetch
        )


class StockdexAdapter(SourceAdapter):
    def __init__(self, cache: DataCache, rate_limiter: RateLimiter):
        super().__init__(cache, rate_limiter)
        self.source = DataSource.STOCKDEX
        # Import stockdx
        try:
            from stockdex import Ticker

            self.Ticker = Ticker
            self.available = True
        except ImportError:
            logger.error("Stockdex package not available")
            self.available = False

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get current quote from Stockdex"""
        if not self.available:
            return None

        def fetch():
            try:
                ticker = self.Ticker(symbol)

                # Try to get current price data
                price_data = ticker.yahoo_api_price()
                if price_data is not None and not price_data.empty:
                    # Get the most recent data
                    latest = price_data.iloc[-1]

                    # Try different column names for price
                    current_price = 0
                    for col in [
                        "Close",
                        "close",
                        "Price",
                        "price",
                        "Adj Close",
                        "adj_close",
                    ]:
                        if col in latest and pd.notna(latest[col]):
                            current_price = float(latest[col])
                            break

                    # If still 0, try the first numeric column
                    if current_price == 0:
                        for col in price_data.columns:
                            if pd.api.types.is_numeric_dtype(
                                price_data[col]
                            ) and pd.notna(latest[col]):
                                current_price = float(latest[col])
                                break

                    # Calculate change if we have enough data
                    change = 0
                    change_percent = 0
                    if len(price_data) > 1:
                        prev = price_data.iloc[-2]
                        prev_price = 0
                        for col in [
                            "Close",
                            "close",
                            "Price",
                            "price",
                            "Adj Close",
                            "adj_close",
                        ]:
                            if col in prev and pd.notna(prev[col]):
                                prev_price = float(prev[col])
                                break

                        if prev_price > 0:
                            change = current_price - prev_price
                            change_percent = (change / prev_price) * 100

                    # Extract other data
                    volume = 0
                    for col in ["Volume", "volume", "Vol"]:
                        if col in latest and pd.notna(latest[col]):
                            volume = int(latest[col])
                            break

                    day_low = None
                    for col in ["Low", "low"]:
                        if col in latest and pd.notna(latest[col]):
                            day_low = float(latest[col])
                            break

                    day_high = None
                    for col in ["High", "high"]:
                        if col in latest and pd.notna(latest[col]):
                            day_high = float(latest[col])
                            break

                    return Quote(
                        symbol=symbol,
                        price=Decimal(str(current_price)),
                        change=Decimal(str(change)),
                        change_percent=Decimal(str(change_percent)),
                        volume=volume,
                        day_low=Decimal(str(day_low)) if day_low else None,
                        day_high=Decimal(str(day_high)) if day_high else None,
                        timestamp=datetime.now(),
                        source=self.source.value,
                    )
                return None
            except Exception as e:
                logger.error(f"Error getting quote from Stockdex: {e}")
                return None

        return self._get_cached_or_fetch(f"quote_{symbol}", "quote", fetch)

    def get_historical(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data from Stockdex"""
        if not self.available:
            return None

        def fetch():
            try:
                ticker = self.Ticker(symbol)

                # Get historical price data
                hist_data = ticker.yahoo_api_price()
                if hist_data is not None and not hist_data.empty:
                    # Ensure we have the right columns
                    if (
                        "Date" not in hist_data.columns
                        and hist_data.index.name != "Date"
                    ):
                        hist_data.reset_index(inplace=True)

                    # Standardize column names
                    hist_data = hist_data.rename(
                        columns={
                            "date": "Date",
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                        }
                    )

                    # Ensure Date column exists and is datetime
                    if "Date" in hist_data.columns:
                        hist_data["Date"] = pd.to_datetime(hist_data["Date"])
                    elif hist_data.index.name == "Date" or isinstance(
                        hist_data.index, pd.DatetimeIndex
                    ):
                        hist_data.reset_index(inplace=True)
                        hist_data.rename(columns={"index": "Date"}, inplace=True)
                        hist_data["Date"] = pd.to_datetime(hist_data["Date"])

                    hist_data["source"] = self.source.value

                    # Filter by period if needed
                    if period != "max":
                        days_map = {
                            "1d": 1,
                            "5d": 5,
                            "1mo": 30,
                            "3mo": 90,
                            "6mo": 180,
                            "1y": 365,
                            "2y": 730,
                            "5y": 1825,
                        }
                        days = days_map.get(period, 365)
                        cutoff_date = datetime.now() - timedelta(days=days)

                        if "Date" in hist_data.columns:
                            hist_data = hist_data[hist_data["Date"] >= cutoff_date]

                    return (
                        hist_data.sort_values("Date").reset_index(drop=True)
                        if "Date" in hist_data.columns
                        else hist_data
                    )

                return None
            except Exception as e:
                logger.error(f"Error getting historical data from Stockdex: {e}")
                return None

        return self._get_cached_or_fetch(
            f"historical_{symbol}_{period}", "historical_daily", fetch
        )

    def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        """Get company info from Stockdx"""
        if not self.available:
            return None

        def fetch():
            try:
                ticker = self.Ticker(symbol)

                # Try to get company info from financials
                financials = ticker.yahoo_api_financials()
                if financials is not None and not financials.empty:
                    # Basic company info from ticker
                    return CompanyInfo(
                        symbol=symbol,
                        name=symbol,  # Stockdx may not provide detailed company names
                        source=self.source.value,
                    )
                return None
            except Exception as e:
                logger.error(f"Error getting company info from Stockdx: {e}")
                return None

        return self._get_cached_or_fetch(
            f"company_info_{symbol}", "company_info", fetch
        )

    def get_financial_statements(
        self, symbol: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Get financial statements from Stockdx"""
        if not self.available:
            return {}

        def fetch():
            try:
                ticker = self.Ticker(symbol)

                results = {}

                # Get income statement
                try:
                    income_stmt = ticker.yahoo_api_income_statement()
                    results["income_statement"] = (
                        income_stmt
                        if income_stmt is not None and not income_stmt.empty
                        else None
                    )
                except Exception as e:
                    logger.warning(f"Error getting income statement: {e}")
                    results["income_statement"] = None

                # Get balance sheet
                try:
                    balance_sheet = ticker.yahoo_api_balance_sheet()
                    results["balance_sheet"] = (
                        balance_sheet
                        if balance_sheet is not None and not balance_sheet.empty
                        else None
                    )
                except Exception as e:
                    logger.warning(f"Error getting balance sheet: {e}")
                    results["balance_sheet"] = None

                # Get cash flow
                try:
                    cash_flow = ticker.yahoo_api_cash_flow()
                    results["cash_flow"] = (
                        cash_flow
                        if cash_flow is not None and not cash_flow.empty
                        else None
                    )
                except Exception as e:
                    logger.warning(f"Error getting cash flow: {e}")
                    results["cash_flow"] = None

                return results
            except Exception as e:
                logger.error(f"Error getting financial statements from Stockdx: {e}")
                return {}

        result = self._get_cached_or_fetch(f"financials_{symbol}", "financials", fetch)
        return result if result is not None else {}

        # ============================================================================
        # Main Consolidated Data Feed
        # ============================================================================

        self.cache_service = CacheService()
        # self.cache = DataCache()  # Deprecated, use cache_service


class ConsolidatedDataFeed:
    def __init__(self):
        self.cache = DataCache()
        self.rate_limiter = RateLimiter()

        # Initialize adapters
        self.yfinance = YFinanceAdapter(self.cache, self.rate_limiter)
        self.finnhub = FinnhubAdapter(self.cache, self.rate_limiter)
        self.fmp = FMPAdapter(self.cache, self.rate_limiter)
        self.finance_db = FinanceDatabaseAdapter(self.cache, self.rate_limiter)
        self.investiny = InvestinyAdapter(self.cache, self.rate_limiter)
        self.stockdex = StockdexAdapter(self.cache, self.rate_limiter)

        # Source priority for different data types
        self.source_priority = {
            "quote": [
                self.yfinance,
                self.fmp,
                self.finnhub,
                self.investiny,
                self.stockdex,
            ],
            "historical": [self.yfinance, self.fmp, self.investiny, self.stockdex],
            "company_info": [
                self.yfinance,
                self.fmp,
                self.finnhub,
                self.investiny,
                self.stockdex,
            ],
            "news": [self.finnhub, self.yfinance],
        }

        logger.info("ConsolidatedDataFeed initialized")

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get real-time quote with automatic fallback"""
        for adapter in self.source_priority["quote"]:
            try:
                quote = adapter.get_quote(symbol)
                if quote:
                    logger.info(f"Quote for {symbol} from {quote.source}")
                    return quote
            except Exception as e:
                logger.warning(f"Failed to get quote from {adapter.source.value}: {e}")
                continue

        logger.error(f"Failed to get quote for {symbol} from all sources")
        return None

    def get_historical(
        self,
        symbol: str,
        period: str = "1y",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Get historical data with automatic fallback"""
        for adapter in self.source_priority["historical"]:
            try:
                if hasattr(adapter, "get_historical"):
                    if adapter.source == DataSource.YFINANCE:
                        hist = adapter.get_historical(symbol, period)
                    else:
                        hist = adapter.get_historical(symbol, from_date, to_date)

                    if hist is not None and not hist.empty:
                        logger.info(
                            f"Historical data for {symbol} from {adapter.source.value}"
                        )
                        return hist
            except Exception as e:
                logger.warning(
                    f"Failed to get historical data from {adapter.source.value}: {e}"
                )
                continue

        logger.error(f"Failed to get historical data for {symbol} from all sources")
        return None

    def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        """Get company information with automatic fallback"""
        for adapter in self.source_priority["company_info"]:
            try:
                info = adapter.get_company_info(symbol)
                if info:
                    logger.info(f"Company info for {symbol} from {info.source}")
                    return info
            except Exception as e:
                logger.warning(
                    f"Failed to get company info from {adapter.source.value}: {e}"
                )
                continue

        logger.error(f"Failed to get company info for {symbol} from all sources")
        return None

    def get_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        """Get news with automatic fallback"""
        for adapter in self.source_priority["news"]:
            try:
                if hasattr(adapter, "get_news"):
                    news = adapter.get_news(symbol, limit)
                    if news:
                        logger.info(f"News for {symbol} from {adapter.source.value}")
                        return news
            except Exception as e:
                logger.warning(f"Failed to get news from {adapter.source.value}: {e}")
                continue

        logger.error(f"Failed to get news for {symbol} from all sources")
        return []

    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols efficiently"""
        results = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                results[symbol] = quote
        return results

    def search_securities(self, query: Optional[str] = None, **filters) -> Dict:
        """Search for securities using finance database"""
        try:
            results = {}

            # Search equities
            equities = self.finance_db.search_equities(**filters)
            if equities:
                results["equities"] = equities

            # Search ETFs
            etfs = self.finance_db.search_etfs(**filters)
            if etfs:
                results["etfs"] = etfs

            return results
        except Exception as e:
            logger.error(f"Failed to search securities: {e}")
            return {}

    def get_financial_statements(
        self, symbol: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Get financial statements (income statement, balance sheet, cash flow) with automatic fallback"""
        # Try Stockdex first as it has comprehensive financial data
        if (
            hasattr(self.stockdex, "get_financial_statements")
            and self.stockdex.available
        ):
            try:
                financials = self.stockdex.get_financial_statements(symbol)
                if financials and any(
                    v is not None and not v.empty
                    for v in financials.values()
                    if hasattr(v, "empty")
                ):
                    logger.info(
                        f"Financial statements for {symbol} from {self.stockdex.source.value}"
                    )
                    return financials
            except Exception as e:
                logger.warning(f"Failed to get financial statements from Stockdex: {e}")

        # Return empty dict if no financial data available
        logger.error(f"Failed to get financial statements for {symbol}")
        return {}

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "total_cached_items": len(self.cache.cache),
            "cache_hit_rate": getattr(self.cache, "hit_rate", 0),
        }

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.cache.clear()
        logger.info("Cache cleared")


# ============================================================================
# Convenience Functions
# ============================================================================

# Global instance for easy access
_data_feed = None


def get_data_feed() -> ConsolidatedDataFeed:
    """Get the global data feed instance"""
    global _data_feed
    if _data_feed is None:
        _data_feed = ConsolidatedDataFeed()
    return _data_feed


def get_quote(symbol: str) -> Optional[Quote]:
    """Convenience function to get a quote"""
    return get_data_feed().get_quote(symbol)


def get_historical(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Convenience function to get historical data"""
    return get_data_feed().get_historical(symbol, period)


def get_company_info(symbol: str) -> Optional[CompanyInfo]:
    """Convenience function to get company info"""
    return get_data_feed().get_company_info(symbol)


def get_news(symbol: str, limit: int = 10) -> List[NewsItem]:
    """Convenience function to get news"""
    return get_data_feed().get_news(symbol, limit)


def get_financial_statements(symbol: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Convenience function to get financial statements"""
    return get_data_feed().get_financial_statements(symbol)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example usage
    feed = ConsolidatedDataFeed()

    # Test quote
    print("Testing quote...")
    quote = feed.get_quote("AAPL")
    if quote:
        print(
            f"AAPL: ${quote.price} ({quote.change:+.2f}, {quote.change_percent:+.2f}%) from {quote.source}"
        )

    # Test historical data
    print("\nTesting historical data...")
    hist = feed.get_historical("AAPL", period="1mo")
    if hist is not None:
        print(f"Historical data shape: {hist.shape}, source: {hist['source'].iloc[0]}")

    # Test company info
    print("\nTesting company info...")
    info = feed.get_company_info("AAPL")
    if info:
        print(f"Company: {info.name}, Sector: {info.sector}, from {info.source}")

    # Test news
    print("\nTesting news...")
    news = feed.get_news("AAPL", limit=3)
    for item in news:
        print(f"News: {item.title[:50]}... from {item.source}")

    # Test multiple quotes
    print("\nTesting multiple quotes...")
    quotes = feed.get_multiple_quotes(["AAPL", "GOOGL", "MSFT"])
    for symbol, quote in quotes.items():
        print(f"{symbol}: ${quote.price} from {quote.source}")

    # Test financial statements (new feature)
    print("\nTesting financial statements...")
    financials = feed.get_financial_statements("AAPL")
    for stmt_type, data in financials.items():
        if data is not None and not data.empty:
            print(f"{stmt_type}: {data.shape}")
        else:
            print(f"{stmt_type}: No data available")

    # Test new adapters availability
    print("\nTesting new adapters...")
    print(f"Investiny available: {feed.investiny.available}")
    print(f"Stockdex available: {feed.stockdex.available}")

    print(f"\nCache stats: {feed.get_cache_stats()}")
