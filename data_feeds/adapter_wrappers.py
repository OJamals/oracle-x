"""
Thin wrapper adapters conforming to SourceAdapterProtocol.

These wrappers adapt the existing consolidated adapters to a standardized
protocol without altering core logic or data shapes. They accept
(cache, rate_limiter, performance_tracker) for orchestrator ownership but
delegate implementation to the underlying consolidated adapters.

Wrappers implemented:
- YFinanceAdapterWrapper
- FMPAdapterWrapper
- FinnhubAdapterWrapper
- FinanceDatabaseAdapterWrapper

Notes:
- capabilities() reflects the underlying adapter features.
- Unsupported fetch_* methods raise NotImplementedError as required.
- fetch_historical returns the original pandas DataFrame from consolidated adapters.
- health() returns a minimal dict using performance/rate limit info when available.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Set, List
import logging

import pandas as pd

from data_feeds.adapter_protocol import SourceAdapterProtocol
from data_feeds.consolidated_data_feed import (
    YFinanceAdapter as ConsolidatedYF,
    FMPAdapter as ConsolidatedFMP,
    FinnhubAdapter as ConsolidatedFinnhub,
    FinanceDatabaseAdapter as ConsolidatedFinanceDB,
    DataCache,  # Import the DataCache class
    Quote,
    CompanyInfo,
    NewsItem,
)

logger = logging.getLogger(__name__)


class _BaseWrapper(SourceAdapterProtocol):
    def __init__(self, cache, rate_limiter, performance_tracker) -> None:
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker
        self._last_error: Optional[str] = None
        # Create the DataCache that the consolidated adapters expect
        self._data_cache = DataCache()
        # Underlying consolidated adapter instance created by subclasses
        self._adapter = None  # type: ignore

    def capabilities(self) -> Set[str]:
        raise NotImplementedError

    def fetch_quote(self, symbol: str) -> Optional[Quote]:
        raise NotImplementedError

    def fetch_historical(
        self,
        symbol: str,
        period: str = "1y",
        interval: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        raise NotImplementedError

    def fetch_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        raise NotImplementedError

    def fetch_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        raise NotImplementedError

    def fetch_sentiment(self, symbol: str, **kwargs):
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        rate_info: Dict[str, Any] = {}
        try:
            # Attempt to expose simple rate limit counters if the injected rate_limiter provides them
            if hasattr(self.rate_limiter, "limits"):
                rate_info["limits"] = {str(getattr(k, "value", str(k))): v for k, v in getattr(self.rate_limiter, "limits", {}).items()}
            if hasattr(self.rate_limiter, "daily_usage"):
                rate_info["daily_usage"] = {str(getattr(k, "value", str(k))): v for k, v in getattr(self.rate_limiter, "daily_usage", {}).items()}
        except Exception:
            # Best-effort only
            rate_info = {}

        status = "ok" if self._last_error is None else "degraded"
        return {
            "source": self.__class__.__name__.replace("AdapterWrapper", "").lower(),
            "status": status,
            "last_error": self._last_error,
            "rate_limits": rate_info or None,
        }


class YFinanceAdapterWrapper(_BaseWrapper):
    def __init__(self, cache, rate_limiter, performance_tracker) -> None:
        super().__init__(cache, rate_limiter, performance_tracker)
        # Consolidated adapter expects its own cache/ratelimiter types; we pass through
        self._adapter = ConsolidatedYF(self._data_cache, rate_limiter)

    def capabilities(self) -> Set[str]:
        return {"quote", "historical", "company_info", "news"}

    def fetch_quote(self, symbol: str) -> Optional[Quote]:
        try:
            return self._adapter.get_quote(symbol)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"YFinanceAdapterWrapper.fetch_quote error for {symbol}: {e}")
            return None

    def fetch_historical(
        self,
        symbol: str,
        period: str = "1y",
        interval: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        # Consolidated YF supports get_historical(symbol, period)
        try:
            return self._adapter.get_historical(symbol, period=period)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"YFinanceAdapterWrapper.fetch_historical error for {symbol}: {e}")
            return None

    def fetch_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        try:
            return self._adapter.get_company_info(symbol)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"YFinanceAdapterWrapper.fetch_company_info error for {symbol}: {e}")
            return None

    def fetch_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        try:
            return self._adapter.get_news(symbol, limit) or []
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"YFinanceAdapterWrapper.fetch_news error for {symbol}: {e}")
            return []

    def fetch_sentiment(self, symbol: str, **kwargs):
        raise NotImplementedError("YFinance adapter does not support sentiment.")


class FMPAdapterWrapper(_BaseWrapper):
    def __init__(self, cache, rate_limiter, performance_tracker) -> None:
        super().__init__(cache, rate_limiter, performance_tracker)
        self._adapter = ConsolidatedFMP(self._data_cache, rate_limiter)

    def capabilities(self) -> Set[str]:
        return {"quote", "historical", "company_info"}

    def fetch_quote(self, symbol: str) -> Optional[Quote]:
        try:
            return self._adapter.get_quote(symbol)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"FMPAdapterWrapper.fetch_quote error for {symbol}: {e}")
            return None

    def fetch_historical(
        self,
        symbol: str,
        period: str = "1y",
        interval: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        # Consolidated FMP expects (symbol, from_date, to_date)
        try:
            return self._adapter.get_historical(symbol, from_date, to_date)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"FMPAdapterWrapper.fetch_historical error for {symbol}: {e}")
            return None

    def fetch_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        try:
            return self._adapter.get_company_info(symbol)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"FMPAdapterWrapper.fetch_company_info error for {symbol}: {e}")
            return None

    def fetch_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        raise NotImplementedError("FMP adapter does not provide company news in consolidated adapter.")

    def fetch_sentiment(self, symbol: str, **kwargs):
        raise NotImplementedError("FMP adapter does not support sentiment.")


class FinnhubAdapterWrapper(_BaseWrapper):
    def __init__(self, cache, rate_limiter, performance_tracker) -> None:
        super().__init__(cache, rate_limiter, performance_tracker)
        self._adapter = ConsolidatedFinnhub(self._data_cache, rate_limiter)

    def capabilities(self) -> Set[str]:
        return {"quote", "company_info", "news"}

    def fetch_quote(self, symbol: str) -> Optional[Quote]:
        try:
            return self._adapter.get_quote(symbol)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"FinnhubAdapterWrapper.fetch_quote error for {symbol}: {e}")
            return None

    def fetch_historical(
        self,
        symbol: str,
        period: str = "1y",
        interval: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        raise NotImplementedError("Finnhub consolidated adapter does not expose historical fetch in this module.")

    def fetch_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        try:
            return self._adapter.get_company_info(symbol)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"FinnhubAdapterWrapper.fetch_company_info error for {symbol}: {e}")
            return None

    def fetch_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        try:
            return self._adapter.get_news(symbol, limit) or []
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"FinnhubAdapterWrapper.fetch_news error for {symbol}: {e}")
            return []

    def fetch_sentiment(self, symbol: str, **kwargs):
        raise NotImplementedError("Finnhub adapter does not support sentiment in this module.")


class FinanceDatabaseAdapterWrapper(_BaseWrapper):
    def __init__(self, cache, rate_limiter, performance_tracker) -> None:
        super().__init__(cache, rate_limiter, performance_tracker)
        self._adapter = ConsolidatedFinanceDB(cache, rate_limiter)

    def capabilities(self) -> Set[str]:
        # Exposes search capabilities primarily (mapped to fundamentals/discovery umbrella)
        return {"fundamentals"}

    def fetch_quote(self, symbol: str) -> Optional[Quote]:
        raise NotImplementedError("FinanceDatabase does not provide quotes.")

    def fetch_historical(
        self,
        symbol: str,
        period: str = "1y",
        interval: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        raise NotImplementedError("FinanceDatabase does not provide OHLCV time series.")

    def fetch_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        # Not directly supported via this adapter in consolidated module
        raise NotImplementedError("FinanceDatabase wrapper does not provide single-company info here.")

    def fetch_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        raise NotImplementedError("FinanceDatabase does not provide company news.")

    def fetch_sentiment(self, symbol: str, **kwargs):
        raise NotImplementedError("FinanceDatabase does not provide sentiment.")

    # Expose underlying search features in case the orchestrator uses them later
    def search_equities(self, **kwargs) -> Dict:
        try:
            return self._adapter.search_equities(**kwargs) or {}
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"FinanceDatabaseAdapterWrapper.search_equities error: {e}")
            return {}

    def search_etfs(self, **kwargs) -> Dict:
        try:
            return self._adapter.search_etfs(**kwargs) or {}
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"FinanceDatabaseAdapterWrapper.search_etfs error: {e}")
            return {}