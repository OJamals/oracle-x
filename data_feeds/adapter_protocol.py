"""
Adapter Protocols for standardized source integration.

SourceAdapterProtocol defines a uniform interface so the DataFeedOrchestrator
can manage caching, retries, rate limits, health, and capability discovery
consistently across all data sources.

Implementation notes:
- Adapters SHOULD implement only the fetch_* methods they natively support.
- For any fetch_* method that is not supported by a given source, the adapter
  MUST raise NotImplementedError. The orchestrator will use capabilities()
  to avoid calling unsupported methods, but raising ensures defensive behavior.
- __init__ should accept cache, rate_limiter, performance_tracker so that
  ownership remains with the orchestrator.
"""

from __future__ import annotations

from typing import Protocol, Set, Optional, Dict, Any, runtime_checkable
from pandas import DataFrame

# Reuse shared dataclasses and types where applicable
from data_feeds.consolidated_data_feed import Quote, CompanyInfo, NewsItem  # type: ignore
from data_feeds.data_types import MarketData  # type: ignore


@runtime_checkable
class SourceAdapterProtocol(Protocol):
    """
    Standard interface all source adapters should follow.

    Required constructor:
      __init__(cache, rate_limiter, performance_tracker) - stores references,
      but orchestrator owns their lifecycle.

    Capabilities:
      capabilities() - returns a set of supported features for the adapter.
      Example members: {"quote", "historical", "company_info", "news", "sentiment", "fundamentals"}

    Fetch methods:
      Each fetch_* method should be implemented if supported by the underlying source.
      If not supported, the method MUST raise NotImplementedError.

      fetch_quote(symbol: str) -> Optional[Quote]
      fetch_historical(
          symbol: str,
          period: str = "1y",
          interval: Optional[str] = None,
          from_date: Optional[str] = None,
          to_date: Optional[str] = None
      ) -> Optional[MarketData | DataFrame]
        - For sources returning a pandas DataFrame directly (e.g., legacy consolidated),
          return the DataFrame. For orchestrator-native sources, return MarketData.

      fetch_company_info(symbol: str) -> Optional[CompanyInfo]
      fetch_news(symbol: str, limit: int = 10) -> list[NewsItem]
      fetch_sentiment(symbol: str, **kwargs) -> Any

    Health:
      health() - returns a minimal dictionary with health metadata such as:
        {
          "source": "<name>",
          "status": "ok" | "degraded" | "error",
          "last_error": str | None,
          "rate_limits": { ... }  # if available from the injected rate_limiter
        }
    """

    def __init__(self, cache, rate_limiter, performance_tracker) -> None: ...

    def capabilities(self) -> Set[str]: ...

    def fetch_quote(self, symbol: str) -> Optional[Quote]: ...

    def fetch_historical(
        self,
        symbol: str,
        period: str = "1y",
        interval: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[MarketData | DataFrame]: ...

    def fetch_company_info(self, symbol: str) -> Optional[CompanyInfo]: ...

    def fetch_news(self, symbol: str, limit: int = 10) -> list[NewsItem]: ...

    def fetch_sentiment(self, symbol: str, **kwargs) -> Any: ...

    def health(self) -> Dict[str, Any]: ...
