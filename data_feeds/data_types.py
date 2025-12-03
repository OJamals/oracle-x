"""
Shared data types and dataclasses for the data feeds module.

This module contains shared dataclasses and types that are used across multiple
data feed components to avoid circular import issues.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class DataSource(Enum):
    """Available data sources in priority order"""

    YFINANCE = "yfinance"
    FINNHUB = "finnhub"
    IEX_CLOUD = "iex_cloud"
    REDDIT = "reddit"
    TWITTER = "twitter"
    GOOGLE_TRENDS = "google_trends"
    YAHOO_NEWS = "yahoo_news"
    NEWS = "news"
    FRED = "fred"
    SEC_EDGAR = "sec_edgar"
    TWELVE_DATA = "twelve_data"  # Deprecated (legacy placeholder)
    FINVIZ = "finviz"


class DataQuality(Enum):
    """Data quality levels"""

    EXCELLENT = "excellent"  # 95-100% quality score
    GOOD = "good"  # 80-94% quality score
    FAIR = "fair"  # 60-79% quality score
    POOR = "poor"  # 40-59% quality score
    UNUSABLE = "unusable"  # <40% quality score


@dataclass
class Quote:
    symbol: str
    price: Optional[Decimal]
    change: Optional[Decimal]
    change_percent: Optional[Decimal]
    volume: Optional[int]
    market_cap: Optional[int] = None
    pe_ratio: Optional[Decimal] = None
    day_low: Optional[Decimal] = None
    day_high: Optional[Decimal] = None
    year_low: Optional[Decimal] = None
    year_high: Optional[Decimal] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    quality_score: Optional[float] = None


@dataclass
class MarketData:
    symbol: str
    data: pd.DataFrame
    timeframe: str
    source: str
    timestamp: datetime
    quality_score: float


@dataclass
class SentimentData:
    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    source: str
    timestamp: datetime
    sample_size: Optional[int] = None
    raw_data: Optional[Dict] = None


@dataclass
class DataQualityMetrics:
    source: str
    quality_score: float
    latency_ms: float
    success_rate: float
    last_updated: datetime
    issues: List[str]
