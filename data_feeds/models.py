from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal, Optional


def ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class MarketBreadth:
    exchange: Optional[str]
    advancers: int
    decliners: int
    unchanged: Optional[int]
    new_highs: Optional[int]
    new_lows: Optional[int]
    as_of: datetime  # UTC-aware
    source: str

    def __post_init__(self):
        self.as_of = ensure_utc(self.as_of)


@dataclass
class GroupPerformance:
    group_type: Literal["sector", "industry", "country"]
    group_name: str
    perf_1d: Optional[Decimal]
    perf_1w: Optional[Decimal]
    perf_1m: Optional[Decimal]
    perf_3m: Optional[Decimal]
    perf_6m: Optional[Decimal]
    perf_1y: Optional[Decimal]
    perf_ytd: Optional[Decimal]
    as_of: datetime  # UTC-aware
    source: str

    def __post_init__(self):
        self.as_of = ensure_utc(self.as_of)
