from datetime import datetime, timezone
from typing import List, Optional
from decimal import Decimal

from data_feeds.finviz_scraper import fetch_finviz_breadth
from data_feeds.models import MarketBreadth, GroupPerformance


class FinVizAdapter:
    def __init__(self, settings: Optional[dict] = None):
        self.settings = settings or {}

    def get_market_breadth(self) -> Optional[MarketBreadth]:
        stats = fetch_finviz_breadth() or {}
        adv = int(stats.get("advancers") or 0)
        dec = int(stats.get("decliners") or 0)
        # Finviz homepage scrape currently provides advancers/decliners; others unavailable in MVP
        breadth = MarketBreadth(
            exchange=None,
            advancers=adv,
            decliners=dec,
            unchanged=None,
            new_highs=None,
            new_lows=None,
            as_of=datetime.now(timezone.utc),
            source="finviz",
        )
        return breadth

    def get_sector_performance(self) -> List[GroupPerformance]:
        # MVP: not implemented safely yet; returning empty list as a placeholder.
        # TODO: Implement sector performance parsing from FinViz groups with robust mapping and UA/proxy/retry.
        return []