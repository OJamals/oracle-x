from datetime import datetime, timezone
from typing import List, Optional, Dict
from decimal import Decimal
import pandas as pd

from data_feeds.finviz_scraper import (
    fetch_finviz_breadth, 
    fetch_finviz_sector_performance,
    fetch_finviz_news,
    fetch_finviz_insider_trading,
    fetch_finviz_earnings,
    fetch_finviz_forex,
    fetch_finviz_crypto
)
from data_feeds.models import MarketBreadth, GroupPerformance


class FinVizAdapter:
    def __init__(self, settings: Optional[dict] = None):
        self.settings = settings or {}

    def get_market_breadth(self) -> Optional[MarketBreadth]:
        stats = fetch_finviz_breadth() or {}
        adv = int(stats.get("advancers") or 0)
        dec = int(stats.get("decliners") or 0)
        highs = int(stats.get("new_highs") or 0)
        lows = int(stats.get("new_lows") or 0)
        
        breadth = MarketBreadth(
            exchange=None,
            advancers=adv,
            decliners=dec,
            unchanged=None,  # Not available in current data
            new_highs=highs,
            new_lows=lows,
            as_of=datetime.now(timezone.utc),
            source="finviz",
        )
        return breadth

    def get_sector_performance(self) -> List[GroupPerformance]:
        sectors = fetch_finviz_sector_performance()
        return [
            GroupPerformance(
                group_type="sector",
                group_name=str(sector['sector_name']),  # Ensure string type
                perf_1d=sector['perf_1d'],
                perf_1w=sector['perf_1w'],
                perf_1m=sector['perf_1m'],
                perf_3m=sector['perf_3m'],
                perf_6m=sector['perf_6m'],
                perf_1y=sector['perf_1y'],
                perf_ytd=sector['perf_ytd'],
                as_of=datetime.now(timezone.utc),
                source="finviz"
            )
            for sector in sectors
        ]
    
    def get_news(self) -> Dict[str, pd.DataFrame]:
        """
        Get news and blog data from Finviz.
        Returns a dictionary with 'news' and 'blogs' DataFrames.
        """
        return fetch_finviz_news()
    
    def get_insider_trading(self) -> pd.DataFrame:
        """
        Get insider trading data from Finviz.
        Returns a DataFrame with insider trading information.
        """
        return fetch_finviz_insider_trading()
    
    def get_earnings(self) -> Dict[str, pd.DataFrame]:
        """
        Get earnings data from Finviz.
        Returns a dictionary with earnings data partitioned by day.
        """
        return fetch_finviz_earnings()
    
    def get_forex(self) -> pd.DataFrame:
        """
        Get forex performance data from Finviz.
        Returns a DataFrame with forex performance data.
        """
        return fetch_finviz_forex()
    
    def get_crypto(self) -> pd.DataFrame:
        """
        Get crypto performance data from Finviz.
        Returns a DataFrame with crypto performance data.
        """
        return fetch_finviz_crypto()