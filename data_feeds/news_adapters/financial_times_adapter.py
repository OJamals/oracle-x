"""
Financial Times News Adapter
Fetches financial news from Financial Times RSS feeds with advanced sentiment analysis
"""

import logging
from typing import Dict, List, Optional, Any
from .base_news_adapter import BaseNewsAdapter

logger = logging.getLogger(__name__)

class FinancialTimesAdapter(BaseNewsAdapter):
    """Financial Times news adapter with enhanced sentiment analysis"""
    
    def __init__(self):
        # Financial Times Companies RSS feed (working feed from FeedSpot)
        super().__init__(
            source_name="financial_times",
            rss_url="https://www.ft.com/rss/home"
        )
        
    def fetch_news_articles(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch Financial Times articles
        """
        # Get base articles from RSS
        articles = self._fetch_from_rss(symbol, limit)
        
        # FT articles are already finance-focused from companies section
        logger.info(f"Fetched {len(articles)} Financial Times articles for {symbol}")
        return articles[:limit]
        
    def _is_relevant_to_symbol(self, article: Dict[str, Any], symbol: str) -> bool:
        """Enhanced symbol relevance check for Financial Times"""
        # Use base class logic first
        if super()._is_relevant_to_symbol(article, symbol):
            return True
            
        # Additional FT-specific logic
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        # FT often uses formal company names
        ft_style_mentions = [
            f"{symbol.lower()} corp",
            f"{symbol.lower()} inc",
            f"{symbol.lower()} ltd",
            f"{symbol.upper()} corp",
            f"{symbol.upper()} inc", 
            f"{symbol.upper()} ltd",
            f"ticker: {symbol.lower()}",
            f"ticker: {symbol.upper()}",
            f"({symbol.upper()})"
        ]
        
        return any(mention in text for mention in ft_style_mentions)
