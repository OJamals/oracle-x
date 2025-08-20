"""
Fortune News Adapter
Fetches business and financial news from Fortune RSS feeds
"""

import logging
from typing import Dict, List, Optional, Any
from .base_news_adapter import BaseNewsAdapter

logger = logging.getLogger(__name__)

class FortuneAdapter(BaseNewsAdapter):
    """Fortune business and financial news adapter"""
    
    def __init__(self):
        # Fortune main RSS feed (working feed from FeedSpot)
        super().__init__(
            source_name="fortune",
            rss_url="https://fortune.com/feed/fortune-feeds/?id=3230629"
        )
        
    def fetch_news_articles(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch Fortune articles with additional filtering for financial relevance
        """
        # Get base articles from RSS
        articles = self._fetch_from_rss(symbol, limit)
        
        # Additional Fortune-specific filtering
        filtered_articles = []
        for article in articles:
            if self._is_financial_article(article):
                filtered_articles.append(article)
                
        logger.info(f"Filtered {len(filtered_articles)} financial articles from {len(articles)} Fortune articles for {symbol}")
        return filtered_articles[:limit]
    
    def _is_financial_article(self, article: Dict[str, Any]) -> bool:
        """Check if article is finance-related"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        financial_keywords = [
            'stock', 'shares', 'earnings', 'revenue', 'profit', 'financial',
            'market', 'trading', 'investor', 'investment', 'analyst',
            'sec', 'ipo', 'dividend', 'quarterly', 'guidance', 'forecast',
            'wall street', 'nasdaq', 'nyse', 'sp 500', 's&p 500',
            'ceo', 'fortune 500', 'business', 'corporate', 'company',
            'acquisition', 'merger', 'valuation', 'startup', 'funding'
        ]
        
        return any(keyword in text for keyword in financial_keywords)
        
    def _is_relevant_to_symbol(self, article: Dict[str, Any], symbol: str) -> bool:
        """Enhanced symbol relevance check for Fortune"""
        # Use base class logic first
        if super()._is_relevant_to_symbol(article, symbol):
            return True
            
        # Additional Fortune-specific logic
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        # Check for ticker format variations
        symbol_variations = [
            f"{symbol.lower()}",
            f"${symbol.lower()}",
            f"({symbol.lower()})",
            f"{symbol.upper()}",
            f"${symbol.upper()}",
            f"({symbol.upper()})"
        ]
        
        return any(variation in text for variation in symbol_variations)
