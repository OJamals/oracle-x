"""
Reuters News Adapter
Fetches financial news from Reuters RSS feeds with advanced sentiment analysis
"""

import logging
from typing import Dict, List, Optional, Any
from .base_news_adapter import BaseNewsAdapter

logger = logging.getLogger(__name__)

class ReutersAdapter(BaseNewsAdapter):
    """Reuters financial news adapter with enhanced sentiment analysis"""
    
    def __init__(self):
        # Reuters Official RSS feed (confirmed working with current content)
        super().__init__(
            source_name="reuters",
            rss_url="https://www.reuters.com/arc/outboundfeeds/rss/?outputType=xml"
        )
        
    def fetch_news_articles(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch Reuters articles with additional filtering for financial relevance
        """
        # Get base articles from RSS
        articles = self._fetch_from_rss(symbol, limit)
        
        # Additional Reuters-specific filtering
        filtered_articles = []
        for article in articles:
            if self._is_financial_article(article):
                filtered_articles.append(article)
                
        logger.info(f"Filtered {len(filtered_articles)} financial articles from {len(articles)} Reuters articles for {symbol}")
        return filtered_articles[:limit]
    
    def _is_financial_article(self, article: Dict[str, Any]) -> bool:
        """Check if article is finance-related"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        financial_keywords = [
            'stock', 'stocks', 'shares', 'earnings', 'revenue', 'profit', 'financial',
            'market', 'trading', 'investor', 'investment', 'analyst',
            'sec', 'ipo', 'dividend', 'quarterly', 'guidance', 'forecast',
            'wall street', 'nasdaq', 'nyse', 'sp 500', 's&p 500',
            'business', 'company', 'corporation', 'corporate', 'ceo', 'cfo',
            'fund', 'bank', 'banking', 'finance', 'economics', 'economic'
        ]
        
        return any(keyword in text for keyword in financial_keywords)
        
    def _is_relevant_to_symbol(self, article: Dict[str, Any], symbol: str) -> bool:
        """Enhanced symbol relevance check for Reuters"""
        # Use base class logic first (this includes the financial keywords fallback)
        if super()._is_relevant_to_symbol(article, symbol):
            return True
            
        # Additional Reuters-specific logic for stricter matching if needed
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
