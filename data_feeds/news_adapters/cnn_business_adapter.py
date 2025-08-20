"""
CNBC Business News Adapter (formerly CNN Business)
Fetches financial news from CNBC Business RSS feeds with advanced sentiment analysis
"""

import logging
from typing import Dict, List, Optional, Any
from .base_news_adapter import BaseNewsAdapter

logger = logging.getLogger(__name__)

class CNNBusinessAdapter(BaseNewsAdapter):
    """CNBC Business news adapter with enhanced sentiment analysis (replacing CNN Business)"""
    
    def __init__(self):
        # CNBC Business RSS feed (replacing stale CNN feed)
        super().__init__(
            source_name="cnbc_business",
            rss_url="https://www.cnbc.com/id/10001147/device/rss/rss.html"
        )
        
    def fetch_news_articles(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch CNBC Business articles with business focus
        """
        # Get base articles from RSS
        articles = self._fetch_from_rss(symbol, limit)
        
        # Additional filtering for business relevance
        filtered_articles = []
        for article in articles:
            if self._is_business_article(article):
                filtered_articles.append(article)
                
        logger.info(f"Filtered {len(filtered_articles)} business articles from {len(articles)} CNBC articles for {symbol}")
        return filtered_articles[:limit]
    
    def _is_business_article(self, article: Dict[str, Any]) -> bool:
        """Check if article is business-related"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        business_keywords = [
            'business', 'company', 'corporate', 'stock', 'shares', 'market',
            'earnings', 'revenue', 'profit', 'financial', 'economy',
            'ceo', 'executive', 'investor', 'investment', 'trading',
            'wall street', 'nasdaq', 'dow', 'sp 500'
        ]
        
        return any(keyword in text for keyword in business_keywords)
        
    def _is_relevant_to_symbol(self, article: Dict[str, Any], symbol: str) -> bool:
        """Enhanced symbol relevance check for CNN Business"""
        # Use base class logic first
        if super()._is_relevant_to_symbol(article, symbol):
            return True
            
        # Additional CNN Business-specific logic
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        # CNN Business style mentions
        cnn_style_mentions = [
            f"{symbol.lower()} stock",
            f"{symbol.lower()} shares",
            f"{symbol.upper()} stock",
            f"{symbol.upper()} shares",
            f"shares of {symbol.lower()}",
            f"shares of {symbol.upper()}"
        ]
        
        return any(mention in text for mention in cnn_style_mentions)
