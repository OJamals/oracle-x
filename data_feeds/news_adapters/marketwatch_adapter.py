"""
MarketWatch News Adapter
Fetches financial news from MarketWatch RSS feeds with advanced sentiment analysis
"""

import logging
from typing import Dict, List, Optional, Any
from .base_news_adapter import BaseNewsAdapter

logger = logging.getLogger(__name__)

class MarketWatchAdapter(BaseNewsAdapter):
    """MarketWatch financial news adapter with enhanced sentiment analysis"""
    
    def __init__(self):
        # MarketWatch Top Stories RSS feed
        super().__init__(
            source_name="marketwatch",
            rss_url="https://feeds.marketwatch.com/marketwatch/topstories/"
        )
        
    def fetch_news_articles(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch MarketWatch articles with financial focus
        """
        # Get base articles from RSS
        articles = self._fetch_from_rss(symbol, limit)
        
        # MarketWatch articles are already finance-focused, so minimal additional filtering
        logger.info(f"Fetched {len(articles)} MarketWatch articles for {symbol}")
        return articles[:limit]
        
    def _is_relevant_to_symbol(self, article: Dict[str, Any], symbol: str) -> bool:
        """Enhanced symbol relevance check for MarketWatch"""
        # Use base class logic first
        if super()._is_relevant_to_symbol(article, symbol):
            return True
            
        # Additional MarketWatch-specific logic
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        # MarketWatch often uses company names without tickers
        company_mappings = {
            'AAPL': ['apple inc', 'apple computer'],
            'TSLA': ['tesla motors', 'tesla inc'],
            'MSFT': ['microsoft corp', 'microsoft corporation'],
            'GOOGL': ['alphabet inc', 'google parent'],
            'GOOG': ['alphabet inc', 'google parent'],
            'AMZN': ['amazon.com', 'amazon inc'],
            'META': ['meta platforms', 'facebook inc'],
            'NVDA': ['nvidia corp', 'nvidia corporation'],
            'JPM': ['jpmorgan chase', 'jp morgan'],
            'JNJ': ['johnson & johnson', 'johnson and johnson'],
            'V': ['visa inc', 'visa corp'],
            'PG': ['procter & gamble', 'procter and gamble'],
            'UNH': ['unitedhealth group', 'united health'],
            'HD': ['home depot'],
            'MA': ['mastercard inc', 'mastercard incorporated'],
            'BAC': ['bank of america', 'bofa'],
            'PFE': ['pfizer inc'],
            'DIS': ['walt disney', 'disney company'],
            'ADBE': ['adobe inc', 'adobe systems'],
            'NFLX': ['netflix inc'],
            'KO': ['coca-cola', 'coca cola'],
            'XOM': ['exxon mobil', 'exxonmobil'],
            'ABT': ['abbott laboratories'],
            'CRM': ['salesforce.com', 'salesforce inc'],
            'TMO': ['thermo fisher scientific'],
            'COST': ['costco wholesale'],
            'AVGO': ['broadcom inc'],
            'ACN': ['accenture plc'],
            'DHR': ['danaher corp'],
            'TXN': ['texas instruments'],
            'NEE': ['nextera energy'],
            'NKE': ['nike inc'],
            'LIN': ['linde plc'],
            'WMT': ['walmart inc'],
            'BMY': ['bristol-myers squibb']
        }
        
        if symbol.upper() in company_mappings:
            company_names = company_mappings[symbol.upper()]
            if any(name in text for name in company_names):
                return True
                
        return False
