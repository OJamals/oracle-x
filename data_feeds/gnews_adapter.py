#!/usr/bin/env python3
"""
GNews Adapter for Oracle-X Integration
Professional implementation following Oracle-X patterns and best practices
"""

import time
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Async I/O utilities import with fallback
AsyncHTTPClient = None
ASYNC_IO_AVAILABLE = False
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.async_io_utils import AsyncHTTPClient
    ASYNC_IO_AVAILABLE = True
except ImportError:
    pass

# Install gnews if needed
try:
    from gnews import GNews
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "gnews"], check=True)
    from gnews import GNews

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Oracle-X compatible SentimentData structure
@dataclass
class SentimentData:
    """SentimentData compatible with Oracle-X DataFeedOrchestrator"""
    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float      # 0 to 1
    source: str
    timestamp: datetime
    sample_size: Optional[int] = None
    raw_data: Optional[Dict] = None

@dataclass
class GNewsConfig:
    """Configuration for GNews adapter"""
    language: str = 'en'
    country: str = 'US'
    period: str = '24h'  # 1h, 7d, 1M, etc.
    max_results: int = 50
    timeout: int = 10
    cache_ttl: int = 1800  # 30 minutes
    retry_attempts: int = 3
    retry_delay: float = 1.0
    quality_threshold: float = 0.5
    relevance_threshold: float = 0.3

class GNewsAdapter:
    """
    GNews adapter for Oracle-X integration
    Provides high-quality financial news sentiment analysis
    """
    
    def __init__(self, config: Optional[GNewsConfig] = None):
        self.config = config or GNewsConfig()
        self.gnews = GNews(
            language=self.config.language,
            country=self.config.country,
            period=self.config.period,
            max_results=self.config.max_results
        )
        self.analyzer = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Cache for articles to avoid redundant API calls
        self._cache = {}
        self._cache_timestamps = {}
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[key]
        return (datetime.now() - cache_time).total_seconds() < self.config.cache_ttl
    
    def _get_cached_or_fetch(self, key: str, fetch_func) -> List[Dict]:
        """Get data from cache or fetch new data"""
        if self._is_cache_valid(key):
            self.logger.debug(f"Using cached data for {key}")
            return self._cache[key]
        
        try:
            data = fetch_func()
            self._cache[key] = data
            self._cache_timestamps[key] = datetime.now()
            self.logger.debug(f"Fetched and cached {len(data)} articles for {key}")
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data for {key}: {e}")
            # Return cached data if available, even if stale
            return self._cache.get(key, [])
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic"""
        for attempt in range(self.config.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise e
                
                wait_time = self.config.retry_delay * (2 ** attempt)
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                time.sleep(wait_time)
    
    def _search_symbol_news(self, symbol: str) -> List[Dict]:
        """Search for symbol-specific news articles"""
        search_terms = [
            symbol,
            f"${symbol}",
            f"{symbol} stock",
            f"{symbol} earnings",
            f"{symbol} shares"
        ]
        
        all_articles = []
        seen_titles = set()
        
        for term in search_terms:
            try:
                articles = self._retry_with_backoff(self.gnews.get_news, term)
                if articles:
                    for article in articles:
                        title = article.get('title', '').strip()
                        if title and title not in seen_titles:
                            article['search_term'] = term
                            article['relevance_score'] = self._calculate_relevance(article, symbol)
                            all_articles.append(article)
                            seen_titles.add(title)
                
                # Brief delay between searches to be respectful
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error searching for {term}: {e}")
                continue
        
        # Sort by relevance and return top articles
        all_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return all_articles[:self.config.max_results]
    
    def _get_topic_news(self, topic: str) -> List[Dict]:
        """Get news articles by topic"""
        try:
            articles = self._retry_with_backoff(self.gnews.get_news_by_topic, topic)
            if articles:
                for article in articles:
                    article['topic'] = topic
                    article['relevance_score'] = 0.5  # Default topic relevance
                return articles[:self.config.max_results]
        except Exception as e:
            self.logger.error(f"Error getting {topic} news: {e}")
        
        return []
    
    def _calculate_relevance(self, article: Dict, symbol: str) -> float:
        """Calculate relevance score for an article"""
        title = article.get('title', '').lower()
        content = article.get('snippet', '').lower()
        
        # Symbol mentions
        symbol_lower = symbol.lower()
        relevance_score = 0.0
        
        # Direct symbol mentions (high weight)
        if symbol_lower in title:
            relevance_score += 0.5
        if f"${symbol_lower}" in title:
            relevance_score += 0.6
        if symbol_lower in content:
            relevance_score += 0.3
        
        # Financial keywords (medium weight)
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'stock', 'shares', 'trading',
            'analyst', 'upgrade', 'downgrade', 'price target', 'investment',
            'market', 'financial', 'quarterly', 'guidance', 'forecast'
        ]
        
        for keyword in financial_keywords:
            if keyword in title:
                relevance_score += 0.2
            elif keyword in content:
                relevance_score += 0.1
        
        # News source quality (bonus)
        source = article.get('publisher', {}).get('title', '').lower()
        quality_sources = [
            'reuters', 'bloomberg', 'wsj', 'financial times', 'cnbc',
            'marketwatch', 'yahoo finance', 'seeking alpha', 'barrons'
        ]
        
        for quality_source in quality_sources:
            if quality_source in source:
                relevance_score += 0.2
                break
        
        return min(1.0, relevance_score)
    
    def _analyze_sentiment_batch(self, articles: List[Dict]) -> Tuple[float, float, List[float]]:
        """Analyze sentiment for a batch of articles"""
        if not articles:
            return 0.0, 0.0, []
        
        sentiment_scores = []
        relevant_articles = []
        
        for article in articles:
            title = article.get('title', '')
            snippet = article.get('snippet', '')
            
            # Combine title and snippet for analysis
            text = f"{title}. {snippet}".strip()
            if not text:
                continue
            
            # Check relevance threshold
            relevance = article.get('relevance_score', 0.5)
            if relevance < self.config.relevance_threshold:
                continue
            
            # Analyze sentiment
            sentiment = self.analyzer.polarity_scores(text)
            compound_score = sentiment['compound']
            
            # Weight by relevance
            weighted_score = compound_score * relevance
            sentiment_scores.append(weighted_score)
            relevant_articles.append(article)
        
        if not sentiment_scores:
            return 0.0, 0.0, []
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Calculate confidence based on sample size and score variance
        variance = sum((score - avg_sentiment) ** 2 for score in sentiment_scores) / len(sentiment_scores)
        confidence = min(0.95, len(sentiment_scores) / 20.0) * (1.0 - min(0.5, variance))
        
        return avg_sentiment, confidence, sentiment_scores
    
    def get_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """
        Get sentiment data for a specific symbol
        Main interface for Oracle-X integration
        """
        try:
            start_time = time.time()
            
            # Get symbol-specific news
            cache_key_symbol = f"symbol_{symbol}_{self.config.period}"
            symbol_articles = self._get_cached_or_fetch(
                cache_key_symbol,
                lambda: self._search_symbol_news(symbol)
            )
            
            # Get general financial news for context
            cache_key_business = f"topic_BUSINESS_{self.config.period}"
            business_articles = self._get_cached_or_fetch(
                cache_key_business,
                lambda: self._get_topic_news('BUSINESS')
            )
            
            # Combine and prioritize symbol-specific content
            all_articles = symbol_articles + business_articles[:10]  # Limit general news
            
            if not all_articles:
                self.logger.warning(f"No articles found for {symbol}")
                return SentimentData(
                    symbol=symbol,
                    sentiment_score=0.0,
                    confidence=0.0,
                    source='gnews',
                    timestamp=datetime.now(),
                    sample_size=0,
                    raw_data={'error': 'No articles found', 'processing_time': time.time() - start_time}
                )
            
            # Analyze sentiment
            avg_sentiment, confidence, scores = self._analyze_sentiment_batch(all_articles)
            
            # Prepare sample headlines
            sample_headlines = [
                article.get('title', '')[:100] + ('...' if len(article.get('title', '')) > 100 else '')
                for article in all_articles[:5]
                if article.get('title')
            ]
            
            # Calculate quality score
            avg_relevance = sum(article.get('relevance_score', 0.5) for article in all_articles) / len(all_articles)
            quality_score = (confidence * 0.6 + avg_relevance * 0.4) * 100
            
            processing_time = time.time() - start_time
            
            return SentimentData(
                symbol=symbol,
                sentiment_score=avg_sentiment,
                confidence=confidence,
                source='gnews',
                timestamp=datetime.now(),
                sample_size=len(all_articles),
                raw_data={
                    'sample_texts': sample_headlines,
                    'quality_score': quality_score,
                    'processing_time': processing_time,
                    'symbol_articles': len(symbol_articles),
                    'business_articles': len(business_articles),
                    'avg_relevance': avg_relevance,
                    'source': 'gnews'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment for {symbol}: {e}")
            return SentimentData(
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                source='gnews',
                timestamp=datetime.now(),
                sample_size=0,
                raw_data={'error': str(e), 'source': 'gnews'}
            )
    
    def get_news_headlines(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get recent news headlines for a symbol"""
        try:
            cache_key = f"headlines_{symbol}_{limit}_{self.config.period}"
            articles = self._get_cached_or_fetch(
                cache_key,
                lambda: self._search_symbol_news(symbol)
            )
            
            # Filter and format headlines
            headlines = []
            for article in articles[:limit]:
                if article.get('relevance_score', 0) >= self.config.relevance_threshold:
                    headlines.append({
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'published': article.get('published date', ''),
                        'source': article.get('publisher', {}).get('title', ''),
                        'relevance': article.get('relevance_score', 0),
                        'snippet': article.get('snippet', '')
                    })
            
            return headlines
            
        except Exception as e:
            self.logger.error(f"Error getting headlines for {symbol}: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on GNews service"""
        try:
            # Try a simple search
            test_articles = self.gnews.get_news('test')
            return {
                'status': 'healthy',
                'service': 'gnews',
                'test_results': len(test_articles) if test_articles else 0,
                'config': {
                    'language': self.config.language,
                    'country': self.config.country,
                    'period': self.config.period,
                    'max_results': self.config.max_results
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'service': 'gnews',
                'error': str(e)
            }

# Integration example for DataFeedOrchestrator
def integrate_gnews_with_orchestrator():
    """
    Example integration with Oracle-X DataFeedOrchestrator
    """
    config = GNewsConfig(
        max_results=30,
        cache_ttl=1800,  # 30 minutes
        quality_threshold=0.4,
        relevance_threshold=0.3
    )
    
    gnews_adapter = GNewsAdapter(config)
    
    # Test the adapter
    symbols = ['AAPL', 'TSLA', 'NVDA']
    
    for symbol in symbols:
        print(f"\n--- Testing GNews for {symbol} ---")
        sentiment = gnews_adapter.get_sentiment(symbol)
        
        if sentiment:
            print(f"Sentiment: {sentiment.sentiment_score:.3f}")
            print(f"Confidence: {sentiment.confidence:.3f}")
            print(f"Articles: {sentiment.sample_size}")
            
            if sentiment.raw_data:
                print(f"Quality Score: {sentiment.raw_data.get('quality_score', 0):.1f}")
                print(f"Processing Time: {sentiment.raw_data.get('processing_time', 0):.2f}s")
                
                headlines = sentiment.raw_data.get('sample_texts', [])
                if headlines:
                    print(f"Sample: {headlines[0]}")
        else:
            print("No sentiment data available")
        
        # Brief delay between symbols
        time.sleep(1)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run integration example
    integrate_gnews_with_orchestrator()
