"""
Twitter Adapter for Data Feed Orchestrator
Integrates TwitterSentimentFeed with the unified data architecture
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .base_adapters import BaseFeedAdapter, SentimentData
from .twitter_feed import TwitterSentimentFeed
from .cache import SmartCache
from .rate_limiter import RateLimiter 
from .performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

class TwitterAdapter(BaseFeedAdapter):
    """Twitter sentiment data adapter"""
    
    def __init__(self, cache: SmartCache, rate_limiter: RateLimiter, 
                 performance_tracker: PerformanceTracker):
        super().__init__(cache, rate_limiter, performance_tracker)
        self.feed = TwitterSentimentFeed()
        self.source_name = "twitter_advanced"
        
    def get_sentiment(self, symbol: str, limit: int = 50) -> Optional[SentimentData]:
        """
        Get Twitter sentiment data for a symbol
        
        Args:
            symbol: Stock symbol to analyze
            limit: Maximum number of tweets to fetch
            
        Returns:
            SentimentData object with Twitter sentiment analysis
        """
        cache_key = f"twitter_sentiment_{symbol}_{limit}"
        
        # Check cache first (valid for 30 minutes)
        cached_data = self.cache.get(cache_key, max_age_minutes=30)
        if cached_data:
            return cached_data
            
        # Check rate limits
        if not self.rate_limiter.can_make_request("twitter", symbol):
            logger.warning(f"Rate limit exceeded for Twitter sentiment: {symbol}")
            return None
            
        try:
            with self.performance_tracker.track_operation("twitter_sentiment"):
                # Fetch tweets using TwitterSentimentFeed
                tweets = self.feed.fetch(symbol, limit=limit)
                
                if not tweets:
                    logger.warning(f"No Twitter data found for {symbol}")
                    return None
                    
                # Extract sentiment data
                sentiments = []
                texts = []
                for tweet in tweets:
                    if 'sentiment' in tweet and 'text' in tweet:
                        # Twitter feed already provides VADER sentiment
                        sentiment_score = tweet['sentiment'].get('compound', 0.0)
                        sentiments.append(sentiment_score)
                        texts.append(tweet['text'])
                
                if not sentiments:
                    logger.warning(f"No sentiment data extracted from Twitter for {symbol}")
                    return None
                
                # Calculate aggregate sentiment
                overall_sentiment = sum(sentiments) / len(sentiments)
                
                # Calculate confidence based on sample size and variance
                sentiment_variance = sum((s - overall_sentiment) ** 2 for s in sentiments) / len(sentiments)
                confidence = min(0.95, max(0.1, 1.0 - sentiment_variance))
                
                # Classify sentiment direction
                if overall_sentiment > 0.1:
                    direction = "positive"
                elif overall_sentiment < -0.1:
                    direction = "negative"
                else:
                    direction = "neutral"
                
                # Create sentiment data object
                sentiment_data = SentimentData(
                    symbol=symbol,
                    overall_sentiment=overall_sentiment,
                    confidence=confidence,
                    sample_size=len(sentiments),
                    timestamp=datetime.now(),
                    source="twitter_advanced",
                    raw_data={
                        'tweets': tweets,
                        'sample_texts': texts,
                        'individual_sentiments': sentiments,
                        'direction': direction,
                        'variance': sentiment_variance
                    }
                )
                
                # Cache the result
                self.cache.set(cache_key, sentiment_data)
                
                logger.info(f"Twitter sentiment for {symbol}: {overall_sentiment:.3f} "
                          f"(confidence: {confidence:.3f}, samples: {len(sentiments)})")
                
                return sentiment_data
                
        except Exception as e:
            logger.error(f"Failed to fetch Twitter sentiment for {symbol}: {e}")
            return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get Twitter adapter health status"""
        return {
            'source': 'twitter_advanced',
            'status': 'operational',
            'last_request': self.rate_limiter.get_last_request_time("twitter"),
            'cache_size': len(self.cache._cache),
            'performance': self.performance_tracker.get_operation_stats("twitter_sentiment")
        }
