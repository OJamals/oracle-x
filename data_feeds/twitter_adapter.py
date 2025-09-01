"""
Enhanced Twitter Adapter for Data Feed Orchestrator
Integrates TwitterSentimentFeed with advanced multi-model sentiment analysis
Uses FinBERT + VADER + Financial Lexicon ensemble for superior accuracy
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sys
import os

# Import from data_feed_orchestrator for proper base classes
from data_feeds.data_feed_orchestrator import SentimentData
from .twitter_feed import TwitterSentimentFeed

# Import advanced sentiment analysis engine
try:
    from data_feeds.advanced_sentiment import get_sentiment_engine, analyze_symbol_sentiment, SentimentSummary
    ADVANCED_SENTIMENT_AVAILABLE = True
    _get_sentiment_engine = get_sentiment_engine
    _analyze_symbol_sentiment = analyze_symbol_sentiment
except ImportError as e:
    logging.warning(f"Advanced sentiment analysis not available: {e}")
    ADVANCED_SENTIMENT_AVAILABLE = False
    _get_sentiment_engine = None
    _analyze_symbol_sentiment = None

logger = logging.getLogger(__name__)

class EnhancedTwitterAdapter:
    """Enhanced Twitter sentiment data adapter with advanced multi-model analysis"""
    
    def __init__(self):
        self.feed = TwitterSentimentFeed()
        self.source_name = "twitter_enhanced"
        
        # Initialize advanced sentiment engine if available
        if ADVANCED_SENTIMENT_AVAILABLE and _get_sentiment_engine:
            try:
                self.sentiment_engine = _get_sentiment_engine()
                logger.info("Advanced sentiment engine initialized for Twitter adapter")
            except Exception as e:
                logger.error(f"Failed to initialize advanced sentiment engine: {e}")
                self.sentiment_engine = None
        else:
            self.sentiment_engine = None
            
    def get_sentiment(self, symbol: str, limit: int = 50) -> Optional[SentimentData]:
        """
        Get enhanced Twitter sentiment data for a symbol using advanced analysis
        
        Args:
            symbol: Stock symbol to analyze
            limit: Maximum number of tweets to fetch
            
        Returns:
            SentimentData object with advanced Twitter sentiment analysis
        """
        try:
            # Fetch tweets using TwitterSentimentFeed
            tweets = self.feed.fetch(symbol, limit=limit)
            
            if not tweets:
                logger.warning(f"No Twitter data found for {symbol}")
                return None
                
            # Extract text content for advanced analysis
            texts = []
            tweet_metadata = []
            
            for tweet in tweets:
                if 'text' in tweet:
                    texts.append(tweet['text'])
                    tweet_metadata.append({
                        'timestamp': tweet.get('timestamp', datetime.now()),
                        'user': tweet.get('user', 'unknown'),
                        'retweets': tweet.get('retweets', 0),
                        'likes': tweet.get('likes', 0)
                    })
            
            if not texts:
                logger.warning(f"No text content extracted from Twitter for {symbol}")
                return None
            
            # Use advanced sentiment analysis if available
            if self.sentiment_engine and ADVANCED_SENTIMENT_AVAILABLE and _analyze_symbol_sentiment:
                try:
                    # Get symbol sentiment summary using advanced engine
                    sentiment_summary = _analyze_symbol_sentiment(
                        symbol=symbol,
                        texts=texts,
                        sources=["twitter"] * len(texts)
                    )
                    
                    # Create enhanced sentiment data object
                    sentiment_data = SentimentData(
                        symbol=symbol,
                        sentiment_score=sentiment_summary.overall_sentiment,
                        confidence=sentiment_summary.confidence,
                        source="twitter_enhanced",
                        timestamp=datetime.now(),
                        sample_size=sentiment_summary.sample_size,
                        raw_data={
                            'tweets': tweets,
                            'sample_texts': texts,
                            'tweet_metadata': tweet_metadata,
                            'bullish_mentions': sentiment_summary.bullish_mentions,
                            'bearish_mentions': sentiment_summary.bearish_mentions,
                            'neutral_mentions': sentiment_summary.neutral_mentions,
                            'trending_direction': sentiment_summary.trending_direction,
                            'quality_score': sentiment_summary.quality_score,
                            'analysis_method': 'advanced_multi_model'
                        }
                    )
                    
                    logger.info(f"Enhanced Twitter sentiment for {symbol}: {sentiment_summary.overall_sentiment:.3f} "
                              f"(confidence: {sentiment_summary.confidence:.3f}, samples: {sentiment_summary.sample_size}, "
                              f"trend: {sentiment_summary.trending_direction})")
                    
                    return sentiment_data
                    
                except Exception as e:
                    logger.error(f"Advanced sentiment analysis failed for {symbol}, falling back to basic: {e}")
                    # Fall through to basic analysis
            
            # Fallback to basic sentiment analysis (original logic)
            return self._basic_sentiment_analysis(symbol, tweets, texts, tweet_metadata)
                
        except Exception as e:
            logger.error(f"Failed to fetch Twitter sentiment for {symbol}: {e}")
            return None
    
    def _basic_sentiment_analysis(self, symbol: str, tweets: List[Dict], texts: List[str], 
                                 tweet_metadata: List[Dict]) -> Optional[SentimentData]:
        """Fallback basic sentiment analysis using original VADER logic"""
        try:
            # Extract basic VADER sentiment from tweets
            sentiments = []
            for tweet in tweets:
                if 'sentiment' in tweet:
                    sentiment_score = tweet['sentiment'].get('compound', 0.0)
                    sentiments.append(sentiment_score)
            
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
            
            # Create basic sentiment data object
            sentiment_data = SentimentData(
                symbol=symbol,
                sentiment_score=overall_sentiment,
                confidence=confidence,
                source="twitter_basic",
                timestamp=datetime.now(),
                sample_size=len(sentiments),
                raw_data={
                    'tweets': tweets,
                    'sample_texts': texts,
                    'tweet_metadata': tweet_metadata,
                    'individual_sentiments': sentiments,
                    'direction': direction,
                    'variance': sentiment_variance,
                    'analysis_method': 'basic_vader'
                }
            )
            
            logger.info(f"Basic Twitter sentiment for {symbol}: {overall_sentiment:.3f} "
                      f"(confidence: {confidence:.3f}, samples: {len(sentiments)})")
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Basic sentiment analysis failed for {symbol}: {e}")
            return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get enhanced Twitter adapter health status"""
        return {
            'source': 'twitter_enhanced',
            'status': 'operational',
            'advanced_sentiment_available': ADVANCED_SENTIMENT_AVAILABLE,
            'sentiment_engine_loaded': self.sentiment_engine is not None,
            'twitter_feed_status': 'available'
        }

# Maintain backward compatibility with old class name
class TwitterAdapter(EnhancedTwitterAdapter):
    """Backward compatibility alias for EnhancedTwitterAdapter"""
    pass
