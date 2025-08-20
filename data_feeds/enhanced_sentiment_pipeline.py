"""
Enhanced Sentiment Pipeline Orchestrator
Coordinates multiple sentiment sources with parallel processing and advanced analysis
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import sys
import os

# Import enhanced adapters
from data_feeds.twitter_adapter import EnhancedTwitterAdapter
from data_feeds.news_adapters import (
    ReutersAdapter,
    MarketWatchAdapter, 
    CNNBusinessAdapter,
    FinancialTimesAdapter
)

# Import new adapters
try:
    from data_feeds.news_adapters.seeking_alpha_adapter import SeekingAlphaAdapter
    SEEKING_ALPHA_AVAILABLE = True
except ImportError:
    SeekingAlphaAdapter = None
    SEEKING_ALPHA_AVAILABLE = False
    
try:
    from data_feeds.news_adapters.benzinga_adapter import BenzingaAdapter
    BENZINGA_AVAILABLE = True
except ImportError:
    BenzingaAdapter = None
    BENZINGA_AVAILABLE = False
    
try:
    from data_feeds.news_adapters.fortune_adapter import FortuneAdapter
    FORTUNE_AVAILABLE = True
except ImportError:
    FortuneAdapter = None
    FORTUNE_AVAILABLE = False

from data_feeds.data_feed_orchestrator import SentimentData

# Import Reddit sentiment
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from data_feeds.reddit_sentiment import fetch_reddit_sentiment
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    
# Import Yahoo News (if available)
try:
    from data_feeds.yahoo_news_adapter import YahooNewsAdapter
    YAHOO_NEWS_AVAILABLE = True
except ImportError:
    YAHOO_NEWS_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedSentimentPipeline:
    """
    Orchestrates multiple sentiment sources with parallel processing and advanced analysis
    """
    
    def __init__(self, max_workers: int = 4, timeout_seconds: int = 15):
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        
        # Initialize all sentiment sources
        self.sentiment_sources = {}
        
        # Enhanced Twitter adapter
        try:
            self.sentiment_sources['twitter'] = EnhancedTwitterAdapter()
            logger.info("Enhanced Twitter adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Twitter adapter: {e}")
            
        # News adapters
        try:
            self.sentiment_sources['reuters'] = ReutersAdapter()
            logger.info("Reuters adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Reuters adapter: {e}")
            
        try:
            self.sentiment_sources['marketwatch'] = MarketWatchAdapter()
            logger.info("MarketWatch adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MarketWatch adapter: {e}")
            
        try:
            self.sentiment_sources['cnn_business'] = CNNBusinessAdapter()
            logger.info("CNN Business adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize CNN Business adapter: {e}")
            
        try:
            self.sentiment_sources['financial_times'] = FinancialTimesAdapter()
            logger.info("Financial Times adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Financial Times adapter: {e}")
            
        # New high-quality financial news adapters
        if SEEKING_ALPHA_AVAILABLE and SeekingAlphaAdapter:
            try:
                self.sentiment_sources['seeking_alpha'] = SeekingAlphaAdapter()
                logger.info("Seeking Alpha adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Seeking Alpha adapter: {e}")
                
        if BENZINGA_AVAILABLE and BenzingaAdapter:
            try:
                self.sentiment_sources['benzinga'] = BenzingaAdapter()
                logger.info("Benzinga adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Benzinga adapter: {e}")
                
        if FORTUNE_AVAILABLE and FortuneAdapter:
            try:
                self.sentiment_sources['fortune'] = FortuneAdapter()
                logger.info("Fortune adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Fortune adapter: {e}")
            
        # Yahoo News adapter (if available)
        if YAHOO_NEWS_AVAILABLE:
            try:
                self.sentiment_sources['yahoo_news'] = YahooNewsAdapter()
                logger.info("Yahoo News adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Yahoo News adapter: {e}")
        
        logger.info(f"Enhanced sentiment pipeline initialized with {len(self.sentiment_sources)} sources")
    
    def get_sentiment_analysis(self, symbol: str, include_reddit: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis from multiple sources in parallel
        
        Args:
            symbol: Stock symbol to analyze
            include_reddit: Whether to include Reddit sentiment (can be slow)
            
        Returns:
            Dictionary with aggregated sentiment analysis
        """
        start_time = time.time()
        
        # Prepare sentiment source tasks
        sentiment_tasks = []
        
        # Add adapter-based sources
        for source_name, adapter in self.sentiment_sources.items():
            sentiment_tasks.append((source_name, adapter))
        
        # Execute sentiment analysis in parallel
        sentiment_results = {}
        reddit_result = None
        
        # Run adapter-based sources in parallel
        if sentiment_tasks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all adapter tasks
                future_to_source = {
                    executor.submit(self._get_adapter_sentiment, adapter, symbol): source_name
                    for source_name, adapter in sentiment_tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_source, timeout=self.timeout_seconds):
                    source_name = future_to_source[future]
                    try:
                        result = future.result(timeout=5)  # Individual timeout
                        if result:
                            sentiment_results[source_name] = result
                            logger.info(f"Sentiment from {source_name}: {result.sentiment_score:.3f} "
                                      f"(confidence: {result.confidence:.3f})")
                    except Exception as e:
                        logger.warning(f"Failed to get sentiment from {source_name}: {e}")
                        
                # Handle any remaining futures that didn't complete
                for future in future_to_source:
                    if not future.done():
                        future.cancel()
                        source_name = future_to_source[future]
                        logger.warning(f"Timeout cancelled future for {source_name}")
        
        # Handle Reddit separately (can be slow)
        if include_reddit and REDDIT_AVAILABLE:
            try:
                reddit_data = fetch_reddit_sentiment(limit=100)
                if symbol.upper() in reddit_data:
                    reddit_info = reddit_data[symbol.upper()]
                    reddit_result = SentimentData(
                        symbol=symbol,
                        sentiment_score=reddit_info.get('sentiment_score', 0.0),
                        confidence=reddit_info.get('confidence', 0.5),
                        source="reddit_enhanced",
                        timestamp=datetime.now(),
                        sample_size=reddit_info.get('sample_size', 0),
                        raw_data=reddit_info
                    )
                    sentiment_results['reddit'] = reddit_result
                    logger.info(f"Reddit sentiment for {symbol}: {reddit_result.sentiment_score:.3f}")
            except Exception as e:
                logger.error(f"Failed to get Reddit sentiment: {e}")
        
        # Aggregate sentiment results
        aggregated_sentiment = self._aggregate_sentiment_results(symbol, sentiment_results)
        
        processing_time = time.time() - start_time
        aggregated_sentiment['processing_time_seconds'] = processing_time
        
        logger.info(f"Enhanced sentiment pipeline completed for {symbol} in {processing_time:.2f}s "
                   f"with {len(sentiment_results)} sources")
        
        return aggregated_sentiment
    
    def _get_adapter_sentiment(self, adapter: Any, symbol: str) -> Optional[SentimentData]:
        """Get sentiment from a single adapter with error handling"""
        try:
            return adapter.get_sentiment(symbol)
        except Exception as e:
            logger.error(f"Adapter sentiment error for {adapter.__class__.__name__}: {e}")
            return None
    
    def _aggregate_sentiment_results(self, symbol: str, sentiment_results: Dict[str, SentimentData]) -> Dict[str, Any]:
        """
        Aggregate sentiment results from multiple sources with confidence weighting
        """
        if not sentiment_results:
            return {
                'symbol': symbol,
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'sources_count': 0,
                'trending_direction': 'uncertain',
                'source_breakdown': {},
                'quality_score': 0.0
            }
        
        # Extract sentiment scores and confidences
        sentiments = []
        confidences = []
        source_breakdown = {}
        
        for source_name, sentiment_data in sentiment_results.items():
            sentiment_score = sentiment_data.sentiment_score
            confidence = sentiment_data.confidence
            
            sentiments.append(sentiment_score)
            confidences.append(confidence)
            
            source_breakdown[source_name] = {
                'sentiment': sentiment_score,
                'confidence': confidence,
                'sample_size': sentiment_data.sample_size,
                'source': sentiment_data.source,
                'analysis_method': sentiment_data.raw_data.get('analysis_method', 'unknown') if sentiment_data.raw_data else 'unknown'
            }
        
        # Calculate confidence-weighted overall sentiment
        if sum(confidences) > 0:
            overall_sentiment = sum(s * c for s, c in zip(sentiments, confidences)) / sum(confidences)
            average_confidence = sum(confidences) / len(confidences)
        else:
            overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            average_confidence = 0.5
        
        # Determine trending direction
        if overall_sentiment > 0.2 and average_confidence > 0.6:
            trending_direction = "bullish"
        elif overall_sentiment < -0.2 and average_confidence > 0.6:
            trending_direction = "bearish"
        elif abs(overall_sentiment) < 0.1:
            trending_direction = "neutral"
        else:
            trending_direction = "uncertain"
        
        # Calculate quality score based on source diversity and confidence
        source_diversity_bonus = min(20, len(sentiment_results) * 3)  # Max 20 points for source diversity
        confidence_score = average_confidence * 60  # Max 60 points for confidence
        sample_size_bonus = min(20, sum(sd.sample_size or 0 for sd in sentiment_results.values()) / 10)  # Max 20 points for sample size
        
        quality_score = source_diversity_bonus + confidence_score + sample_size_bonus
        
        # Count sentiment directions
        bullish_sources = sum(1 for s in sentiments if s > 0.1)
        bearish_sources = sum(1 for s in sentiments if s < -0.1)
        neutral_sources = len(sentiments) - bullish_sources - bearish_sources
        
        return {
            'symbol': symbol,
            'overall_sentiment': overall_sentiment,
            'confidence': average_confidence,
            'sources_count': len(sentiment_results),
            'trending_direction': trending_direction,
            'quality_score': quality_score,
            'source_breakdown': source_breakdown,
            'sentiment_distribution': {
                'bullish_sources': bullish_sources,
                'bearish_sources': bearish_sources,
                'neutral_sources': neutral_sources
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all sentiment sources"""
        health_status = {
            'pipeline_status': 'operational',
            'sources_count': len(self.sentiment_sources),
            'reddit_available': REDDIT_AVAILABLE,
            'yahoo_news_available': YAHOO_NEWS_AVAILABLE,
            'max_workers': self.max_workers,
            'timeout_seconds': self.timeout_seconds,
            'sources': {}
        }
        
        # Get health status from each source
        for source_name, adapter in self.sentiment_sources.items():
            try:
                if hasattr(adapter, 'get_health_status'):
                    health_status['sources'][source_name] = adapter.get_health_status()
                else:
                    health_status['sources'][source_name] = {'status': 'unknown'}
            except Exception as e:
                health_status['sources'][source_name] = {'status': 'error', 'error': str(e)}
        
        return health_status

# Global pipeline instance
_sentiment_pipeline = None

def get_enhanced_sentiment_pipeline() -> EnhancedSentimentPipeline:
    """Get global enhanced sentiment pipeline instance"""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = EnhancedSentimentPipeline()
    return _sentiment_pipeline

def get_enhanced_sentiment(symbol: str, include_reddit: bool = True) -> Dict[str, Any]:
    """Get enhanced sentiment analysis for a symbol"""
    return get_enhanced_sentiment_pipeline().get_sentiment_analysis(symbol, include_reddit)
