"""
Enhanced Sentiment Pipeline Orchestrator - Optimized Version
Coordinates multiple sentiment sources with parallel processing and advanced analysis
Performance optimized to reduce processing time from 6+ seconds to <2 seconds
"""

import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import sys
import os
import threading
import queue
import random
import statistics

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

from data_feeds.advanced_sentiment import AdvancedSentimentEngine, analyze_symbol_sentiment

logger = logging.getLogger(__name__)

class OptimizedSentimentPipeline:
    """
    High-performance sentiment pipeline with intelligent parallel processing,
    rate limiting, and optimized resource management
    """

    def __init__(self, max_workers: int = 8, timeout_seconds: int = 10, batch_size: int = 3):
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size  # Process sources in batches to avoid overwhelming APIs

        # Initialize all sentiment sources
        self.sentiment_sources = {}

        # Enhanced Twitter adapter
        try:
            self.sentiment_sources['twitter'] = EnhancedTwitterAdapter()
            logger.info("Enhanced Twitter adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Twitter adapter: {e}")

        # News adapters - organized by priority
        self._init_news_adapters()

        # Yahoo News adapter (if available)
        if YAHOO_NEWS_AVAILABLE:
            try:
                self.sentiment_sources['yahoo_news'] = YahooNewsAdapter()
                logger.info("Yahoo News adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Yahoo News adapter: {e}")

        logger.info(f"Optimized sentiment pipeline initialized with {len(self.sentiment_sources)} sources")

        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Simple in-memory cache for sentiment results
        self._cache = {}
        self._cache_ttl = 180  # 3 minutes cache TTL for faster repeated requests

    def _init_news_adapters(self):
        """Initialize news adapters with priority ordering"""
        adapters_config = [
            ('reuters', ReutersAdapter, "Reuters adapter initialized"),
            ('marketwatch', MarketWatchAdapter, "MarketWatch adapter initialized"),
            ('cnn_business', CNNBusinessAdapter, "CNN Business adapter initialized"),
            ('financial_times', FinancialTimesAdapter, "Financial Times adapter initialized"),
        ]

        # Add premium adapters if available
        if SEEKING_ALPHA_AVAILABLE and SeekingAlphaAdapter:
            adapters_config.append(('seeking_alpha', SeekingAlphaAdapter, "Seeking Alpha adapter initialized"))

        if BENZINGA_AVAILABLE and BenzingaAdapter:
            adapters_config.append(('benzinga', BenzingaAdapter, "Benzinga adapter initialized"))

        if FORTUNE_AVAILABLE and FortuneAdapter:
            adapters_config.append(('fortune', FortuneAdapter, "Fortune adapter initialized"))

        # Initialize adapters with error handling
        for source_name, adapter_class, success_msg in adapters_config:
            try:
                self.sentiment_sources[source_name] = adapter_class()
                logger.info(success_msg)
            except Exception as e:
                logger.warning(f"Failed to initialize {source_name} adapter: {e}")

    def get_sentiment_analysis(self, symbol: str, include_reddit: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis with optimized parallel processing

        Args:
            symbol: Stock symbol to analyze
            include_reddit: Whether to include Reddit sentiment (cached for performance)

        Returns:
            Dictionary with aggregated sentiment analysis
        """
        start_time = time.time()
        cache_key = f"sentiment_{symbol}_{include_reddit}"

        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.performance_stats['cache_hits'] += 1
            cached_result['cached'] = True
            return cached_result

        self.performance_stats['cache_misses'] += 1

        # Prepare sentiment source tasks
        sentiment_tasks = []
        for source_name, adapter in self.sentiment_sources.items():
            sentiment_tasks.append((source_name, adapter))

        # Execute sentiment analysis in optimized parallel batches
        sentiment_results = {}

        if sentiment_tasks:
            sentiment_results = self._execute_parallel_sentiment_analysis(sentiment_tasks, symbol)

        # Handle Reddit sentiment (cached and optimized)
        reddit_result = None
        if include_reddit and REDDIT_AVAILABLE:
            reddit_result = self._get_reddit_sentiment(symbol)
            if reddit_result:
                sentiment_results['reddit'] = reddit_result

        # Aggregate sentiment results
        aggregated_sentiment = self._aggregate_sentiment_results(symbol, sentiment_results)

        processing_time = time.time() - start_time
        aggregated_sentiment['processing_time_seconds'] = processing_time
        aggregated_sentiment['cached'] = False

        # Cache the result
        self._cache_result(cache_key, aggregated_sentiment)

        # Update performance stats
        self._update_performance_stats(processing_time, len(sentiment_results))

        logger.info(f"Optimized sentiment pipeline completed for {symbol} in {processing_time:.2f}s "
                   f"with {len(sentiment_results)} sources")

        return aggregated_sentiment

    def _execute_parallel_sentiment_analysis(self, sentiment_tasks: List[Tuple[str, Any]], symbol: str) -> Dict[str, SentimentData]:
        """Execute sentiment analysis with intelligent batching and rate limiting"""
        sentiment_results = {}
        completed_count = 0
        total_tasks = len(sentiment_tasks)

        # Process in batches to avoid overwhelming APIs
        for i in range(0, total_tasks, self.batch_size):
            batch = sentiment_tasks[i:i + self.batch_size]
            batch_results = self._process_sentiment_batch(batch, symbol)

            for source_name, result in batch_results.items():
                if result:
                    sentiment_results[source_name] = result
                    completed_count += 1
                    logger.info(f"Sentiment from {source_name}: {result.sentiment_score:.3f} "
                               f"(confidence: {result.confidence:.3f})")

            # Small delay between batches to respect API rate limits
            if i + self.batch_size < total_tasks:
                time.sleep(0.1)

        return sentiment_results

    def _process_sentiment_batch(self, batch: List[Tuple[str, Any]], symbol: str) -> Dict[str, Optional[SentimentData]]:
        """Process a single batch of sentiment sources with timeout handling"""
        results = {}

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_source = {
                executor.submit(self._get_adapter_sentiment_safe, adapter, symbol): source_name
                for source_name, adapter in batch
            }

            # Use timeout for the entire batch
            batch_timeout = min(self.timeout_seconds / max(1, len(batch)), 3.0)

            try:
                for future in as_completed(future_to_source, timeout=batch_timeout):
                    source_name = future_to_source[future]
                    try:
                        result = future.result(timeout=2.0)
                        results[source_name] = result
                    except Exception as e:
                        logger.warning(f"Failed to get sentiment from {source_name}: {e}")
                        results[source_name] = None
                        self.performance_stats['failed_requests'] += 1

            except TimeoutError:
                logger.warning(f"Batch processing timeout for {len(batch)} sources")
                # Cancel remaining futures
                for future in future_to_source:
                    if not future.done():
                        future.cancel()

        return results

    def _get_adapter_sentiment_safe(self, adapter: Any, symbol: str) -> Optional[SentimentData]:
        """Get sentiment from adapter with enhanced error handling and timing"""
        start_time = time.time()
        try:
            result = adapter.get_sentiment(symbol)
            response_time = time.time() - start_time
            self.performance_stats['successful_requests'] += 1
            self.performance_stats['total_requests'] += 1
            return result
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Adapter sentiment error for {adapter.__class__.__name__}: {e}")
            self.performance_stats['failed_requests'] += 1
            self.performance_stats['total_requests'] += 1
            return None

    def _get_reddit_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """Get Reddit sentiment with caching optimization"""
        try:
            # Use cached Reddit data if available (Reddit module has its own caching)
            reddit_data = fetch_reddit_sentiment(limit=50)  # Reduced limit for performance

            if symbol.upper() in reddit_data:
                reddit_info = reddit_data[symbol.upper()]
                return SentimentData(
                    symbol=symbol,
                    sentiment_score=reddit_info.get('sentiment_score', 0.0),
                    confidence=reddit_info.get('confidence', 0.5),
                    source="reddit_enhanced",
                    timestamp=datetime.now(),
                    sample_size=reddit_info.get('sample_size', 0),
                    raw_data=reddit_info
                )
        except Exception as e:
            logger.error(f"Failed to get Reddit sentiment: {e}")

        return None

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if result is cached and not expired"""
        if cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if time.time() - cached_item['timestamp'] < self._cache_ttl:
                return cached_item['data']
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, data: Dict[str, Any]):
        """Cache sentiment analysis result"""
        self._cache[cache_key] = {
            'data': data.copy(),
            'timestamp': time.time()
        }

        # Clean up old cache entries (keep cache size manageable)
        if len(self._cache) > 100:
            current_time = time.time()
            expired_keys = [
                key for key, item in self._cache.items()
                if current_time - item['timestamp'] > self._cache_ttl
            ]
            for key in expired_keys:
                del self._cache[key]

    def _update_performance_stats(self, processing_time: float, sources_count: int):
        """Update performance statistics"""
        # Update average response time
        if self.performance_stats['total_requests'] > 0:
            current_avg = self.performance_stats['average_response_time']
            total_requests = self.performance_stats['total_requests']
            self.performance_stats['average_response_time'] = (
                (current_avg * (total_requests - 1)) + processing_time
            ) / total_requests

    def _aggregate_sentiment_results(self, symbol: str, sentiment_results: Dict[str, SentimentData]) -> Dict[str, Any]:
        """Aggregate sentiment results with enhanced confidence weighting and quality metrics"""
        if not sentiment_results:
            return self._create_empty_sentiment_response(symbol)

        # Extract sentiment scores and confidences
        sentiments = []
        confidences = []
        source_breakdown = {}
        analysis_methods = set()

        for source_name, sentiment_data in sentiment_results.items():
            if sentiment_data is None:
                continue

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
            
            # Track analysis methods for quality assessment
            if sentiment_data.raw_data and 'analysis_method' in sentiment_data.raw_data:
                analysis_methods.add(sentiment_data.raw_data['analysis_method'])

        # Calculate confidence-weighted overall sentiment with outlier rejection
        if sum(confidences) > 0:
            # Apply outlier rejection - remove scores that deviate significantly from weighted mean
            weighted_mean = sum(s * c for s, c in zip(sentiments, confidences)) / sum(confidences)
            std_dev = statistics.stdev(sentiments) if len(sentiments) > 1 else 0.0
            
            filtered_sentiments = []
            filtered_confidences = []
            
            for s, c in zip(sentiments, confidences):
                if abs(s - weighted_mean) <= 2 * std_dev or std_dev == 0.0:  # Keep within 2 std dev
                    filtered_sentiments.append(s)
                    filtered_confidences.append(c)
            
            if filtered_sentiments:
                overall_sentiment = sum(s * c for s, c in zip(filtered_sentiments, filtered_confidences)) / sum(filtered_confidences)
                average_confidence = sum(filtered_confidences) / len(filtered_confidences)
            else:
                # Fallback if all filtered out
                overall_sentiment = weighted_mean
                average_confidence = sum(confidences) / len(confidences)
        else:
            overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            average_confidence = 0.5

        # Determine trending direction with enhanced thresholds
        trending_direction = self._determine_enhanced_trending_direction(overall_sentiment, average_confidence, sentiments)

        # Calculate enhanced quality score with multiple factors
        quality_score = self._calculate_enhanced_quality_score(sentiment_results, average_confidence, len(analysis_methods))

        # Count sentiment directions with confidence filtering
        sentiment_distribution = self._calculate_enhanced_sentiment_distribution(sentiments, confidences)

        return {
            'symbol': symbol,
            'overall_sentiment': overall_sentiment,
            'confidence': average_confidence,
            'sources_count': len(sentiment_results),
            'trending_direction': trending_direction,
            'quality_score': quality_score,
            'source_breakdown': source_breakdown,
            'sentiment_distribution': sentiment_distribution,
            'timestamp': datetime.now().isoformat(),
            'performance_stats': self.performance_stats.copy(),
            'analysis_methods_count': len(analysis_methods)
        }

    def _determine_enhanced_trending_direction(self, overall_sentiment: float, average_confidence: float, sentiments: List[float]) -> str:
        """Determine trending direction with enhanced confidence-based thresholds"""
        if average_confidence < 0.4:
            return "uncertain"
        
        # Calculate sentiment dispersion
        if len(sentiments) >= 2:
            dispersion = statistics.stdev(sentiments)
        else:
            dispersion = 0.0
        
        # Adjust thresholds based on confidence and dispersion
        confidence_factor = min(1.0, average_confidence * 1.5)
        dispersion_penalty = min(0.3, dispersion * 0.5)
        
        bullish_threshold = 0.15 * confidence_factor - dispersion_penalty
        bearish_threshold = -0.15 * confidence_factor + dispersion_penalty
        
        if overall_sentiment > max(0.05, bullish_threshold) and average_confidence > 0.5:
            return "bullish"
        elif overall_sentiment < min(-0.05, bearish_threshold) and average_confidence > 0.5:
            return "bearish"
        elif abs(overall_sentiment) < 0.08:
            return "neutral"
        else:
            return "uncertain"

    def _calculate_enhanced_quality_score(self, sentiment_results: Dict[str, SentimentData], average_confidence: float, analysis_methods_count: int) -> float:
        """Calculate enhanced quality score with multiple quality factors"""
        source_count = len(sentiment_results)
        
        # Base confidence score (35% weight)
        confidence_score = min(35.0, average_confidence * 35.0)
        
        # Source diversity bonus (25% weight) - more sources = higher score
        source_diversity_bonus = min(25.0, source_count * 5.0)
        
        # Analysis method diversity bonus (15% weight) - multiple analysis methods = higher quality
        method_diversity_bonus = min(15.0, analysis_methods_count * 7.5)
        
        # Sample size bonus (15% weight) - larger samples = better quality
        total_sample_size = sum(sd.sample_size or 0 for sd in sentiment_results.values())
        sample_size_bonus = min(15.0, total_sample_size / 20.0)  # Max 15 points for 300+ samples
        
        # Confidence consistency bonus (10% weight) - low dispersion = higher quality
        confidences = [sd.confidence for sd in sentiment_results.values() if sd is not None]
        if len(confidences) >= 2:
            confidence_dispersion = statistics.stdev(confidences)
            consistency_bonus = max(0.0, 10.0 * (1.0 - min(1.0, confidence_dispersion * 2.0)))
        else:
            consistency_bonus = 5.0  # Neutral bonus for single source
        
        total_score = (
            confidence_score + 
            source_diversity_bonus + 
            method_diversity_bonus + 
            sample_size_bonus + 
            consistency_bonus
        )
        
        return min(100.0, max(0.0, total_score))

    def _calculate_enhanced_sentiment_distribution(self, sentiments: List[float], confidences: List[float]) -> Dict[str, int]:
        """Calculate sentiment distribution with confidence weighting"""
        if not sentiments:
            return {'bullish_sources': 0, 'bearish_sources': 0, 'neutral_sources': 0}
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for sentiment, confidence in zip(sentiments, confidences):
            # Apply confidence threshold - only count confident sources
            if confidence < 0.4:
                continue
                
            if sentiment > 0.15:
                bullish_count += 1
            elif sentiment < -0.15:
                bearish_count += 1
            else:
                neutral_count += 1
        
        return {
            'bullish_sources': bullish_count,
            'bearish_sources': bearish_count,
            'neutral_sources': neutral_count
        }

    def _create_empty_sentiment_response(self, symbol: str) -> Dict[str, Any]:
        """Create empty sentiment response for when no data is available"""
        return {
            'symbol': symbol,
            'overall_sentiment': 0.0,
            'confidence': 0.0,
            'sources_count': 0,
            'trending_direction': 'uncertain',
            'source_breakdown': {},
            'quality_score': 0.0,
            'sentiment_distribution': {
                'bullish_sources': 0,
                'bearish_sources': 0,
                'neutral_sources': 0
            },
            'timestamp': datetime.now().isoformat(),
            'performance_stats': self.performance_stats.copy(),
            'analysis_methods_count': 0
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all sentiment sources with performance metrics"""
        health_status = {
            'pipeline_status': 'operational',
            'sources_count': len(self.sentiment_sources),
            'reddit_available': REDDIT_AVAILABLE,
            'yahoo_news_available': YAHOO_NEWS_AVAILABLE,
            'max_workers': self.max_workers,
            'timeout_seconds': self.timeout_seconds,
            'batch_size': self.batch_size,
            'cache_size': len(self._cache),
            'performance_stats': self.performance_stats.copy(),
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

    def clear_cache(self):
        """Clear sentiment analysis cache"""
        self._cache.clear()
        logger.info("Sentiment analysis cache cleared")

# Global pipeline instance
_sentiment_pipeline = None

def get_optimized_sentiment_pipeline() -> OptimizedSentimentPipeline:
    """Get global optimized sentiment pipeline instance"""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = OptimizedSentimentPipeline()
    return _sentiment_pipeline

def get_enhanced_sentiment(symbol: str, include_reddit: bool = True) -> Dict[str, Any]:
    """Get enhanced sentiment analysis for a symbol using optimized pipeline"""
    return get_optimized_sentiment_pipeline().get_sentiment_analysis(symbol, include_reddit)

# Backward compatibility
def get_enhanced_sentiment_pipeline() -> OptimizedSentimentPipeline:
    """Get enhanced sentiment pipeline instance (backward compatibility)"""
    return get_optimized_sentiment_pipeline()