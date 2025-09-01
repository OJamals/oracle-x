#!/usr/bin/env python3
"""
News Sentiment System Evaluation and GNews Integration Comparison
Comprehensive analysis of current news sources vs gnews package integration
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append('/Users/omar/Documents/Projects/oracle-x')

# Install gnews if not available
try:
    from gnews import GNews
except ImportError:
    print("Installing gnews package...")
    os.system("pip install gnews")
    from gnews import GNews

# Import Oracle-X components
try:
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the Oracle-X root directory")
    sys.exit(1)

@dataclass
class NewsSource:
    """News source metadata and results"""
    name: str
    description: str
    articles_count: int
    sentiment_score: float
    confidence: float
    sample_headlines: List[str]
    processing_time: float
    quality_score: float
    error: Optional[str] = None

@dataclass
class ComparisonResult:
    """Complete comparison results"""
    symbol: str
    timestamp: str
    current_sources: List[NewsSource]
    gnews_results: List[NewsSource]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]

class GNewsAdapter:
    """GNews adapter for Oracle-X integration"""
    
    def __init__(self):
        self.gnews = GNews(
            language='en',
            country='US',
            period='24h',  # Last 24 hours
            max_results=50
        )
        self.analyzer = SentimentIntensityAnalyzer()
    
    def get_news_by_keyword(self, symbol: str, limit: int = 30) -> NewsSource:
        """Get news using keyword search"""
        start_time = time.time()
        
        try:
            # Search for symbol-related news
            articles = self.gnews.get_news(symbol)
            
            if not articles:
                return NewsSource(
                    name="GNews-Keyword",
                    description=f"GNews keyword search for {symbol}",
                    articles_count=0,
                    sentiment_score=0.0,
                    confidence=0.0,
                    sample_headlines=[],
                    processing_time=time.time() - start_time,
                    quality_score=0.0,
                    error="No articles found"
                )
            
            # Limit results
            articles = articles[:limit]
            
            # Analyze sentiment
            headlines = [article.get('title', '') for article in articles if article.get('title')]
            scores = []
            
            for headline in headlines:
                sentiment = self.analyzer.polarity_scores(headline)
                scores.append(sentiment['compound'])
            
            avg_sentiment = sum(scores) / len(scores) if scores else 0.0
            confidence = min(0.95, len(scores) / 20.0)  # Higher confidence with more articles
            quality_score = (confidence * 0.7 + min(1.0, len(articles) / 30.0) * 0.3) * 100
            
            return NewsSource(
                name="GNews-Keyword",
                description=f"GNews keyword search for {symbol}",
                articles_count=len(articles),
                sentiment_score=avg_sentiment,
                confidence=confidence,
                sample_headlines=headlines[:5],
                processing_time=time.time() - start_time,
                quality_score=quality_score
            )
            
        except Exception as e:
            return NewsSource(
                name="GNews-Keyword",
                description=f"GNews keyword search for {symbol}",
                articles_count=0,
                sentiment_score=0.0,
                confidence=0.0,
                sample_headlines=[],
                processing_time=time.time() - start_time,
                quality_score=0.0,
                error=str(e)
            )
    
    def get_news_by_topic(self, topic: str = "BUSINESS", limit: int = 30) -> NewsSource:
        """Get news by business topic"""
        start_time = time.time()
        
        try:
            articles = self.gnews.get_news_by_topic(topic)
            
            if not articles:
                return NewsSource(
                    name=f"GNews-{topic}",
                    description=f"GNews {topic} topic news",
                    articles_count=0,
                    sentiment_score=0.0,
                    confidence=0.0,
                    sample_headlines=[],
                    processing_time=time.time() - start_time,
                    quality_score=0.0,
                    error="No articles found"
                )
            
            articles = articles[:limit]
            headlines = [article.get('title', '') for article in articles if article.get('title')]
            
            scores = []
            for headline in headlines:
                sentiment = self.analyzer.polarity_scores(headline)
                scores.append(sentiment['compound'])
            
            avg_sentiment = sum(scores) / len(scores) if scores else 0.0
            confidence = min(0.9, len(scores) / 25.0)
            quality_score = (confidence * 0.6 + min(1.0, len(articles) / 30.0) * 0.4) * 100
            
            return NewsSource(
                name=f"GNews-{topic}",
                description=f"GNews {topic} topic news",
                articles_count=len(articles),
                sentiment_score=avg_sentiment,
                confidence=confidence,
                sample_headlines=headlines[:5],
                processing_time=time.time() - start_time,
                quality_score=quality_score
            )
            
        except Exception as e:
            return NewsSource(
                name=f"GNews-{topic}",
                description=f"GNews {topic} topic news",
                articles_count=0,
                sentiment_score=0.0,
                confidence=0.0,
                sample_headlines=[],
                processing_time=time.time() - start_time,
                quality_score=0.0,
                error=str(e)
            )

class CurrentSourcesAnalyzer:
    """Analyzer for current Oracle-X news sources"""
    
    def __init__(self):
        # Set up RSS feeds for testing
        os.environ.setdefault("RSS_FEEDS", 
            "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en,"
            "http://feeds.benzinga.com/benzinga,"
            "https://feeds.marketwatch.com/marketwatch/topstories/,"
            "https://www.cnbc.com/id/10001147/device/rss/rss.html")
        os.environ.setdefault("RSS_INCLUDE_ALL", "1")
        os.environ.setdefault("ADVANCED_SENTIMENT_MAX_PER_SOURCE", "50")
        
        self.orchestrator = DataFeedOrchestrator()
    
    def get_current_sentiment_sources(self, symbol: str) -> List[NewsSource]:
        """Get sentiment data from all current sources"""
        sources = []
        
        # Get available sentiment sources
        sentiment_sources = self.orchestrator.list_available_sentiment_sources()
        print(f"Available sentiment sources: {sentiment_sources}")
        
        # Get sentiment data from each source
        try:
            sentiment_map = self.orchestrator.get_sentiment_data(symbol)
            
            for source_name, sentiment_data in sentiment_map.items():
                if sentiment_data:
                    start_time = time.time()
                    
                    sample_headlines = []
                    if hasattr(sentiment_data, 'raw_data') and sentiment_data.raw_data:
                        sample_headlines = sentiment_data.raw_data.get('sample_texts', [])[:5]
                    
                    sources.append(NewsSource(
                        name=f"Current-{source_name}",
                        description=f"Oracle-X {source_name} source",
                        articles_count=getattr(sentiment_data, 'sample_size', 0),
                        sentiment_score=getattr(sentiment_data, 'sentiment_score', 0.0),
                        confidence=getattr(sentiment_data, 'confidence', 0.0),
                        sample_headlines=sample_headlines,
                        processing_time=time.time() - start_time,
                        quality_score=75.0  # Default quality score
                    ))
                    
        except Exception as e:
            print(f"Error getting current sentiment sources: {e}")
            sources.append(NewsSource(
                name="Current-Error",
                description="Error retrieving current sources",
                articles_count=0,
                sentiment_score=0.0,
                confidence=0.0,
                sample_headlines=[],
                processing_time=0.0,
                quality_score=0.0,
                error=str(e)
            ))
        
        # Get advanced sentiment data
        try:
            start_time = time.time()
            advanced_sentiment = self.orchestrator.get_advanced_sentiment_data(symbol)
            
            if advanced_sentiment:
                sample_headlines = []
                if hasattr(advanced_sentiment, 'raw_data') and advanced_sentiment.raw_data:
                    sample_headlines = advanced_sentiment.raw_data.get('sample_texts', [])[:5]
                
                sources.append(NewsSource(
                    name="Current-Advanced",
                    description="Oracle-X Advanced Sentiment (aggregated)",
                    articles_count=getattr(advanced_sentiment, 'sample_size', 0),
                    sentiment_score=getattr(advanced_sentiment, 'sentiment_score', 0.0),
                    confidence=getattr(advanced_sentiment, 'confidence', 0.0),
                    sample_headlines=sample_headlines,
                    processing_time=time.time() - start_time,
                    quality_score=85.0  # Higher quality for aggregated data
                ))
                
        except Exception as e:
            print(f"Error getting advanced sentiment: {e}")
        
        return sources

def run_comparison(symbol: str = "AAPL") -> ComparisonResult:
    """Run comprehensive comparison between current sources and gnews"""
    
    print(f"🔍 Starting news sentiment comparison for {symbol}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize analyzers
    current_analyzer = CurrentSourcesAnalyzer()
    gnews_adapter = GNewsAdapter()
    
    # Get current sources data
    print("📊 Analyzing current Oracle-X news sources...")
    current_sources = current_analyzer.get_current_sentiment_sources(symbol)
    
    # Get gnews data
    print("🆕 Testing GNews integration...")
    gnews_results = []
    
    # Test different gnews approaches
    gnews_results.append(gnews_adapter.get_news_by_keyword(symbol))
    gnews_results.append(gnews_adapter.get_news_by_keyword(f"${symbol}"))  # With $ prefix
    gnews_results.append(gnews_adapter.get_news_by_topic("BUSINESS"))
    gnews_results.append(gnews_adapter.get_news_by_topic("TECHNOLOGY"))
    gnews_results.append(gnews_adapter.get_news_by_topic("FINANCE"))
    
    # Calculate performance metrics
    total_current_articles = sum(s.articles_count for s in current_sources)
    total_gnews_articles = sum(s.articles_count for s in gnews_results)
    
    avg_current_quality = sum(s.quality_score for s in current_sources) / len(current_sources) if current_sources else 0
    avg_gnews_quality = sum(s.quality_score for s in gnews_results) / len(gnews_results) if gnews_results else 0
    
    avg_current_time = sum(s.processing_time for s in current_sources) / len(current_sources) if current_sources else 0
    avg_gnews_time = sum(s.processing_time for s in gnews_results) / len(gnews_results) if gnews_results else 0
    
    performance_metrics = {
        "total_current_articles": total_current_articles,
        "total_gnews_articles": total_gnews_articles,
        "avg_current_quality": avg_current_quality,
        "avg_gnews_quality": avg_gnews_quality,
        "avg_current_processing_time": avg_current_time,
        "avg_gnews_processing_time": avg_gnews_time,
        "current_sources_count": len(current_sources),
        "gnews_sources_count": len(gnews_results)
    }
    
    # Generate recommendations
    recommendations = []
    
    if total_gnews_articles > total_current_articles * 1.5:
        recommendations.append("✅ GNews provides significantly more articles (>50% increase)")
    
    if avg_gnews_quality > avg_current_quality:
        recommendations.append("✅ GNews shows higher average quality scores")
    elif avg_current_quality > avg_gnews_quality:
        recommendations.append("⚠️ Current sources have higher quality scores")
    
    if avg_gnews_time < avg_current_time:
        recommendations.append("✅ GNews is faster than current sources")
    elif avg_gnews_time > avg_current_time * 2:
        recommendations.append("⚠️ GNews is significantly slower than current sources")
    
    # Check for errors
    current_errors = [s for s in current_sources if s.error]
    gnews_errors = [s for s in gnews_results if s.error]
    
    if len(gnews_errors) < len(current_errors):
        recommendations.append("✅ GNews has fewer errors than current sources")
    
    if any(s.articles_count > 20 for s in gnews_results):
        recommendations.append("✅ GNews provides good article volume per topic")
    
    # Create comparison result
    result = ComparisonResult(
        symbol=symbol,
        timestamp=datetime.now().isoformat(),
        current_sources=current_sources,
        gnews_results=gnews_results,
        performance_metrics=performance_metrics,
        recommendations=recommendations
    )
    
    return result

def print_comparison_results(result: ComparisonResult):
    """Print detailed comparison results"""
    
    print("\n" + "="*80)
    print(f"📊 NEWS SENTIMENT COMPARISON RESULTS FOR {result.symbol}")
    print("="*80)
    
    print(f"\n🕒 Analysis completed at: {result.timestamp}")
    
    # Performance Overview
    print("\n📈 PERFORMANCE OVERVIEW")
    print("-" * 40)
    metrics = result.performance_metrics
    print(f"Current Sources: {metrics['current_sources_count']} sources, {metrics['total_current_articles']} articles")
    print(f"GNews Sources:   {metrics['gnews_sources_count']} sources, {metrics['total_gnews_articles']} articles")
    print(f"Quality Scores:  Current={metrics['avg_current_quality']:.1f}, GNews={metrics['avg_gnews_quality']:.1f}")
    print(f"Processing Time: Current={metrics['avg_current_processing_time']:.2f}s, GNews={metrics['avg_gnews_processing_time']:.2f}s")
    
    # Current Sources Detail
    print("\n🔄 CURRENT ORACLE-X SOURCES")
    print("-" * 40)
    for source in result.current_sources:
        status = "❌ ERROR" if source.error else "✅ OK"
        print(f"{status} {source.name}")
        print(f"    Articles: {source.articles_count}, Sentiment: {source.sentiment_score:.3f}, Confidence: {source.confidence:.3f}")
        print(f"    Quality: {source.quality_score:.1f}, Time: {source.processing_time:.2f}s")
        if source.sample_headlines:
            print(f"    Sample: {source.sample_headlines[0][:80]}...")
        if source.error:
            print(f"    Error: {source.error}")
        print()
    
    # GNews Sources Detail
    print("\n🆕 GNEWS SOURCES")
    print("-" * 40)
    for source in result.gnews_results:
        status = "❌ ERROR" if source.error else "✅ OK"
        print(f"{status} {source.name}")
        print(f"    Articles: {source.articles_count}, Sentiment: {source.sentiment_score:.3f}, Confidence: {source.confidence:.3f}")
        print(f"    Quality: {source.quality_score:.1f}, Time: {source.processing_time:.2f}s")
        if source.sample_headlines:
            print(f"    Sample: {source.sample_headlines[0][:80]}...")
        if source.error:
            print(f"    Error: {source.error}")
        print()
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    for rec in result.recommendations:
        print(f"  {rec}")
    
    if not result.recommendations:
        print("  No specific recommendations - both approaches have similar performance")

def save_results(result: ComparisonResult, filename: str = None):
    """Save results to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_sentiment_comparison_{result.symbol}_{timestamp}.json"
    
    # Convert to dict for JSON serialization
    result_dict = asdict(result)
    
    with open(filename, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")

def main():
    """Main execution function"""
    
    # Test symbols
    test_symbols = ["AAPL", "TSLA", "NVDA"]
    
    for symbol in test_symbols:
        print(f"\n{'🚀 TESTING SYMBOL: ' + symbol:^80}")
        
        try:
            # Run comparison
            result = run_comparison(symbol)
            
            # Print results
            print_comparison_results(result)
            
            # Save results
            save_results(result)
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait between symbols
        if symbol != test_symbols[-1]:
            print("\n⏱️  Waiting 10 seconds before next symbol...")
            time.sleep(10)
    
    print(f"\n{'✅ COMPARISON COMPLETE':^80}")

if __name__ == "__main__":
    main()
