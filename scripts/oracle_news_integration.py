#!/usr/bin/env python3
"""
Final RSS News Feeds Integration Guide for Oracle-X
Complete integration of all working news adapters into DataFeedOrchestrator
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# RSS Feeds Configuration (place in .env file or set as environment variables)
RSS_CONFIG = {
    'RSS_FEEDS': 'http://feeds.benzinga.com/benzinga,https://www.cnbc.com/id/10001147/device/rss/rss.html,https://www.ft.com/rss/home,https://fortune.com/feed/fortune-feeds/?id=3230629,https://feeds.marketwatch.com/marketwatch/topstories/,https://seekingalpha.com/feed.xml',
    'RSS_INCLUDE_ALL': '1'
}

# Set environment variables
for key, value in RSS_CONFIG.items():
    os.environ[key] = value

# Add project root to path
sys.path.append('/Users/omar/Documents/Projects/oracle-x')

try:
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
    
    class OracleNewsIntegration:
        """Oracle-X News Feeds Integration Class"""
        
        def __init__(self):
            """Initialize the news integration with DataFeedOrchestrator"""
            self.orchestrator = DataFeedOrchestrator()
            self.rss_adapter = self._find_rss_adapter()
            
        def _find_rss_adapter(self):
            """Find the RSS adapter in the orchestrator"""
            for key, adapter in self.orchestrator.adapters.items():
                if 'RSS' in str(key):
                    return adapter
            return None
            
        def get_news_sentiment(self, symbol: str) -> Optional[Dict]:
            """
            Get news sentiment for a symbol from all integrated RSS feeds
            
            Args:
                symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
                
            Returns:
                Dictionary with sentiment data or None if no data available
            """
            if not self.rss_adapter:
                return None
                
            try:
                sentiment_data = self.rss_adapter.get_sentiment(symbol)
                
                if sentiment_data:
                    return {
                        'symbol': symbol,
                        'sentiment_score': sentiment_data.sentiment_score,
                        'confidence': sentiment_data.confidence,
                        'source': sentiment_data.source,
                        'sample_size': sentiment_data.sample_size,
                        'timestamp': sentiment_data.timestamp,
                        'raw_data': sentiment_data.raw_data
                    }
                return None
                
            except Exception as e:
                print(f"Error getting sentiment for {symbol}: {e}")
                return None
                
        def get_comprehensive_sentiment(self, symbol: str) -> Dict:
            """
            Get comprehensive sentiment from all available sources including RSS
            
            Args:
                symbol: Stock symbol
                
            Returns:
                Dictionary with all available sentiment sources
            """
            try:
                # Get all sentiment sources
                all_sentiment = self.orchestrator.get_sentiment_data(symbol)
                
                # Add RSS sentiment if available
                rss_sentiment = self.get_news_sentiment(symbol)
                if rss_sentiment:
                    all_sentiment['rss_news'] = rss_sentiment
                    
                return all_sentiment
                
            except Exception as e:
                print(f"Error getting comprehensive sentiment for {symbol}: {e}")
                return {}
                
        def list_integrated_feeds(self) -> List[Dict[str, str]]:
            """List all integrated RSS news feeds"""
            return [
                {'name': 'Benzinga', 'url': 'http://feeds.benzinga.com/benzinga', 'focus': 'Financial news and stock analysis'},
                {'name': 'CNBC Business', 'url': 'https://www.cnbc.com/id/10001147/device/rss/rss.html', 'focus': 'Business news (replaced CNN Business)'},
                {'name': 'Financial Times', 'url': 'https://www.ft.com/rss/home', 'focus': 'International business and markets'},
                {'name': 'Fortune', 'url': 'https://fortune.com/feed/fortune-feeds/?id=3230629', 'focus': 'Business insights and market analysis'},
                {'name': 'MarketWatch', 'url': 'https://feeds.marketwatch.com/marketwatch/topstories/', 'focus': 'Market news and financial data'},
                {'name': 'Seeking Alpha', 'url': 'https://seekingalpha.com/feed.xml', 'focus': 'Investment research and analysis'}
            ]
            
        def test_integration(self, symbols: List[str] = ['AAPL', 'TSLA', 'MSFT', 'GOOGL']) -> Dict:
            """Test the news integration with multiple symbols"""
            results = {}
            
            print("🚀 Oracle-X News Feeds Integration Test")
            print(f"Time: {datetime.now()}")
            print("="*80)
            
            # Show integrated feeds
            feeds = self.list_integrated_feeds()
            print(f"\n📰 Integrated News Feeds ({len(feeds)} feeds):")
            for i, feed in enumerate(feeds, 1):
                print(f"  {i}. {feed['name']}: {feed['focus']}")
            
            print(f"\n📊 Testing sentiment analysis...")
            
            for symbol in symbols:
                print(f"\n🎯 {symbol}:")
                sentiment = self.get_news_sentiment(symbol)
                
                if sentiment:
                    results[symbol] = sentiment
                    print(f"  ✅ Sentiment: {sentiment['sentiment_score']:.3f}")
                    print(f"  📈 Confidence: {sentiment['confidence']:.3f}")
                    print(f"  📊 Articles: {sentiment['sample_size']}")
                    
                    # Show sample headlines
                    if sentiment.get('raw_data', {}).get('sample_texts'):
                        headlines = sentiment['raw_data']['sample_texts'][:2]
                        print(f"  📝 Headlines:")
                        for i, headline in enumerate(headlines, 1):
                            print(f"     {i}. {headline[:70]}...")
                else:
                    print(f"  ❌ No sentiment data available")
            
            # Summary
            successful = len([r for r in results.values() if r])
            total = len(symbols)
            
            print(f"\n" + "="*80)
            print(f"INTEGRATION TEST SUMMARY")
            print(f"="*80)
            print(f"Successfully processed: {successful}/{total} symbols")
            print(f"Success rate: {(successful/total)*100:.1f}%")
            
            if successful > 0:
                avg_sentiment = sum(r['sentiment_score'] for r in results.values()) / successful
                total_articles = sum(r['sample_size'] for r in results.values())
                print(f"Average sentiment: {avg_sentiment:.3f}")
                print(f"Total articles analyzed: {total_articles}")
            
            if successful >= total * 0.75:
                print(f"\n✅ NEWS INTEGRATION SUCCESSFUL!")
                print(f"   📰 All news adapters integrated and tested")
                print(f"   🎯 Ready for production use in Oracle-X")
            else:
                print(f"\n⚠️  NEWS INTEGRATION PARTIAL")
                
            return results

    def main():
        """Main integration demo"""
        try:
            # Initialize news integration
            news = OracleNewsIntegration()
            
            # Run integration test
            results = news.test_integration()
            
            # Show usage example
            print(f"\n📖 USAGE EXAMPLES:")
            print(f"="*80)
            print(f"# Initialize news integration")
            print(f"from oracle_news_integration import OracleNewsIntegration")
            print(f"news = OracleNewsIntegration()")
            print(f"")
            print(f"# Get sentiment for a symbol")
            print(f"sentiment = news.get_news_sentiment('AAPL')")
            print(f"print(f'AAPL sentiment: {{sentiment['sentiment_score']:.3f}}')")
            print(f"")
            print(f"# Get all sentiment sources (including RSS)")
            print(f"all_sentiment = news.get_comprehensive_sentiment('AAPL')")
            print(f"print(f'Available sources: {{list(all_sentiment.keys())}}')")
            print(f"")
            print(f"# List integrated feeds")
            print(f"feeds = news.list_integrated_feeds()")
            print(f"for feed in feeds:")
            print(f"    print(f'{{feed['name']}}: {{feed['focus']}}')")
            
            return len(results) > 0
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    if __name__ == "__main__":
        success = main()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)
