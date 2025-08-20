#!/usr/bin/env python3
"""
Debug RSS Adapter Registration in DataFeedOrchestrator
"""

import os
import sys
import logging

# Set up RSS feeds configuration
os.environ['RSS_FEEDS'] = 'http://feeds.benzinga.com/benzinga,https://www.cnbc.com/id/10001147/device/rss/rss.html,https://www.ft.com/rss/home,https://fortune.com/feed/fortune-feeds/?id=3230629,https://feeds.marketwatch.com/marketwatch/topstories/,https://seekingalpha.com/feed.xml'
os.environ['RSS_INCLUDE_ALL'] = '1'

# Add project root to path
sys.path.append('/Users/omar/Documents/Projects/oracle-x')

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator, GenericRSSSentimentAdapter
    
    print("🔍 Debug RSS Adapter Registration")
    print("="*50)
    
    # Test RSS adapter initialization independently
    print("\n1. Testing RSS Adapter Initialization...")
    try:
        # Create a minimal RSS adapter to test
        cache = None  # We'll pass None for debug
        rate_limiter = None
        performance_tracker = None
        
        rss_adapter = GenericRSSSentimentAdapter(cache, rate_limiter, performance_tracker)
        print(f"✅ RSS Adapter created")
        print(f"   Feed URLs: {rss_adapter.feed_urls}")
        print(f"   Include All: {rss_adapter.include_all}")
        print(f"   Feedparser Available: {rss_adapter.feedparser_available}")
        
        # Test if it has the right interface
        if hasattr(rss_adapter, 'get_sentiment'):
            print(f"✅ RSS Adapter has get_sentiment method")
        else:
            print(f"❌ RSS Adapter missing get_sentiment method")
            
    except Exception as e:
        print(f"❌ RSS Adapter initialization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. Testing DataFeedOrchestrator Initialization...")
    try:
        orchestrator = DataFeedOrchestrator()
        print(f"✅ DataFeedOrchestrator created")
        print(f"   Adapters registered: {len(orchestrator.adapters)}")
        print(f"   Adapter keys: {list(orchestrator.adapters.keys())}")
        
        # Look for RSS adapter specifically
        rss_found = False
        for key, adapter in orchestrator.adapters.items():
            if 'rss' in str(key).lower() or 'RSS' in str(key):
                print(f"   🎯 Found RSS adapter: {key} -> {type(adapter).__name__}")
                rss_found = True
                
                # Test the adapter
                if hasattr(adapter, 'get_sentiment'):
                    print(f"      ✅ Has get_sentiment method")
                    try:
                        print(f"      Testing with AAPL...")
                        result = adapter.get_sentiment('AAPL')
                        if result:
                            print(f"      ✅ RSS sentiment test passed: {result.sentiment_score:.3f}")
                        else:
                            print(f"      ⚠️  RSS sentiment test returned None")
                    except Exception as e:
                        print(f"      ❌ RSS sentiment test failed: {e}")
                else:
                    print(f"      ❌ Missing get_sentiment method")
        
        if not rss_found:
            print(f"   ❌ No RSS adapter found in orchestrator")
            print(f"   Available adapters by type:")
            for key, adapter in orchestrator.adapters.items():
                print(f"      {key}: {type(adapter).__name__}")
        
    except Exception as e:
        print(f"❌ DataFeedOrchestrator initialization failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n3. Testing Environment Variables...")
    print(f"   RSS_FEEDS: {os.environ.get('RSS_FEEDS', 'NOT SET')[:100]}...")
    print(f"   RSS_INCLUDE_ALL: {os.environ.get('RSS_INCLUDE_ALL', 'NOT SET')}")
    
    print("\n4. Testing Feedparser Availability...")
    try:
        import feedparser
        print(f"✅ feedparser available: {feedparser.__version__}")
    except ImportError:
        print(f"❌ feedparser not available")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
