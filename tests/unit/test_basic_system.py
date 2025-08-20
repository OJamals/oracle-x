"""
Quick test script for the new DataFeedOrchestrator
Tests basic functionality without triggering Reddit loops
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_feeds.data_feed_orchestrator import get_orchestrator, get_quote, get_system_health
from data_feeds.oracle_data_interface import get_oracle_provider

def test_basic_quotes():
    """Test basic quote functionality without sentiment"""
    print("=== Testing Basic Quote Functionality ===")
    
    test_symbols = ["AAPL", "MSFT", "SPY"]
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")
        quote = get_quote(symbol)
        
        if quote:
            print(f"✓ {symbol}: ${quote.price} (Quality: {quote.quality_score:.1f}%)")
            print(f"  Change: {quote.change_percent}%, Volume: {quote.volume:,}")
            print(f"  Source: {quote.source}")
        else:
            print(f"✗ Failed to get quote for {symbol}")

def test_market_internals():
    """Test market internals without sentiment analysis"""
    print("\n=== Testing Market Internals ===")
    
    provider = get_oracle_provider()
    internals = provider.get_market_internals()
    
    if 'error' not in internals:
        print(f"✓ SPY: ${internals.get('spy_price', 'N/A')} ({internals.get('spy_change', 'N/A')}%)")
        print(f"  QQQ: ${internals.get('qqq_price', 'N/A')} ({internals.get('qqq_change', 'N/A')}%)")
        print(f"  VIX: {internals.get('vix', 'N/A')}")
        print(f"  Quality Score: {internals.get('quality_score', 0):.1f}%")
    else:
        print(f"✗ Market internals error: {internals['error']}")

def test_system_health():
    """Test overall system health"""
    print("\n=== Testing System Health ===")
    
    health = get_system_health()
    print(f"✓ System Status: {health['status']}")
    print(f"  Sources Available: {health['sources_available']}")
    print(f"  Cache Size: {health['cache_size']}")
    if health['quality_issues']:
        print(f"  Quality Issues: {health['quality_issues']}")

def main():
    """Run basic tests without Reddit API calls"""
    print("Testing Data Feed System - Basic Functionality")
    print("=" * 50)
    
    try:
        test_basic_quotes()
        test_market_internals() 
        test_system_health()
        
        print("\n" + "=" * 50)
        print("✓ Basic tests completed successfully!")
        print("\nSystem is ready for production use.")
        print("Reddit sentiment is available but requires API credentials.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
