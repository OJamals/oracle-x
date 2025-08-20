"""
Test script for the new DataFeedOrchestrator and OracleDataInterface
Validates that our consolidated system works and provides quality data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_feeds.data_feed_orchestrator import get_orchestrator, get_quote, get_system_health
from data_feeds.oracle_data_interface import get_oracle_provider, get_signals_from_scrapers_v2

def test_basic_functionality():
    """Test basic data feed functionality"""
    print("=== Testing DataFeedOrchestrator ===")
    
    # Test quote retrieval
    print("\n1. Testing quote retrieval...")
    quote = get_quote("AAPL")
    if quote:
        print(f"✓ AAPL Quote: ${quote.price} (Quality: {quote.quality_score:.1f}%)")
        print(f"  Change: {quote.change_percent}%, Volume: {quote.volume:,}")
        print(f"  Source: {quote.source}, Timestamp: {quote.timestamp}")
    else:
        print("✗ Failed to get AAPL quote")
    
    # Test system health
    print("\n2. Testing system health...")
    health = get_system_health()
    print(f"✓ System Status: {health['status']}")
    print(f"  Sources Available: {health['sources_available']}")
    print(f"  Cache Size: {health['cache_size']}")
    if health['quality_issues']:
        print(f"  Quality Issues: {health['quality_issues']}")

def test_oracle_interface():
    """Test the Oracle data interface"""
    print("\n=== Testing Oracle Data Interface ===")
    
    # Test market intelligence
    print("\n1. Testing comprehensive market intelligence...")
    provider = get_oracle_provider()
    intelligence = provider.get_comprehensive_market_intelligence(["AAPL", "MSFT", "GOOGL"])
    
    print(f"✓ Market Intelligence Quality Score: {intelligence.quality_score:.1f}%")
    print(f"  Tickers Analyzed: {len(intelligence.market_data)}")
    print(f"  Data Sources Used: {intelligence.data_sources_used}")
    if intelligence.warnings:
        print(f"  Warnings: {intelligence.warnings}")
    
    # Test individual components
    print("\n2. Testing market internals...")
    internals = provider.get_market_internals()
    if 'error' not in internals:
        print(f"✓ SPY: ${internals.get('spy_price', 'N/A')} ({internals.get('spy_change', 'N/A')}%)")
        print(f"  VIX: {internals.get('vix', 'N/A')}")
        print(f"  Quality Score: {internals.get('quality_score', 0):.1f}%")
    else:
        print(f"✗ Market internals error: {internals['error']}")
    
    print("\n3. Testing sentiment analysis...")
    sentiment = provider.get_sentiment_analysis(["AAPL", "TSLA"])
    print(f"✓ Overall Sentiment: {sentiment['overall_sentiment']:.2f}")
    print(f"  Confidence: {sentiment['confidence']:.2f}")
    print(f"  Quality Score: {sentiment['quality_score']:.1f}%")
    
    print("\n4. Testing options analysis...")
    options = provider.get_options_analysis(["AAPL", "TSLA"])
    print(f"✓ Options Analysis Quality: {options['quality_score']}%")
    print(f"  Unusual Volume Stocks: {len(options['unusual_volume'])}")

def test_backward_compatibility():
    """Test backward compatibility functions"""
    print("\n=== Testing Backward Compatibility ===")
    
    # Test the new signals function
    print("\n1. Testing enhanced signals function...")
    try:
        signals = get_signals_from_scrapers_v2("Market analysis test")
        
        print(f"✓ Signals generated successfully")
        print(f"  Overall Quality Score: {signals['data_quality']['overall_score']:.1f}%")
        print(f"  Sources Used: {signals['data_quality']['sources_used']}")
        print(f"  System Health: {signals['system_health']['status']}")
        
        # Check data sections
        sections = ['market_internals', 'options_flow', 'sentiment_web', 'earnings']
        for section in sections:
            if section in signals:
                print(f"  ✓ {section}: Available")
            else:
                print(f"  ✗ {section}: Missing")
    
    except Exception as e:
        print(f"✗ Error testing signals: {e}")

def main():
    """Run all tests"""
    print("Testing New Data Feed System")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_oracle_interface()
        test_backward_compatibility()
        
        print("\n" + "=" * 40)
        print("✓ All tests completed!")
        print("\nNext Steps:")
        print("1. Update oracle_engine/prompt_chain.py to use new interface")
        print("2. Remove old placeholder feed files")
        print("3. Update imports across the codebase")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
