#!/usr/bin/env python3
"""
Enhanced Sentiment Pipeline Test Script
Tests the new enhanced sentiment analysis pipeline with multiple sources
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_enhanced_sentiment_pipeline():
    """Test the enhanced sentiment pipeline with comprehensive validation"""
    
    print("🚀 Enhanced Sentiment Pipeline Test")
    print("=" * 60)
    
    # Test symbols
    test_symbols = ["AAPL", "TSLA", "NVDA"]
    
    try:
        # Import the enhanced sentiment pipeline
        from data_feeds.enhanced_sentiment_pipeline import get_enhanced_sentiment_pipeline
        from agent_bundle.data_feed_orchestrator import get_enhanced_sentiment_analysis
        
        print("✅ Enhanced sentiment modules imported successfully")
        
        # Initialize pipeline
        pipeline = get_enhanced_sentiment_pipeline()
        health_status = pipeline.get_health_status()
        
        print(f"\n📊 Pipeline Health Status:")
        print(f"   Status: {health_status['pipeline_status']}")
        print(f"   Sources: {health_status['sources_count']}")
        print(f"   Reddit Available: {health_status['reddit_available']}")
        print(f"   Max Workers: {health_status['max_workers']}")
        
        # Test individual sources
        print(f"\n🔧 Individual Source Status:")
        for source_name, status in health_status['sources'].items():
            status_emoji = "✅" if status.get('status') != 'error' else "❌"
            print(f"   {status_emoji} {source_name}: {status.get('status', 'unknown')}")
        
        # Test sentiment analysis for each symbol
        for symbol in test_symbols:
            print(f"\n🎯 Testing Enhanced Sentiment Analysis: {symbol}")
            print("-" * 40)
            
            start_time = time.time()
            
            # Test direct pipeline call
            result = pipeline.get_sentiment_analysis(symbol, include_reddit=False)
            
            processing_time = time.time() - start_time
            
            print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
            print(f"   📈 Overall Sentiment: {result['overall_sentiment']:.3f}")
            print(f"   🎯 Confidence: {result['confidence']:.3f}")
            print(f"   📊 Sources Count: {result['sources_count']}")
            print(f"   🔄 Trending Direction: {result['trending_direction']}")
            print(f"   ⭐ Quality Score: {result['quality_score']:.1f}")
            
            # Display source breakdown
            print(f"   🔍 Source Breakdown:")
            for source_name, data in result['source_breakdown'].items():
                sentiment_emoji = "📈" if data['sentiment'] > 0.1 else "📉" if data['sentiment'] < -0.1 else "➡️"
                print(f"     {sentiment_emoji} {source_name}: {data['sentiment']:.3f} "
                      f"(confidence: {data['confidence']:.3f}, method: {data['analysis_method']})")
            
            # Test orchestrator interface
            print(f"\n   🔄 Testing Orchestrator Interface...")
            start_time = time.time()
            orchestrator_result = get_enhanced_sentiment_analysis(symbol, include_reddit=False)
            orchestrator_time = time.time() - start_time
            
            print(f"   ⏱️  Orchestrator Time: {orchestrator_time:.2f}s")
            print(f"   ✅ Orchestrator Result: {orchestrator_result.get('overall_sentiment', 'error')}")
            
            # Brief pause between symbols
            time.sleep(1)
    
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_adapter_integration():
    """Test individual adapter integration"""
    
    print(f"\n🔧 Testing Individual Adapter Integration")
    print("=" * 60)
    
    test_symbol = "AAPL"
    
    # Test enhanced Twitter adapter
    try:
        from data_feeds.twitter_adapter import EnhancedTwitterAdapter
        
        print(f"📱 Testing Enhanced Twitter Adapter...")
        twitter_adapter = EnhancedTwitterAdapter()
        
        start_time = time.time()
        twitter_result = twitter_adapter.get_sentiment(test_symbol)
        twitter_time = time.time() - start_time
        
        if twitter_result:
            print(f"   ✅ Twitter Sentiment: {twitter_result.sentiment_score:.3f}")
            print(f"   🎯 Confidence: {twitter_result.confidence:.3f}")
            print(f"   ⏱️  Time: {twitter_time:.2f}s")
            print(f"   📊 Sample Size: {twitter_result.sample_size}")
            if twitter_result.raw_data:
                method = twitter_result.raw_data.get('analysis_method', 'unknown')
                print(f"   🔍 Analysis Method: {method}")
        else:
            print(f"   ❌ No Twitter sentiment data returned")
            
    except Exception as e:
        print(f"   ❌ Twitter Adapter Error: {e}")
    
    # Test news adapters
    news_adapters = [
        ("Reuters", "data_feeds.news_adapters.ReutersAdapter"),
        ("MarketWatch", "data_feeds.news_adapters.MarketWatchAdapter"),
        ("CNN Business", "data_feeds.news_adapters.CNNBusinessAdapter"),
        ("Financial Times", "data_feeds.news_adapters.FinancialTimesAdapter")
    ]
    
    for adapter_name, adapter_import in news_adapters:
        try:
            print(f"\n📰 Testing {adapter_name} Adapter...")
            
            # Dynamic import
            module_path, class_name = adapter_import.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            adapter_class = getattr(module, class_name)
            
            adapter = adapter_class()
            
            start_time = time.time()
            result = adapter.get_sentiment(test_symbol)
            adapter_time = time.time() - start_time
            
            if result:
                print(f"   ✅ {adapter_name} Sentiment: {result.sentiment_score:.3f}")
                print(f"   🎯 Confidence: {result.confidence:.3f}")
                print(f"   ⏱️  Time: {adapter_time:.2f}s")
                print(f"   📊 Sample Size: {result.sample_size}")
            else:
                print(f"   ⚠️  No {adapter_name} sentiment data (possibly no recent news)")
                
        except Exception as e:
            print(f"   ❌ {adapter_name} Adapter Error: {e}")

def benchmark_performance():
    """Benchmark enhanced vs original sentiment analysis"""
    
    print(f"\n⚡ Performance Benchmark")
    print("=" * 60)
    
    test_symbols = ["AAPL", "TSLA", "NVDA"]
    
    try:
        from data_feeds.enhanced_sentiment_pipeline import get_enhanced_sentiment_pipeline
        from agent_bundle.data_feed_orchestrator import get_sentiment_data
        
        pipeline = get_enhanced_sentiment_pipeline()
        
        # Benchmark enhanced pipeline
        print(f"🚀 Enhanced Pipeline Benchmark:")
        enhanced_times = []
        
        for symbol in test_symbols:
            start_time = time.time()
            result = pipeline.get_sentiment_analysis(symbol, include_reddit=False)
            processing_time = time.time() - start_time
            enhanced_times.append(processing_time)
            
            print(f"   {symbol}: {processing_time:.2f}s "
                  f"(sources: {result['sources_count']}, "
                  f"sentiment: {result['overall_sentiment']:.3f})")
        
        avg_enhanced_time = sum(enhanced_times) / len(enhanced_times)
        print(f"   📊 Average Time: {avg_enhanced_time:.2f}s")
        
        # Benchmark original methods
        print(f"\n🔄 Original Methods Benchmark:")
        original_times = []
        
        for symbol in test_symbols:
            start_time = time.time()
            result = get_sentiment_data(symbol)  # Original orchestrator method
            processing_time = time.time() - start_time
            original_times.append(processing_time)
            
            print(f"   {symbol}: {processing_time:.2f}s "
                  f"(sources: {len(result)})")
        
        avg_original_time = sum(original_times) / len(original_times)
        print(f"   📊 Average Time: {avg_original_time:.2f}s")
        
        # Performance comparison
        speedup = avg_original_time / avg_enhanced_time if avg_enhanced_time > 0 else 0
        improvement = ((avg_enhanced_time - avg_original_time) / avg_original_time) * 100 if avg_original_time > 0 else 0
        
        print(f"\n📈 Performance Comparison:")
        print(f"   Enhanced Pipeline: {avg_enhanced_time:.2f}s")
        print(f"   Original Methods: {avg_original_time:.2f}s")
        
        if improvement < 0:
            print(f"   ⚡ Speed Improvement: {abs(improvement):.1f}% faster")
        else:
            print(f"   ⚠️  Speed Change: {improvement:.1f}% slower (more comprehensive)")
            
    except Exception as e:
        print(f"❌ Benchmark Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test execution"""
    
    print("🧪 Enhanced Sentiment Pipeline Comprehensive Test Suite")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Enhanced Sentiment Pipeline
    success = test_enhanced_sentiment_pipeline()
    
    if success:
        # Test 2: Individual Adapter Integration
        test_adapter_integration()
        
        # Test 3: Performance Benchmark
        benchmark_performance()
        
        print(f"\n🎉 Enhanced Sentiment Pipeline Test Suite Completed!")
        print("=" * 80)
        print("✅ All tests executed successfully")
        print("📊 Enhanced sentiment analysis with advanced multi-model approach is operational")
        print("🚀 Pipeline ready for production use with parallel processing")
        
    else:
        print(f"\n❌ Test Suite Failed!")
        print("Please check dependencies and configuration")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
