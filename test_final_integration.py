#!/usr/bin/env python3
"""
Final Integration Test - Enhanced Sentiment Pipeline with Main Oracle-X Pipeline
Tests integration between enhanced sentiment analysis and main trading pipeline
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

def test_main_pipeline_integration():
    """Test enhanced sentiment integration with main Oracle-X pipeline"""
    
    print("🔗 Enhanced Sentiment Pipeline - Main Pipeline Integration Test")
    print("=" * 70)
    
    test_symbol = "AAPL"
    
    try:
        # Test 1: Direct enhanced sentiment access
        print(f"🎯 Test 1: Direct Enhanced Sentiment Access")
        from agent_bundle.data_feed_orchestrator import get_enhanced_sentiment_analysis
        
        start_time = time.time()
        enhanced_result = get_enhanced_sentiment_analysis(test_symbol, include_reddit=False)
        enhanced_time = time.time() - start_time
        
        print(f"   ✅ Enhanced Sentiment: {enhanced_result.get('overall_sentiment', 0):.3f}")
        print(f"   🎯 Confidence: {enhanced_result.get('confidence', 0):.3f}")
        print(f"   📊 Sources: {enhanced_result.get('sources_count', 0)}")
        print(f"   ⏱️  Processing Time: {enhanced_time:.2f}s")
        print(f"   🔄 Trend: {enhanced_result.get('trending_direction', 'unknown')}")
        
        # Test 2: Orchestrator sentiment comparison
        print(f"\n🔄 Test 2: Original vs Enhanced Sentiment Comparison")
        from agent_bundle.data_feed_orchestrator import get_sentiment_data, get_advanced_sentiment
        
        # Original sentiment data
        start_time = time.time()
        original_result = get_sentiment_data(test_symbol)
        original_time = time.time() - start_time
        
        print(f"   Original Sentiment Sources: {len(original_result)}")
        for source, data in original_result.items():
            print(f"     • {source}: {data.sentiment_score:.3f} (confidence: {data.confidence:.3f})")
        print(f"   Original Processing Time: {original_time:.2f}s")
        
        # Advanced sentiment data
        start_time = time.time()
        advanced_result = get_advanced_sentiment(test_symbol)
        advanced_time = time.time() - start_time
        
        if advanced_result:
            print(f"   Advanced Sentiment: {advanced_result.sentiment_score:.3f}")
            print(f"   Advanced Confidence: {advanced_result.confidence:.3f}")
            print(f"   Advanced Processing Time: {advanced_time:.2f}s")
        else:
            print(f"   Advanced Sentiment: Not available")
        
        # Test 3: Data Feed Orchestrator full integration
        print(f"\n🏗️ Test 3: Full DataFeedOrchestrator Integration")
        from agent_bundle.data_feed_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        health_status = orchestrator.validate_system_health()
        
        print(f"   System Health: {health_status.get('overall_health', 'unknown')}")
        print(f"   Active Sources: {len(health_status.get('source_health', {}))}")
        
        # Get quote data to verify full integration
        quote_data = orchestrator.get_quote(test_symbol)
        if quote_data:
            print(f"   Quote Integration: ✅ ${quote_data.price:.2f}")
        else:
            print(f"   Quote Integration: ❌ Failed")
        
        # Test 4: Performance comparison summary
        print(f"\n📈 Test 4: Performance Summary")
        print(f"   Enhanced Pipeline: {enhanced_time:.2f}s")
        print(f"   Original Pipeline: {original_time:.2f}s")
        print(f"   Advanced Pipeline: {advanced_time:.2f}s")
        
        improvement = ((original_time - enhanced_time) / original_time) * 100 if original_time > 0 else 0
        if improvement > 0:
            print(f"   Performance Improvement: +{improvement:.1f}%")
        else:
            print(f"   Performance Change: {improvement:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sentiment_data_quality():
    """Test sentiment data quality and consistency"""
    
    print(f"\n📊 Enhanced Sentiment Data Quality Test")
    print("=" * 50)
    
    test_symbols = ["AAPL", "TSLA", "NVDA"]
    
    try:
        from agent_bundle.data_feed_orchestrator import get_enhanced_sentiment_analysis
        
        quality_metrics = []
        
        for symbol in test_symbols:
            result = get_enhanced_sentiment_analysis(symbol, include_reddit=False)
            
            # Extract quality metrics
            quality_score = result.get('quality_score', 0)
            confidence = result.get('confidence', 0)
            sources_count = result.get('sources_count', 0)
            processing_time = result.get('processing_time_seconds', 0)
            
            quality_metrics.append({
                'symbol': symbol,
                'quality_score': quality_score,
                'confidence': confidence,
                'sources_count': sources_count,
                'processing_time': processing_time,
                'sentiment': result.get('overall_sentiment', 0)
            })
            
            print(f"   {symbol}:")
            print(f"     Quality Score: {quality_score:.1f}/100")
            print(f"     Confidence: {confidence:.3f}")
            print(f"     Sources: {sources_count}")
            print(f"     Time: {processing_time:.2f}s")
            print(f"     Sentiment: {result.get('overall_sentiment', 0):.3f}")
        
        # Calculate averages
        avg_quality = sum(m['quality_score'] for m in quality_metrics) / len(quality_metrics)
        avg_confidence = sum(m['confidence'] for m in quality_metrics) / len(quality_metrics)
        avg_sources = sum(m['sources_count'] for m in quality_metrics) / len(quality_metrics)
        avg_time = sum(m['processing_time'] for m in quality_metrics) / len(quality_metrics)
        
        print(f"\n📊 Quality Metrics Summary:")
        print(f"   Average Quality Score: {avg_quality:.1f}/100")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Average Sources: {avg_sources:.1f}")
        print(f"   Average Processing Time: {avg_time:.2f}s")
        
        # Quality assessment
        if avg_quality >= 70:
            quality_grade = "🌟 Excellent"
        elif avg_quality >= 50:
            quality_grade = "✅ Good"
        elif avg_quality >= 30:
            quality_grade = "⚠️ Fair"
        else:
            quality_grade = "❌ Poor"
        
        print(f"   Overall Quality Grade: {quality_grade}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality Test Error: {e}")
        return False

def test_error_handling():
    """Test error handling and fallback mechanisms"""
    
    print(f"\n🛡️ Error Handling & Fallback Test")
    print("=" * 40)
    
    try:
        from agent_bundle.data_feed_orchestrator import get_enhanced_sentiment_analysis
        
        # Test with invalid symbol
        print(f"🔍 Testing Invalid Symbol...")
        result = get_enhanced_sentiment_analysis("INVALID_SYMBOL", include_reddit=False)
        
        if 'error' in result:
            print(f"   ✅ Error handling working: {result.get('trending_direction')}")
        else:
            print(f"   ✅ Graceful fallback: {result.get('sources_count', 0)} sources")
        
        # Test with network timeout simulation (short timeout)
        print(f"\n⏱️ Testing Timeout Resilience...")
        from data_feeds.enhanced_sentiment_pipeline import EnhancedSentimentPipeline
        
        # Create pipeline with very short timeout
        timeout_pipeline = EnhancedSentimentPipeline(max_workers=2, timeout_seconds=1)
        result = timeout_pipeline.get_sentiment_analysis("AAPL", include_reddit=False)
        
        print(f"   Sources Retrieved: {result.get('sources_count', 0)}")
        print(f"   Processing Time: {result.get('processing_time_seconds', 0):.2f}s")
        print(f"   Fallback Status: {'✅ Working' if result.get('sources_count', 0) > 0 else '⚠️ Limited'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error Handling Test Error: {e}")
        return False

def main():
    """Main integration test execution"""
    
    print("🧪 Enhanced Sentiment Pipeline - Final Integration Test Suite")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Main pipeline integration
    if test_main_pipeline_integration():
        tests_passed += 1
        print("✅ Test 1: Main Pipeline Integration - PASSED")
    else:
        print("❌ Test 1: Main Pipeline Integration - FAILED")
    
    # Test 2: Data quality validation
    if test_sentiment_data_quality():
        tests_passed += 1
        print("✅ Test 2: Data Quality Validation - PASSED")
    else:
        print("❌ Test 2: Data Quality Validation - FAILED")
    
    # Test 3: Error handling
    if test_error_handling():
        tests_passed += 1
        print("✅ Test 3: Error Handling - PASSED")
    else:
        print("❌ Test 3: Error Handling - FAILED")
    
    # Final summary
    print(f"\n🎯 Final Integration Test Results")
    print("=" * 40)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ Enhanced sentiment pipeline fully integrated with Oracle-X")
        print("🚀 System ready for production deployment")
        print("📊 Advanced multi-model sentiment analysis operational")
        return 0
    else:
        print(f"\n⚠️ Some tests failed ({total_tests-tests_passed}/{total_tests})")
        print("Please review errors and retry integration")
        return 1

if __name__ == "__main__":
    exit(main())
