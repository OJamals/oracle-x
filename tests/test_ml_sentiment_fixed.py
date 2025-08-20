#!/usr/bin/env python3
"""
Comprehensive test to validate ML engine integration with Twitter sentiment analysis
"""

import logging
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator, DataSource
from data_feeds.advanced_sentiment import AdvancedSentimentEngine
from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine, PredictionType

def test_ml_training_with_sentiment():
    """Test ML model training with actual Twitter sentiment data"""
    print("🔧 Testing ML Training with Twitter Sentiment Integration...")
    
    try:
        # Initialize components
        data_orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        ml_engine = EnsemblePredictionEngine(data_orchestrator, sentiment_engine)
        
        # Test symbol
        symbol = "TSLA"
        
        print(f"📊 Testing ML training for {symbol}...")
        
        # Test data collection
        print("1. Fetching market data...")
        market_data = data_orchestrator.get_market_data(symbol, period="60d", interval="1d")
        if not market_data or market_data.data.empty:
            print(f"   ❌ No market data available for {symbol}")
            return False
        print(f"   ✅ Got {len(market_data.data)} days of market data")
        
        # Test sentiment data collection
        print("2. Fetching sentiment data...")
        sentiment_data = data_orchestrator.get_sentiment_data(symbol)
        
        if not sentiment_data:
            print(f"   ❌ No sentiment data available for {symbol}")
            return False
        
        # Check what sentiment data we got
        total_texts = 0
        for source_name, sentiment_obj in sentiment_data.items():
            if hasattr(sentiment_obj, 'raw_data') and sentiment_obj.raw_data:
                if 'sample_texts' in sentiment_obj.raw_data:
                    texts = sentiment_obj.raw_data['sample_texts']
                    if texts:
                        total_texts += len(texts)
                        print(f"   ✅ {source_name}: {len(texts)} texts")
                        # Show first few texts
                        for i, text in enumerate(texts[:3]):
                            if isinstance(text, dict):
                                text_content = text.get('text', str(text)[:100])
                            else:
                                text_content = str(text)[:100]
                            print(f"      - Text {i+1}: {text_content}...")
                    else:
                        print(f"   ⚠️  {source_name}: No texts in sample_texts")
                else:
                    print(f"   ⚠️  {source_name}: No sample_texts in raw_data")
            else:
                print(f"   ⚠️  {source_name}: No raw_data available")
        
        if total_texts == 0:
            print(f"   ❌ No actual texts found in sentiment data")
            return False
        
        print(f"   ✅ Total texts available: {total_texts}")
        
        # Test ML training
        print("3. Training ML models...")
        start_time = time.time()
        
        training_results = ml_engine.train_models([symbol], lookback_days=60)
        
        training_time = time.time() - start_time
        
        if not training_results:
            print(f"   ❌ ML Training failed - no results")
            return False
        
        print(f"   ✅ ML Training completed in {training_time:.2f}s")
        
        # Analyze training results
        success_count = 0
        error_count = 0
        
        for target_key, results in training_results.items():
            print(f"   📊 Target: {target_key}")
            for model_key, result in results.items():
                if isinstance(result, dict) and 'error' not in result:
                    success_count += 1
                    print(f"      ✅ {model_key}: Success")
                    if 'validation_metrics' in result:
                        metrics = result['validation_metrics']
                        print(f"         Metrics: {metrics}")
                else:
                    error_count += 1
                    error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                    print(f"      ❌ {model_key}: {error_msg}")
        
        print(f"   📈 Training Summary: {success_count} successful, {error_count} errors")
        
        # Test prediction with sentiment
        print("4. Testing ML prediction with sentiment...")
        prediction = ml_engine.predict(symbol, PredictionType.PRICE_DIRECTION, horizon_days=5)
        
        if not prediction:
            print(f"   ❌ ML Prediction failed - no result")
            return False
        
        print(f"   ✅ ML Prediction successful")
        print(f"      Symbol: {prediction.symbol}")
        print(f"      Prediction: {prediction.prediction:.4f}")
        print(f"      Confidence: {prediction.confidence:.3f}")
        print(f"      Sentiment available: {prediction.prediction_context.get('sentiment_available', False)}")
        print(f"      Models used: {prediction.prediction_context.get('models_used', 0)}")
        
        # Test model performance
        print("5. Checking model performance...")
        performance = ml_engine.get_model_performance()
        
        if not performance:
            print(f"   ❌ No model performance data available")
            return False
        
        print(f"   ✅ Performance data available for {len(performance)} models")
        for model_key, perf in performance.items():
            print(f"      {model_key}: Weight={perf['weight']:.3f}, Accuracy={perf['accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sentiment_engine_directly():
    """Test sentiment engine directly with sample texts"""
    print("🧠 Testing Sentiment Engine Directly...")
    
    try:
        sentiment_engine = AdvancedSentimentEngine()
        
        # Sample texts (simulating tweets)
        sample_texts = [
            "TSLA is going to the moon! 🚀 Elon is a genius",
            "Tesla stock is overvalued, expecting a crash soon",
            "Just bought more TSLA, love the innovation",
            "Tesla earnings were disappointing, selling my shares"
        ]
        
        print(f"📝 Analyzing {len(sample_texts)} sample texts...")
        
        sentiment_summary = sentiment_engine.get_symbol_sentiment_summary(
            "TSLA", sample_texts, ["test"] * len(sample_texts)
        )
        
        if not sentiment_summary:
            print("   ❌ Sentiment analysis failed")
            return False
        
        print("   ✅ Sentiment analysis successful")
        print(f"      Overall sentiment: {sentiment_summary.overall_sentiment:.3f}")
        print(f"      Confidence: {sentiment_summary.confidence:.3f}")
        print(f"      Sample size: {sentiment_summary.sample_size}")
        print(f"      Bullish mentions: {sentiment_summary.bullish_mentions}")
        print(f"      Bearish mentions: {sentiment_summary.bearish_mentions}")
        print(f"      Neutral mentions: {sentiment_summary.neutral_mentions}")
        print(f"      Trending direction: {sentiment_summary.trending_direction}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive ML + sentiment integration tests"""
    print("🚀 Oracle-X ML + Sentiment Integration Test")
    print("=" * 60)
    
    tests = [
        ("Sentiment Engine Direct Test", test_sentiment_engine_directly),
        ("ML Training with Sentiment", test_ml_training_with_sentiment),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}")
        print("-" * 40)
        
        start_time = time.time()
        success = test_func()
        duration = time.time() - start_time
        
        results.append((test_name, success, duration))
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status} ({duration:.2f}s)")
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, duration in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {duration:.2f}s")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ML + Sentiment integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
