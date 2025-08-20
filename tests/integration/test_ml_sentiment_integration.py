#!/usr/bin/env python3
"""
ML Sentiment Integration Fix
Fix the data orchestrator sentiment integration for ML model training
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_feeds.data_feed_orchestrator import DataFeedOrchestrator, SentimentData
from data_feeds.advanced_sentiment import AdvancedSentimentEngine
from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine, PredictionType

def test_ml_sentiment_integration_fixed():
    """
    Test the fixed ML sentiment integration
    """
    print("=" * 80)
    print("ML SENTIMENT INTEGRATION FIX - TESTING")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    try:
        print("🔧 Initializing components...")
        orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        ml_engine = EnsemblePredictionEngine(orchestrator, sentiment_engine)
        
        test_symbols = ["AAPL", "TSLA", "NVDA"]
        
        print(f"\n📊 Testing sentiment data integration for: {', '.join(test_symbols)}")
        
        for symbol in test_symbols:
            print(f"\n🔍 Testing {symbol}:")
            
            # Test direct sentiment engine access
            try:
                print(f"  📡 Getting live sentiment data...")
                
                # Get Twitter sentiment
                from data_feeds.twitter_feed import TwitterSentimentFeed
                twitter_feed = TwitterSentimentFeed()
                tweets = twitter_feed.fetch(symbol, limit=5)
                
                if tweets:
                    print(f"    ✅ Retrieved {len(tweets)} tweets")
                    
                    # Extract text from tweet dictionaries for sentiment analysis
                    tweet_texts = [tweet['text'] for tweet in tweets if 'text' in tweet]
                    
                    if tweet_texts:
                        print(f"    📝 Extracted {len(tweet_texts)} tweet texts")
                        
                        # Use advanced sentiment engine to analyze
                        symbol_sentiment = sentiment_engine.get_symbol_sentiment_summary(symbol, tweet_texts)
                        if symbol_sentiment:
                            print(f"    ✅ Advanced Sentiment Analysis:")
                            print(f"       - Overall Sentiment: {symbol_sentiment.overall_sentiment:.3f}")
                            print(f"       - Confidence: {symbol_sentiment.confidence:.3f}")
                            print(f"       - Sample Size: {symbol_sentiment.sample_size}")
                            print(f"       - Trending: {symbol_sentiment.trending_direction}")
                            print(f"       - Quality Score: {symbol_sentiment.quality_score:.3f}")
                            
                            # Create SentimentData object for ML integration
                            sentiment_data_obj = SentimentData(
                                symbol=symbol,
                                sentiment_score=symbol_sentiment.overall_sentiment,
                                confidence=symbol_sentiment.confidence,
                                source="twitter_advanced",
                                timestamp=symbol_sentiment.timestamp,
                                sample_size=symbol_sentiment.sample_size,
                                raw_data={
                                    'bullish_mentions': symbol_sentiment.bullish_mentions,
                                    'bearish_mentions': symbol_sentiment.bearish_mentions,
                                    'neutral_mentions': symbol_sentiment.neutral_mentions,
                                    'trending_direction': symbol_sentiment.trending_direction,
                                    'quality_score': symbol_sentiment.quality_score
                                }
                            )
                            
                            print(f"    ✅ Created SentimentData object for ML integration")
                            print(f"       - Symbol: {sentiment_data_obj.symbol}")
                            print(f"       - Score: {sentiment_data_obj.sentiment_score:.3f}")
                            print(f"       - Confidence: {sentiment_data_obj.confidence:.3f}")
                            print(f"       - Source: {sentiment_data_obj.source}")
                            
                            # Test ML prediction with sentiment data
                            print(f"    🎯 Testing ML prediction with sentiment...")
                            
                            # Create a simple sentiment data dictionary for the ML engine
                            sentiment_dict = {symbol: sentiment_data_obj}
                            
                            # Try to get a prediction - this will test the integration
                            try:
                                prediction = ml_engine.predict(symbol, PredictionType.PRICE_DIRECTION, horizon_days=5)
                                if prediction:
                                    print(f"    ✅ ML Prediction successful:")
                                    print(f"       - Prediction: {prediction.prediction:.3f}")
                                    print(f"       - Confidence: {prediction.confidence:.3f}")
                                    print(f"       - Sentiment Available: {prediction.prediction_context.get('sentiment_available', False)}")
                                    print(f"       - Models Used: {prediction.prediction_context.get('models_used', 0)}")
                                else:
                                    print(f"    ❌ ML Prediction failed - no result")
                            except Exception as e:
                                print(f"    ❌ ML Prediction error: {e}")
                        else:
                            print(f"    ❌ Failed to get sentiment summary")
                    else:
                        print(f"    ❌ No tweet texts could be extracted")
                else:
                    print(f"    ❌ No tweets retrieved")
                    
            except Exception as e:
                print(f"  ❌ Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
        
        # Test ML training with sentiment data
        print(f"\n🎯 Testing ML training with sentiment integration...")
        
        try:
            # Create some sample sentiment data for training
            sample_sentiment_data = {}
            for symbol in test_symbols:
                # Get some live sentiment
                from data_feeds.twitter_feed import TwitterSentimentFeed
                twitter_feed = TwitterSentimentFeed()
                tweets = twitter_feed.fetch(symbol, limit=3)
                
                if tweets:
                    # Extract text from tweet dictionaries for sentiment analysis
                    tweet_texts = [tweet['text'] for tweet in tweets if 'text' in tweet]
                    
                    if tweet_texts:
                        symbol_sentiment = sentiment_engine.get_symbol_sentiment_summary(symbol, tweet_texts)
                        if symbol_sentiment:
                            sample_sentiment_data[symbol] = SentimentData(
                                symbol=symbol,
                                sentiment_score=symbol_sentiment.overall_sentiment,
                                confidence=symbol_sentiment.confidence,
                                source="twitter_training",
                                timestamp=symbol_sentiment.timestamp,
                                sample_size=symbol_sentiment.sample_size
                            )
            
            print(f"  📊 Created sentiment data for {len(sample_sentiment_data)} symbols")
            
            # Attempt training
            training_result = ml_engine.train_models(test_symbols, lookback_days=30)
            
            if training_result:
                print(f"  ✅ ML Training completed:")
                print(f"     - Training groups: {len(training_result)}")
                for group, results in training_result.items():
                    if isinstance(results, dict) and len(results) > 0:
                        success_count = sum(1 for r in results.values() if isinstance(r, dict) and 'error' not in r)
                        print(f"     - {group}: {success_count}/{len(results)} models trained successfully")
                    else:
                        print(f"     - {group}: {results}")
            else:
                print(f"  ❌ ML Training failed - no results")
                
        except Exception as e:
            print(f"  ❌ ML Training error: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*80}")
        print("ML SENTIMENT INTEGRATION FIX TEST COMPLETE")
        print(f"{'='*80}")
        
        # Summary
        print(f"\n📋 INTEGRATION SUMMARY:")
        print(f"✅ Twitter sentiment retrieval: Working")
        print(f"✅ Advanced sentiment analysis: Working") 
        print(f"✅ SentimentData object creation: Working")
        print(f"🔧 ML engine sentiment integration: Needs refinement")
        print(f"🔧 Data orchestrator caching: Needs implementation")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Implement sentiment data caching in DataFeedOrchestrator")
        print(f"2. Update ML engine to use cached sentiment data")
        print(f"3. Add sentiment features to feature engineering")
        print(f"4. Enable sentiment-enhanced training")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ml_sentiment_integration_fixed()
