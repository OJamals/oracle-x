#!/usr/bin/env python3
"""
Twitter Sentiment Validation Script
Demonstrates actual Twitter posts being evaluated and their sentiment analysis
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_feeds.twitter_feed import TwitterSentimentFeed
from data_feeds.advanced_sentiment import AdvancedSentimentEngine

def validate_twitter_sentiment_feed():
    """
    Validate Twitter sentiment feed by showing actual posts and their analysis
    """
    print("=" * 80)
    print("TWITTER SENTIMENT VALIDATION - LIVE DEMONSTRATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # Initialize Twitter feed
    print("🔧 Initializing Twitter sentiment feed...")
    twitter_feed = TwitterSentimentFeed(min_likes=0, min_retweets=0, sample_size=2000)
    
    # Initialize advanced sentiment engine
    print("🔧 Initializing advanced sentiment engine...")
    sentiment_engine = AdvancedSentimentEngine()
    
    # Test symbols to search
    test_symbols = ["AAPL", "TSLA", "NVDA", "SPY", "Bitcoin"]
    
    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"🔍 TESTING TWITTER SENTIMENT FOR: {symbol}")
        print(f"{'='*60}")
        
        try:
            # Fetch tweets
            print(f"\n📡 Fetching tweets for '{symbol}'...")
            tweets = twitter_feed.fetch(symbol, limit=10)
            
            if not tweets:
                print(f"❌ No tweets found for {symbol}")
                continue
            
            print(f"✅ Found {len(tweets)} tweets for {symbol}")
            
            # Display actual tweets and their sentiment analysis
            for i, tweet in enumerate(tweets[:5]):  # Show first 5 tweets
                print(f"\n📱 TWEET #{i+1}:")
                print("-" * 40)
                print(f"Text: {tweet['text'][:200]}{'...' if len(tweet['text']) > 200 else ''}")
                print(f"Language: {tweet['lang']}")
                print(f"Tickers found: {tweet['tickers']}")
                print(f"Likes: {tweet['likes']}, Retweets: {tweet['retweets']}")
                
                # Show VADER sentiment
                vader_sentiment = tweet['sentiment']
                print(f"VADER Sentiment:")
                print(f"  - Positive: {vader_sentiment['pos']:.3f}")
                print(f"  - Negative: {vader_sentiment['neg']:.3f}")
                print(f"  - Neutral: {vader_sentiment['neu']:.3f}")
                print(f"  - Compound: {vader_sentiment['compound']:.3f}")
                
                # Show TextBlob sentiment if available
                if 'textblob_polarity' in vader_sentiment and vader_sentiment['textblob_polarity'] is not None:
                    print(f"TextBlob Sentiment:")
                    print(f"  - Polarity: {vader_sentiment['textblob_polarity']:.3f}")
                    print(f"  - Subjectivity: {vader_sentiment['textblob_subjectivity']:.3f}")
                
                # Use advanced sentiment engine for comparison
                try:
                    advanced_result = sentiment_engine.analyze_text(tweet['text'], symbol, "twitter")
                    print(f"Advanced Engine (VADER+FinBERT):")
                    print(f"  - Ensemble Score: {advanced_result.ensemble_score:.3f}")
                    print(f"  - VADER Score: {advanced_result.vader_score:.3f}")
                    print(f"  - FinBERT Score: {advanced_result.finbert_score:.3f}")
                    print(f"  - Confidence: {advanced_result.confidence:.3f}")
                except Exception as e:
                    print(f"Advanced Engine Error: {e}")
                
                print()
            
            # Aggregate sentiment for symbol
            print(f"\n📊 AGGREGATE SENTIMENT ANALYSIS FOR {symbol}:")
            print("-" * 50)
            
            if tweets:
                # Calculate averages
                total_compound = sum(t['sentiment']['compound'] for t in tweets)
                avg_compound = total_compound / len(tweets)
                
                positive_tweets = sum(1 for t in tweets if t['sentiment']['compound'] > 0.1)
                negative_tweets = sum(1 for t in tweets if t['sentiment']['compound'] < -0.1)
                neutral_tweets = len(tweets) - positive_tweets - negative_tweets
                
                print(f"Total tweets analyzed: {len(tweets)}")
                print(f"Average compound sentiment: {avg_compound:.3f}")
                print(f"Positive tweets: {positive_tweets} ({positive_tweets/len(tweets)*100:.1f}%)")
                print(f"Negative tweets: {negative_tweets} ({negative_tweets/len(tweets)*100:.1f}%)")
                print(f"Neutral tweets: {neutral_tweets} ({neutral_tweets/len(tweets)*100:.1f}%)")
                
                # Use advanced sentiment engine for symbol aggregate
                try:
                    symbol_sentiment = sentiment_engine.get_symbol_sentiment_summary(symbol, tweets)
                    if symbol_sentiment:
                        print(f"\nAdvanced Engine Symbol Summary:")
                        print(f"  - Overall Sentiment: {symbol_sentiment.overall_sentiment:.3f}")
                        print(f"  - Confidence: {symbol_sentiment.confidence:.3f}")
                        print(f"  - Sample Size: {symbol_sentiment.sample_size}")
                        print(f"  - Bullish: {symbol_sentiment.bullish_mentions}")
                        print(f"  - Bearish: {symbol_sentiment.bearish_mentions}")
                        print(f"  - Neutral: {symbol_sentiment.neutral_mentions}")
                        print(f"  - Trending: {symbol_sentiment.trending_direction}")
                        print(f"  - Quality Score: {symbol_sentiment.quality_score:.3f}")
                        print(f"  - Timestamp: {symbol_sentiment.timestamp}")
                except Exception as e:
                    print(f"Advanced Engine Symbol Summary Error: {e}")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("TWITTER SENTIMENT VALIDATION COMPLETE")
    print(f"{'='*80}")

def test_ml_sentiment_integration():
    """
    Test that ML models are actively collecting sentiment data for training
    """
    print("\n" + "=" * 80)
    print("ML SENTIMENT INTEGRATION VALIDATION")
    print("=" * 80)
    
    try:
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine
        
        print("🔧 Initializing data orchestrator...")
        orchestrator = DataFeedOrchestrator()
        
        print("🔧 Initializing sentiment engine...")
        sentiment_engine = AdvancedSentimentEngine()
        
        print("🔧 Initializing ML prediction engine...")
        ml_engine = EnsemblePredictionEngine(orchestrator, sentiment_engine)
        
        # Test symbols
        test_symbols = ["AAPL", "TSLA", "NVDA"]
        
        print(f"\n📊 Testing ML sentiment data collection for: {', '.join(test_symbols)}")
        
        for symbol in test_symbols:
            print(f"\n🔍 Testing {symbol}:")
            
            # Test data orchestrator sentiment access
            try:
                sentiment_data = orchestrator.get_sentiment_data(symbol)
                if sentiment_data and symbol in sentiment_data:
                    symbol_sentiment = sentiment_data[symbol]
                    print(f"  ✅ Data Orchestrator: Retrieved sentiment data")
                    print(f"     - Score: {symbol_sentiment.sentiment_score:.3f}")
                    print(f"     - Confidence: {symbol_sentiment.confidence:.3f}")
                    print(f"     - Source: {symbol_sentiment.source}")
                    print(f"     - Sample Size: {getattr(symbol_sentiment, 'sample_size', 'N/A')}")
                else:
                    print(f"  ❌ Data Orchestrator: No sentiment data available")
            except Exception as e:
                print(f"  ❌ Data Orchestrator Error: {e}")
            
            # Test ML engine sentiment integration
            try:
                from oracle_engine.ensemble_ml_engine import PredictionType
                prediction = ml_engine.predict(symbol, PredictionType.PRICE_DIRECTION, horizon_days=5)
                if prediction:
                    print(f"  ✅ ML Engine: Successfully generated prediction with sentiment")
                    print(f"     - Prediction: {prediction.prediction:.3f}")
                    print(f"     - Confidence: {prediction.confidence:.3f}")
                    print(f"     - Sentiment Available: {prediction.prediction_context.get('sentiment_available', False)}")
                else:
                    print(f"  ❌ ML Engine: Failed to generate prediction")
            except Exception as e:
                print(f"  ❌ ML Engine Error: {e}")
        
        # Test ML training with sentiment data
        print(f"\n🎯 Testing ML training with sentiment data...")
        try:
            training_result = ml_engine.train_models(test_symbols, lookback_days=30)
            if training_result:
                print(f"  ✅ ML Training: Successfully trained models with sentiment data")
                print(f"     - Training results: {len(training_result)} model groups trained")
                for model_group, result in training_result.items():
                    print(f"     - {model_group}: {len(result)} models")
            else:
                print(f"  ❌ ML Training: Failed to train models")
        except Exception as e:
            print(f"  ❌ ML Training Error: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ ML Integration Test Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting comprehensive Twitter sentiment validation...")
    
    # Run Twitter sentiment validation
    validate_twitter_sentiment_feed()
    
    # Test ML integration
    test_ml_sentiment_integration()
    
    print("\n🎉 Validation complete!")
