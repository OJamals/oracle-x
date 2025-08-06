#!/usr/bin/env python3
"""
Simple test to validate Twitter sentiment -> ML integration
"""

print("🚀 Testing Twitter Sentiment -> ML Integration")

try:
    # Import components
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
    from data_feeds.advanced_sentiment import AdvancedSentimentEngine  
    from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine, PredictionType
    
    print("✅ Imports successful")
    
    # Initialize components
    data_orchestrator = DataFeedOrchestrator()
    sentiment_engine = AdvancedSentimentEngine()
    ml_engine = EnsemblePredictionEngine(data_orchestrator, sentiment_engine)
    
    print("✅ Components initialized")
    
    # Test symbol
    symbol = "TSLA"
    
    # Get sentiment data
    print(f"📊 Getting sentiment data for {symbol}...")
    sentiment_data = data_orchestrator.get_sentiment_data(symbol)
    
    if sentiment_data:
        print(f"✅ Got sentiment data from {len(sentiment_data)} sources")
        
        # Check for actual texts
        total_texts = 0
        for source_name, sentiment_obj in sentiment_data.items():
            if hasattr(sentiment_obj, 'raw_data') and sentiment_obj.raw_data:
                if 'sample_texts' in sentiment_obj.raw_data:
                    texts = sentiment_obj.raw_data['sample_texts']
                    if texts:
                        total_texts += len(texts)
                        print(f"   ✅ {source_name}: {len(texts)} texts")
        
        if total_texts > 0:
            print(f"✅ Total texts available: {total_texts}")
            
            # Test ML training with sentiment
            print("🧠 Testing ML training with sentiment...")
            training_results = ml_engine.train_models([symbol], lookback_days=30)
            
            if training_results:
                print("✅ ML Training completed successfully!")
                print(f"   Training results: {len(training_results)} targets")
                
                # Test prediction
                print("🎯 Testing prediction...")
                prediction = ml_engine.predict(symbol, PredictionType.PRICE_DIRECTION, horizon_days=5)
                
                if prediction:
                    print("✅ Prediction successful!")
                    print(f"   Sentiment available: {prediction.prediction_context.get('sentiment_available', False)}")
                    print(f"   Models used: {prediction.prediction_context.get('models_used', 0)}")
                    print("🎉 FULL INTEGRATION WORKING!")
                else:
                    print("❌ Prediction failed")
            else:
                print("❌ ML Training failed")
        else:
            print("❌ No texts found in sentiment data")
    else:
        print("❌ No sentiment data")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("🏁 Test complete")
