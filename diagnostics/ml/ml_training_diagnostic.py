#!/usr/bin/env python3
"""
ML Training Diagnostic Script - UPDATED
Tests fixes for ML training and Twitter sentiment integration
"""

import sys
import logging
from datetime import datetime
sys.path.append('/Users/omar/Documents/Projects/oracle-x')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce verbose logging for external libraries
logging.getLogger('httpx').setLevel(logging.WARNING)  # Suppress verbose URL logs
logging.getLogger('twscrape').setLevel(logging.WARNING)  # Suppress Twitter warnings

def test_fixed_ml_training_pipeline():
    """Test the FIXED ML training pipeline with Twitter sentiment"""
    
    print("� Testing FIXED ML Training Pipeline with Twitter Sentiment")
    print("=" * 70)
    
    try:
        # Import components
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        from data_feeds.advanced_sentiment import AdvancedSentimentEngine
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine, PredictionType
        
        print("✅ Imported all components successfully")
        
        # Initialize components
        data_orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        
        print("✅ Initialized data orchestrator and sentiment engine")
        print(f"📊 Available adapters: {list(data_orchestrator.adapters.keys())}")
        
        # Create ML engine
        ml_engine = EnsemblePredictionEngine(data_orchestrator, sentiment_engine)
        
        print("✅ Created EnsemblePredictionEngine")
        print(f"📊 Available models: {list(ml_engine.models.keys())}")
        
        # Test Twitter + Reddit sentiment integration
        test_symbol = "TSLA"
        
        print(f"\n🐦 Testing Twitter + Reddit sentiment for {test_symbol}")
        
        # Get sentiment data from both sources
        sentiment_data = data_orchestrator.get_sentiment_data(test_symbol)
        
        print(f"📊 Sentiment data sources: {list(sentiment_data.keys())}")
        
        for source, data in sentiment_data.items():
            print(f"✅ {source}:")
            print(f"   Sentiment Score: {data.sentiment_score:.3f}")
            print(f"   Confidence: {data.confidence:.3f}")
            print(f"   Sample Size: {data.sample_size}")
            if data.raw_data and 'sample_texts' in data.raw_data:
                sample_texts = data.raw_data['sample_texts']
                print(f"   Sample texts: {len(sample_texts)}")
                if sample_texts:
                    print(f"   First sample: {sample_texts[0][:100]}...")
        
        # Test ML training with sentiment data
        print(f"\n🤖 Testing ML training with sentiment data for {test_symbol}")
        
        training_results = ml_engine.train_models([test_symbol], lookback_days=60)
        
        print(f"📈 Training results: {training_results}")
        
        # Check if models are now trained
        trained_models = [key for key, model in ml_engine.models.items() 
                         if hasattr(model, 'is_trained') and model.is_trained]
        print(f"🤖 Trained models after training: {trained_models}")
        
        # Test prediction with sentiment
        print(f"\n🔮 Testing prediction with sentiment data for {test_symbol}")
        
        result = ml_engine.predict(test_symbol, PredictionType.PRICE_DIRECTION, horizon_days=5)
        
        if result:
            print(f"✅ Prediction successful!")
            print(f"   Symbol: {result.symbol}")
            print(f"   Prediction: {result.prediction}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Models used: {result.prediction_context.get('models_used', 0)}")
            print(f"   Sentiment available: {result.prediction_context.get('sentiment_available', False)}")
            print(f"   Data quality: {result.data_quality_score:.3f}")
        else:
            print("❌ Prediction failed")
            
        # Test feature engineering with sentiment
        print(f"\n⚙️ Testing enhanced feature engineering with sentiment")
        
        if ml_engine.feature_engineer:
            # Get some test data
            market_data = data_orchestrator.get_market_data(test_symbol, period="30d", interval="1d")
            sentiment_summary = sentiment_engine.get_symbol_sentiment_summary(test_symbol, [])
            
            if market_data and not market_data.data.empty:
                test_data = {test_symbol: market_data.data}
                test_sentiment = {test_symbol: sentiment_summary} if sentiment_summary else {}
                
                features_df = ml_engine.feature_engineer.engineer_features(test_data, test_sentiment)
                
                print(f"✅ Feature engineering successful!")
                print(f"   Features shape: {features_df.shape}")
                print(f"   Feature columns: {list(features_df.columns)}")
                
                # Check for target columns
                target_cols = [col for col in features_df.columns if col.startswith('target_')]
                print(f"   Target columns: {target_cols}")
                
                # Check for sentiment columns
                sentiment_cols = [col for col in features_df.columns if 'sentiment' in col]
                print(f"   Sentiment columns: {sentiment_cols}")
                
            else:
                print("❌ No market data available for feature engineering test")
        else:
            print("❌ No feature engineer available")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_ml_training_pipeline()
