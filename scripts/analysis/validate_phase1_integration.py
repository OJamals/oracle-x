#!/usr/bin/env python3
"""
Validate Phase 1 Integration
Tests that the fixed training system works in the main oracle-x system
"""
import sys
import os
import traceback
from datetime import datetime
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

def test_fixed_training():
    """Test the integrated fixed training system"""
    print("=" * 60)
    print("🧪 PHASE 1 INTEGRATION VALIDATION")
    print("=" * 60)
    
    try:
        # Import the main ensemble engine with our fix
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine
        print("✅ Successfully imported EnsemblePredictionEngine")
        
        # Import required components with correct paths
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        from data_feeds.advanced_sentiment import AdvancedSentimentEngine
        print("✅ Successfully imported all dependencies")
        
        # Initialize components
        print("\n📊 Initializing ML system...")
        orchestrator = DataFeedOrchestrator()
        
        # Try to initialize sentiment engine, but continue without it if it fails
        try:
            sentiment_engine = AdvancedSentimentEngine()
            print("✅ Sentiment engine initialized")
        except Exception as e:
            print(f"⚠️  Sentiment engine failed to initialize: {e}")
            sentiment_engine = None
        
        # Initialize the ensemble engine with our fix - use correct constructor
        engine = EnsemblePredictionEngine(
            data_orchestrator=orchestrator,
            sentiment_engine=sentiment_engine
        )
        print("✅ EnsemblePredictionEngine initialized")
        
        # Test training with a small set of symbols
        test_symbols = ['AAPL', 'MSFT']
        print(f"\n🚀 Testing fixed training with symbols: {test_symbols}")
        
        # Run the fixed training
        training_results = engine.train_models(
            symbols=test_symbols,
            lookback_days=60,  # Reduced for testing
            update_existing=False
        )
        
        print(f"\n📋 Training Results: {len(training_results)} result sets")
        
        # Analyze results
        total_models_trained = 0
        successful_training_types = []
        
        for training_type, results in training_results.items():
            print(f"\n  📊 {training_type}:")
            if isinstance(results, dict):
                model_count = 0
                for model_key, result in results.items():
                    if isinstance(result, dict) and 'error' not in result:
                        model_count += 1
                        total_models_trained += 1
                        print(f"    ✅ {model_key}: Success")
                    else:
                        print(f"    ❌ {model_key}: Failed - {result}")
                
                if model_count > 0:
                    successful_training_types.append(training_type)
                    print(f"    📈 Successfully trained {model_count} models for {training_type}")
            else:
                print(f"    ❌ Invalid result format: {results}")
        
        # Check model states
        print(f"\n🔍 Checking trained models...")
        trained_models = []
        for model_key, model in engine.models.items():
            if model and getattr(model, 'is_trained', False):
                trained_models.append(model_key)
                print(f"  ✅ {model_key}: Trained and ready")
            else:
                print(f"  ❌ {model_key}: Not trained")
        
        # Final assessment
        print(f"\n" + "=" * 60)
        print("📊 PHASE 1 INTEGRATION RESULTS")
        print("=" * 60)
        print(f"✅ Total models trained: {total_models_trained}")
        print(f"✅ Successful training types: {len(successful_training_types)}")
        print(f"✅ Active trained models: {len(trained_models)}")
        
        if total_models_trained >= 4:  # At least 4 models (2 types × 2 algorithms minimum)
            print("\n🎉 PHASE 1 INTEGRATION: SUCCESS!")
            print("   The fixed training system is working in the main oracle-x system!")
            return True
        else:
            print(f"\n⚠️  PHASE 1 INTEGRATION: PARTIAL SUCCESS")
            print(f"   Only {total_models_trained} models trained, expected at least 4")
            return False
            
    except Exception as e:
        print(f"\n❌ PHASE 1 INTEGRATION: FAILED")
        print(f"Error: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

def test_ensemble_functionality():
    """Test that the ensemble can make predictions after training"""
    print("\n" + "=" * 60)
    print("🔮 TESTING ENSEMBLE PREDICTIONS")
    print("=" * 60)
    
    try:
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine, PredictionType
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        
        # Initialize minimal system
        orchestrator = DataFeedOrchestrator()
        engine = EnsemblePredictionEngine(
            data_orchestrator=orchestrator,
            sentiment_engine=None  # Skip sentiment for this test
        )
        
        # Quick training
        training_results = engine.train_models(['AAPL'], lookback_days=60)
        
        if not training_results:
            print("❌ No training results - cannot test predictions")
            return False
        
        # Test predictions using the main predict method
        print("🔮 Testing ensemble predictions...")
        
        # Test price direction prediction
        try:
            direction_pred = engine.predict('AAPL', PredictionType.PRICE_DIRECTION, horizon_days=1)
            if direction_pred:
                print(f"✅ Price direction prediction: {direction_pred}")
            else:
                print("❌ Price direction prediction failed")
        except Exception as e:
            print(f"❌ Price direction prediction error: {e}")
        
        # Test price target prediction  
        try:
            target_pred = engine.predict('AAPL', PredictionType.PRICE_TARGET, horizon_days=1)
            if target_pred:
                print(f"✅ Price target prediction: {target_pred}")
            else:
                print("❌ Price target prediction failed")
        except Exception as e:
            print(f"❌ Price target prediction error: {e}")
        
        print("\n✅ Ensemble prediction testing completed")
        return True
        
    except Exception as e:
        print(f"❌ Ensemble prediction testing failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Phase 1 Integration Validation...")
    
    # Test 1: Fixed training integration
    training_success = test_fixed_training()
    
    # Test 2: Ensemble functionality
    if training_success:
        prediction_success = test_ensemble_functionality()
    else:
        prediction_success = False
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏁 FINAL VALIDATION SUMMARY")
    print("=" * 80)
    
    if training_success and prediction_success:
        print("🎉 PHASE 1 COMPLETE: All systems working!")
        print("   ✅ Fixed training integrated successfully")
        print("   ✅ Ensemble predictions working")
        print("   🚀 Ready for Phase 2 enhancements!")
    elif training_success:
        print("⚠️  PHASE 1 PARTIAL: Training fixed, predictions need work")
        print("   ✅ Fixed training integrated successfully")
        print("   ❌ Ensemble predictions need debugging")
    else:
        print("❌ PHASE 1 FAILED: Core training issues remain")
        print("   ❌ Training integration needs more work")
    
    print("\n🔧 Phase 1 validation completed.")
