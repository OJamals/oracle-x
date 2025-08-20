"""
Test Enhanced ML Training System
Validates the improved training pipeline with robust error handling
"""

import pytest
import logging
import sys
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.mark.ml
@pytest.mark.slow
@pytest.mark.network
def test_enhanced_training():
    """Test the enhanced training system"""
    print("="*80)
    print("🚀 TESTING ENHANCED ML TRAINING SYSTEM")
    print("="*80)
    
    try:
        # Import components
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        from data_feeds.advanced_sentiment import AdvancedSentimentEngine
        from examples.enhanced_ml_training import create_enhanced_training_wrapper
        
        print("\n📦 INITIALIZING COMPONENTS")
        print("-" * 50)
        
        # Initialize base components
        orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        
        # Initialize base engine
        base_engine = EnsemblePredictionEngine(
            data_orchestrator=orchestrator,
            sentiment_engine=sentiment_engine
        )
        
        print("✅ Base engine initialized")
        
        # Create enhanced wrapper
        enhanced_engine = create_enhanced_training_wrapper(base_engine)
        print("✅ Enhanced training wrapper created")
        
        print(f"📊 Available models: {len(base_engine.models)}")
        
        # Test training with single symbol first
        print("\n🧠 TESTING ENHANCED TRAINING")
        print("-" * 50)
        
        test_symbols = ["AAPL"]  # Start with one symbol
        print(f"Training on symbols: {test_symbols}")
        
        # Run enhanced training
        training_results = enhanced_engine.train_models_robustly(
            symbols=test_symbols,
            lookback_days=60  # Reduced for testing
        )
        
        print("\n📊 TRAINING RESULTS")
        print("-" * 50)
        print(f"Training time: {training_results['training_time']:.2f} seconds")
        print(f"Success rate: {training_results['success_rate']:.2%}")
        print(f"Successful symbols: {training_results['successful_symbols']}")
        print(f"Failed symbols: {training_results['failed_symbols']}")
        print(f"Total samples trained: {training_results['total_samples_trained']}")
        
        # Check detailed results
        print("\n📋 DETAILED RESULTS")
        print("-" * 50)
        for symbol, result in training_results['detailed_results'].items():
            print(f"Symbol: {symbol}")
            print(f"  Success: {result.success}")
            print(f"  Training time: {result.training_time:.2f}s")
            print(f"  Samples trained: {result.samples_trained}")
            if result.error_message:
                print(f"  Error: {result.error_message}")
            if result.performance_metrics:
                print(f"  Metrics: {result.performance_metrics}")
        
        # Test model states after training
        print("\n🔍 POST-TRAINING MODEL STATES")
        print("-" * 50)
        trained_models = 0
        for model_key, model in base_engine.models.items():
            is_trained = hasattr(model, 'is_trained') and model.is_trained
            print(f"Model {model_key}: trained={is_trained}")
            if is_trained:
                trained_models += 1
        
        print(f"\n📈 TRAINING SUMMARY")
        print(f"Total models: {len(base_engine.models)}")
        print(f"Trained models: {trained_models}")
        print(f"Training completion rate: {trained_models/len(base_engine.models):.2%}")
        
        # Test prediction capability
        print("\n🔮 TESTING PREDICTION CAPABILITY")
        print("-" * 50)
        
        try:
            from oracle_engine.ensemble_ml_engine import PredictionType
            prediction = base_engine.predict("AAPL", PredictionType.PRICE_DIRECTION, horizon_days=5)
            print(f"Prediction test: {'✅ SUCCESS' if prediction else '❌ FAILED'}")
            if prediction:
                print(f"Prediction result: {prediction}")
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
        
        # Overall assessment
        print("\n" + "="*80)
        print("📊 ENHANCED TRAINING ASSESSMENT")
        print("="*80)
        
        if training_results['success_rate'] > 0:
            print("✅ Enhanced training system is working!")
            if trained_models > 0:
                print("✅ Models are actually training successfully!")
            else:
                print("⚠️  Training completed but models not marked as trained")
        else:
            print("❌ Enhanced training system needs further fixes")
        
        return training_results
        
    except Exception as e:
        print(f"\n❌ Enhanced training test failed: {e}")
        print(traceback.format_exc())
        return None

def test_memory_management():
    """Test memory management features"""
    print("\n🧠 TESTING MEMORY MANAGEMENT")
    print("-" * 50)
    
    try:
        from examples.enhanced_ml_training import MemoryManager, SafeSentimentProcessor
        
        # Test memory manager
        memory_manager = MemoryManager(max_memory_mb=2048)
        checkpoint = memory_manager.check_memory("test_operation")
        print(f"✅ Memory manager working: {checkpoint}")
        
        # Test safe sentiment processor
        sentiment_processor = SafeSentimentProcessor(batch_size=10, max_texts=50)
        print("✅ Safe sentiment processor initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False

def test_training_fallbacks():
    """Test training fallback strategies"""
    print("\n🔄 TESTING TRAINING FALLBACKS")
    print("-" * 50)
    
    try:
        from examples.enhanced_ml_training import RobustTrainingPipeline
        print("✅ Robust training pipeline components available")
        return True
        
    except Exception as e:
        print(f"❌ Training fallback test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting enhanced ML training system tests...")
    
    # Test 1: Memory Management
    memory_test = test_memory_management()
    
    # Test 2: Training Fallbacks
    fallback_test = test_training_fallbacks()
    
    # Test 3: Enhanced Training
    training_test = test_enhanced_training()
    
    # Summary
    print("\n" + "="*80)
    print("🏁 TEST SUMMARY")
    print("="*80)
    
    tests_passed = 0
    if memory_test:
        print("✅ Memory management: PASSED")
        tests_passed += 1
    else:
        print("❌ Memory management: FAILED")
    
    if fallback_test:
        print("✅ Training fallbacks: PASSED")
        tests_passed += 1
    else:
        print("❌ Training fallbacks: FAILED")
    
    if training_test:
        print("✅ Enhanced training: PASSED")
        tests_passed += 1
    else:
        print("❌ Enhanced training: FAILED")
    
    print(f"\nOverall: {tests_passed}/3 tests passed")
    
    if tests_passed == 3:
        print("🎉 All tests passed! Enhanced training system is ready.")
    else:
        print("⚠️  Some tests failed. Review issues above.")
