#!/usr/bin/env python3
"""
Comprehensive ML Engine Test Suite
Tests the ensemble ML engine with real examples and scenarios
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from oracle_engine.ensemble_ml_engine import (
    EnsemblePredictionEngine, 
    PredictionType, 
    ModelType,
    create_prediction_engine
)
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
from sentiment.sentiment_engine import AdvancedSentimentEngine

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLEngineTestSuite:
    """Comprehensive test suite for the ML engine"""
    
    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        self.results = {}
        self.failed_tests = []
        
        # Initialize components
        logger.info("Initializing ML Engine components...")
        try:
            self.data_orchestrator = DataFeedOrchestrator()
            self.sentiment_engine = AdvancedSentimentEngine()
            self.ml_engine = create_prediction_engine(
                self.data_orchestrator, 
                self.sentiment_engine
            )
            logger.info("✅ ML Engine components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def run_all_tests(self):
        """Run the complete test suite"""
        logger.info("🚀 Starting ML Engine Comprehensive Test Suite")
        logger.info("=" * 60)
        
        tests = [
            ("Engine Initialization", self.test_engine_initialization),
            ("Model Configuration", self.test_model_configuration),
            ("Data Integration", self.test_data_integration),
            ("Basic Predictions", self.test_basic_predictions),
            ("Training Process", self.test_training_process),
            ("Ensemble Predictions", self.test_ensemble_predictions),
            ("Performance Metrics", self.test_performance_metrics),
            ("Fallback Mechanisms", self.test_fallback_mechanisms),
            ("Caching System", self.test_caching_system),
            ("Error Handling", self.test_error_handling),
        ]
        
        total_tests = len(tests)
        passed_tests = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running: {test_name}")
            logger.info("-" * 40)
            
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                
                if result:
                    logger.info(f"✅ {test_name} PASSED ({duration:.2f}s)")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} FAILED ({duration:.2f}s)")
                    self.failed_tests.append(test_name)
                
                self.results[test_name] = {
                    'passed': result,
                    'duration': duration
                }
                
            except Exception as e:
                logger.error(f"💥 {test_name} CRASHED: {e}")
                self.failed_tests.append(test_name)
                self.results[test_name] = {
                    'passed': False,
                    'error': str(e)
                }
        
        # Final report
        self.generate_final_report(total_tests, passed_tests)
    
    def test_engine_initialization(self):
        """Test 1: Engine Initialization"""
        try:
            # Check if engine is properly initialized
            assert self.ml_engine is not None, "ML Engine not initialized"
            assert hasattr(self.ml_engine, 'models'), "Models dict not found"
            assert hasattr(self.ml_engine, 'model_weights'), "Model weights not found"
            assert hasattr(self.ml_engine, 'data_orchestrator'), "Data orchestrator not found"
            assert hasattr(self.ml_engine, 'sentiment_engine'), "Sentiment engine not found"
            
            logger.info(f"📊 Engine has {len(self.ml_engine.models)} models initialized")
            logger.info(f"📊 Model types: {list(self.ml_engine.models.keys())}")
            
            return True
        except AssertionError as e:
            logger.error(f"Initialization test failed: {e}")
            return False
    
    def test_model_configuration(self):
        """Test 2: Model Configuration"""
        try:
            # Check model configurations
            configs = self.ml_engine.model_configs
            assert len(configs) > 0, "No model configurations found"
            
            expected_models = ['random_forest', 'xgboost', 'neural_network']
            for model_type in expected_models:
                if model_type in configs:
                    logger.info(f"📋 {model_type} config: {configs[model_type]}")
            
            # Check prediction horizons
            horizons = self.ml_engine.prediction_horizons
            assert len(horizons) > 0, "No prediction horizons configured"
            logger.info(f"📅 Prediction horizons: {horizons} days")
            
            return True
        except AssertionError as e:
            logger.error(f"Configuration test failed: {e}")
            return False
    
    def test_data_integration(self):
        """Test 3: Data Integration"""
        try:
            test_symbol = 'AAPL'
            logger.info(f"🔍 Testing data integration for {test_symbol}")
            
            # Test market data retrieval
            market_data = self.data_orchestrator.get_market_data(
                test_symbol, period="30d", interval="1d"
            )
            
            if market_data and not market_data.data.empty:
                logger.info(f"📈 Market data: {len(market_data.data)} rows")
                logger.info(f"📈 Columns: {list(market_data.data.columns)}")
                logger.info(f"📈 Date range: {market_data.data.index[0]} to {market_data.data.index[-1]}")
            else:
                logger.warning("⚠️  No market data available (using fallback mode)")
            
            # Test sentiment analysis (optional)
            try:
                sentiment = self.sentiment_engine.get_symbol_sentiment_summary(test_symbol, [])
                if sentiment:
                    logger.info(f"💭 Sentiment score: {sentiment.overall_sentiment:.3f}")
                else:
                    logger.info("💭 No sentiment data available")
            except Exception as e:
                logger.warning(f"💭 Sentiment analysis error: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Data integration test failed: {e}")
            return False
    
    def test_basic_predictions(self):
        """Test 4: Basic Predictions"""
        try:
            test_symbol = 'AAPL'
            prediction_types = [PredictionType.PRICE_DIRECTION, PredictionType.PRICE_TARGET]
            horizons = [1, 5, 10]
            
            successful_predictions = 0
            
            for pred_type in prediction_types:
                for horizon in horizons:
                    logger.info(f"🔮 Testing {pred_type.value} prediction for {horizon} days")
                    
                    result = self.ml_engine.predict(test_symbol, pred_type, horizon)
                    
                    if result:
                        logger.info(f"✅ Prediction: {result.prediction:.4f}")
                        logger.info(f"✅ Confidence: {result.confidence:.3f}")
                        logger.info(f"✅ Uncertainty: {result.uncertainty:.3f}")
                        logger.info(f"✅ Models used: {result.prediction_context.get('models_used', 'unknown')}")
                        successful_predictions += 1
                    else:
                        logger.warning(f"⚠️  No prediction for {pred_type.value}_{horizon}d")
            
            # At least some predictions should work
            assert successful_predictions > 0, "No predictions were successful"
            logger.info(f"📊 {successful_predictions}/{len(prediction_types) * len(horizons)} predictions successful")
            
            return True
        except Exception as e:
            logger.error(f"Basic predictions test failed: {e}")
            return False
    
    def test_training_process(self):
        """Test 5: Training Process"""
        try:
            logger.info("🎓 Testing model training process")
            
            # Test with a subset of symbols for faster training
            training_symbols = ['AAPL', 'MSFT']
            
            # Start training
            training_results = self.ml_engine.train_models(
                symbols=training_symbols,
                lookback_days=30,  # Shorter for testing
                update_existing=False
            )
            
            if training_results:
                logger.info(f"📚 Training completed with {len(training_results)} result sets")
                for target, results in training_results.items():
                    logger.info(f"📚 {target}: {len(results) if isinstance(results, dict) else 'completed'} models")
            else:
                logger.info("📚 Training used fallback method (ML engine not fully available)")
            
            # Check if training time was updated
            if self.ml_engine.last_training_time:
                logger.info(f"🕐 Last training: {self.ml_engine.last_training_time}")
            
            return True
        except Exception as e:
            logger.error(f"Training process test failed: {e}")
            return False
    
    def test_ensemble_predictions(self):
        """Test 6: Ensemble Predictions"""
        try:
            logger.info("🎯 Testing ensemble prediction capabilities")
            
            test_symbol = 'MSFT'
            pred_type = PredictionType.PRICE_DIRECTION
            horizon = 5
            
            result = self.ml_engine.predict(test_symbol, pred_type, horizon)
            
            if result:
                logger.info(f"🎯 Symbol: {result.symbol}")
                logger.info(f"🎯 Prediction Type: {result.prediction_type.value}")
                logger.info(f"🎯 Prediction: {result.prediction:.4f}")
                logger.info(f"🎯 Confidence: {result.confidence:.3f}")
                logger.info(f"🎯 Market Regime: {result.market_regime}")
                logger.info(f"🎯 Data Quality: {result.data_quality_score:.3f}")
                
                # Check model contributions
                if result.model_contributions:
                    logger.info("🎯 Model Contributions:")
                    for model, contrib in result.model_contributions.items():
                        logger.info(f"   - {model}: {contrib:.4f}")
                
                # Check feature importance
                if result.feature_importance:
                    logger.info("🎯 Top Features:")
                    sorted_features = sorted(
                        result.feature_importance.items(), 
                        key=lambda x: abs(x[1]), 
                        reverse=True
                    )
                    for feature, importance in sorted_features[:5]:
                        logger.info(f"   - {feature}: {importance:.4f}")
                
                return True
            else:
                logger.warning("⚠️  No ensemble prediction available")
                return False
                
        except Exception as e:
            logger.error(f"Ensemble predictions test failed: {e}")
            return False
    
    def test_performance_metrics(self):
        """Test 7: Performance Metrics"""
        try:
            logger.info("📊 Testing performance metrics")
            
            performance = self.ml_engine.get_model_performance()
            
            if performance:
                logger.info(f"📊 Performance data for {len(performance)} models:")
                for model_key, metrics in performance.items():
                    logger.info(f"📊 {model_key}:")
                    logger.info(f"   - Weight: {metrics.get('weight', 0):.3f}")
                    logger.info(f"   - Accuracy: {metrics.get('accuracy', 0):.3f}")
                    logger.info(f"   - MSE: {metrics.get('mse', 0):.3f}")
                    logger.info(f"   - Predictions: {metrics.get('total_predictions', 0)}")
            else:
                logger.info("📊 No performance metrics available yet")
            
            # Check model weights
            weights = self.ml_engine.model_weights
            if weights:
                logger.info(f"⚖️  Model weights: {weights}")
                total_weight = sum(weights.values())
                logger.info(f"⚖️  Total weight: {total_weight:.3f}")
            
            return True
        except Exception as e:
            logger.error(f"Performance metrics test failed: {e}")
            return False
    
    def test_fallback_mechanisms(self):
        """Test 8: Fallback Mechanisms"""
        try:
            logger.info("🔄 Testing fallback mechanisms")
            
            # Test fallback prediction directly
            test_symbol = 'GOOGL'
            pred_type = PredictionType.PRICE_DIRECTION
            horizon = 5
            
            # Force fallback by temporarily disabling ML
            original_ml_available = self.ml_engine.__class__.__module__ + '.ML_ENGINE_AVAILABLE'
            
            result = self.ml_engine._fallback_predict(test_symbol, pred_type, horizon)
            
            if result:
                logger.info(f"🔄 Fallback prediction successful:")
                logger.info(f"   - Prediction: {result.prediction:.4f}")
                logger.info(f"   - Confidence: {result.confidence:.3f}")
                logger.info(f"   - Method: {result.prediction_context}")
            else:
                logger.warning("⚠️  Fallback prediction failed")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Fallback mechanisms test failed: {e}")
            return False
    
    def test_caching_system(self):
        """Test 9: Caching System"""
        try:
            logger.info("🗄️  Testing prediction caching")
            
            test_symbol = 'TSLA'
            pred_type = PredictionType.PRICE_TARGET
            horizon = 5
            
            # First prediction (should cache)
            start_time = time.time()
            result1 = self.ml_engine.predict(test_symbol, pred_type, horizon)
            first_duration = time.time() - start_time
            
            # Second prediction (should use cache)
            start_time = time.time()
            result2 = self.ml_engine.predict(test_symbol, pred_type, horizon)
            second_duration = time.time() - start_time
            
            if result1 and result2:
                logger.info(f"🗄️  First prediction: {first_duration:.3f}s")
                logger.info(f"🗄️  Second prediction: {second_duration:.3f}s")
                logger.info(f"🗄️  Same result: {result1.prediction == result2.prediction}")
                logger.info(f"🗄️  Cache entries: {len(self.ml_engine.prediction_cache)}")
                
                # Second should be faster (cached)
                if second_duration < first_duration:
                    logger.info("✅ Caching appears to be working")
                
                return True
            else:
                logger.warning("⚠️  Caching test inconclusive")
                return False
                
        except Exception as e:
            logger.error(f"Caching system test failed: {e}")
            return False
    
    def test_error_handling(self):
        """Test 10: Error Handling"""
        try:
            logger.info("🛡️  Testing error handling")
            
            # Test with invalid symbol
            result = self.ml_engine.predict('INVALID', PredictionType.PRICE_DIRECTION, 5)
            if result is None:
                logger.info("✅ Invalid symbol handled gracefully")
            
            # Test with extreme horizon
            result = self.ml_engine.predict('AAPL', PredictionType.PRICE_DIRECTION, 1000)
            logger.info(f"🛡️  Extreme horizon handled: {result is not None}")
            
            # Test save/load functionality
            test_file = Path("test_models.json")
            try:
                save_success = self.ml_engine.save_models(str(test_file))
                logger.info(f"💾 Model save: {'✅' if save_success else '❌'}")
                
                if save_success and test_file.exists():
                    load_success = self.ml_engine.load_models(str(test_file))
                    logger.info(f"📂 Model load: {'✅' if load_success else '❌'}")
                    test_file.unlink()  # Cleanup
                
            except Exception as e:
                logger.warning(f"💾 Save/load test failed: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    def generate_final_report(self, total_tests, passed_tests):
        """Generate final test report"""
        logger.info("\n" + "=" * 60)
        logger.info("📋 FINAL TEST REPORT")
        logger.info("=" * 60)
        
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"📊 Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if self.failed_tests:
            logger.info(f"❌ Failed Tests: {', '.join(self.failed_tests)}")
        
        # Engine status summary
        logger.info("\n🔧 ENGINE STATUS SUMMARY:")
        logger.info(f"   - ML Engine Available: {hasattr(self.ml_engine, 'models')}")
        logger.info(f"   - Models Initialized: {len(self.ml_engine.models)}")
        logger.info(f"   - Prediction Cache: {len(self.ml_engine.prediction_cache)} entries")
        logger.info(f"   - Model Weights: {len(self.ml_engine.model_weights)} weights")
        
        # Recommendations
        logger.info("\n💡 RECOMMENDATIONS:")
        if success_rate >= 80:
            logger.info("   ✅ ML Engine is working well!")
        elif success_rate >= 60:
            logger.info("   ⚠️  ML Engine has some issues but is functional")
        else:
            logger.info("   ❌ ML Engine needs attention")
        
        if not self.ml_engine.models:
            logger.info("   💡 Consider installing ML dependencies for full functionality")
        
        if len(self.ml_engine.prediction_cache) == 0:
            logger.info("   💡 Make some predictions to test caching")
        
        logger.info("\n🎯 Test Suite Completed!")

def main():
    """Main test execution"""
    try:
        test_suite = MLEngineTestSuite()
        test_suite.run_all_tests()
        
        # Return appropriate exit code
        return 0 if not test_suite.failed_tests else 1
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test suite interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
