"""
Comprehensive Phase 2 ML Enhancement Validation Test
Tests all Phase 2 systems: advanced feature engineering, meta-learning, real-time adaptation, and diagnostics
"""

import pytest
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

# Add the project root to path
sys.path.append("/Users/omar/Documents/Projects/oracle-x")

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_phase2_system_availability():
    """Test that all Phase 2 systems are available and importable"""
    print("\n🔍 Phase 2 System Availability Test")

    # Test advanced feature engineering
    try:
        from oracle_engine.advanced_feature_engineering import AdvancedFeatureEngineer

        feature_engineer = AdvancedFeatureEngineer()
        print("✅ Advanced Feature Engineering: Available")
        advanced_features_available = True
    except Exception as e:
        print(f"❌ Advanced Feature Engineering: {e}")
        advanced_features_available = False

    # Test advanced learning techniques
    try:
        from oracle_engine.advanced_learning_techniques import (
            AdvancedLearningOrchestrator,
            MetaLearningConfig,
        )

        config = MetaLearningConfig()
        orchestrator = AdvancedLearningOrchestrator(config)
        print("✅ Advanced Learning Techniques: Available")
        advanced_learning_available = True
    except Exception as e:
        print(f"❌ Advanced Learning Techniques: {e}")
        advanced_learning_available = False

    # Test real-time learning engine
    try:
        from oracle_engine.realtime_learning_engine import (
            RealTimeLearningEngine,
            OnlineLearningConfig,
        )

        config = OnlineLearningConfig()
        engine = RealTimeLearningEngine(config)
        print("✅ Real-time Learning Engine: Available")
        realtime_learning_available = True
    except Exception as e:
        print(f"❌ Real-time Learning Engine: {e}")
        realtime_learning_available = False

    # Test enhanced ML diagnostics
    try:
        from oracle_engine.enhanced_ml_diagnostics import EnhancedMLDiagnostics

        diagnostics = EnhancedMLDiagnostics()
        print("✅ Enhanced ML Diagnostics: Available")
        diagnostics_available = True
    except Exception as e:
        print(f"❌ Enhanced ML Diagnostics: {e}")
        diagnostics_available = False

    # Test neural network enhancements
    try:
        from oracle_engine.ml_prediction_engine import NeuralNetworkPredictor

        nn = NeuralNetworkPredictor()
        print("✅ Enhanced Neural Network: Available")
        neural_network_available = True
    except Exception as e:
        print(f"❌ Enhanced Neural Network: {e}")
        neural_network_available = False

    print(f"\n📊 Phase 2 Availability Summary:")
    print(f"   Advanced Features: {'✅' if advanced_features_available else '❌'}")
    print(f"   Meta-Learning: {'✅' if advanced_learning_available else '❌'}")
    print(f"   Real-time Learning: {'✅' if realtime_learning_available else '❌'}")
    print(f"   ML Diagnostics: {'✅' if diagnostics_available else '❌'}")
    print(f"   Enhanced Neural Network: {'✅' if neural_network_available else '❌'}")

    return {
        "advanced_features": advanced_features_available,
        "advanced_learning": advanced_learning_available,
        "realtime_learning": realtime_learning_available,
        "diagnostics": diagnostics_available,
        "neural_network": neural_network_available,
    }


def create_test_data(num_samples=1000, num_features=20):
    """Create synthetic test data for ML validation"""
    np.random.seed(42)

    # Create feature data
    X = pd.DataFrame(
        {f"feature_{i}": np.random.randn(num_samples) for i in range(num_features)}
    )

    # Add some realistic financial features
    X["price"] = 100 + np.cumsum(np.random.randn(num_samples) * 0.01)
    X["volume"] = np.random.lognormal(10, 1, num_samples)
    X["returns"] = X["price"].pct_change().fillna(0)
    X["volatility"] = X["returns"].rolling(20).std().fillna(0.01)

    # Create targets
    y_classification = (X["returns"] > 0).astype(int)  # Price direction
    y_regression = X["price"].shift(-1).fillna(X["price"].iloc[-1])  # Price target

    return X, y_classification, y_regression


def test_advanced_feature_engineering():
    """Test advanced feature engineering capabilities"""
    print("\n🔧 Advanced Feature Engineering Test")

    try:
        from oracle_engine.advanced_feature_engineering import AdvancedFeatureEngineer

        # Create test data
        X, y_class, y_reg = create_test_data(500, 10)

        # Initialize feature engineer
        feature_engineer = AdvancedFeatureEngineer(
            technical_indicators=True, sentiment_features=True, automated_selection=True
        )

        # Test feature creation
        enhanced_features = feature_engineer.create_technical_features(X)
        print(f"✅ Technical features created: {enhanced_features.shape[1]} features")

        # Test feature selection
        selected_features = feature_engineer.select_features(
            enhanced_features, y_class, method="statistical"
        )
        print(
            f"✅ Feature selection completed: {len(selected_features)} features selected"
        )

        # Test feature ranking
        rankings = feature_engineer.rank_features(enhanced_features, y_class)
        print(f"✅ Feature ranking completed: {len(rankings)} features ranked")

        return True

    except Exception as e:
        print(f"❌ Advanced Feature Engineering Test Failed: {e}")
        return False


def test_meta_learning_system():
    """Test meta-learning and ensemble stacking"""
    print("\n🧠 Meta-Learning System Test")

    try:
        from oracle_engine.advanced_learning_techniques import (
            AdvancedLearningOrchestrator,
            MetaLearningConfig,
        )

        # Create test data
        X, y_class, y_reg = create_test_data(300, 8)

        # Configure meta-learning
        config = MetaLearningConfig(
            enable_stacking=True, enable_blending=True, enable_auto_ml=True
        )

        # Initialize orchestrator
        orchestrator = AdvancedLearningOrchestrator(config)

        # Test model registration (simulated)
        class DummyModel:
            def __init__(self):
                self.is_trained = False

            def fit(self, X, y):
                self.is_trained = True
                return self

            def predict(self, X):
                return np.random.randint(0, 2, len(X))

        # Register dummy models
        orchestrator.meta_learner.register_base_model(
            DummyModel(), "model1", "classification"
        )
        orchestrator.meta_learner.register_base_model(
            DummyModel(), "model2", "classification"
        )

        # Test ensemble creation
        ensemble = orchestrator.meta_learner.create_stacked_ensemble(
            X, y_class, "classification"
        )
        print(f"✅ Stacked ensemble created: {type(ensemble).__name__}")

        # Test AutoML pipeline
        results = orchestrator.meta_learner.auto_ml_pipeline(
            X, y_class, "classification"
        )
        print(f"✅ AutoML pipeline completed: {len(results)} results")

        return True

    except Exception as e:
        print(f"❌ Meta-Learning System Test Failed: {e}")
        return False


def test_realtime_learning_engine():
    """Test real-time learning and adaptation"""
    print("\n⚡ Real-time Learning Engine Test")

    try:
        from oracle_engine.realtime_learning_engine import (
            RealTimeLearningEngine,
            OnlineLearningConfig,
        )

        # Configure real-time learning
        config = OnlineLearningConfig(
            batch_size=50, adaptation_threshold=0.1, drift_detection_enabled=True
        )

        # Initialize engine
        engine = RealTimeLearningEngine(config)

        # Test model registration
        class DummyOnlineModel:
            def __init__(self):
                self.is_trained = True

            def predict(self, X):
                return np.random.rand(len(X))

            def fit(self, X, y):
                return self

        engine.register_model(DummyOnlineModel(), "dummy_model", "regression", 1.0)
        print("✅ Model registered for real-time learning")

        # Test sample processing
        X, y_class, y_reg = create_test_data(100, 5)

        results = []
        for i in range(10):
            sample_result = engine.process_new_sample(X.iloc[i], y_reg.iloc[i])
            results.append(sample_result)

        print(f"✅ Processed {len(results)} samples in real-time")

        # Test performance update
        true_values = y_reg.iloc[:10].tolist()
        engine.update_performance(true_values)
        print("✅ Performance metrics updated")

        # Test system status
        status = engine.get_system_status()
        print(
            f"✅ System status retrieved: {status['total_samples_processed']} samples processed"
        )

        return True

    except Exception as e:
        print(f"❌ Real-time Learning Engine Test Failed: {e}")
        return False


def test_ml_diagnostics_system():
    """Test enhanced ML diagnostics and monitoring"""
    print("\n📊 ML Diagnostics System Test")

    try:
        from oracle_engine.enhanced_ml_diagnostics import EnhancedMLDiagnostics

        # Initialize diagnostics
        diagnostics = EnhancedMLDiagnostics()

        # Create test data and model
        X, y_class, y_reg = create_test_data(200, 5)
        X_test = X.iloc[150:]
        y_test = y_class.iloc[150:]

        class TestModel:
            def __init__(self):
                self.is_trained = True

            def predict(self, X):
                return np.random.randint(0, 2, len(X))

        model = TestModel()

        # Test performance calculation
        metrics = diagnostics.calculate_model_performance(
            model, X_test, y_test, "classification"
        )
        print(f"✅ Performance metrics calculated: accuracy = {metrics.accuracy:.3f}")

        # Test drift detection
        ref_data = X.iloc[:100]
        curr_data = X.iloc[100:150]
        drift_metrics = diagnostics.detect_model_drift(
            "test_model", ref_data, curr_data
        )
        print(
            f"✅ Drift detection completed: drift score = {drift_metrics.drift_score:.3f}"
        )

        # Test system health monitoring
        models = {"test_model": model}
        health = diagnostics.monitor_system_health(models)
        print(
            f"✅ System health monitored: {health.trained_models}/{health.total_models} models trained"
        )

        # Test performance summary
        diagnostics.add_performance_record(metrics)
        summary = diagnostics.get_performance_summary()
        print(f"✅ Performance summary generated: {len(summary)} models tracked")

        return True

    except Exception as e:
        print(f"❌ ML Diagnostics System Test Failed: {e}")
        return False


def test_enhanced_neural_network():
    """Test enhanced neural network with Phase 2 improvements"""
    print("\n🧠 Enhanced Neural Network Test")

    try:
        from oracle_engine.ml_prediction_engine import NeuralNetworkPredictor

        # Create test data
        X, y_class, y_reg = create_test_data(200, 8)

        # Test classification neural network
        nn_classifier = NeuralNetworkPredictor(
            prediction_type="classification",
            input_size=X.shape[1],
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            batch_normalization=True,
            learning_rate_scheduling=True,
        )

        # Test training with enhanced features
        training_result = nn_classifier.train(X, y_class)
        print(f"✅ Neural network training completed: {training_result['status']}")

        # Test prediction
        predictions, confidence = nn_classifier.predict(X.iloc[:10])
        print(f"✅ Neural network predictions generated: {len(predictions)} predictions")

        # Test regression neural network
        nn_regressor = NeuralNetworkPredictor(
            prediction_type="regression",
            input_size=X.shape[1],
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            batch_normalization=True,
        )

        training_result = nn_regressor.train(X, y_reg)
        print(f"✅ Regression neural network training: {training_result['status']}")

        return True

    except Exception as e:
        print(f"❌ Enhanced Neural Network Test Failed: {e}")
        return False


def test_integration_with_ensemble():
    """Test Phase 2 integration with ensemble engine"""
    print("\n🔗 Ensemble Integration Test")

    try:
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator

        # Create mock orchestrator
        class MockOrchestrator:
            def __init__(self):
                pass

        orchestrator = MockOrchestrator()

        # Initialize ensemble with Phase 2 enhancements
        ensemble = EnsemblePredictionEngine(orchestrator)

        # Test Phase 2 status
        phase2_status = ensemble.get_phase2_status()
        print(f"✅ Phase 2 status retrieved:")
        for system, available in phase2_status.items():
            status_icon = "✅" if available else "❌"
            print(f"   {system}: {status_icon}")

        print(
            f"✅ Phase 2 fully operational: {'✅' if phase2_status['phase2_fully_operational'] else '❌'}"
        )

        return True

    except Exception as e:
        print(f"❌ Ensemble Integration Test Failed: {e}")
        return False


def run_comprehensive_phase2_test():
    """Run comprehensive Phase 2 validation test suite"""
    print("🚀 Oracle-X Phase 2 ML Enhancement Comprehensive Test Suite")
    print("=" * 70)

    test_results = {}

    # Test 1: System Availability
    test_results["availability"] = test_phase2_system_availability()

    # Test 2: Advanced Feature Engineering
    test_results["feature_engineering"] = test_advanced_feature_engineering()

    # Test 3: Meta-Learning System
    test_results["meta_learning"] = test_meta_learning_system()

    # Test 4: Real-time Learning
    test_results["realtime_learning"] = test_realtime_learning_engine()

    # Test 5: ML Diagnostics
    test_results["diagnostics"] = test_ml_diagnostics_system()

    # Test 6: Enhanced Neural Network
    test_results["neural_network"] = test_enhanced_neural_network()

    # Test 7: Ensemble Integration
    test_results["ensemble_integration"] = test_integration_with_ensemble()

    # Final Summary
    print("\n" + "=" * 70)
    print("🎯 Phase 2 Test Results Summary")
    print("=" * 70)

    passed_tests = 0
    total_tests = 0

    for test_name, result in test_results.items():
        if test_name == "availability":
            # Handle availability test specially
            available_systems = sum(1 for v in result.values() if v)
            total_systems = len(result)
            print(
                f"System Availability: {available_systems}/{total_systems} systems available"
            )
            passed_tests += available_systems
            total_tests += total_systems
        else:
            status_icon = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status_icon}")
            if result:
                passed_tests += 1
            total_tests += 1

    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    print(
        f"\n📈 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)"
    )

    if success_rate >= 80:
        print("🎉 Phase 2 Enhancement Implementation: SUCCESS!")
        print("🚀 Oracle-X ML system is ready for advanced operations!")
    elif success_rate >= 60:
        print("⚠️  Phase 2 Enhancement Implementation: PARTIAL SUCCESS")
        print("🔧 Some systems need attention but core functionality available")
    else:
        print("❌ Phase 2 Enhancement Implementation: NEEDS WORK")
        print("🛠️  Multiple systems require debugging and fixes")

    return test_results, success_rate


if __name__ == "__main__":
    # Run the comprehensive test suite
    results, success_rate = run_comprehensive_phase2_test()

    # Save results to file
    import json
    from datetime import datetime

    results_file = (
        f"phase2_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "success_rate": success_rate,
                "test_results": results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n📁 Test results saved to: {results_file}")
