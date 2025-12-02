#!/usr/bin/env python3
"""
Comprehensive Diagnostic Test for EnsemblePredictionEngine Integration
Tests MLModelManager initialization, model training, predictions, and error handling
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_test_result(
    test_name: str, status: str, message: str = "", details: Dict[str, Any] = None
):
    """Log a test result with consistent formatting"""
    timestamp = datetime.now().isoformat()
    status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"

    print(f"\n{status_icon} {test_name}: {status}")
    print(f"   Time: {timestamp}")
    print(f"   Message: {message}")

    if details:
        for key, value in details.items():
            print(f"   {key}: {value}")

    return {
        "test": test_name,
        "status": status,
        "message": message,
        "details": details,
        "timestamp": timestamp,
    }


class EnsemblePredictionDiagnostic:
    """Comprehensive diagnostic for EnsemblePredictionEngine integration"""

    def __init__(self):
        self.test_results = []
        self.ml_manager = None
        self.ensemble_engine = None

    def run_full_diagnostic(self):
        """Run complete diagnostic suite"""
        print("=" * 80)
        print("ENSEMBLE PREDICTION ENGINE INTEGRATION DIAGNOSTIC")
        print("=" * 80)

        # Test 1: Dependency Check
        self.test_dependencies()

        # Test 2: MLModelManager Initialization
        self.test_ml_manager_initialization()

        # Test 3: Ensemble Engine Creation
        self.test_ensemble_engine_creation()

        # Test 4: Model Initialization
        self.test_model_initialization()

        # Test 5: Training Process
        self.test_training_process()

        # Test 6: Prediction Functionality
        self.test_prediction_functionality()

        # Test 7: Error Handling
        self.test_error_handling()

        # Test 8: Performance Metrics
        self.test_performance_metrics()

        # Generate summary report
        self.generate_summary_report()

    def test_dependencies(self):
        """Test availability of required dependencies"""
        print("\n🔍 Testing Dependencies...")

        dependencies = {
            "DataFeedOrchestrator": "data_feeds.data_feed_orchestrator",
            "AdvancedSentimentEngine": "data_feeds.advanced_sentiment",
            "EnsemblePredictionEngine": "oracle_engine.ensemble_ml_engine",
            "MLModelManager": "oracle_engine.ml_model_manager",
        }

        available_deps = {}
        missing_deps = {}

        for name, module in dependencies.items():
            try:
                __import__(module)
                available_deps[name] = True
            except ImportError as e:
                missing_deps[name] = str(e)

        status = "PASS" if not missing_deps else "FAIL"
        message = f"Available: {len(available_deps)}, Missing: {len(missing_deps)}"

        details = {
            "available_dependencies": list(available_deps.keys()),
            "missing_dependencies": missing_deps,
        }

        self.test_results.append(
            log_test_result("dependency_check", status, message, details)
        )
        return status == "PASS"

    def test_ml_manager_initialization(self):
        """Test MLModelManager initialization"""
        print("\n🔧 Testing MLModelManager Initialization...")

        try:
            from oracle_engine.ml_model_manager import (
                MLModelManager,
                DEPENDENCIES_AVAILABLE,
            )

            # Check dependency availability first
            if not DEPENDENCIES_AVAILABLE:
                self.test_results.append(
                    log_test_result(
                        "ml_manager_initialization",
                        "FAIL",
                        "Required dependencies not available",
                        {"dependencies_available": False},
                    )
                )
                return False

            # Create MLModelManager
            self.ml_manager = MLModelManager(
                models_dir="models", monitoring_db="data/databases/test_monitoring.db"
            )

            if self.ml_manager is None:
                raise Exception("MLModelManager creation returned None")

            # Check if ensemble_engine was initialized
            ensemble_status = (
                "initialized"
                if self.ml_manager.ensemble_engine is not None
                else "not_initialized"
            )

            status = "PASS" if self.ml_manager.ensemble_engine is not None else "FAIL"
            message = f"MLModelManager created, ensemble_engine {ensemble_status}"

            details = {
                "ml_manager_created": True,
                "ensemble_engine_initialized": self.ml_manager.ensemble_engine
                is not None,
                "dependencies_available": DEPENDENCIES_AVAILABLE,
            }

            self.test_results.append(
                log_test_result("ml_manager_initialization", status, message, details)
            )
            return status == "PASS"

        except Exception as e:
            self.test_results.append(
                log_test_result(
                    "ml_manager_initialization",
                    "FAIL",
                    f"Exception during initialization: {e}",
                    {"exception": str(e), "traceback": traceback.format_exc()},
                )
            )
            return False

    def test_ensemble_engine_creation(self):
        """Test ensemble engine creation and basic functionality"""
        print("\n🏗️  Testing Ensemble Engine Creation...")

        if not self.ml_manager:
            self.test_results.append(
                log_test_result(
                    "ensemble_engine_creation", "SKIP", "MLModelManager not available"
                )
            )
            return False

        try:
            # Try to initialize ensemble engine manually
            test_symbols = ["AAPL", "GOOGL", "MSFT"]
            init_success = self.ml_manager.initialize_ensemble(test_symbols)

            if not init_success:
                self.test_results.append(
                    log_test_result(
                        "ensemble_engine_creation",
                        "FAIL",
                        "initialize_ensemble returned False",
                        {"init_success": False},
                    )
                )
                return False

            self.ensemble_engine = self.ml_manager.ensemble_engine

            if self.ensemble_engine is None:
                self.test_results.append(
                    log_test_result(
                        "ensemble_engine_creation",
                        "FAIL",
                        "Ensemble engine is None after initialization",
                    )
                )
                return False

            # Test basic attributes
            phase2_status = self.ensemble_engine.get_phase2_status()

            status = "PASS"
            message = "Ensemble engine created successfully"
            details = {
                "ensemble_engine_type": type(self.ensemble_engine).__name__,
                "phase2_status": phase2_status,
                "available_models": (
                    len(self.ensemble_engine.models)
                    if hasattr(self.ensemble_engine, "models")
                    else 0
                ),
            }

            self.test_results.append(
                log_test_result("ensemble_engine_creation", status, message, details)
            )
            return True

        except Exception as e:
            self.test_results.append(
                log_test_result(
                    "ensemble_engine_creation",
                    "FAIL",
                    f"Exception during ensemble creation: {e}",
                    {"exception": str(e), "traceback": traceback.format_exc()},
                )
            )
            return False

    def test_model_initialization(self):
        """Test model initialization within ensemble engine"""
        print("\n🤖 Testing Model Initialization...")

        if not self.ensemble_engine:
            self.test_results.append(
                log_test_result(
                    "model_initialization", "SKIP", "Ensemble engine not available"
                )
            )
            return False

        try:
            # Check if models are initialized
            models = getattr(self.ensemble_engine, "models", {})
            model_weights = getattr(self.ensemble_engine, "model_weights", {})

            initialized_models = len(models)
            trained_models = sum(
                1 for model in models.values() if getattr(model, "is_trained", False)
            )

            # Check for specific model types
            expected_model_types = ["random_forest", "xgboost", "neural_network"]
            available_types = []

            for model_key in models.keys():
                for model_type in expected_model_types:
                    if model_type in model_key:
                        available_types.append(model_type)
                        break

            status = "PASS" if initialized_models > 0 else "WARN"
            message = (
                f"Models initialized: {initialized_models}, Trained: {trained_models}"
            )

            details = {
                "total_models": initialized_models,
                "trained_models": trained_models,
                "available_model_types": list(set(available_types)),
                "model_keys": list(models.keys()),
                "model_weights": model_weights,
            }

            self.test_results.append(
                log_test_result("model_initialization", status, message, details)
            )
            return initialized_models > 0

        except Exception as e:
            self.test_results.append(
                log_test_result(
                    "model_initialization",
                    "FAIL",
                    f"Exception during model check: {e}",
                    {"exception": str(e), "traceback": traceback.format_exc()},
                )
            )
            return False

    def test_training_process(self):
        """Test model training process"""
        print("\n🎓 Testing Training Process...")

        if not self.ensemble_engine:
            self.test_results.append(
                log_test_result(
                    "training_process", "SKIP", "Ensemble engine not available"
                )
            )
            return False

        try:
            # Test training with limited data for speed
            test_symbols = ["AAPL"]
            start_time = time.time()

            training_results = self.ensemble_engine.train_models(
                symbols=test_symbols,
                lookback_days=30,  # Reduced for speed
                update_existing=False,
            )

            training_time = time.time() - start_time

            if training_results is None:
                self.test_results.append(
                    log_test_result(
                        "training_process", "FAIL", "Training returned None"
                    )
                )
                return False

            # Check training results
            successful_trains = sum(
                1
                for result in training_results.values()
                if isinstance(result, dict) and "error" not in result
            )

            status = "PASS" if successful_trains > 0 else "WARN"
            message = f"Training completed in {training_time:.2f}s, {successful_trains} models trained"

            details = {
                "training_time_seconds": training_time,
                "total_results": len(training_results),
                "successful_trains": successful_trains,
                "training_results_keys": list(training_results.keys()),
                "last_training_time": getattr(
                    self.ensemble_engine, "last_training_time", None
                ),
            }

            self.test_results.append(
                log_test_result("training_process", status, message, details)
            )
            return successful_trains > 0

        except Exception as e:
            self.test_results.append(
                log_test_result(
                    "training_process",
                    "FAIL",
                    f"Exception during training: {e}",
                    {"exception": str(e), "traceback": traceback.format_exc()},
                )
            )
            return False

    def test_prediction_functionality(self):
        """Test prediction functionality"""
        print("\n🔮 Testing Prediction Functionality...")

        if not self.ml_manager:
            self.test_results.append(
                log_test_result(
                    "prediction_functionality", "SKIP", "MLModelManager not available"
                )
            )
            return False

        try:
            from oracle_engine.ensemble_ml_engine import PredictionType

            # Test prediction for a symbol
            test_symbol = "AAPL"
            start_time = time.time()

            prediction = self.ml_manager.predict(
                symbol=test_symbol,
                prediction_type=PredictionType.PRICE_DIRECTION,
                horizon_days=5,
            )

            prediction_time = time.time() - start_time

            if prediction is None:
                self.test_results.append(
                    log_test_result(
                        "prediction_functionality", "FAIL", "Prediction returned None"
                    )
                )
                return False

            # Check prediction structure
            required_fields = [
                "symbol",
                "prediction_type",
                "prediction",
                "confidence",
                "source",
            ]
            missing_fields = [
                field for field in required_fields if field not in prediction
            ]

            if missing_fields:
                self.test_results.append(
                    log_test_result(
                        "prediction_functionality",
                        "FAIL",
                        f"Missing required fields: {missing_fields}",
                        {"prediction": prediction},
                    )
                )
                return False

            # Check if it's using default values (indicating fallback)
            is_default = (
                prediction.get("prediction", 0) == 0.0
                and prediction.get("confidence", 0) == 0.5
                and prediction.get("source") == "default"
            )

            status = "PASS" if not is_default else "WARN"
            message = f"Prediction completed in {prediction_time:.3f}s, source: {prediction.get('source')}"

            details = {
                "prediction_time_ms": prediction_time * 1000,
                "prediction": prediction,
                "is_default_prediction": is_default,
                "prediction_source": prediction.get("source"),
                "confidence": prediction.get("confidence"),
            }

            self.test_results.append(
                log_test_result("prediction_functionality", status, message, details)
            )
            return not is_default

        except Exception as e:
            self.test_results.append(
                log_test_result(
                    "prediction_functionality",
                    "FAIL",
                    f"Exception during prediction: {e}",
                    {"exception": str(e), "traceback": traceback.format_exc()},
                )
            )
            return False

    def test_error_handling(self):
        """Test error handling with missing dependencies"""
        print("\n🛡️  Testing Error Handling...")

        try:
            # from oracle_engine.ml_model_manager import MLModelManager
            # Import commented out to avoid unused import warning

            # Test with invalid parameters
            test_cases = [
                ("Invalid symbol", {"symbol": "", "prediction_type": "invalid"}),
                ("None prediction type", {"symbol": "AAPL", "prediction_type": None}),
                (
                    "Extreme horizon",
                    {
                        "symbol": "AAPL",
                        "prediction_type": "price_direction",
                        "horizon_days": 999,
                    },
                ),
            ]

            error_handling_results = {}

            for test_name, params in test_cases:
                try:
                    if self.ml_manager:
                        # This should handle errors gracefully
                        result = self.ml_manager.predict(**params)
                        error_handling_results[test_name] = {
                            "handled": True,
                            "result": result,
                        }
                    else:
                        error_handling_results[test_name] = {
                            "handled": False,
                            "error": "No ML manager",
                        }
                except Exception as e:
                    error_handling_results[test_name] = {
                        "handled": False,
                        "error": str(e),
                    }

            # Check if errors were handled properly
            all_handled = all(
                result.get("handled", False)
                for result in error_handling_results.values()
            )

            status = "PASS" if all_handled else "WARN"
            message = f"Error handling tested: {sum(1 for r in error_handling_results.values() if r.get('handled'))}/{len(test_cases)} cases handled"

            details = {
                "error_test_results": error_handling_results,
                "all_errors_handled": all_handled,
            }

            self.test_results.append(
                log_test_result("error_handling", status, message, details)
            )
            return all_handled

        except Exception as e:
            self.test_results.append(
                log_test_result(
                    "error_handling",
                    "FAIL",
                    f"Exception during error handling test: {e}",
                    {"exception": str(e), "traceback": traceback.format_exc()},
                )
            )
            return False

    def test_performance_metrics(self):
        """Test performance and timing metrics"""
        print("\n📊 Testing Performance Metrics...")

        if not self.ensemble_engine:
            self.test_results.append(
                log_test_result(
                    "performance_metrics", "SKIP", "Ensemble engine not available"
                )
            )
            return False

        try:
            # Test multiple predictions for timing
            test_symbol = "AAPL"
            num_predictions = 5

            prediction_times = []

            for i in range(num_predictions):
                start_time = time.time()
                self.ml_manager.predict(
                    symbol=test_symbol,
                    prediction_type=self.ensemble_engine.PredictionType.PRICE_DIRECTION,
                    horizon_days=5,
                )  # Make prediction but don't store result
                prediction_time = time.time() - start_time
                prediction_times.append(prediction_time)

            avg_time = sum(prediction_times) / len(prediction_times)
            min_time = min(prediction_times)
            max_time = max(prediction_times)

            # Get model performance if available
            model_performance = {}
            try:
                if hasattr(self.ensemble_engine, "get_model_performance"):
                    model_performance = self.ensemble_engine.get_model_performance()
            except Exception as e:
                logger.warning(f"Could not get model performance: {e}")

            status = "PASS" if avg_time < 5.0 else "WARN"  # Should be under 5 seconds
            message = f"Avg prediction time: {avg_time:.3f}s, Range: {min_time:.3f}-{max_time:.3f}s"

            details = {
                "num_predictions": num_predictions,
                "avg_prediction_time": avg_time,
                "min_prediction_time": min_time,
                "max_prediction_time": max_time,
                "model_performance_available": len(model_performance) > 0,
                "performance_metrics": model_performance,
            }

            self.test_results.append(
                log_test_result("performance_metrics", status, message, details)
            )
            return avg_time < 5.0

        except Exception as e:
            self.test_results.append(
                log_test_result(
                    "performance_metrics",
                    "FAIL",
                    f"Exception during performance test: {e}",
                    {"exception": str(e), "traceback": traceback.format_exc()},
                )
            )
            return False

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 80)
        print("DIAGNOSTIC SUMMARY REPORT")
        print("=" * 80)

        passed = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed = sum(1 for result in self.test_results if result["status"] == "FAIL")
        warned = sum(1 for result in self.test_results if result["status"] == "WARN")
        skipped = sum(1 for result in self.test_results if result["status"] == "SKIP")

        total = len(self.test_results)

        print("\n📈 Test Results Summary:")
        print(f"   Total Tests: {total}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⚠️  Warnings: {warned}")
        print(f"   ⏭️  Skipped: {skipped}")

        # Overall health assessment
        if failed > 0:
            overall_status = "CRITICAL"
            overall_message = "Critical issues found - integration may not be working"
        elif warned > 0:
            overall_status = "DEGRADED"
            overall_message = "Some issues found - integration partially working"
        elif passed == total:
            overall_status = "HEALTHY"
            overall_message = (
                "All tests passed - integration appears to be working correctly"
            )
        else:
            overall_status = "UNKNOWN"
            overall_message = "Unable to determine integration status"

        print(f"\n🏥 Overall Integration Status: {overall_status}")
        print(f"   Assessment: {overall_message}")

        # Detailed analysis of issues
        if failed > 0 or warned > 0:
            print("\n🔍 Issues Found:")

            for result in self.test_results:
                if result["status"] in ["FAIL", "WARN"]:
                    print(f"   • {result['test']}: {result['message']}")

        # Recommendations
        print("\n💡 Recommendations:")

        if failed > 0:
            print("   1. Address critical failures first")
            print("   2. Check dependency availability")
            print("   3. Verify model initialization process")
            print("   4. Review error handling mechanisms")

        if warned > 0:
            print("   1. Monitor warning conditions")
            print("   2. Optimize performance where needed")
            print("   3. Consider fallback mechanisms")

        if passed == total:
            print("   1. Integration appears to be working correctly")
            print("   2. Consider adding monitoring and alerting")
            print("   3. Test with production data loads")

        # Export detailed results
        print("\n📄 Detailed results exported to: ensemble_diagnostic_report.json")

        # Save detailed report
        try:
            import json

            report_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "passed": passed,
                    "failed": failed,
                    "warned": warned,
                    "skipped": skipped,
                    "total": total,
                    "overall_status": overall_status,
                },
                "test_results": self.test_results,
                "recommendations": overall_message,
            }

            with open("ensemble_diagnostic_report.json", "w") as f:
                json.dump(report_data, f, indent=2, default=str)

        except Exception as e:
            print(f"❌ Failed to save detailed report: {e}")

        return report_data


def main():
    """Main diagnostic execution"""
    diagnostic = EnsemblePredictionDiagnostic()
    diagnostic.run_full_diagnostic()


if __name__ == "__main__":
    main()
