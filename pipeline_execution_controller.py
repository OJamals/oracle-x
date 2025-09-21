#!/usr/bin/env python3
"""
Oracle-X Pipeline Execution Controller
Comprehensive execution environment for running all Oracle-X pipelines with:
- Detailed logging and monitoring
- Performance metrics capture
- Error tracking and recovery
- Data quality validation
- Output analysis and reporting
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import Oracle-X components
from common_utils import setup_logging, CLIFormatter, PerformanceTimer
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator, get_orchestrator
from oracle_engine.ml_prediction_engine import create_ml_model, ModelType, PredictionType

# Setup comprehensive logging
class ExecutionLogger:
    """Comprehensive logging system for pipeline execution"""

    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.log_dir = Path(f"execution_logs/{execution_id}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup main logger
        self.logger = setup_logging(f"execution-{execution_id}")
        self.logger.setLevel(logging.DEBUG)

        # Setup file handlers
        self.log_file = self.log_dir / "execution.log"
        self.error_file = self.log_dir / "errors.log"
        self.performance_file = self.log_dir / "performance.log"
        self.data_quality_file = self.log_dir / "data_quality.log"

        # Create file handlers
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        error_handler = logging.FileHandler(self.error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n'
        ))

        performance_handler = logging.FileHandler(self.performance_file)
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(performance_handler)

        # Setup data quality logger
        self.data_quality_logger = logging.getLogger(f"data-quality-{execution_id}")
        quality_handler = logging.FileHandler(self.data_quality_file)
        quality_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.data_quality_logger.addHandler(quality_handler)
        self.data_quality_logger.setLevel(logging.INFO)

    def log_execution_start(self, pipeline_name: str, config: Dict[str, Any]):
        """Log pipeline execution start"""
        self.logger.info(f"🚀 Starting pipeline: {pipeline_name}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        self.data_quality_logger.info(f"Pipeline {pipeline_name} started")

    def log_execution_end(self, pipeline_name: str, success: bool, execution_time: float, results: Dict[str, Any]):
        """Log pipeline execution completion"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.logger.info(f"{status} Pipeline {pipeline_name} completed in {execution_time:.2f}s")
        self.logger.info(f"Results: {json.dumps(results, indent=2)}")
        self.data_quality_logger.info(f"Pipeline {pipeline_name} completed - Success: {success}")

    def log_error(self, pipeline_name: str, error: Exception, context: str = ""):
        """Log pipeline error"""
        error_msg = f"❌ Error in {pipeline_name}: {str(error)}"
        if context:
            error_msg += f" (Context: {context})"

        self.logger.error(error_msg)
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        self.data_quality_logger.error(f"Pipeline {pipeline_name} error: {str(error)}")

    def log_performance_metric(self, metric_name: str, value: float, unit: str = "", context: str = ""):
        """Log performance metric"""
        msg = f"📊 {metric_name}: {value}{unit}"
        if context:
            msg += f" ({context})"

        self.logger.info(msg)
        # Also log to performance file
        with open(self.performance_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {metric_name}: {value}{unit} - {context}\n")

    def log_data_quality(self, pipeline_name: str, quality_score: float, issues: List[str]):
        """Log data quality metrics"""
        status = "✅ GOOD" if quality_score >= 80 else "⚠️  POOR" if quality_score >= 60 else "❌ BAD"
        self.data_quality_logger.info(f"{status} {pipeline_name} - Quality Score: {quality_score:.2f}%")
        if issues:
            self.data_quality_logger.info(f"Issues: {', '.join(issues)}")

class SystemMonitor:
    """Monitor system resources during execution"""

    def __init__(self, logger: ExecutionLogger):
        self.logger = logger
        self.start_time = time.time()
        self.cpu_percentages = []
        self.memory_usages = []
        self.disk_ios = []
        self.monitoring = True
        self.monitor_thread = None

    def start_monitoring(self):
        """Start system monitoring"""
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        self.logger.logger.info("🔍 System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        # Log final statistics
        if self.cpu_percentages:
            avg_cpu = sum(self.cpu_percentages) / len(self.cpu_percentages)
            max_cpu = max(self.cpu_percentages)
            self.logger.log_performance_metric("Average CPU Usage", avg_cpu, "%", "system")

        if self.memory_usages:
            avg_memory = sum(self.memory_usages) / len(self.memory_usages)
            max_memory = max(self.memory_usages)
            self.logger.log_performance_metric("Average Memory Usage", avg_memory, "MB", "system")
            self.logger.log_performance_metric("Peak Memory Usage", max_memory, "MB", "system")

    def _monitor_resources(self):
        """Monitor system resources in background"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_percentages.append(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / 1024 / 1024
                self.memory_usages.append(memory_mb)

                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.disk_ios.append(disk_io.read_bytes + disk_io.write_bytes)

                # Log if usage is high
                if cpu_percent > 80:
                    self.logger.logger.warning(f"High CPU usage detected: {cpu_percent}%")

                if memory.percent > 80:
                    self.logger.logger.warning(f"High memory usage detected: {memory.percent}%")

            except Exception as e:
                self.logger.logger.error(f"Error monitoring system resources: {e}")

            time.sleep(5)  # Monitor every 5 seconds

class PipelineExecutionController:
    """Main controller for executing Oracle-X pipelines"""

    def __init__(self, execution_id: Optional[str] = None):
        self.execution_id = execution_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = ExecutionLogger(self.execution_id)
        self.system_monitor = SystemMonitor(self.logger)
        self.results = {}
        self.errors = []
        self.performance_metrics = {}
        self.data_quality_results = {}

        # Initialize components
        self.data_orchestrator = None
        self.ml_models = {}

        self.logger.logger.info(f"🎯 Pipeline Execution Controller initialized - ID: {self.execution_id}")

    def setup_environment(self) -> bool:
        """Setup execution environment"""
        try:
            self.logger.log_execution_start("environment_setup", {})

            # Start system monitoring
            self.system_monitor.start_monitoring()

            # Initialize data orchestrator
            self.data_orchestrator = get_orchestrator()
            self.logger.logger.info("✅ Data orchestrator initialized")

            # Initialize ML models
            self._initialize_ml_models()

            # Log system information
            self._log_system_info()

            self.logger.log_execution_end("environment_setup", True, 0, {"status": "ready"})
            return True

        except Exception as e:
            self.logger.log_error("environment_setup", e)
            return False

    def _initialize_ml_models(self):
        """Initialize ML models for testing"""
        try:
            available_models = []
            if hasattr(create_ml_model, '__wrapped__'):
                # This would be the actual function if decorated
                available_models = [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.NEURAL_NETWORK]
            else:
                available_models = [ModelType.RANDOM_FOREST]  # Fallback

            for model_type in available_models:
                try:
                    model = create_ml_model(model_type, PredictionType.PRICE_DIRECTION)
                    self.ml_models[model_type.value] = model
                    self.logger.logger.info(f"✅ ML Model initialized: {model_type.value}")
                except Exception as e:
                    self.logger.logger.warning(f"⚠️  Failed to initialize {model_type.value}: {e}")

        except Exception as e:
            self.logger.logger.error(f"Failed to initialize ML models: {e}")

    def _log_system_info(self):
        """Log system information"""
        try:
            # System info
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            self.logger.log_performance_metric("Total Memory", memory.total / 1024 / 1024 / 1024, "GB", "system")
            self.logger.log_performance_metric("Available Memory", memory.available / 1024 / 1024 / 1024, "GB", "system")
            self.logger.log_performance_metric("CPU Cores", float(psutil.cpu_count() or 1), "", "system")
            self.logger.log_performance_metric("Disk Total", disk.total / 1024 / 1024 / 1024, "GB", "system")
            self.logger.log_performance_metric("Disk Free", disk.free / 1024 / 1024 / 1024, "GB", "system")

        except Exception as e:
            self.logger.logger.error(f"Failed to log system info: {e}")

    def execute_main_pipeline(self) -> Dict[str, Any]:
        """Execute main data collection pipeline"""
        self.logger.log_execution_start("main_pipeline", {"mode": "standard"})

        start_time = time.time()
        success = False
        results = {}

        try:
            # Import and run main pipeline
            from main import OracleXPipeline

            # Create pipeline instance
            pipeline = OracleXPipeline(mode="standard")

            # Execute pipeline
            result_file = pipeline.run()

            if result_file and os.path.exists(result_file):
                # Read results
                with open(result_file, 'r') as f:
                    results = json.load(f)

                success = True
                self.logger.logger.info(f"✅ Main pipeline completed successfully - Output: {result_file}")
            else:
                raise Exception("Pipeline execution failed - no output file generated")

        except Exception as e:
            self.logger.log_error("main_pipeline", e)
            self.errors.append({"pipeline": "main", "error": str(e), "traceback": traceback.format_exc()})
            results = {"error": str(e)}

        execution_time = time.time() - start_time
        self.logger.log_execution_end("main_pipeline", success, execution_time, results)

        self.results["main_pipeline"] = {
            "success": success,
            "execution_time": execution_time,
            "results": results
        }

        return self.results["main_pipeline"]

    def execute_oracle_cli_pipeline(self) -> Dict[str, Any]:
        """Execute Oracle CLI pipeline"""
        self.logger.log_execution_start("oracle_cli_pipeline", {"command": "validate"})

        start_time = time.time()
        success = False
        results = {}

        try:
            # Import and run CLI validation
            from oracle_cli import handle_validate_system

            # Capture output by redirecting stdout/stderr
            import io
            from contextlib import redirect_stdout, redirect_stderr

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Run system validation
                handle_validate_system(type('Args', (), {
                    'comprehensive': True,
                    'verbose': True
                })())

            # Get captured output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            results = {
                "stdout": stdout_output,
                "stderr": stderr_output,
                "validation_completed": len(stdout_output) > 0
            }

            success = "✅" in stdout_output or "completed successfully" in stdout_output.lower()
            self.logger.logger.info("✅ Oracle CLI pipeline completed")

        except Exception as e:
            self.logger.log_error("oracle_cli_pipeline", e)
            self.errors.append({"pipeline": "oracle_cli", "error": str(e), "traceback": traceback.format_exc()})
            results = {"error": str(e)}

        execution_time = time.time() - start_time
        self.logger.log_execution_end("oracle_cli_pipeline", success, execution_time, results)

        self.results["oracle_cli_pipeline"] = {
            "success": success,
            "execution_time": execution_time,
            "results": results
        }

        return self.results["oracle_cli_pipeline"]

    def execute_signal_collection_pipeline(self) -> Dict[str, Any]:
        """Execute signal collection and processing pipeline"""
        self.logger.log_execution_start("signal_collection_pipeline", {"symbols": ["AAPL", "TSLA", "NVDA"]})

        start_time = time.time()
        success = False
        results = {}

        try:
            if not self.data_orchestrator:
                raise Exception("Data orchestrator not initialized")

            # Test signal collection
            symbols = ["AAPL", "TSLA", "NVDA"]
            signals = self.data_orchestrator.get_signals_from_scrapers(symbols)

            if signals:
                results = {
                    "symbols_processed": len(symbols),
                    "signals_count": signals.get("signals_count", 0),
                    "timestamp": signals.get("timestamp"),
                    "data_source": signals.get("data_source"),
                    "sample_signals": {
                        k: v for k, v in list(signals.items())[:5]  # First 5 signals
                        if not k.startswith('_')
                    }
                }
                success = True
                self.logger.logger.info(f"✅ Signal collection completed for {len(symbols)} symbols")
            else:
                raise Exception("No signals collected")

        except Exception as e:
            self.logger.log_error("signal_collection_pipeline", e)
            self.errors.append({"pipeline": "signal_collection", "error": str(e), "traceback": traceback.format_exc()})
            results = {"error": str(e)}

        execution_time = time.time() - start_time
        self.logger.log_execution_end("signal_collection_pipeline", success, execution_time, results)

        self.results["signal_collection_pipeline"] = {
            "success": success,
            "execution_time": execution_time,
            "results": results
        }

        return self.results["signal_collection_pipeline"]

    def execute_ml_prediction_pipeline(self) -> Dict[str, Any]:
        """Execute ML prediction pipeline"""
        self.logger.log_execution_start("ml_prediction_pipeline", {"model_types": list(self.ml_models.keys())})

        start_time = time.time()
        success = False
        results = {}

        try:
            if not self.ml_models:
                raise Exception("No ML models available")

            # Test ML models with sample data
            sample_features = {
                "feature_1": [1.0, 2.0, 3.0],
                "feature_2": [0.5, 1.5, 2.5],
                "feature_3": [0.1, 0.2, 0.3]
            }

            import pandas as pd
            sample_df = pd.DataFrame(sample_features)

            model_results = {}
            for model_name, model in self.ml_models.items():
                try:
                    # Test prediction
                    predictions, uncertainties = model.predict(sample_df)

                    model_results[model_name] = {
                        "predictions_shape": predictions.shape,
                        "uncertainties_shape": uncertainties.shape,
                        "sample_prediction": float(predictions[0]) if len(predictions) > 0 else None,
                        "sample_uncertainty": float(uncertainties[0]) if len(uncertainties) > 0 else None,
                        "model_available": True
                    }

                except Exception as e:
                    model_results[model_name] = {
                        "error": str(e),
                        "model_available": False
                    }

            results = {
                "models_tested": len(self.ml_models),
                "models_working": len([m for m in model_results.values() if m.get("model_available", False)]),
                "model_results": model_results
            }

            success = len([m for m in model_results.values() if m.get("model_available", False)]) > 0
            self.logger.logger.info(f"✅ ML prediction pipeline completed - {results['models_working']}/{results['models_tested']} models working")

        except Exception as e:
            self.logger.log_error("ml_prediction_pipeline", e)
            self.errors.append({"pipeline": "ml_prediction", "error": str(e), "traceback": traceback.format_exc()})
            results = {"error": str(e)}

        execution_time = time.time() - start_time
        self.logger.log_execution_end("ml_prediction_pipeline", success, execution_time, results)

        self.results["ml_prediction_pipeline"] = {
            "success": success,
            "execution_time": execution_time,
            "results": results
        }

        return self.results["ml_prediction_pipeline"]

    def execute_data_validation_pipeline(self) -> Dict[str, Any]:
        """Execute data validation and quality checks"""
        self.logger.log_execution_start("data_validation_pipeline", {"validation_type": "comprehensive"})

        start_time = time.time()
        success = False
        results = {}

        try:
            if not self.data_orchestrator:
                raise Exception("Data orchestrator not initialized")

            # Get data quality report
            quality_report = self.data_orchestrator.get_data_quality_report()

            # Get system health
            system_health = self.data_orchestrator.get_system_health()

            # Get cache statistics
            cache_stats = self.data_orchestrator.get_cache_statistics()

            results = {
                "quality_report": quality_report,
                "system_health": system_health,
                "cache_statistics": cache_stats,
                "validation_timestamp": datetime.now().isoformat()
            }

            # Analyze quality scores
            quality_scores = [q.quality_score for q in quality_report.values()]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                min_quality = min(quality_scores)
                max_quality = max(quality_scores)

                results["quality_analysis"] = {
                    "average_quality_score": avg_quality,
                    "min_quality_score": min_quality,
                    "max_quality_score": max_quality,
                    "quality_distribution": {
                        "excellent": len([q for q in quality_scores if q >= 90]),
                        "good": len([q for q in quality_scores if 80 <= q < 90]),
                        "fair": len([q for q in quality_scores if 60 <= q < 80]),
                        "poor": len([q for q in quality_scores if q < 60])
                    }
                }

                # Log data quality metrics
                for source, metrics in quality_report.items():
                    self.logger.log_data_quality(
                        source,
                        metrics.quality_score,
                        metrics.issues
                    )

            success = True
            self.logger.logger.info("✅ Data validation pipeline completed")

        except Exception as e:
            self.logger.log_error("data_validation_pipeline", e)
            self.errors.append({"pipeline": "data_validation", "error": str(e), "traceback": traceback.format_exc()})
            results = {"error": str(e)}

        execution_time = time.time() - start_time
        self.logger.log_execution_end("data_validation_pipeline", success, execution_time, results)

        self.results["data_validation_pipeline"] = {
            "success": success,
            "execution_time": execution_time,
            "results": results
        }

        return self.results["data_validation_pipeline"]

    def execute_all_pipelines(self) -> Dict[str, Any]:
        """Execute all pipelines in sequence"""
        self.logger.logger.info("🎯 Starting comprehensive pipeline execution")

        # Setup environment
        if not self.setup_environment():
            return {"error": "Failed to setup execution environment"}

        # Execute pipelines in order
        pipeline_order = [
            ("main_pipeline", self.execute_main_pipeline),
            ("oracle_cli_pipeline", self.execute_oracle_cli_pipeline),
            ("signal_collection_pipeline", self.execute_signal_collection_pipeline),
            ("ml_prediction_pipeline", self.execute_ml_prediction_pipeline),
            ("data_validation_pipeline", self.execute_data_validation_pipeline)
        ]

        for pipeline_name, pipeline_func in pipeline_order:
            try:
                self.logger.logger.info(f"🔄 Executing {pipeline_name}...")
                result = pipeline_func()
                self.performance_metrics[pipeline_name] = {
                    "execution_time": result.get("execution_time", 0),
                    "success": result.get("success", False)
                }

            except Exception as e:
                self.logger.logger.error(f"Failed to execute {pipeline_name}: {e}")
                self.errors.append({"pipeline": pipeline_name, "error": str(e)})

        # Stop monitoring
        self.system_monitor.stop_monitoring()

        # Generate final report
        return self.generate_execution_report()

    def generate_execution_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution report"""
        end_time = time.time()
        total_execution_time = end_time - self.system_monitor.start_time

        # Calculate overall statistics
        successful_pipelines = sum(1 for r in self.results.values() if r.get("success", False))
        total_pipelines = len(self.results)
        success_rate = (successful_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0

        # Generate report
        report = {
            "execution_id": self.execution_id,
            "execution_timestamp": datetime.now().isoformat(),
            "total_execution_time": total_execution_time,
            "success_rate": success_rate,
            "successful_pipelines": successful_pipelines,
            "total_pipelines": total_pipelines,
            "pipeline_results": self.results,
            "errors": self.errors,
            "performance_metrics": self.performance_metrics,
            "log_files": {
                "main_log": str(self.logger.log_file),
                "error_log": str(self.logger.error_file),
                "performance_log": str(self.logger.performance_file),
                "data_quality_log": str(self.logger.data_quality_file)
            }
        }

        # Save report to file
        report_file = self.logger.log_dir / "execution_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Log summary
        self.logger.logger.info("📊 EXECUTION SUMMARY:")
        self.logger.logger.info(f"   Total Time: {total_execution_time:.2f}s")
        self.logger.logger.info(f"   Success Rate: {success_rate:.1f}%")
        self.logger.logger.info(f"   Successful: {successful_pipelines}/{total_pipelines}")
        self.logger.logger.info(f"   Errors: {len(self.errors)}")
        self.logger.logger.info(f"   Report saved to: {report_file}")

        return report

def main():
    """Main execution function"""
    print("🚀 Oracle-X Pipeline Execution Controller")
    print(f"   Execution ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print("=" * 50)

    # Create controller
    controller = PipelineExecutionController()

    try:
        # Execute all pipelines
        report = controller.execute_all_pipelines()

        # Print summary
        print("\n" + "=" * 50)
        print("📊 EXECUTION SUMMARY")
        print(f"   Total Time: {report['total_execution_time']:.2f}s")
        print(f"   Success Rate: {report['success_rate']:.1f}%")
        print(f"   Successful: {report['successful_pipelines']}/{report['total_pipelines']}")
        print(f"   Errors: {len(report['errors'])}")

        if report['errors']:
            print("\n❌ ERRORS ENCOUNTERED:")
            for error in report['errors']:
                print(f"   • {error['pipeline']}: {error['error']}")

        print(f"\n📁 Detailed logs saved to: execution_logs/{controller.execution_id}/")
        print(f"📄 Full report: execution_logs/{controller.execution_id}/execution_report.json")

        return 0 if report['success_rate'] > 50 else 1

    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        controller.system_monitor.stop_monitoring()
        return 1
    except Exception as e:
        print(f"\n💥 Execution failed: {e}")
        controller.logger.log_error("main_execution", e)
        return 1

if __name__ == "__main__":
    sys.exit(main())