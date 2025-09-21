#!/usr/bin/env python3
"""
Focused Pipeline Execution Script
Executes Oracle-X pipelines individually with comprehensive logging and monitoring
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import psutil
import warnings
warnings.filterwarnings('ignore')

# Setup logging
def setup_execution_logging(execution_id: str) -> logging.Logger:
    """Setup comprehensive logging for pipeline execution"""
    log_dir = Path(f"execution_logs/{execution_id}")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"pipeline-execution-{execution_id}")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler for all logs
    log_file = log_dir / "focused_execution.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Error file handler
    error_file = log_dir / "errors.log"
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n'
    ))

    # Performance file handler
    perf_file = log_dir / "performance.log"
    perf_handler = logging.FileHandler(perf_file)
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(perf_handler)

    return logger

def log_system_info(logger: logging.Logger):
    """Log system information"""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        logger.info("📊 System Information:")
        logger.info(f"   Total Memory: {memory.total / 1024 / 1024 / 1024:.1f}GB")
        logger.info(f"   Available Memory: {memory.available / 1024 / 1024 / 1024:.1f}GB")
        logger.info(f"   CPU Cores: {psutil.cpu_count()}")
        logger.info(f"   Disk Total: {disk.total / 1024 / 1024 / 1024:.1f}GB")
        logger.info(f"   Disk Free: {disk.free / 1024 / 1024 / 1024:.1f}GB")
    except Exception as e:
        logger.error(f"Failed to log system info: {e}")

def execute_main_pipeline(logger: logging.Logger) -> Dict[str, Any]:
    """Execute main data collection pipeline"""
    logger.info("🚀 Executing Main Pipeline (main.py)")

    start_time = time.time()
    success = False
    results = {}

    try:
        # Import and run main pipeline
        from main import OracleXPipeline

        logger.info("   Initializing OracleXPipeline...")
        pipeline = OracleXPipeline(mode="standard")

        logger.info("   Running standard pipeline...")
        result_file = pipeline.run()

        if result_file and os.path.exists(result_file):
            # Read results
            with open(result_file, 'r') as f:
                results = json.load(f)

            success = True
            logger.info(f"   ✅ Main pipeline completed successfully - Output: {result_file}")
            logger.info(f"   Execution time: {time.time() - start_time:.2f}s")
        else:
            raise Exception("Pipeline execution failed - no output file generated")

    except Exception as e:
        logger.error(f"   ❌ Main pipeline failed: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        results = {"error": str(e)}

    execution_time = time.time() - start_time
    return {
        "pipeline": "main",
        "success": success,
        "execution_time": execution_time,
        "results": results
    }

def execute_oracle_cli_pipeline(logger: logging.Logger) -> Dict[str, Any]:
    """Execute Oracle CLI pipeline"""
    logger.info("🚀 Executing Oracle CLI Pipeline")

    start_time = time.time()
    success = False
    results = {}

    try:
        # Import and run CLI validation
        from oracle_cli import handle_validate_system

        logger.info("   Running system validation...")
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
        logger.info("   ✅ Oracle CLI pipeline completed")
        logger.info(f"   Execution time: {time.time() - start_time:.2f}s")

    except Exception as e:
        logger.error(f"   ❌ Oracle CLI pipeline failed: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        results = {"error": str(e)}

    execution_time = time.time() - start_time
    return {
        "pipeline": "oracle_cli",
        "success": success,
        "execution_time": execution_time,
        "results": results
    }

def execute_signal_collection_pipeline(logger: logging.Logger) -> Dict[str, Any]:
    """Execute signal collection pipeline"""
    logger.info("🚀 Executing Signal Collection Pipeline")

    start_time = time.time()
    success = False
    results = {}

    try:
        # Import data orchestrator
        from data_feeds.data_feed_orchestrator import get_orchestrator

        logger.info("   Initializing data orchestrator...")
        orchestrator = get_orchestrator()

        logger.info("   Collecting signals for test symbols...")
        # Test signal collection
        symbols = ["AAPL", "TSLA", "NVDA"]
        signals = orchestrator.get_signals_from_scrapers(symbols)

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
            logger.info(f"   ✅ Signal collection completed for {len(symbols)} symbols")
            logger.info(f"   Execution time: {time.time() - start_time:.2f}s")
        else:
            raise Exception("No signals collected")

    except Exception as e:
        logger.error(f"   ❌ Signal collection pipeline failed: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        results = {"error": str(e)}

    execution_time = time.time() - start_time
    return {
        "pipeline": "signal_collection",
        "success": success,
        "execution_time": execution_time,
        "results": results
    }

def execute_ml_prediction_pipeline(logger: logging.Logger) -> Dict[str, Any]:
    """Execute ML prediction pipeline"""
    logger.info("🚀 Executing ML Prediction Pipeline")

    start_time = time.time()
    success = False
    results = {}

    try:
        # Import ML components
        from oracle_engine.ml_prediction_engine import create_ml_model, ModelType, PredictionType

        logger.info("   Testing ML models...")
        # Test ML models with sample data
        sample_features = {
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [0.5, 1.5, 2.5],
            "feature_3": [0.1, 0.2, 0.3]
        }

        import pandas as pd
        sample_df = pd.DataFrame(sample_features)

        model_results = {}
        available_models = [ModelType.RANDOM_FOREST]  # Start with basic model

        for model_type in available_models:
            try:
                logger.info(f"   Testing {model_type.value} model...")
                model = create_ml_model(model_type, PredictionType.PRICE_DIRECTION)

                # Test prediction
                predictions, uncertainties = model.predict(sample_df)

                model_results[model_type.value] = {
                    "predictions_shape": predictions.shape,
                    "uncertainties_shape": uncertainties.shape,
                    "sample_prediction": float(predictions[0]) if len(predictions) > 0 else None,
                    "sample_uncertainty": float(uncertainties[0]) if len(uncertainties) > 0 else None,
                    "model_available": True
                }
                logger.info(f"   ✅ {model_type.value} model working")

            except Exception as e:
                logger.warning(f"   ⚠️  {model_type.value} model failed: {e}")
                model_results[model_type.value] = {
                    "error": str(e),
                    "model_available": False
                }

        results = {
            "models_tested": len(available_models),
            "models_working": len([m for m in model_results.values() if m.get("model_available", False)]),
            "model_results": model_results
        }

        success = len([m for m in model_results.values() if m.get("model_available", False)]) > 0
        logger.info(f"   ✅ ML prediction pipeline completed - {results['models_working']}/{results['models_tested']} models working")
        logger.info(f"   Execution time: {time.time() - start_time:.2f}s")

    except Exception as e:
        logger.error(f"   ❌ ML prediction pipeline failed: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        results = {"error": str(e)}

    execution_time = time.time() - start_time
    return {
        "pipeline": "ml_prediction",
        "success": success,
        "execution_time": execution_time,
        "results": results
    }

def execute_data_validation_pipeline(logger: logging.Logger) -> Dict[str, Any]:
    """Execute data validation pipeline"""
    logger.info("🚀 Executing Data Validation Pipeline")

    start_time = time.time()
    success = False
    results = {}

    try:
        # Import data orchestrator
        from data_feeds.data_feed_orchestrator import get_orchestrator

        logger.info("   Initializing data orchestrator...")
        orchestrator = get_orchestrator()

        logger.info("   Getting data quality report...")
        # Get data quality report
        quality_report = orchestrator.get_data_quality_report()

        logger.info("   Getting system health...")
        # Get system health
        system_health = orchestrator.get_system_health()

        logger.info("   Getting cache statistics...")
        # Get cache statistics
        cache_stats = orchestrator.get_cache_statistics()

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

            logger.info(f"   ✅ Data validation completed - Avg quality: {avg_quality:.1f}%")
            logger.info(f"   Execution time: {time.time() - start_time:.2f}s")
        else:
            logger.warning("   ⚠️  No quality scores available")
            results["quality_analysis"] = {"error": "No quality data available"}

        success = True

    except Exception as e:
        logger.error(f"   ❌ Data validation pipeline failed: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        results = {"error": str(e)}

    execution_time = time.time() - start_time
    return {
        "pipeline": "data_validation",
        "success": success,
        "execution_time": execution_time,
        "results": results
    }

def generate_execution_report(execution_id: str, pipeline_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive execution report"""
    total_execution_time = sum(r.get("execution_time", 0) for r in pipeline_results)
    successful_pipelines = sum(1 for r in pipeline_results if r.get("success", False))
    total_pipelines = len(pipeline_results)
    success_rate = (successful_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0

    # Generate report
    report = {
        "execution_id": execution_id,
        "execution_timestamp": datetime.now().isoformat(),
        "total_execution_time": total_execution_time,
        "success_rate": success_rate,
        "successful_pipelines": successful_pipelines,
        "total_pipelines": total_pipelines,
        "pipeline_results": pipeline_results,
        "log_files": {
            "main_log": f"execution_logs/{execution_id}/focused_execution.log",
            "error_log": f"execution_logs/{execution_id}/errors.log",
            "performance_log": f"execution_logs/{execution_id}/performance.log"
        }
    }

    # Save report to file
    log_dir = Path(f"execution_logs/{execution_id}")
    report_file = log_dir / "execution_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return report

def main():
    """Main execution function"""
    execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🚀 Oracle-X Focused Pipeline Execution")
    print(f"   Execution ID: {execution_id}")
    print("=" * 50)

    # Setup logging
    logger = setup_execution_logging(execution_id)
    logger.info(f"🎯 Pipeline Execution started - ID: {execution_id}")

    # Log system info
    log_system_info(logger)

    # Execute pipelines
    pipeline_functions = [
        ("Main Pipeline", execute_main_pipeline),
        ("Oracle CLI Pipeline", execute_oracle_cli_pipeline),
        ("Signal Collection Pipeline", execute_signal_collection_pipeline),
        ("ML Prediction Pipeline", execute_ml_prediction_pipeline),
        ("Data Validation Pipeline", execute_data_validation_pipeline)
    ]

    pipeline_results = []

    for pipeline_name, pipeline_func in pipeline_functions:
        try:
            logger.info(f"🔄 Executing {pipeline_name}...")
            print(f"🔄 Executing {pipeline_name}...")

            result = pipeline_func(logger)
            pipeline_results.append(result)

            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            print(f"   {status} - {result['execution_time']:.2f}s")

        except Exception as e:
            logger.error(f"Failed to execute {pipeline_name}: {e}")
            print(f"   ❌ ERROR - {e}")
            pipeline_results.append({
                "pipeline": pipeline_name.lower().replace(" ", "_"),
                "success": False,
                "execution_time": 0,
                "results": {"error": str(e)}
            })

    # Generate report
    report = generate_execution_report(execution_id, pipeline_results)

    # Print summary
    print("\n" + "=" * 50)
    print("📊 EXECUTION SUMMARY")
    print(f"   Total Time: {report['total_execution_time']:.2f}s")
    print(f"   Success Rate: {report['success_rate']:.1f}%")
    print(f"   Successful: {report['successful_pipelines']}/{report['total_pipelines']}")

    # Print individual results
    print("\n📋 PIPELINE RESULTS:")
    for result in pipeline_results:
        status = "✅" if result["success"] else "❌"
        print(f"   {status} {result['pipeline']}: {result['execution_time']:.2f}s")

    print(f"\n📁 Detailed logs saved to: execution_logs/{execution_id}/")
    print(f"📄 Full report: execution_logs/{execution_id}/execution_report.json")

    return 0 if report['success_rate'] > 50 else 1

if __name__ == "__main__":
    sys.exit(main())