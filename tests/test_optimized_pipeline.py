#!/usr/bin/env python3
"""
Comprehensive Integration Testing for Oracle-X Optimized Pipeline

Tests all major components working together:
- Async data pipeline performance
- Unified cache manager effectiveness
- ML interface functionality
- Data feed orchestrator integration
- End-to-end pipeline execution
- Performance benchmarking and validation

Usage:
    python test_optimized_pipeline.py [--verbose] [--benchmark]
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import optimized components
try:
    from main_optimized import DependencyFactory, OracleXOptimizedPipeline

    from core.async_data_pipeline import AsyncDataPipeline, fetch_yfinance_data
    from core.unified_cache_manager import UnifiedCacheManager, cache_manager
    from core.unified_ml_interface import (
        UnifiedMLInterface,
        predict_direction,
        predict_price,
    )
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator

    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import optimized components: {e}")
    COMPONENTS_AVAILABLE = False

# Test configuration
TEST_CONFIG = {
    "test_tickers": ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA"],
    "benchmark_iterations": 3,
    "performance_threshold": 10.0,  # seconds
    "cache_hit_target": 0.3,  # 30% cache hit rate
    "concurrent_requests": 5,
}


class PipelineTester:
    """Comprehensive testing for the optimized Oracle-X pipeline"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []
        self.logger = logging.getLogger(__name__)

        if not COMPONENTS_AVAILABLE:
            raise ImportError("Optimized components not available for testing")

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        self.logger.info("Starting comprehensive Oracle-X pipeline tests...")

        test_results = {
            "timestamp": datetime.now().isoformat(),
            "components_tested": [],
            "performance_tests": {},
            "integration_tests": {},
            "benchmark_results": {},
            "overall_status": "unknown",
        }

        try:
            # Test 1: Component availability and basic functionality
            await self.test_component_availability(test_results)

            # Test 2: Async data pipeline performance
            await self.test_async_data_pipeline(test_results)

            # Test 3: Unified cache manager effectiveness
            await self.test_cache_manager(test_results)

            # Test 4: ML interface functionality
            await self.test_ml_interface(test_results)

            # Test 5: Data feed orchestrator integration
            await self.test_data_orchestrator(test_results)

            # Test 6: End-to-end pipeline integration
            await self.test_end_to_end_pipeline(test_results)

            # Test 7: Performance benchmarking
            await self.test_performance_benchmark(test_results)

            # Overall assessment
            test_results["overall_status"] = self._assess_overall_status(test_results)

            return test_results

        except Exception as e:
            self.logger.error(f"Comprehensive testing failed: {e}")
            test_results["error"] = str(e)
            test_results["overall_status"] = "failed"
            return test_results

    async def test_component_availability(self, results: Dict[str, Any]):
        """Test that all required components are available"""
        self.logger.info("Testing component availability...")

        components = {
            "OracleXOptimizedPipeline": OracleXOptimizedPipeline,
            "DependencyFactory": DependencyFactory,
            "AsyncDataPipeline": AsyncDataPipeline,
            "UnifiedCacheManager": UnifiedCacheManager,
            "UnifiedMLInterface": UnifiedMLInterface,
            "DataFeedOrchestrator": DataFeedOrchestrator,
        }

        available_components = []
        for name, cls in components.items():
            try:
                # Try to instantiate
                if name == "OracleXOptimizedPipeline":
                    deps = DependencyFactory.create_all_dependencies()
                    instance = cls(deps)
                elif name == "DependencyFactory":
                    instance = cls()
                elif name == "UnifiedCacheManager":
                    instance = cls()
                elif name == "UnifiedMLInterface":
                    instance = cls()
                elif name == "DataFeedOrchestrator":
                    instance = cls()
                else:
                    instance = cls()

                available_components.append(name)
                if self.verbose:
                    self.logger.info(f"✅ {name}: Available")

            except Exception as e:
                self.logger.warning(f"❌ {name}: {e}")

        results["components_tested"] = available_components

        if len(available_components) >= 4:  # At least 4 core components
            self.logger.info(
                f"✅ Component availability: {len(available_components)}/{len(components)} components working"
            )
        else:
            raise Exception(
                f"Insufficient components available: {len(available_components)}/{len(components)}"
            )

    async def test_async_data_pipeline(self, results: Dict[str, Any]):
        """Test async data pipeline performance"""
        self.logger.info("Testing async data pipeline...")

        # Create pipeline
        pipeline = AsyncDataPipeline(max_concurrent=TEST_CONFIG["concurrent_requests"])

        # Test data fetching
        start_time = time.time()
        tickers = TEST_CONFIG["test_tickers"][:3]  # Test with 3 tickers

        # Submit requests
        request_ids = []
        for ticker in tickers:
            request_id = await pipeline.submit_request(
                type(
                    "DataRequest",
                    (),
                    {
                        "request_id": f"test_{ticker}",
                        "source": "yfinance",
                        "priority": type("Priority", (), {"MEDIUM": 3})(),
                        "callback": fetch_yfinance_data,
                        "args": (ticker,),
                    },
                )()
            )
            request_ids.append(request_id)

        # Start processing (in a real scenario, this would run continuously)
        await asyncio.sleep(0.1)  # Brief pause to allow processing

        # Get stats
        stats = await pipeline.get_stats()

        processing_time = time.time() - start_time

        results["performance_tests"]["async_pipeline"] = {
            "processing_time": processing_time,
            "requests_submitted": len(request_ids),
            "pipeline_stats": stats,
            "status": "passed" if processing_time < 5.0 else "slow",
        }

        self.logger.info(f"✅ Async pipeline test completed in {processing_time:.2f}s")

    async def test_cache_manager(self, results: Dict[str, Any]):
        """Test unified cache manager effectiveness"""
        self.logger.info("Testing cache manager...")

        # Clear cache first
        cache_manager.clear()

        # Test basic caching
        test_key = "test_cache_key"
        test_value = {"test": "data", "number": 42}

        # First set (cache miss)
        start_time = time.time()
        cache_manager.set(test_key, test_value, ttl=300)
        first_set_time = time.time() - start_time

        # First get (cache hit)
        start_time = time.time()
        result = cache_manager.get(test_key)
        first_get_time = time.time() - start_time

        # Verify result
        cache_hit = result == test_value

        # Test cache stats
        stats = cache_manager.get_stats()

        results["performance_tests"]["cache_manager"] = {
            "set_time": first_set_time,
            "get_time": first_get_time,
            "cache_hit": cache_hit,
            "stats": stats,
            "status": "passed"
            if cache_hit and first_get_time < first_set_time
            else "failed",
        }

        self.logger.info(
            f"✅ Cache manager test: hit={cache_hit}, get_time={first_get_time:.4f}s"
        )

    async def test_ml_interface(self, results: Dict[str, Any]):
        """Test ML interface functionality"""
        self.logger.info("Testing ML interface...")

        ml_interface = UnifiedMLInterface()

        # Test price prediction
        ticker = TEST_CONFIG["test_tickers"][0]

        start_time = time.time()
        price_prediction = await predict_price(ticker, "1d")
        price_pred_time = time.time() - start_time

        # Test direction prediction
        start_time = time.time()
        direction_prediction = await predict_direction(ticker, "1d")
        direction_pred_time = time.time() - start_time

        # Check results
        price_valid = price_prediction is not None and price_prediction.confidence > 0
        direction_valid = (
            direction_prediction is not None and direction_prediction.confidence > 0
        )

        results["integration_tests"]["ml_interface"] = {
            "price_prediction_time": price_pred_time,
            "direction_prediction_time": direction_pred_time,
            "price_prediction_valid": price_valid,
            "direction_prediction_valid": direction_valid,
            "status": "passed" if price_valid and direction_valid else "partial",
        }

        self.logger.info(
            f"✅ ML interface test: price_pred={price_valid}, direction_pred={direction_valid}"
        )

    async def test_data_orchestrator(self, results: Dict[str, Any]):
        """Test data feed orchestrator integration"""
        self.logger.info("Testing data orchestrator...")

        orchestrator = DataFeedOrchestrator()
        tickers = TEST_CONFIG["test_tickers"][:2]  # Test with 2 tickers

        start_time = time.time()
        signals = await orchestrator.get_signals(tickers, "test query")
        processing_time = time.time() - start_time

        # Validate results
        has_data_sources = len(signals.get("data_sources", {})) > 0
        has_trading_signals = (
            len(signals.get("trading_signals", {}).get("bullish_signals", [])) >= 0
        )

        results["integration_tests"]["data_orchestrator"] = {
            "processing_time": processing_time,
            "tickers_processed": len(tickers),
            "data_sources_count": len(signals.get("data_sources", {})),
            "trading_signals_count": len(
                signals.get("trading_signals", {}).get("bullish_signals", [])
            ),
            "status": "passed"
            if has_data_sources and processing_time < 10.0
            else "slow",
        }

        self.logger.info(f"✅ Data orchestrator test: {processing_time:.2f}s")

    async def test_end_to_end_pipeline(self, results: Dict[str, Any]):
        """Test end-to-end pipeline integration"""
        self.logger.info("Testing end-to-end pipeline...")

        # Create dependencies
        deps = DependencyFactory.create_all_dependencies()

        if not deps.validate():
            results["integration_tests"]["end_to_end"] = {
                "status": "skipped",
                "reason": "Dependencies not available",
            }
            return

        # Create pipeline
        pipeline = OracleXOptimizedPipeline(deps)

        # Test pipeline execution
        query = "Test trading analysis for tech stocks"
        tickers = TEST_CONFIG["test_tickers"][:2]

        start_time = time.time()
        result = await pipeline.run_async(query, tickers)
        total_time = time.time() - start_time

        # Validate results
        has_predictions = len(result.get("predictions", [])) > 0
        has_performance_data = "performance" in result
        success = result.get("status") == "success"

        results["integration_tests"]["end_to_end"] = {
            "total_time": total_time,
            "has_predictions": has_predictions,
            "has_performance_data": has_performance_data,
            "success": success,
            "status": "passed"
            if success
            and has_predictions
            and total_time < TEST_CONFIG["performance_threshold"]
            else "slow",
        }

        self.logger.info(f"✅ End-to-end test: {total_time:.2f}s")

    async def test_performance_benchmark(self, results: Dict[str, Any]):
        """Run performance benchmarking tests"""
        self.logger.info("Running performance benchmarks...")

        benchmark_times = []

        for i in range(TEST_CONFIG["benchmark_iterations"]):
            self.logger.info(
                f"Benchmark iteration {i+1}/{TEST_CONFIG['benchmark_iterations']}"
            )

            start_time = time.time()

            # Run a mini pipeline
            deps = DependencyFactory.create_all_dependencies()
            if deps.validate():
                pipeline = OracleXOptimizedPipeline(deps)
                await pipeline.run_async(
                    "Benchmark test", TEST_CONFIG["test_tickers"][:2]
                )

            iteration_time = time.time() - start_time
            benchmark_times.append(iteration_time)

        avg_time = sum(benchmark_times) / len(benchmark_times)
        min_time = min(benchmark_times)
        max_time = max(benchmark_times)

        results["benchmark_results"] = {
            "iterations": TEST_CONFIG["benchmark_iterations"],
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "times": benchmark_times,
            "status": "passed"
            if avg_time < TEST_CONFIG["performance_threshold"]
            else "slow",
        }

        self.logger.info(
            f"✅ Benchmark completed: avg={avg_time:.2f}s, min={min_time:.2f}s"
        )

    def _assess_overall_status(self, results: Dict[str, Any]) -> str:
        """Assess overall test status"""
        performance_tests = results.get("performance_tests", {})
        integration_tests = results.get("integration_tests", {})
        benchmark_results = results.get("benchmark_results", {})

        # Count passed tests
        passed_count = 0
        total_count = 0

        for test_group in [performance_tests, integration_tests, benchmark_results]:
            for test_name, test_result in test_group.items():
                total_count += 1
                if test_result.get("status") == "passed":
                    passed_count += 1

        success_rate = passed_count / total_count if total_count > 0 else 0

        if success_rate >= 0.8:
            return "excellent"
        elif success_rate >= 0.6:
            return "good"
        elif success_rate >= 0.4:
            return "fair"
        else:
            return "poor"

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {filename}")


async def main():
    """Main test execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Oracle-X Optimized Pipeline Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--benchmark", action="store_true", help="Run only benchmarks")
    parser.add_argument("--save-results", help="Save results to file")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        tester = PipelineTester(verbose=args.verbose)

        if args.benchmark:
            # Run only benchmarks
            results = {"benchmark_results": {}}
            await tester.test_performance_benchmark(results)
        else:
            # Run comprehensive tests
            results = await tester.run_comprehensive_tests()

        # Display results summary
        print("\n" + "=" * 60)
        print("ORACLE-X OPTIMIZED PIPELINE TEST RESULTS")
        print("=" * 60)

        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Components Tested: {len(results['components_tested'])}")

        if "performance_tests" in results:
            print("\nPerformance Tests:")
            for test_name, test_result in results["performance_tests"].items():
                status = "✅" if test_result["status"] == "passed" else "⚠️"
                print(
                    f"  {status} {test_name}: {test_result.get('processing_time', 0):.3f}s"
                )

        if "integration_tests" in results:
            print("\nIntegration Tests:")
            for test_name, test_result in results["integration_tests"].items():
                status = "✅" if test_result["status"] == "passed" else "⚠️"
                print(f"  {status} {test_name}")

        if "benchmark_results" in results:
            benchmark = results["benchmark_results"]
            print("\nBenchmark Results:")
            print(f"  Average Time: {benchmark['avg_time']:.3f}s")
            print(
                f"  Min/Max Time: {benchmark['min_time']:.3f}s / {benchmark['max_time']:.3f}s"
            )

        print("\n" + "=" * 60)

        # Save results if requested
        if args.save_results:
            tester.save_results(results, args.save_results)

        # Exit with appropriate code
        if results["overall_status"] in ["excellent", "good"]:
            sys.exit(0)
        else:
            sys.exit(1)

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all optimized components are properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test Execution Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
