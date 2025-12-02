#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark for Oracle-X Phase 1 & Phase 1.5 Optimizations

This script validates the performance improvements from:
- Phase 1: Database pooling, HTTP client optimization, memory processing, async I/O
- Phase 1.5: HTTP operations & concurrent processing

Tests include:
- Database operations: Query/insert performance with connection pooling
- HTTP operations: API call throughput and latency with connection pooling
- Memory usage: Memory optimization effectiveness
- Concurrent processing: Async HTTP operations with multiple symbols
- End-to-end pipeline: Complete system performance comparison

Usage:
    python comprehensive_benchmark.py [--baseline] [--optimized] [--iterations N]

Results are saved to benchmark_results.json for analysis.
"""

import asyncio
import json
import os
import sys
import time
import threading
import psutil
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Import optimized components
from core.database_pool import DatabasePool, execute_query, execute_update, execute_many
from core.http_client import get_http_client_manager, optimized_get
from core.memory_processor import get_memory_processor, optimize_dataframe_memory
from core.async_io_utils import AsyncIOManager, get_async_io_manager

# Import data feeds for realistic testing
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
from data_feeds.twelvedata_adapter import TwelveDataAdapter
from data_feeds.finhub import FinnhubAdapter

# Test configuration
TEST_DB_PATH = "data/databases/benchmark_test.db"
TEST_SYMBOLS = ["AAPL", "NVDA", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NFLX"]
ITERATIONS = 10


class BenchmarkResults:
    """Collect and analyze benchmark results"""

    def __init__(self):
        self.results = {}
        self.baseline_metrics = {}
        self.optimized_metrics = {}

    def record_metric(
        self,
        test_name: str,
        metric_name: str,
        value: float,
        test_type: str = "optimized",
    ):
        """Record a performance metric"""
        if test_name not in self.results:
            self.results[test_name] = {}

        if test_type not in self.results[test_name]:
            self.results[test_name][test_type] = {}

        self.results[test_name][test_type][metric_name] = value

    def get_improvement(self, test_name: str, metric_name: str) -> Optional[float]:
        """Calculate improvement percentage for a metric"""
        if (
            test_name in self.results
            and "baseline" in self.results[test_name]
            and "optimized" in self.results[test_name]
            and metric_name in self.results[test_name]["baseline"]
            and metric_name in self.results[test_name]["optimized"]
        ):

            baseline = self.results[test_name]["baseline"][metric_name]
            optimized = self.results[test_name]["optimized"][metric_name]

            if baseline > 0:
                return ((baseline - optimized) / baseline) * 100
        return None

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON file"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
            "summary": self.generate_summary(),
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Results saved to {filename}")

    def generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        summary = {"overall_improvements": {}, "test_details": {}}

        for test_name in self.results:
            summary["test_details"][test_name] = {}

            for metric_name in [
                "avg_response_time",
                "throughput",
                "memory_usage",
                "cpu_usage",
            ]:
                improvement = self.get_improvement(test_name, metric_name)
                if improvement is not None:
                    summary["test_details"][test_name][
                        f"{metric_name}_improvement"
                    ] = improvement

                    # Track overall improvements
                    if metric_name not in summary["overall_improvements"]:
                        summary["overall_improvements"][metric_name] = []
                    summary["overall_improvements"][metric_name].append(improvement)

        # Calculate averages
        for metric_name, improvements in summary["overall_improvements"].items():
            if improvements:
                summary["overall_improvements"][metric_name] = {
                    "avg_improvement": sum(improvements) / len(improvements),
                    "min_improvement": min(improvements),
                    "max_improvement": max(improvements),
                }

        return summary


class DatabaseBenchmark:
    """Benchmark database operations"""

    def __init__(self, results: BenchmarkResults):
        self.results = results
        self.db_path = TEST_DB_PATH

    def setup_test_data(self):
        """Create test database with sample data"""
        # Create test table
        execute_update(
            self.db_path,
            """
            CREATE TABLE IF NOT EXISTS benchmark_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                price REAL,
                volume INTEGER,
                timestamp REAL
            )
        """,
        )

        # Insert test data
        test_data = []
        for i in range(1000):
            for symbol in TEST_SYMBOLS:
                test_data.append(
                    (
                        symbol,
                        100 + np.random.random() * 200,
                        int(np.random.random() * 1000000),
                        time.time() + i,
                    )
                )

        execute_many(
            self.db_path,
            """
            INSERT INTO benchmark_data (symbol, price, volume, timestamp)
            VALUES (?, ?, ?, ?)
        """,
            test_data,
        )

    def benchmark_queries(self, test_type: str = "optimized"):
        """Benchmark database query performance"""
        print(f"Running {test_type} database query benchmark...")

        start_time = time.time()
        total_queries = 0

        for _ in range(ITERATIONS):
            for symbol in TEST_SYMBOLS:
                # Execute multiple queries per symbol
                results = execute_query(
                    self.db_path,
                    "SELECT * FROM benchmark_data WHERE symbol = ? ORDER BY timestamp DESC LIMIT 100",
                    (symbol,),
                )
                total_queries += 1

                # Simulate data processing
                if results:
                    prices = [row["price"] for row in results]
                    avg_price = sum(prices) / len(prices)

        end_time = time.time()
        total_time = end_time - start_time

        self.results.record_metric(
            "database_queries", "total_time", total_time, test_type
        )
        self.results.record_metric(
            "database_queries",
            "avg_response_time",
            total_time / total_queries,
            test_type,
        )
        self.results.record_metric(
            "database_queries", "throughput", total_queries / total_time, test_type
        )

        print(
            f"Database query performance: {total_time:.2f}s for {total_queries} queries"
        )

    def benchmark_inserts(self, test_type: str = "optimized"):
        """Benchmark database insert performance"""
        print(f"Running {test_type} database insert benchmark...")

        start_time = time.time()
        total_inserts = 0

        for _ in range(ITERATIONS):
            # Bulk insert test data
            insert_data = []
            for symbol in TEST_SYMBOLS:
                for i in range(10):  # 10 inserts per symbol per iteration
                    insert_data.append(
                        (
                            symbol,
                            100 + np.random.random() * 200,
                            int(np.random.random() * 1000000),
                            time.time(),
                        )
                    )

            execute_many(
                self.db_path,
                """
                INSERT INTO benchmark_data (symbol, price, volume, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                insert_data,
            )

            total_inserts += len(insert_data)

        end_time = time.time()
        total_time = end_time - start_time

        self.results.record_metric(
            "database_inserts", "total_time", total_time, test_type
        )
        self.results.record_metric(
            "database_inserts", "throughput", total_inserts / total_time, test_type
        )

        print(
            f"Database insert performance: {total_time:.2f}s for {total_inserts} inserts"
        )

    def run_baseline_database_test(self):
        """Run baseline database test using direct sqlite3"""
        import sqlite3

        print("Running baseline database query benchmark...")

        start_time = time.time()
        total_queries = 0

        for _ in range(ITERATIONS):
            for symbol in TEST_SYMBOLS:
                # Direct sqlite3 connection for each query (baseline)
                conn = sqlite3.connect(self.db_path)
                cursor = conn.execute(
                    "SELECT * FROM benchmark_data WHERE symbol = ? ORDER BY timestamp DESC LIMIT 100",
                    (symbol,),
                )
                results = cursor.fetchall()
                conn.close()
                total_queries += 1

        end_time = time.time()
        total_time = end_time - start_time

        self.results.record_metric(
            "database_queries", "total_time", total_time, "baseline"
        )
        self.results.record_metric(
            "database_queries",
            "avg_response_time",
            total_time / total_queries,
            "baseline",
        )
        self.results.record_metric(
            "database_queries", "throughput", total_queries / total_time, "baseline"
        )

        print(
            f"Database (baseline) query performance: {total_time:.2f}s for {total_queries} queries"
        )


class HTTPBenchmark:
    """Benchmark HTTP operations"""

    def __init__(self, results: BenchmarkResults):
        self.results = results
        self.http_client = get_http_client_manager()

    def benchmark_http_requests(self, test_type: str = "optimized"):
        """Benchmark HTTP request performance"""
        print(f"Running {test_type} HTTP request benchmark...")

        # Test URLs (using mock endpoints for benchmarking)
        test_urls = [
            "https://httpbin.org/get",
            "https://httpbin.org/json",
            "https://httpbin.org/uuid",
        ]

        start_time = time.time()
        total_requests = 0
        successful_requests = 0

        for _ in range(ITERATIONS):
            for url in test_urls:
                try:
                    if test_type == "optimized":
                        response = optimized_get(url, timeout=5)
                    else:
                        # Baseline: direct requests.get
                        import requests

                        response = requests.get(url, timeout=5)

                    if response.status_code == 200:
                        successful_requests += 1
                    total_requests += 1

                except Exception as e:
                    total_requests += 1
                    print(f"Request failed: {e}")

        end_time = time.time()
        total_time = end_time - start_time

        self.results.record_metric("http_requests", "total_time", total_time, test_type)
        self.results.record_metric(
            "http_requests", "avg_response_time", total_time / total_requests, test_type
        )
        self.results.record_metric(
            "http_requests", "throughput", total_requests / total_time, test_type
        )
        self.results.record_metric(
            "http_requests",
            "success_rate",
            successful_requests / total_requests * 100,
            test_type,
        )

        print(
            f"HTTP request performance: {total_time:.2f}s for {total_requests} requests"
        )

    async def benchmark_concurrent_http(self, test_type: str = "optimized"):
        """Benchmark concurrent HTTP requests"""
        print(f"Running {test_type} concurrent HTTP benchmark...")

        test_requests = []
        for symbol in TEST_SYMBOLS:
            test_requests.extend(
                [
                    {
                        "method": "GET",
                        "url": f"https://httpbin.org/get?symbol={symbol}&i={i}",
                    }
                    for i in range(5)  # 5 requests per symbol
                ]
            )

        start_time = time.time()

        if test_type == "optimized":
            # Use optimized async HTTP client
            async with AsyncIOManager() as io_manager:
                results = await io_manager.concurrent_http_requests(test_requests)
        else:
            # Baseline: synchronous requests with ThreadPoolExecutor
            import concurrent.futures
            import requests

            def sync_request(req):
                try:
                    return requests.get(req["url"], timeout=5)
                except:
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(sync_request, test_requests))

        end_time = time.time()
        total_time = end_time - start_time

        successful_requests = sum(
            1
            for r in results
            if r is not None and getattr(r, "status_code", None) == 200
        )

        self.results.record_metric(
            "concurrent_http", "total_time", total_time, test_type
        )
        self.results.record_metric(
            "concurrent_http", "throughput", len(test_requests) / total_time, test_type
        )
        self.results.record_metric(
            "concurrent_http",
            "success_rate",
            successful_requests / len(test_requests) * 100,
            test_type,
        )

        print(
            f"Concurrent HTTP performance: {total_time:.2f}s for {len(test_requests)} requests"
        )


class MemoryBenchmark:
    """Benchmark memory usage"""

    def __init__(self, results: BenchmarkResults):
        self.results = results
        self.memory_processor = get_memory_processor()

    def benchmark_dataframe_processing(self, test_type: str = "optimized"):
        """Benchmark DataFrame memory optimization"""
        print(f"Running {test_type} DataFrame processing benchmark...")

        # Create large test DataFrame
        n_rows = 100000
        test_data = {
            "symbol": np.random.choice(TEST_SYMBOLS, n_rows),
            "price": 100 + np.random.random(n_rows) * 200,
            "volume": np.random.randint(1000, 1000000, n_rows),
            "timestamp": np.random.random(n_rows) * time.time(),
        }
        df = pd.DataFrame(test_data)

        # Measure memory usage before processing
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        if test_type == "optimized":
            # Use optimized processing
            optimized_df = optimize_dataframe_memory(df)
            # Simulate processing operations
            result = self.memory_processor.process_large_dataframe(
                optimized_df, lambda chunk: chunk.groupby("symbol")["price"].mean()
            )
        else:
            # Baseline: direct pandas operations
            result = df.groupby("symbol")["price"].mean()

        end_time = time.time()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        processing_time = end_time - start_time
        memory_used = memory_after - memory_before

        self.results.record_metric(
            "dataframe_processing", "processing_time", processing_time, test_type
        )
        self.results.record_metric(
            "dataframe_processing", "memory_usage", memory_used, test_type
        )
        self.results.record_metric(
            "dataframe_processing", "peak_memory_mb", memory_after, test_type
        )

        print(
            f"DataFrame processing: Took {processing_time:.2f}s with {memory_used:.2f}MB memory usage"
        )


class SystemBenchmark:
    """End-to-end system benchmark"""

    def __init__(self, results: BenchmarkResults):
        self.results = results

    def benchmark_end_to_end_pipeline(self, test_type: str = "optimized"):
        """Benchmark complete data pipeline"""
        print(f"Running {test_type} end-to-end pipeline benchmark...")

        start_time = time.time()

        try:
            # Initialize orchestrator
            orchestrator = DataFeedOrchestrator()

            total_requests = 0
            successful_requests = 0

            for symbol in TEST_SYMBOLS[:3]:  # Test with fewer symbols for end-to-end
                try:
                    # Get market data
                    market_data = orchestrator.get_market_data(
                        symbol, period="1d", interval="1h"
                    )
                    if market_data and not market_data.data.empty:
                        successful_requests += 1
                    total_requests += 1

                    # Get quote data
                    quote_data = orchestrator.get_quote(symbol)
                    if quote_data:
                        successful_requests += 1
                    total_requests += 1

                except Exception as e:
                    total_requests += 2  # Count both attempts
                    print(f"Pipeline test failed for {symbol}: {e}")

        except Exception as e:
            print(f"Pipeline benchmark failed: {e}")
            return

        end_time = time.time()
        total_time = end_time - start_time

        self.results.record_metric(
            "end_to_end_pipeline", "total_time", total_time, test_type
        )
        self.results.record_metric(
            "end_to_end_pipeline",
            "avg_response_time",
            total_time / len(TEST_SYMBOLS[:3]),
            test_type,
        )
        self.results.record_metric(
            "end_to_end_pipeline",
            "success_rate",
            successful_requests / total_requests * 100,
            test_type,
        )

        print(
            f"End-to-end pipeline: {total_time:.2f}s with {successful_requests}/{total_requests} successful requests"
        )


async def run_comprehensive_benchmark(args):
    """Run all benchmark tests"""

    print("=" * 80)
    print("ORACLE-X COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"Test Symbols: {', '.join(TEST_SYMBOLS)}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = BenchmarkResults()

    # Initialize components
    db_benchmark = DatabaseBenchmark(results)
    http_benchmark = HTTPBenchmark(results)
    memory_benchmark = MemoryBenchmark(results)
    system_benchmark = SystemBenchmark(results)

    # Setup test database
    print("Setting up test database...")
    db_benchmark.setup_test_data()

    # Run benchmarks
    if args.baseline or args.all:
        print("\n" + "=" * 50)
        print("BASELINE TESTS (Pre-optimization)")
        print("=" * 50)

        # Database baseline tests
        db_benchmark.run_baseline_database_test()
        db_benchmark.benchmark_inserts("baseline")

        # HTTP baseline tests
        http_benchmark.benchmark_http_requests("baseline")
        await http_benchmark.benchmark_concurrent_http("baseline")

        # Memory baseline tests
        memory_benchmark.benchmark_dataframe_processing("baseline")

        # System baseline tests
        system_benchmark.benchmark_end_to_end_pipeline("baseline")

    if args.optimized or args.all:
        print("\n" + "=" * 50)
        print("OPTIMIZED TESTS (Post-Phase 1 & 1.5)")
        print("=" * 50)

        # Database optimized tests
        db_benchmark.benchmark_queries("optimized")
        db_benchmark.benchmark_inserts("optimized")

        # HTTP optimized tests
        http_benchmark.benchmark_http_requests("optimized")
        await http_benchmark.benchmark_concurrent_http("optimized")

        # Memory optimized tests
        memory_benchmark.benchmark_dataframe_processing("optimized")

        # System optimized tests
        system_benchmark.benchmark_end_to_end_pipeline("optimized")

    # Generate and save results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 50)

    summary = results.generate_summary()

    print("Overall Performance Improvements:")
    for metric, data in summary["summary"]["overall_improvements"].items():
        if isinstance(data, dict) and "avg_improvement" in data:
            print(f"{metric}: Overall {data['avg_improvement']:.1f}% improvement")

    print("\nDetailed Test Results:")
    for test_name, metrics in summary["summary"]["test_details"].items():
        print(f"\n{test_name.upper()}:")
        for metric_name, improvement in metrics.items():
            print(f"    {metric_name}: {improvement:.1f}% improvement")

    # Save detailed results
    results.save_results()

    print(f"\nBenchmark completed! Results saved to benchmark_results.json")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive performance benchmarks for Oracle-X."
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Run baseline benchmarks only"
    )
    parser.add_argument(
        "--optimized", action="store_true", help="Run optimized benchmarks only"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run both baseline and optimized benchmarks"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=ITERATIONS,
        help="Number of iterations per test",
    )

    args = parser.parse_args()

    # Set global iterations
    global ITERATIONS
    ITERATIONS = args.iterations

    asyncio.run(run_comprehensive_benchmark(args))


if __name__ == "__main__":
    main()
