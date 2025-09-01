#!/usr/bin/env python3
"""
Simple Performance Benchmark for Oracle-X Phase 1 & Phase 1.5 Optimizations

This script provides a basic validation of the performance improvements from:
- Phase 1: Database pooling, HTTP client optimization, memory processing
- Phase 1.5: HTTP operations & concurrent processing

Usage:
    python simple_benchmark.py
"""

import time
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import optimized components
from core.database_pool import execute_query, execute_update, execute_many
from core.http_client import optimized_get
from core.memory_processor import optimize_dataframe_memory

def benchmark_database():
    """Benchmark database operations"""
    print("Benchmarking database operations...")

    db_path = "data/databases/benchmark_test.db"

    # Setup test data
    execute_update(db_path, """
        CREATE TABLE IF NOT EXISTS test_data (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            price REAL,
            timestamp REAL
        )
    """)

    # Insert test data
    test_data = [("AAPL", 150.0 + i, time.time()) for i in range(100)]
    execute_many(db_path, "INSERT INTO test_data (symbol, price, timestamp) VALUES (?, ?, ?)", test_data)

    # Benchmark queries
    start_time = time.time()
    for _ in range(50):
        results = execute_query(db_path, "SELECT * FROM test_data WHERE symbol = ?", ("AAPL",))
    query_time = time.time() - start_time

    return {
        "database_queries": {
            "total_time": query_time,
            "avg_response_time": query_time / 50,
            "throughput": 50 / query_time
        }
    }

def benchmark_http():
    """Benchmark HTTP operations"""
    print("Benchmarking HTTP operations...")

    test_urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/json"
    ]

    start_time = time.time()
    successful_requests = 0

    for _ in range(20):
        for url in test_urls:
            try:
                response = optimized_get(url, timeout=5)
                if response.status_code == 200:
                    successful_requests += 1
            except:
                pass

    http_time = time.time() - start_time

    return {
        "http_requests": {
            "total_time": http_time,
            "successful_requests": successful_requests,
            "throughput": (successful_requests) / http_time if successful_requests > 0 else 0
        }
    }

def benchmark_memory():
    """Benchmark memory operations"""
    print("Benchmarking memory operations...")

    import pandas as pd
    import numpy as np

    # Create test DataFrame
    df = pd.DataFrame({
        'symbol': ['AAPL'] * 10000,
        'price': np.random.random(10000) * 200,
        'volume': np.random.randint(1000, 100000, 10000)
    })

    start_time = time.time()
    optimized_df = optimize_dataframe_memory(df)
    memory_time = time.time() - start_time

    return {
        "memory_processing": {
            "processing_time": memory_time,
            "rows_processed": len(optimized_df)
        }
    }

def main():
    """Run all benchmarks"""
    print("=" * 60)
    print("ORACLE-X PHASE 1 & 1.5 PERFORMANCE VALIDATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = {}

    try:
        # Run database benchmark
        results.update(benchmark_database())

        # Run HTTP benchmark
        results.update(benchmark_http())

        # Run memory benchmark
        results.update(benchmark_memory())

        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'database_throughput': results.get('database_queries', {}).get('throughput', 0),
                'http_throughput': results.get('http_requests', {}).get('throughput', 0),
                'memory_efficiency': results.get('memory_processing', {}).get('processing_time', 0)
            }
        }

        with open('simple_benchmark_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print("\nBenchmark Results:")
        print(f"Database throughput: {results['database_queries']['throughput']:.1f} queries/sec")
        print(f"HTTP throughput: {results['http_requests']['throughput']:.1f} requests/sec")
        print(f"Memory processing time: {results['memory_processing']['processing_time']:.3f}s")

        print("\n✅ Benchmark completed successfully!")
        print("Results saved to simple_benchmark_results.json")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()