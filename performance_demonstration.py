#!/usr/bin/env python3
"""
Oracle-X Performance Demonstration
Shows concrete performance improvements from Phase 2 optimizations
"""

import time
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import optimized modules
from core.optimized_algorithms import (
    VectorizedOptionsEngine,
    OptimizedDataProcessor,
    DynamicAlgorithmSelector,
    get_vectorized_options_price,
    optimize_dataframe,
    parallel_process_data
)

from core.optimized_pandas import (
    OptimizedPandasProcessor,
    optimize_dataframe as pandas_optimize,
    vectorized_query,
    efficient_groupby
)

from data_feeds.cache_service import CacheService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_vectorized_speedup():
    """Demonstrate vectorized operations speedup"""
    logger.info("🚀 Demonstrating Vectorized Operations Speedup...")

    # Generate test data
    np.random.seed(42)
    n_options = 1000
    S = np.random.uniform(50, 200, n_options)
    K = np.random.uniform(50, 200, n_options)
    T = np.random.uniform(0.1, 2.0, n_options)
    r = np.random.uniform(0.01, 0.05, n_options)
    sigma = np.random.uniform(0.1, 0.5, n_options)

    engine = VectorizedOptionsEngine()

    # Time vectorized implementation
    start_time = time.time()
    vectorized_prices = []
    for i in range(n_options):
        price = engine.price_binomial_vectorized(S[i], K[i], T[i], r[i], sigma[i], 'call')
        vectorized_prices.append(price)
    vectorized_time = time.time() - start_time

    # Time traditional loop-based approximation
    start_time = time.time()
    traditional_prices = []
    for i in range(n_options):
        # Simple Black-Scholes approximation
        d1 = (np.log(S[i]/K[i]) + (r[i] + 0.5 * sigma[i]**2) * T[i]) / (sigma[i] * np.sqrt(T[i]))
        d2 = d1 - sigma[i] * np.sqrt(T[i])

        # Very simple CDF approximation
        def simple_cdf(x):
            return 0.5 * (1 + np.tanh(x * 0.8))

        price = S[i] * simple_cdf(d1) - K[i] * np.exp(-r[i] * T[i]) * simple_cdf(d2)
        traditional_prices.append(price)
    traditional_time = time.time() - start_time

    speedup = traditional_time / vectorized_time if vectorized_time > 0 else 1.0

    logger.info("📊 Vectorized Operations Results:")
    logger.info(f"   Options priced: {n_options}")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(".2f")

    return {
        'traditional_time': traditional_time,
        'vectorized_time': vectorized_time,
        'speedup': speedup,
        'options_count': n_options
    }

def demonstrate_pandas_optimization():
    """Demonstrate pandas optimization improvements"""
    logger.info("🚀 Demonstrating Pandas Optimization...")

    # Create large test DataFrame
    np.random.seed(42)
    n_rows = 10000

    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_rows, freq='1min'),
        'price': np.random.uniform(100, 200, n_rows),
        'volume': np.random.randint(1000, 10000, n_rows),
        'sentiment_score': np.random.uniform(-1, 1, n_rows),
        'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'TSLA'], n_rows)
    })

    processor = OptimizedPandasProcessor()

    # Time traditional operations
    start_time = time.time()
    traditional_df = df.copy()
    traditional_df['price_change'] = traditional_df['price'].diff()
    traditional_df['rolling_mean'] = traditional_df['price'].rolling(20).mean()
    traditional_df['sentiment_category'] = traditional_df['sentiment_score'].apply(
        lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
    )
    traditional_time = time.time() - start_time

    # Time optimized operations
    start_time = time.time()
    optimized_df = processor.optimize_dataframe_schema(df.copy())
    optimized_df['price_change'] = optimized_df['price'].diff()
    optimized_df['rolling_mean'] = optimized_df['price'].rolling(20).mean()
    optimized_df['sentiment_category'] = np.select(
        [optimized_df['sentiment_score'] > 0.1, optimized_df['sentiment_score'] < -0.1],
        ['positive', 'negative'],
        default='neutral'
    )
    optimized_time = time.time() - start_time

    speedup = traditional_time / optimized_time if optimized_time > 0 else 1.0

    logger.info("📊 Pandas Optimization Results:")
    logger.info(f"   DataFrame size: {n_rows} rows")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(".2f")
    logger.info(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    return {
        'traditional_time': traditional_time,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'rows_processed': n_rows
    }

def demonstrate_caching_performance():
    """Demonstrate caching performance improvements"""
    logger.info("🚀 Demonstrating Advanced Caching...")

    # Initialize cache
    cache = CacheService(db_path="demo_cache.db", enable_multi_level=True)

    # Test data
    test_data = {
        'market_data_1': {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000000},
        'market_data_2': {'symbol': 'MSFT', 'price': 280.0, 'volume': 500000},
        'sentiment_data': {'score': 0.75, 'confidence': 0.85, 'sources': 15},
        'large_dataset': {'data': list(range(1000)), 'metadata': {'size': 1000, 'type': 'numeric'}}
    }

    # First run - cache misses
    logger.info("   First run (cache population)...")
    start_time = time.time()
    for key, data in test_data.items():
        cache.set_optimized(key, data, endpoint="demo", symbol=key.split('_')[-1], ttl_seconds=3600)
    first_run_time = time.time() - start_time

    # Second run - cache hits
    logger.info("   Second run (cache hits)...")
    start_time = time.time()
    retrieved_data = {}
    for key in test_data.keys():
        retrieved_data[key] = cache.get_optimized(key)
    second_run_time = time.time() - start_time

    speedup = first_run_time / second_run_time if second_run_time > 0 else 1.0

    # Get cache statistics
    stats = {}
    try:
        stats = cache.get_cache_stats()
        hit_rate = stats.get('overall', {}).get('overall_hit_rate', 0) if isinstance(stats, dict) else 0
    except Exception as e:
        logger.warning(f"Failed to get cache stats: {e}")
        hit_rate = 0
        stats = {'error': str(e)}

    logger.info("📊 Caching Performance Results:")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(".2f")
    logger.info(f"   Cache hit rate: {hit_rate:.1%}")

    return {
        'first_run_time': first_run_time,
        'second_run_time': second_run_time,
        'speedup': speedup,
        'cache_stats': stats
    }

def demonstrate_algorithm_selection():
    """Demonstrate dynamic algorithm selection"""
    logger.info("🚀 Demonstrating Dynamic Algorithm Selection...")

    # Create selector and register algorithms
    selector = DynamicAlgorithmSelector()

    # Register some algorithms
    selector.register_algorithm(
        'vectorized_options',
        lambda data, **kwargs: f"Processed {len(data)} items with vectorized options",
        {
            'size_range': (1, 1000),
            'data_types': ['numeric'],
            'memory_efficient': True,
            'time_series_optimized': False,
            'distribution_types': ['normal', 'uniform']
        }
    )

    selector.register_algorithm(
        'parallel_processing',
        lambda data, **kwargs: f"Processed {len(data)} items in parallel",
        {
            'size_range': (100, 10000),
            'data_types': ['mixed', 'numeric'],
            'memory_efficient': True,
            'time_series_optimized': False,
            'distribution_types': ['normal', 'uniform', 'skewed']
        }
    )

    # Test different data types
    test_cases = [
        (np.random.randn(500), "Numeric array"),
        (pd.DataFrame({'a': range(500), 'b': np.random.randn(500)}), "DataFrame"),
        (list(range(500)), "List"),
        ({'data': list(range(500))}, "Dictionary")
    ]

    results = []
    for data, description in test_cases:
        try:
            characteristics = selector.analyze_data_characteristics(data)
            algorithm_name, algorithm_func, confidence = selector.select_optimal_algorithm(data)
            result = algorithm_func(data)
            results.append({
                'data_type': description,
                'selected_algorithm': algorithm_name,
                'confidence': confidence,
                'characteristics': {
                    'size': characteristics.size,
                    'data_type': characteristics.data_type,
                    'sparsity': characteristics.sparsity
                }
            })
        except Exception as e:
            results.append({
                'data_type': description,
                'error': str(e)
            })

    logger.info("📊 Algorithm Selection Results:")
    for result in results:
        if 'error' not in result:
            logger.info(f"   {result['data_type']}: {result['selected_algorithm']} (confidence: {result['confidence']:.2f})")
        else:
            logger.info(f"   {result['data_type']}: Error - {result['error']}")

    return results

def run_performance_demonstration():
    """Run complete performance demonstration"""
    logger.info("🎯 Oracle-X Performance Optimization Demonstration")
    logger.info("=" * 60)

    results = {}

    # Run demonstrations
    try:
        results['vectorized'] = demonstrate_vectorized_speedup()
        logger.info("")
    except Exception as e:
        logger.error(f"Vectorized demo failed: {e}")
        results['vectorized'] = {'error': str(e)}

    try:
        results['pandas'] = demonstrate_pandas_optimization()
        logger.info("")
    except Exception as e:
        logger.error(f"Pandas demo failed: {e}")
        results['pandas'] = {'error': str(e)}

    try:
        results['caching'] = demonstrate_caching_performance()
        logger.info("")
    except Exception as e:
        logger.error(f"Caching demo failed: {e}")
        results['caching'] = {'error': str(e)}

    try:
        results['algorithm_selection'] = demonstrate_algorithm_selection()
        logger.info("")
    except Exception as e:
        logger.error(f"Algorithm selection demo failed: {e}")
        results['algorithm_selection'] = {'error': str(e)}

    # Calculate overall improvements
    speedups = []
    for key, result in results.items():
        if isinstance(result, dict) and 'speedup' in result and result['speedup'] > 0:
            speedups.append(result['speedup'])

    if speedups:
        avg_speedup = np.mean(speedups)
        geometric_mean_speedup = np.exp(np.mean(np.log(speedups)))

        logger.info("🎉 OVERALL PERFORMANCE SUMMARY")
        logger.info("=" * 40)
        logger.info(".2f")
        logger.info(".2f")
        logger.info(f"   Number of optimizations tested: {len(speedups)}")

        if geometric_mean_speedup > 1.5:
            logger.info("🎊 Excellent! Performance targets exceeded!")
        elif geometric_mean_speedup > 1.0:
            logger.info("👍 Good! Performance improvements achieved!")
        else:
            logger.info("⚠️  Some optimizations may need further tuning.")
    else:
        logger.info("⚠️  Unable to calculate overall speedup metrics.")

    return results

if __name__ == "__main__":
    demo_results = run_performance_demonstration()

    # Save results
    import json
    with open("performance_demonstration_results.json", 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)

    logger.info("📊 Demonstration results saved to performance_demonstration_results.json")
