#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Oracle-X Performance Optimizations
Tests Phase 2.3 Algorithm Optimization, Phase 2.1 Advanced Caching, and Phase 2.2 ML Optimization
"""

import time
import numpy as np
import pandas as pd
import logging
import json
import psutil
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import optimized modules
from core.optimized_algorithms import (
    VectorizedOptionsEngine,
    OptimizedDataProcessor,
    DynamicAlgorithmSelector,
    select_optimal_algorithm,
    get_vectorized_options_price,
    optimize_dataframe,
    parallel_process_data
)

from core.optimized_pandas import (
    OptimizedPandasProcessor,
    optimize_dataframe as pandas_optimize,
    vectorized_query,
    efficient_groupby,
    rolling_window_analysis,
    memory_efficient_merge,
    process_large_csv
)

from data_feeds.cache_service import CacheService
from oracle_engine.optimized_ml_engine import (
    OptimizedMLPredictionEngine,
    ModelConfig,
    PredictionType,
    OptimizationLevel,
    create_optimized_ml_engine
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Performance benchmarking utility"""

    def __init__(self):
        self.results = {}
        self.baseline_results = {}

    def measure_time(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure execution time and memory usage"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            return {
                'execution_time': end_time - start_time,
                'memory_usage': end_memory - start_memory,
                'peak_memory': end_memory,
                'result': result,
                'success': True
            }
        except Exception as e:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            return {
                'execution_time': end_time - start_time,
                'memory_usage': end_memory - start_memory,
                'error': str(e),
                'success': False
            }

    def compare_performance(self, baseline_key: str, optimized_key: str) -> Dict[str, Any]:
        """Compare baseline vs optimized performance"""
        if baseline_key not in self.baseline_results or optimized_key not in self.results:
            return {'error': 'Missing benchmark data'}

        baseline = self.baseline_results[baseline_key]
        optimized = self.results[optimized_key]

        if not baseline.get('success', False) or not optimized.get('success', False):
            return {'error': 'Benchmark failed'}

        speedup = baseline['execution_time'] / optimized['execution_time'] if optimized['execution_time'] > 0 else float('inf')
        memory_reduction = (baseline['memory_usage'] - optimized['memory_usage']) / baseline['memory_usage'] * 100 if baseline['memory_usage'] > 0 else 0

        return {
            'speedup_ratio': speedup,
            'memory_reduction_percent': memory_reduction,
            'baseline_time': baseline['execution_time'],
            'optimized_time': optimized['execution_time'],
            'baseline_memory': baseline['memory_usage'],
            'optimized_memory': optimized['memory_usage']
        }

class OracleXOptimizationTester:
    """Comprehensive tester for all Oracle-X optimizations"""

    def __init__(self):
        self.benchmark = PerformanceBenchmark()
        self.test_data = self._generate_test_data()

    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate comprehensive test data"""
        np.random.seed(42)

        # Generate options pricing data
        n_samples = 1000
        options_data = {
            'S': np.random.uniform(50, 200, n_samples),  # Stock prices
            'K': np.random.uniform(50, 200, n_samples),  # Strike prices
            'T': np.random.uniform(0.1, 2.0, n_samples), # Time to maturity
            'r': np.random.uniform(0.01, 0.05, n_samples), # Risk-free rate
            'sigma': np.random.uniform(0.1, 0.5, n_samples), # Volatility
            'option_type': np.random.choice(['call', 'put'], n_samples)
        }

        # Generate DataFrame for pandas operations
        df_data = {
            'timestamp': pd.date_range('2023-01-01', periods=5000, freq='1min'),
            'price': np.random.uniform(100, 200, 5000),
            'volume': np.random.randint(1000, 10000, 5000),
            'sentiment_score': np.random.uniform(-1, 1, 5000),
            'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'TSLA'], 5000)
        }
        df = pd.DataFrame(df_data)

        # Generate ML features
        n_ml_samples = 1000
        ml_features = np.random.randn(n_ml_samples, 20)
        ml_targets = np.random.randn(n_ml_samples)

        return {
            'options_data': options_data,
            'dataframe': df,
            'ml_features': ml_features,
            'ml_targets': ml_targets,
            'large_dataset': self._generate_large_dataset()
        }

    def _generate_large_dataset(self) -> pd.DataFrame:
        """Generate large dataset for memory efficiency testing"""
        n_rows = 50000
        data = {
            'id': range(n_rows),
            'value1': np.random.randn(n_rows),
            'value2': np.random.randn(n_rows),
            'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
            'timestamp': pd.date_range('2023-01-01', periods=n_rows, freq='1min')
        }
        return pd.DataFrame(data)

    def test_vectorized_algorithms(self) -> Dict[str, Any]:
        """Test Phase 2.3.1: Vectorized Operations"""
        logger.info("🧪 Testing Vectorized Operations...")

        data = self.test_data['options_data']
        n_tests = 100

        # Test vectorized binomial pricing
        engine = VectorizedOptionsEngine()

        def baseline_pricing():
            """Traditional loop-based pricing approximation"""
            results = []
            for i in range(min(n_tests, len(data['S']))):
                S, K, T, r, sigma, option_type = (
                    data['S'][i], data['K'][i], data['T'][i],
                    data['r'][i], data['sigma'][i], data['option_type'][i]
                )

                # Simple Black-Scholes approximation for baseline
                d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)

                # Approximate CDF with simple function
                def approx_cdf(x):
                    return 0.5 * (1 + np.tanh(x * 0.8))

                if option_type == 'call':
                    price = S * approx_cdf(d1) - K * np.exp(-r * T) * approx_cdf(d2)
                else:
                    price = K * np.exp(-r * T) * approx_cdf(-d2) - S * approx_cdf(-d1)

                results.append(price)
            return results

        def optimized_pricing():
            """Vectorized pricing"""
            results = []
            for i in range(min(n_tests, len(data['S']))):
                price = engine.price_binomial_vectorized(
                    data['S'][i], data['K'][i], data['T'][i],
                    data['r'][i], data['sigma'][i], data['option_type'][i]
                )
                results.append(price)
            return results

        # Measure performance
        baseline_result = self.benchmark.measure_time(baseline_pricing)
        optimized_result = self.benchmark.measure_time(optimized_pricing)

        self.benchmark.baseline_results['vectorized_pricing'] = baseline_result
        self.benchmark.results['vectorized_pricing'] = optimized_result

        comparison = self.benchmark.compare_performance('vectorized_pricing', 'vectorized_pricing')

        return {
            'test_name': 'Vectorized Options Pricing',
            'baseline_time': baseline_result['execution_time'],
            'optimized_time': optimized_result['execution_time'],
            'speedup': comparison.get('speedup_ratio', 0),
            'memory_reduction': comparison.get('memory_reduction_percent', 0),
            'success': optimized_result['success']
        }

    def test_optimized_pandas(self) -> Dict[str, Any]:
        """Test Phase 2.3.2: Optimized Pandas Operations"""
        logger.info("🧪 Testing Optimized Pandas Operations...")

        df = self.test_data['dataframe'].copy()

        def baseline_operations():
            """Traditional pandas operations"""
            result = df.copy()

            # Traditional operations
            result['price_change'] = result['price'].diff()
            result['rolling_mean'] = result['price'].rolling(20).mean()
            result['sentiment_category'] = result['sentiment_score'].apply(
                lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
            )

            # Groupby operation
            grouped = result.groupby('symbol').agg({
                'price': 'mean',
                'volume': 'sum',
                'sentiment_score': 'mean'
            })

            return result, grouped

        def optimized_operations():
            """Optimized pandas operations"""
            processor = OptimizedPandasProcessor()

            # Optimize schema
            opt_df = processor.optimize_dataframe_schema(df.copy())

            # Vectorized operations
            opt_df['price_change'] = opt_df['price'].diff()
            opt_df['rolling_mean'] = opt_df['price'].rolling(20).mean()
            opt_df['sentiment_category'] = np.select(
                [opt_df['sentiment_score'] > 0.1, opt_df['sentiment_score'] < -0.1],
                ['positive', 'negative'],
                default='neutral'
            )

            # Efficient groupby
            grouped = processor.efficient_groupby_operations(
                opt_df, ['symbol'],
                {'price': ['mean'], 'volume': ['sum'], 'sentiment_score': ['mean']}
            )

            return opt_df, grouped

        # Measure performance
        baseline_result = self.benchmark.measure_time(baseline_operations)
        optimized_result = self.benchmark.measure_time(optimized_operations)

        self.benchmark.baseline_results['pandas_operations'] = baseline_result
        self.benchmark.results['pandas_operations'] = optimized_result

        comparison = self.benchmark.compare_performance('pandas_operations', 'pandas_operations')

        return {
            'test_name': 'Optimized Pandas Operations',
            'baseline_time': baseline_result['execution_time'],
            'optimized_time': optimized_result['execution_time'],
            'speedup': comparison.get('speedup_ratio', 0),
            'memory_reduction': comparison.get('memory_reduction_percent', 0),
            'success': optimized_result['success']
        }

    def test_dynamic_algorithm_selection(self) -> Dict[str, Any]:
        """Test Phase 2.3.3: Dynamic Algorithm Selection"""
        logger.info("🧪 Testing Dynamic Algorithm Selection...")

        # Test data characteristics analysis
        test_data = [
            np.random.randn(1000),  # Numeric array
            self.test_data['dataframe'],  # DataFrame
            list(range(1000)),  # List
            {"test": "data"}  # Generic object
        ]

        selector = DynamicAlgorithmSelector()

        def baseline_selection():
            """Manual algorithm selection"""
            results = []
            for data in test_data:
                if isinstance(data, np.ndarray):
                    algo_type = 'numeric'
                elif isinstance(data, pd.DataFrame):
                    algo_type = 'dataframe'
                else:
                    algo_type = 'generic'
                results.append(algo_type)
            return results

        def optimized_selection():
            """Dynamic algorithm selection"""
            results = []
            for data in test_data:
                try:
                    characteristics = selector.analyze_data_characteristics(data)
                    algorithm_name, algorithm_func, confidence = selector.select_optimal_algorithm(data)
                    results.append({
                        'characteristics': characteristics,
                        'selected_algorithm': algorithm_name,
                        'confidence': confidence
                    })
                except Exception as e:
                    logger.warning(f"Algorithm selection failed for data type {type(data)}: {e}")
                    results.append({'error': str(e)})
            return results

        # Measure performance
        baseline_result = self.benchmark.measure_time(baseline_selection)
        optimized_result = self.benchmark.measure_time(optimized_selection)

        self.benchmark.baseline_results['algorithm_selection'] = baseline_result
        self.benchmark.results['algorithm_selection'] = optimized_result

        comparison = self.benchmark.compare_performance('algorithm_selection', 'algorithm_selection')

        return {
            'test_name': 'Dynamic Algorithm Selection',
            'baseline_time': baseline_result['execution_time'],
            'optimized_time': optimized_result['execution_time'],
            'speedup': comparison.get('speedup_ratio', 0),
            'memory_reduction': comparison.get('memory_reduction_percent', 0),
            'success': optimized_result['success']
        }

    def test_advanced_caching(self) -> Dict[str, Any]:
        """Test Phase 2.1: Advanced Caching Strategies"""
        logger.info("🧪 Testing Advanced Caching...")

        # Initialize cache service
        cache = CacheService(db_path="test_cache.db", enable_multi_level=True)

        # Test data
        test_data = {
            'market_data': {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000000},
            'sentiment_analysis': {'score': 0.75, 'confidence': 0.85},
            'large_dataset': {'data': list(range(10000))}  # Large data for compression
        }

        def baseline_caching():
            """Simple dictionary-based caching"""
            cache_dict = {}
            results = []

            for key, data in test_data.items():
                # Simple put/get
                cache_dict[key] = data
                retrieved = cache_dict.get(key)
                results.append(retrieved)

            return results

        def optimized_caching():
            """Multi-level caching with compression"""
            results = []

            for key, data in test_data.items():
                # Store with optimization
                cache.set_optimized(key, data, endpoint="test", symbol="TEST", ttl_seconds=3600)

                # Retrieve with optimization
                retrieved = cache.get_optimized(key)
                results.append(retrieved)

            return results

        # Measure performance
        baseline_result = self.benchmark.measure_time(baseline_caching)
        optimized_result = self.benchmark.measure_time(optimized_caching)

        self.benchmark.baseline_results['advanced_caching'] = baseline_result
        self.benchmark.results['advanced_caching'] = optimized_result

        comparison = self.benchmark.compare_performance('advanced_caching', 'advanced_caching')

        # Get cache statistics
        try:
            cache_stats = cache.get_cache_stats()
            compression_stats = cache.get_compression_stats()
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            cache_stats = {'error': str(e)}
            compression_stats = {'error': str(e)}

        return {
            'test_name': 'Advanced Caching',
            'baseline_time': baseline_result['execution_time'],
            'optimized_time': optimized_result['execution_time'],
            'speedup': comparison.get('speedup_ratio', 0),
            'memory_reduction': comparison.get('memory_reduction_percent', 0),
            'cache_stats': cache_stats,
            'compression_stats': compression_stats,
            'success': optimized_result['success']
        }

    def test_ml_optimization(self) -> Dict[str, Any]:
        """Test Phase 2.2: ML Model Optimization"""
        logger.info("🧪 Testing ML Model Optimization...")

        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split

            # Prepare data
            X = self.test_data['ml_features']
            y = self.test_data['ml_targets']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create baseline model
            baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
            baseline_model.fit(X_train, y_train)

            def baseline_prediction():
                """Baseline ML prediction"""
                predictions = []
                for i in range(min(100, len(X_test))):
                    pred = baseline_model.predict(X_test[i:i+1])[0]
                    predictions.append(pred)
                return predictions

            # Create optimized engine
            config = ModelConfig(
                prediction_type=PredictionType.PRICE_MOVEMENT,
                optimization_level=OptimizationLevel.FP16,
                use_onnx=True,
                use_quantization=True,
                batch_size=32
            )

            engine = OptimizedMLPredictionEngine(config)

            # Optimize model
            model_id = "test_optimized_model"
            optimization_result = engine.optimize_model(baseline_model, model_id, X_train.shape)

            def optimized_prediction():
                """Optimized ML prediction"""
                predictions = []
                for i in range(min(100, len(X_test))):
                    result = engine.predict_optimized(model_id, X_test[i])
                    predictions.append(result['prediction'])
                return predictions

            # Measure performance
            baseline_result = self.benchmark.measure_time(baseline_prediction)
            optimized_result = self.benchmark.measure_time(optimized_prediction)

            self.benchmark.baseline_results['ml_optimization'] = baseline_result
            self.benchmark.results['ml_optimization'] = optimized_result

            comparison = self.benchmark.compare_performance('ml_optimization', 'ml_optimization')

            return {
                'test_name': 'ML Model Optimization',
                'baseline_time': baseline_result['execution_time'],
                'optimized_time': optimized_result['execution_time'],
                'speedup': comparison.get('speedup_ratio', 0),
                'memory_reduction': comparison.get('memory_reduction_percent', 0),
                'optimization_metrics': optimization_result.get('optimization_metrics', {}),
                'success': optimized_result['success']
            }

        except ImportError:
            logger.warning("scikit-learn not available, skipping ML optimization test")
            return {
                'test_name': 'ML Model Optimization',
                'error': 'scikit-learn not available',
                'success': False
            }

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        logger.info("🚀 Starting Comprehensive Oracle-X Optimization Tests...")

        test_results = {
            'timestamp': time.time(),
            'tests': {}
        }

        # Run all tests
        test_functions = [
            self.test_vectorized_algorithms,
            self.test_optimized_pandas,
            self.test_dynamic_algorithm_selection,
            self.test_advanced_caching,
            self.test_ml_optimization
        ]

        for test_func in test_functions:
            try:
                result = test_func()
                test_results['tests'][result['test_name']] = result
                logger.info(f"✅ {result['test_name']}: {'PASSED' if result['success'] else 'FAILED'}")
                if result['success'] and 'speedup' in result:
                    logger.info(".2f")
            except Exception as e:
                logger.error(f"❌ {test_func.__name__} failed: {e}")
                test_results['tests'][test_func.__name__] = {
                    'test_name': test_func.__name__,
                    'error': str(e),
                    'success': False
                }

        # Calculate overall metrics
        successful_tests = [r for r in test_results['tests'].values() if r.get('success', False)]
        speedup_values = [r.get('speedup', 1.0) for r in successful_tests if r.get('speedup', 1.0) > 0]

        # Calculate geometric mean of speedups for combined effect
        if speedup_values:
            combined_speedup = np.exp(np.mean(np.log(speedup_values)))
        else:
            combined_speedup = 1.0

        test_results['summary'] = {
            'total_tests': len(test_results['tests']),
            'successful_tests': len(successful_tests),
            'success_rate': len(successful_tests) / len(test_results['tests']) * 100,
            'combined_speedup': combined_speedup,
            'average_memory_reduction': np.mean([r.get('memory_reduction', 0) for r in successful_tests]) if successful_tests else 0
        }

        logger.info("🎯 Test Summary:")
        logger.info(f"   Total Tests: {test_results['summary']['total_tests']}")
        logger.info(f"   Successful: {test_results['summary']['successful_tests']}")
        logger.info(".1f")
        logger.info(".2f")
        logger.info(".1f")

        return test_results

def main():
    """Main test execution"""
    tester = OracleXOptimizationTester()
    results = tester.run_comprehensive_tests()

    # Save results
    output_file = "optimization_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"📊 Results saved to {output_file}")

    # Print key findings
    summary = results.get('summary', {})
    if summary.get('combined_speedup', 1.0) > 1.5:
        logger.info("🎉 Excellent! Combined speedup exceeds 1.5x target!")
    elif summary.get('combined_speedup', 1.0) > 1.0:
        logger.info("👍 Good! Performance improvements achieved.")
    else:
        logger.info("⚠️  Performance improvements may need further optimization.")

if __name__ == "__main__":
    main()
