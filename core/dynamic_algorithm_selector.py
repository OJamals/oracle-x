"""
Dynamic Algorithm Selection Framework for Oracle-X
Automatically selects optimal algorithms based on data characteristics and performance profiling
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataCharacteristics:
    """Data characteristics for algorithm selection"""
    size: int
    dimensionality: int
    data_type: str
    sparsity: float
    memory_usage: int
    has_nulls: bool
    is_time_series: bool
    distribution_type: str  # 'normal', 'skewed', 'uniform', 'unknown'

@dataclass
class AlgorithmPerformance:
    """Performance metrics for algorithm evaluation"""
    execution_time: float
    memory_usage: int
    accuracy_score: float
    scalability_score: float
    timestamp: float

class AlgorithmProfile:
    """Profile for a specific algorithm implementation"""

    def __init__(self, name: str, algorithm_func: Callable, optimal_conditions: Dict[str, Any]):
        self.name = name
        self.algorithm_func = algorithm_func
        self.optimal_conditions = optimal_conditions
        self.performance_history: List[AlgorithmPerformance] = []
        self.usage_count = 0

    def is_optimal_for(self, characteristics: DataCharacteristics) -> float:
        """
        Calculate how optimal this algorithm is for given data characteristics
        Returns score between 0-1 (1 being perfect match)
        """
        score = 0.0
        conditions = self.optimal_conditions

        # Size matching
        if 'size_range' in conditions:
            min_size, max_size = conditions['size_range']
            if min_size <= characteristics.size <= max_size:
                score += 0.3
            elif characteristics.size < min_size:
                score += 0.1  # Suboptimal but workable
            else:
                score += 0.05  # May struggle with large data

        # Data type matching
        if 'data_types' in conditions and characteristics.data_type in conditions['data_types']:
            score += 0.2

        # Sparsity matching
        if 'sparsity_range' in conditions:
            min_sparse, max_sparse = conditions['sparsity_range']
            if min_sparse <= characteristics.sparsity <= max_sparse:
                score += 0.15

        # Memory efficiency
        if 'memory_efficient' in conditions and conditions['memory_efficient']:
            if characteristics.memory_usage < psutil.virtual_memory().available * 0.5:
                score += 0.15

        # Time series optimization
        if 'time_series_optimized' in conditions and conditions['time_series_optimized'] == characteristics.is_time_series:
            score += 0.1

        # Distribution matching
        if 'distribution_types' in conditions and characteristics.distribution_type in conditions['distribution_types']:
            score += 0.1

        return min(score, 1.0)

    def record_performance(self, execution_time: float, memory_usage: int,
                          accuracy_score: float = 1.0):
        """Record performance metrics for this algorithm"""
        performance = AlgorithmPerformance(
            execution_time=execution_time,
            memory_usage=memory_usage,
            accuracy_score=accuracy_score,
            scalability_score=self._calculate_scalability_score(),
            timestamp=time.time()
        )
        self.performance_history.append(performance)
        self.usage_count += 1

    def _calculate_scalability_score(self) -> float:
        """Calculate scalability score based on performance history"""
        if len(self.performance_history) < 2:
            return 0.5

        # Analyze performance trend
        recent_performances = self.performance_history[-10:]
        times = [p.execution_time for p in recent_performances]

        # Calculate coefficient of variation (lower is better for scalability)
        if np.mean(times) > 0:
            cv = np.std(times) / np.mean(times)
            scalability = max(0, 1 - cv)  # Convert to 0-1 scale
        else:
            scalability = 0.5

        return scalability

    def get_average_performance(self) -> Optional[AlgorithmPerformance]:
        """Get average performance metrics"""
        if not self.performance_history:
            return None

        avg_time = np.mean([p.execution_time for p in self.performance_history])
        avg_memory = np.mean([p.memory_usage for p in self.performance_history])
        avg_accuracy = np.mean([p.accuracy_score for p in self.performance_history])
        avg_scalability = np.mean([p.scalability_score for p in self.performance_history])

        return AlgorithmPerformance(
            execution_time=avg_time,
            memory_usage=int(avg_memory),
            accuracy_score=avg_accuracy,
            scalability_score=avg_scalability,
            timestamp=time.time()
        )

class DynamicAlgorithmSelector:
    """Dynamic algorithm selection engine"""

    def __init__(self):
        self.algorithms: Dict[str, AlgorithmProfile] = {}
        self.performance_cache: Dict[str, AlgorithmPerformance] = {}
        self.selection_history: List[Dict[str, Any]] = []

    def register_algorithm(self, name: str, algorithm_func: Callable,
                          optimal_conditions: Dict[str, Any]):
        """Register a new algorithm with its optimal conditions"""
        profile = AlgorithmProfile(name, algorithm_func, optimal_conditions)
        self.algorithms[name] = profile
        logger.info(f"Registered algorithm: {name}")

    def analyze_data_characteristics(self, data: Any) -> DataCharacteristics:
        """Analyze data to determine characteristics for algorithm selection"""
        if isinstance(data, pd.DataFrame):
            return self._analyze_dataframe(data)
        elif isinstance(data, np.ndarray):
            return self._analyze_array(data)
        elif isinstance(data, list):
            return self._analyze_list(data)
        else:
            return self._analyze_generic(data)

    def _analyze_dataframe(self, df: pd.DataFrame) -> DataCharacteristics:
        """Analyze pandas DataFrame characteristics"""
        size = len(df)
        dimensionality = len(df.columns)

        # Determine data type
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > len(df.columns) * 0.7:
            data_type = 'numeric'
        elif df.select_dtypes(include=['datetime']).columns.any():
            data_type = 'time_series'
        else:
            data_type = 'mixed'

        # Calculate sparsity
        total_cells = size * dimensionality
        null_cells = df.isnull().sum().sum()
        sparsity = null_cells / total_cells if total_cells > 0 else 0

        # Estimate memory usage
        memory_usage = df.memory_usage(deep=True).sum()

        # Check for time series characteristics
        is_time_series = False
        if 'timestamp' in df.columns or 'date' in df.columns:
            is_time_series = True

        # Analyze distribution
        distribution_type = 'unknown'
        if len(numeric_cols) > 0:
            sample_col = df[numeric_cols[0]].dropna()
            if len(sample_col) > 10:
                skewness = sample_col.skew()
                if abs(skewness) < 0.5:
                    distribution_type = 'normal'
                elif abs(skewness) > 1:
                    distribution_type = 'skewed'
                else:
                    distribution_type = 'uniform'

        return DataCharacteristics(
            size=size,
            dimensionality=dimensionality,
            data_type=data_type,
            sparsity=sparsity,
            memory_usage=memory_usage,
            has_nulls=null_cells > 0,
            is_time_series=is_time_series,
            distribution_type=distribution_type
        )

    def _analyze_array(self, arr: np.ndarray) -> DataCharacteristics:
        """Analyze numpy array characteristics"""
        size = arr.size
        dimensionality = arr.ndim

        data_type = 'numeric' if np.issubdtype(arr.dtype, np.number) else 'mixed'
        sparsity = np.isnan(arr).sum() / size if size > 0 else 0
        memory_usage = arr.nbytes

        return DataCharacteristics(
            size=size,
            dimensionality=dimensionality,
            data_type=data_type,
            sparsity=sparsity,
            memory_usage=memory_usage,
            has_nulls=np.isnan(arr).any() if np.issubdtype(arr.dtype, np.number) else False,
            is_time_series=False,  # Assume not time series unless specified
            distribution_type='unknown'
        )

    def _analyze_list(self, data_list: list) -> DataCharacteristics:
        """Analyze list characteristics"""
        size = len(data_list)
        dimensionality = 1

        # Sample first few items to determine type
        sample = data_list[:min(10, size)]
        if all(isinstance(x, (int, float)) for x in sample):
            data_type = 'numeric'
        elif all(isinstance(x, str) for x in sample):
            data_type = 'text'
        else:
            data_type = 'mixed'

        sparsity = sum(1 for x in data_list if x is None) / size if size > 0 else 0
        memory_usage = sum(len(str(x)) for x in data_list)  # Rough estimate

        return DataCharacteristics(
            size=size,
            dimensionality=dimensionality,
            data_type=data_type,
            sparsity=sparsity,
            memory_usage=memory_usage,
            has_nulls=any(x is None for x in data_list),
            is_time_series=False,
            distribution_type='unknown'
        )

    def _analyze_generic(self, data: Any) -> DataCharacteristics:
        """Fallback analysis for unknown data types"""
        try:
            size = len(data) if hasattr(data, '__len__') else 1
        except:
            size = 1

        return DataCharacteristics(
            size=size,
            dimensionality=1,
            data_type='unknown',
            sparsity=0.0,
            memory_usage=0,
            has_nulls=False,
            is_time_series=False,
            distribution_type='unknown'
        )

    def select_optimal_algorithm(self, data: Any, algorithm_type: str = 'general') -> Tuple[str, Callable, float]:
        """
        Select the optimal algorithm for given data
        Returns: (algorithm_name, algorithm_function, confidence_score)
        """
        characteristics = self.analyze_data_characteristics(data)

        # Filter algorithms by type if specified
        candidates = {
            name: profile for name, profile in self.algorithms.items()
            if algorithm_type in name or algorithm_type == 'general'
        }

        if not candidates:
            # Fallback to any available algorithm
            candidates = self.algorithms

        if not candidates:
            raise ValueError("No algorithms registered")

        # Score each candidate
        scores = {}
        for name, profile in candidates.items():
            score = profile.is_optimal_for(characteristics)
            scores[name] = score

        # Select best algorithm
        best_algorithm = max(scores, key=scores.get)
        confidence = scores[best_algorithm]

        # Record selection for learning
        selection_record = {
            'timestamp': time.time(),
            'data_characteristics': characteristics,
            'selected_algorithm': best_algorithm,
            'confidence': confidence,
            'algorithm_type': algorithm_type,
            'available_candidates': list(candidates.keys())
        }
        self.selection_history.append(selection_record)

        logger.info(f"Selected algorithm '{best_algorithm}' with confidence {confidence:.2f}")

        return best_algorithm, self.algorithms[best_algorithm].algorithm_func, confidence

    def execute_with_adaptive_selection(self, data: Any, algorithm_type: str = 'general',
                                       **kwargs) -> Any:
        """
        Execute algorithm with automatic selection and performance monitoring
        """
        algorithm_name, algorithm_func, confidence = self.select_optimal_algorithm(data, algorithm_type)

        # Monitor execution
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            result = algorithm_func(data, **kwargs)

            # Record performance
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss
            memory_usage = end_memory - start_memory

            self.algorithms[algorithm_name].record_performance(
                execution_time=execution_time,
                memory_usage=memory_usage,
                accuracy_score=1.0  # Assume success
            )

            logger.info(f"Algorithm '{algorithm_name}' executed in {execution_time:.3f}s")

            return result

        except Exception as e:
            # Record failed performance
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss
            memory_usage = end_memory - start_memory

            self.algorithms[algorithm_name].record_performance(
                execution_time=execution_time,
                memory_usage=memory_usage,
                accuracy_score=0.0  # Failed execution
            )

            logger.error(f"Algorithm '{algorithm_name}' failed: {e}")

            # Try fallback algorithm if available
            if confidence < 0.7 and len(self.algorithms) > 1:
                logger.info("Attempting fallback algorithm...")
                # Remove failed algorithm from candidates
                fallback_candidates = {k: v for k, v in self.algorithms.items() if k != algorithm_name}
                if fallback_candidates:
                    fallback_name = list(fallback_candidates.keys())[0]
                    logger.info(f"Using fallback algorithm: {fallback_name}")
                    return fallback_candidates[fallback_name].algorithm_func(data, **kwargs)

            raise e

    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics for all algorithms"""
        analytics = {}

        for name, profile in self.algorithms.items():
            avg_performance = profile.get_average_performance()
            if avg_performance:
                analytics[name] = {
                    'usage_count': profile.usage_count,
                    'avg_execution_time': avg_performance.execution_time,
                    'avg_memory_usage': avg_performance.memory_usage,
                    'avg_accuracy': avg_performance.accuracy_score,
                    'scalability_score': avg_performance.scalability_score
                }

        return analytics

# Global instance
_algorithm_selector = DynamicAlgorithmSelector()

def register_algorithm(name: str, algorithm_func: Callable, optimal_conditions: Dict[str, Any]):
    """Register a new algorithm for dynamic selection"""
    _algorithm_selector.register_algorithm(name, algorithm_func, optimal_conditions)

def select_optimal_algorithm(data: Any, algorithm_type: str = 'general') -> Tuple[str, Callable, float]:
    """Select optimal algorithm for given data"""
    return _algorithm_selector.select_optimal_algorithm(data, algorithm_type)

def execute_adaptive(data: Any, algorithm_type: str = 'general', **kwargs) -> Any:
    """Execute with automatic algorithm selection"""
    return _algorithm_selector.execute_with_adaptive_selection(data, algorithm_type, **kwargs)

def get_algorithm_analytics() -> Dict[str, Any]:
    """Get performance analytics"""
    return _algorithm_selector.get_performance_analytics()
