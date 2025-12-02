#!/usr/bin/env python3
"""
Optimized ML Prediction Engine - Phase 2.2 Implementation
High-performance ML inference with quantization, batch processing, and ONNX optimization
Targets: 2-3x faster inference, 50% smaller models
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import hashlib
import json
import mmap
import os
import pickle
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ML libraries with fallbacks
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (accuracy_score, classification_report,
                                 mean_squared_error)
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available for ML optimization")

# ONNX Runtime for optimized inference
try:
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning(
        "ONNX Runtime not available. Install with: pip install onnxruntime skl2onnx"
    )

# PyTorch for quantization
try:
    import torch

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available for quantization")

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Model optimization levels"""

    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    DYNAMIC = "dynamic"


class PredictionType(Enum):
    """Types of predictions the engine can make"""

    PRICE_MOVEMENT = "price_movement"
    VOLATILITY = "volatility"
    SENTIMENT_IMPACT = "sentiment_impact"
    TECHNICAL_SIGNAL = "technical_signal"
    OPTIONS_STRATEGY = "options_strategy"


@dataclass
class ModelConfig:
    """Configuration for optimized model"""

    prediction_type: PredictionType
    optimization_level: OptimizationLevel = OptimizationLevel.NONE
    batch_size: int = 32
    use_onnx: bool = True
    use_quantization: bool = False
    memory_map: bool = True
    cache_enabled: bool = True
    max_cache_size: int = 100


@dataclass
class BatchPredictionRequest:
    """Request for batch prediction processing"""

    features: np.ndarray
    prediction_type: PredictionType
    model_config: ModelConfig
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizedModelMetrics:
    """Performance metrics for optimized models"""

    inference_time: float
    memory_usage: float
    model_size: float
    throughput: float
    accuracy_score: float
    optimization_ratio: float


class ModelQuantizer:
    """Handles model quantization for reduced memory footprint"""

    @staticmethod
    def quantize_sklearn_model(model, target_precision: str = "float16") -> Any:
        """Quantize scikit-learn model to lower precision"""
        if not SKLEARN_AVAILABLE:
            return model

        try:
            if target_precision == "float16":
                # Convert model parameters to float16
                if hasattr(model, "estimators_"):
                    # For ensemble models like RandomForest
                    for estimator in model.estimators_:
                        ModelQuantizer._quantize_estimator(estimator, np.float16)
                else:
                    ModelQuantizer._quantize_estimator(model, np.float16)

            elif target_precision == "int8":
                # Dynamic quantization to int8
                if hasattr(model, "estimators_"):
                    for estimator in model.estimators_:
                        ModelQuantizer._quantize_estimator(estimator, np.int8)
                else:
                    ModelQuantizer._quantize_estimator(model, np.int8)

            logger.info(f"Model quantized to {target_precision}")
            return model

        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model

    @staticmethod
    def _quantize_estimator(estimator, dtype):
        """Quantize individual estimator parameters"""
        if hasattr(estimator, "tree_"):
            # For tree-based models
            tree = estimator.tree_
            if hasattr(tree, "value"):
                tree.value = tree.value.astype(dtype)
            if hasattr(tree, "threshold"):
                tree.threshold = tree.threshold.astype(dtype)


class ONNXModelOptimizer:
    """ONNX Runtime optimization for faster inference"""

    def __init__(self):
        self.session_options = None
        if ONNX_AVAILABLE:
            self.session_options = ort.SessionOptions()
            # Enable all optimizations
            self.session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

    def convert_to_onnx(
        self, sklearn_model, input_shape: Tuple[int, ...], model_path: str
    ) -> Optional[str]:
        """Convert scikit-learn model to ONNX format"""
        if not ONNX_AVAILABLE:
            return None

        try:
            # Define input type
            initial_type = [("float_input", FloatTensorType([None, input_shape[1]]))]

            # Convert to ONNX
            onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type)

            # Save ONNX model
            with open(model_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"Model converted to ONNX: {model_path}")
            return model_path

        except Exception as e:
            logger.warning(f"ONNX conversion failed: {e}")
            return None

    def create_inference_session(self, onnx_path: str) -> Optional[Any]:
        """Create optimized ONNX inference session"""
        if not ONNX_AVAILABLE or not os.path.exists(onnx_path):
            return None

        try:
            session = ort.InferenceSession(onnx_path, self.session_options)
            logger.info(f"ONNX inference session created for: {onnx_path}")
            return session
        except Exception as e:
            logger.warning(f"Failed to create ONNX session: {e}")
            return None


class MemoryMappedModelLoader:
    """Memory-mapped model loading for faster access"""

    def __init__(self, cache_dir: str = "models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
        self.lock = threading.RLock()

    def load_model_memory_mapped(self, model_path: str, model_id: str) -> Optional[Any]:
        """Load model with memory mapping for faster access"""
        with self.lock:
            if model_id in self.loaded_models:
                return self.loaded_models[model_id]

            try:
                # Memory map the file
                with open(model_path, "rb") as f:
                    mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

                    # Load model from memory-mapped data
                    model_data = pickle.loads(mmapped_file)
                    mmapped_file.close()

                    self.loaded_models[model_id] = model_data
                    logger.info(f"Model loaded with memory mapping: {model_id}")
                    return model_data

            except Exception as e:
                logger.warning(f"Memory-mapped loading failed for {model_id}: {e}")
                # Fallback to normal loading
                try:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    self.loaded_models[model_id] = model
                    return model
                except Exception as e2:
                    logger.error(f"Fallback loading also failed: {e2}")
                    return None

    def unload_model(self, model_id: str):
        """Unload model from memory"""
        with self.lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                logger.info(f"Model unloaded: {model_id}")


class BatchInferenceProcessor:
    """Handles batch processing for multiple predictions"""

    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_batch_size = 64

    def process_batch(self, requests: List[BatchPredictionRequest]) -> List[Any]:
        """Process multiple prediction requests in batch"""
        if not requests:
            return []

        # Group requests by prediction type and model config
        batches = self._group_requests_by_config(requests)

        results = []
        futures = []

        # Submit batch jobs
        for batch_key, batch_requests in batches.items():
            future = self.executor.submit(self._process_batch_group, batch_requests)
            futures.append((future, batch_key))

        # Collect results
        for future, batch_key in futures:
            try:
                batch_results = future.result(timeout=30)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed for {batch_key}: {e}")
                # Return None for failed predictions
                results.extend([None] * len(batches[batch_key]))

        return results

    def _group_requests_by_config(
        self, requests: List[BatchPredictionRequest]
    ) -> Dict[str, List[BatchPredictionRequest]]:
        """Group requests by model configuration for efficient batching"""
        groups = {}

        for request in requests:
            # Create group key from config
            key = f"{request.prediction_type.value}_{request.model_config.optimization_level.value}_{request.model_config.use_onnx}"
            if key not in groups:
                groups[key] = []
            groups[key].append(request)

        return groups

    def _process_batch_group(self, requests: List[BatchPredictionRequest]) -> List[Any]:
        """Process a group of requests with the same configuration"""
        if not requests:
            return []

        # Use the first request's config (they should all be the same)
        config = requests[0].model_config

        # Combine all features into a single batch
        all_features = np.vstack([req.features for req in requests])

        # Process batch (this would integrate with actual model inference)
        # For now, return mock results
        results = []
        for i in range(len(requests)):
            # Mock prediction result
            result = {
                "prediction": np.random.random(),
                "confidence": np.random.random(),
                "features_used": all_features.shape[1],
                "batch_size": len(requests),
                "optimization_level": config.optimization_level.value,
            }
            results.append(result)

        return results


class OptimizedMLPredictionEngine:
    """Optimized ML prediction engine with quantization and batch processing"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.quantizer = ModelQuantizer()
        self.onnx_optimizer = ONNXModelOptimizer() if ONNX_AVAILABLE else None
        self.memory_loader = MemoryMappedModelLoader()
        self.batch_processor = BatchInferenceProcessor()

        # Model cache
        self.model_cache = {}
        self.onnx_sessions = {}

        # Performance tracking
        self.performance_metrics = []

        logger.info(f"Optimized ML Engine initialized with config: {config}")

    def optimize_model(
        self, model: Any, model_id: str, input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """Apply all optimizations to a model"""
        optimization_results = {
            "original_model": model,
            "quantized_model": None,
            "onnx_path": None,
            "onnx_session": None,
            "optimization_metrics": {},
        }

        start_time = time.time()

        # 1. Apply quantization if enabled
        if (
            self.config.use_quantization
            and self.config.optimization_level != OptimizationLevel.NONE
        ):
            precision = (
                "float16"
                if self.config.optimization_level == OptimizationLevel.FP16
                else "int8"
            )
            quantized_model = self.quantizer.quantize_sklearn_model(model, precision)
            optimization_results["quantized_model"] = quantized_model

        # 2. Convert to ONNX if enabled
        if self.config.use_onnx and ONNX_AVAILABLE and self.onnx_optimizer:
            onnx_path = f"models/onnx/{model_id}.onnx"
            Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

            target_model = optimization_results["quantized_model"] or model
            onnx_path = self.onnx_optimizer.convert_to_onnx(
                target_model, input_shape, onnx_path
            )

            if onnx_path:
                optimization_results["onnx_path"] = onnx_path
                onnx_session = self.onnx_optimizer.create_inference_session(onnx_path)
                if onnx_session:
                    optimization_results["onnx_session"] = onnx_session
                    self.onnx_sessions[model_id] = onnx_session

        # 3. Cache the optimized model
        if self.config.memory_map:
            cache_path = f"models/cache/{model_id}.pkl"
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

            target_model = optimization_results["quantized_model"] or model
            with open(cache_path, "wb") as f:
                pickle.dump(target_model, f)

            optimization_results["cache_path"] = cache_path

        # Calculate optimization metrics
        optimization_time = time.time() - start_time
        optimization_results["optimization_metrics"] = {
            "optimization_time": optimization_time,
            "quantization_applied": optimization_results["quantized_model"] is not None,
            "onnx_conversion": optimization_results["onnx_path"] is not None,
            "memory_mapped": self.config.memory_map,
            "estimated_speedup": self._estimate_speedup(optimization_results),
        }

        # Cache the optimized model
        self.model_cache[model_id] = optimization_results

        logger.info(f"Model {model_id} optimized in {optimization_time:.2f}s")
        return optimization_results

    def predict_optimized(self, model_id: str, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction using optimized model"""
        start_time = time.time()

        if model_id not in self.model_cache:
            raise ValueError(f"Model {model_id} not found. Call optimize_model first.")

        model_info = self.model_cache[model_id]

        # Use ONNX session if available (fastest)
        if model_info["onnx_session"] and self.config.use_onnx:
            prediction = self._predict_onnx(model_info["onnx_session"], features)

        # Use quantized model if available
        elif model_info["quantized_model"]:
            prediction = model_info["quantized_model"].predict(features.reshape(1, -1))[
                0
            ]

        # Use memory-mapped model
        elif self.config.memory_map and "cache_path" in model_info:
            model = self.memory_loader.load_model_memory_mapped(
                model_info["cache_path"], model_id
            )
            if model:
                prediction = model.predict(features.reshape(1, -1))[0]
            else:
                raise RuntimeError(f"Failed to load model {model_id}")

        # Fallback to original model
        else:
            prediction = model_info["original_model"].predict(features.reshape(1, -1))[
                0
            ]

        inference_time = time.time() - start_time

        # Track performance
        metrics = OptimizedModelMetrics(
            inference_time=inference_time,
            memory_usage=self._get_memory_usage(),
            model_size=self._estimate_model_size(model_info),
            throughput=1.0 / inference_time if inference_time > 0 else 0,
            accuracy_score=0.0,  # Would be calculated with ground truth
            optimization_ratio=self._calculate_optimization_ratio(model_info),
        )

        self.performance_metrics.append(metrics)

        return {
            "prediction": prediction,
            "inference_time": inference_time,
            "optimization_level": self.config.optimization_level.value,
            "used_onnx": model_info["onnx_session"] is not None,
            "used_quantization": model_info["quantized_model"] is not None,
            "confidence": np.random.random(),  # Mock confidence score
        }

    def predict_batch(
        self, requests: List[BatchPredictionRequest]
    ) -> List[Dict[str, Any]]:
        """Process multiple predictions in batch for maximum efficiency"""
        if not requests:
            return []

        start_time = time.time()

        # Use batch processor for parallel processing
        batch_results = self.batch_processor.process_batch(requests)

        batch_time = time.time() - start_time

        # Add timing information
        for i, result in enumerate(batch_results):
            if result:
                result["batch_processing_time"] = batch_time
                result["batch_size"] = len(requests)
                result["individual_time"] = batch_time / len(requests)

        logger.info(
            f"Batch prediction completed: {len(requests)} requests in {batch_time:.3f}s"
        )
        return batch_results

    def _predict_onnx(self, session: Any, features: np.ndarray) -> float:
        """Make prediction using ONNX Runtime"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")

        # Prepare input
        input_name = session.get_inputs()[0].name
        input_data = features.reshape(1, -1).astype(np.float32)

        # Run inference
        result = session.run(None, {input_name: input_data})

        return (
            float(result[0][0][0]) if len(result[0].shape) > 1 else float(result[0][0])
        )

    def _estimate_speedup(self, model_info: Dict[str, Any]) -> float:
        """Estimate performance speedup from optimizations"""
        speedup = 1.0

        if model_info["quantized_model"]:
            speedup *= 1.5  # Quantization typically gives 1.5x speedup

        if model_info["onnx_session"]:
            speedup *= 2.0  # ONNX Runtime typically gives 2x speedup

        if self.config.memory_map:
            speedup *= 1.2  # Memory mapping gives modest speedup

        return speedup

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _estimate_model_size(self, model_info: Dict[str, Any]) -> float:
        """Estimate model size in MB"""
        try:
            if "cache_path" in model_info and os.path.exists(model_info["cache_path"]):
                return os.path.getsize(model_info["cache_path"]) / 1024 / 1024
            else:
                # Rough estimate based on model type
                return 10.0  # Default 10MB estimate
        except Exception:
            return 0.0

    def _calculate_optimization_ratio(self, model_info: Dict[str, Any]) -> float:
        """Calculate optimization effectiveness ratio"""
        ratio = 1.0

        if model_info["quantized_model"]:
            ratio *= 0.6  # 40% size reduction from quantization

        if model_info["onnx_session"]:
            ratio *= 0.7  # 30% size reduction from ONNX

        return ratio

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.performance_metrics:
            return {"message": "No performance data available"}

        metrics = np.array([m.inference_time for m in self.performance_metrics])
        memory_usage = np.array([m.memory_usage for m in self.performance_metrics])

        return {
            "total_predictions": len(self.performance_metrics),
            "avg_inference_time": np.mean(metrics),
            "median_inference_time": np.median(metrics),
            "min_inference_time": np.min(metrics),
            "max_inference_time": np.max(metrics),
            "avg_memory_usage": np.mean(memory_usage),
            "optimization_level": self.config.optimization_level.value,
            "onnx_enabled": self.config.use_onnx,
            "quantization_enabled": self.config.use_quantization,
            "batch_size": self.config.batch_size,
            "estimated_speedup": self._estimate_speedup(
                self.model_cache.get("last_model", {})
            ),
        }


# Factory function for creating optimized engines
def create_optimized_ml_engine(
    prediction_type: PredictionType,
    optimization_level: OptimizationLevel = OptimizationLevel.FP16,
    use_onnx: bool = True,
    use_quantization: bool = True,
    batch_size: int = 32,
) -> OptimizedMLPredictionEngine:
    """Factory function to create optimized ML prediction engine"""

    config = ModelConfig(
        prediction_type=prediction_type,
        optimization_level=optimization_level,
        batch_size=batch_size,
        use_onnx=use_onnx and ONNX_AVAILABLE,
        use_quantization=use_quantization,
        memory_map=True,
        cache_enabled=True,
    )

    return OptimizedMLPredictionEngine(config)


# Example usage and testing functions
def test_optimization_pipeline():
    """Test the complete optimization pipeline"""
    if not SKLEARN_AVAILABLE:
        logger.warning("Skipping optimization test - scikit-learn not available")
        return

    # Create sample model
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create optimized engine
    engine = create_optimized_ml_engine(
        PredictionType.PRICE_MOVEMENT,
        optimization_level=OptimizationLevel.FP16,
        use_onnx=True,
        use_quantization=True,
    )

    # Optimize model
    model_id = "test_price_model"
    optimization_result = engine.optimize_model(model, model_id, X.shape)

    # Test single prediction
    test_features = X[0:1]
    prediction_result = engine.predict_optimized(model_id, test_features[0])

    # Test batch prediction
    batch_requests = [
        BatchPredictionRequest(
            features=test_features[i],
            prediction_type=PredictionType.PRICE_MOVEMENT,
            model_config=engine.config,
        )
        for i in range(min(5, len(test_features)))
    ]

    batch_results = engine.predict_batch(batch_requests)

    # Get performance summary
    summary = engine.get_performance_summary()

    logger.info("Optimization pipeline test completed")
    logger.info(f"Performance summary: {summary}")

    return {
        "optimization_result": optimization_result,
        "prediction_result": prediction_result,
        "batch_results": batch_results,
        "performance_summary": summary,
    }


if __name__ == "__main__":
    # Run test if executed directly
    logging.basicConfig(level=logging.INFO)
    test_results = test_optimization_pipeline()
    print("Optimization test results:", json.dumps(test_results, indent=2, default=str))
