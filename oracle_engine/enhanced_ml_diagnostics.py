"""
Enhanced ML Diagnostics and Real-time Monitoring System
Part of Phase 2 Optimization for Oracle-X ML System
"""

import json
import logging
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import ML libraries with fallbacks
try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance metrics"""

    # Common metrics
    model_name: str
    prediction_type: str
    timestamp: datetime

    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None

    # Regression metrics
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None

    # Training metrics
    training_time: Optional[float] = None
    training_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    epochs_completed: Optional[int] = None
    early_stopped: Optional[bool] = None

    # Model health
    is_trained: bool = False
    memory_usage_mb: Optional[float] = None
    prediction_latency_ms: Optional[float] = None


@dataclass
class ModelDriftMetrics:
    """Model drift detection metrics"""

    model_name: str
    drift_score: float
    drift_detected: bool
    reference_period: str
    current_period: str
    drift_method: str
    confidence_threshold: float


@dataclass
class SystemHealthMetrics:
    """Overall system health metrics"""

    timestamp: datetime
    total_models: int
    trained_models: int
    healthy_models: int
    avg_prediction_latency: float
    memory_usage_total_mb: float
    last_training_time: Optional[datetime]
    uptime_hours: float


class EnhancedMLDiagnostics:
    """
    Comprehensive ML diagnostics and monitoring system
    Provides real-time performance tracking, drift detection, and health monitoring
    """

    def __init__(self, log_dir: str = "diagnostics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Performance history
        self.performance_history: List[ModelPerformanceMetrics] = []
        self.drift_history: List[ModelDriftMetrics] = []
        self.health_history: List[SystemHealthMetrics] = []

        # Configuration
        self.max_history_days = 30
        self.drift_threshold = 0.15
        self.min_samples_for_drift = 100

        # Real-time monitoring
        self.monitoring_active = False
        self.last_health_check = None
        self.system_start_time = datetime.now()

        logger.info("Enhanced ML diagnostics system initialized")

    def calculate_model_performance(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, prediction_type: str
    ) -> ModelPerformanceMetrics:
        """Calculate comprehensive performance metrics for a model"""
        start_time = time.time()

        if not SKLEARN_AVAILABLE:
            logger.warning("Sklearn not available, using basic metrics")
            return self._calculate_basic_metrics(model, X_test, y_test, prediction_type)

        try:
            # Get predictions
            prediction_start = time.time()

            if hasattr(model, "predict"):
                if prediction_type == "classification":
                    predictions = model.predict(X_test)
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(X_test)
                        if probabilities.shape[1] > 1:
                            probabilities = probabilities[:, 1]  # Positive class
                    else:
                        probabilities = predictions
                else:
                    predictions = model.predict(X_test)
                    probabilities = None
            else:
                logger.warning(
                    f"Model {getattr(model, '__class__', 'unknown')} has no predict method"
                )
                return self._create_empty_metrics(model, prediction_type)

            prediction_time = (time.time() - prediction_start) * 1000  # Convert to ms

            # Calculate metrics
            metrics = ModelPerformanceMetrics(
                model_name=getattr(model, "__class__", {}).get("__name__", "unknown"),
                prediction_type=prediction_type,
                timestamp=datetime.now(),
                training_samples=len(X_test),
                validation_samples=len(X_test),
                prediction_latency_ms=prediction_time,
                is_trained=getattr(model, "is_trained", True),
            )

            if prediction_type == "classification":
                # Classification metrics
                metrics.accuracy = float(accuracy_score(y_test, predictions))
                metrics.precision = float(
                    precision_score(
                        y_test, predictions, average="weighted", zero_division=0
                    )
                )
                metrics.recall = float(
                    recall_score(
                        y_test, predictions, average="weighted", zero_division=0
                    )
                )
                metrics.f1_score = float(
                    f1_score(y_test, predictions, average="weighted", zero_division=0)
                )

                # AUC-ROC for binary classification
                if probabilities is not None and len(np.unique(y_test)) == 2:
                    try:
                        metrics.auc_roc = float(roc_auc_score(y_test, probabilities))
                    except Exception as e:
                        logger.warning(f"Could not calculate AUC-ROC: {e}")

            else:
                # Regression metrics
                metrics.mse = float(mean_squared_error(y_test, predictions))
                metrics.rmse = float(np.sqrt(metrics.mse))
                metrics.mae = float(mean_absolute_error(y_test, predictions))
                metrics.r2_score = float(r2_score(y_test, predictions))

            # Memory usage estimation
            try:
                import sys

                metrics.memory_usage_mb = sys.getsizeof(model) / (1024 * 1024)
            except:
                metrics.memory_usage_mb = 0.0

            calculation_time = time.time() - start_time
            logger.info(
                f"Performance metrics calculated in {calculation_time:.2f}s for {metrics.model_name}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._create_empty_metrics(model, prediction_type)

    def _calculate_basic_metrics(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, prediction_type: str
    ) -> ModelPerformanceMetrics:
        """Calculate basic metrics without sklearn"""
        try:
            predictions = (
                model.predict(X_test)
                if hasattr(model, "predict")
                else np.zeros(len(y_test))
            )

            metrics = ModelPerformanceMetrics(
                model_name=getattr(model, "__class__", {}).get("__name__", "unknown"),
                prediction_type=prediction_type,
                timestamp=datetime.now(),
                is_trained=getattr(model, "is_trained", True),
            )

            if prediction_type == "classification":
                # Basic accuracy
                metrics.accuracy = float(np.mean(predictions == y_test))
            else:
                # Basic regression metrics
                errors = predictions - y_test
                metrics.mse = float(np.mean(errors**2))
                metrics.rmse = float(np.sqrt(metrics.mse))
                metrics.mae = float(np.mean(np.abs(errors)))

            return metrics

        except Exception as e:
            logger.error(f"Error in basic metrics calculation: {e}")
            return self._create_empty_metrics(model, prediction_type)

    def _create_empty_metrics(
        self, model, prediction_type: str
    ) -> ModelPerformanceMetrics:
        """Create empty metrics for failed calculations"""
        return ModelPerformanceMetrics(
            model_name=getattr(model, "__class__", {}).get("__name__", "unknown"),
            prediction_type=prediction_type,
            timestamp=datetime.now(),
            is_trained=False,
        )

    def detect_model_drift(
        self,
        model_name: str,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        method: str = "statistical",
    ) -> ModelDriftMetrics:
        """
        Detect model drift using statistical methods
        """
        try:
            if (
                len(reference_data) < self.min_samples_for_drift
                or len(current_data) < self.min_samples_for_drift
            ):
                logger.warning(
                    f"Insufficient samples for drift detection: ref={len(reference_data)}, curr={len(current_data)}"
                )
                return ModelDriftMetrics(
                    model_name=model_name,
                    drift_score=0.0,
                    drift_detected=False,
                    reference_period="insufficient_data",
                    current_period="insufficient_data",
                    drift_method=method,
                    confidence_threshold=self.drift_threshold,
                )

            if method == "statistical":
                drift_score = self._calculate_statistical_drift(
                    reference_data, current_data
                )
            elif method == "distribution":
                drift_score = self._calculate_distribution_drift(
                    reference_data, current_data
                )
            else:
                drift_score = self._calculate_simple_drift(reference_data, current_data)

            drift_detected = drift_score > self.drift_threshold

            drift_metrics = ModelDriftMetrics(
                model_name=model_name,
                drift_score=drift_score,
                drift_detected=drift_detected,
                reference_period=f"samples_{len(reference_data)}",
                current_period=f"samples_{len(current_data)}",
                drift_method=method,
                confidence_threshold=self.drift_threshold,
            )

            if drift_detected:
                logger.warning(
                    f"Drift detected for {model_name}: score={drift_score:.3f} > {self.drift_threshold}"
                )
            else:
                logger.info(
                    f"No drift detected for {model_name}: score={drift_score:.3f}"
                )

            return drift_metrics

        except Exception as e:
            logger.error(f"Error detecting drift for {model_name}: {e}")
            return ModelDriftMetrics(
                model_name=model_name,
                drift_score=1.0,  # Assume drift on error
                drift_detected=True,
                reference_period="error",
                current_period="error",
                drift_method=method,
                confidence_threshold=self.drift_threshold,
            )

    def _calculate_statistical_drift(
        self, ref_data: pd.DataFrame, curr_data: pd.DataFrame
    ) -> float:
        """Calculate drift using statistical tests"""
        from scipy import stats

        drift_scores = []

        # Get common numeric columns
        common_cols = set(ref_data.select_dtypes(include=[np.number]).columns) & set(
            curr_data.select_dtypes(include=[np.number]).columns
        )

        for col in common_cols:
            try:
                ref_values = ref_data[col].dropna()
                curr_values = curr_data[col].dropna()

                if len(ref_values) < 10 or len(curr_values) < 10:
                    continue

                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.ks_2samp(ref_values, curr_values)
                drift_scores.append(ks_stat)

            except Exception as e:
                logger.warning(f"Error calculating drift for column {col}: {e}")

        return np.mean(drift_scores) if drift_scores else 0.0

    def _calculate_distribution_drift(
        self, ref_data: pd.DataFrame, curr_data: pd.DataFrame
    ) -> float:
        """Calculate drift using distribution comparison"""
        drift_scores = []

        # Get common numeric columns
        common_cols = set(ref_data.select_dtypes(include=[np.number]).columns) & set(
            curr_data.select_dtypes(include=[np.number]).columns
        )

        for col in common_cols:
            try:
                ref_values = ref_data[col].dropna()
                curr_values = curr_data[col].dropna()

                if len(ref_values) < 10 or len(curr_values) < 10:
                    continue

                # Compare means and standard deviations
                ref_mean, ref_std = ref_values.mean(), ref_values.std()
                curr_mean, curr_std = curr_values.mean(), curr_values.std()

                mean_drift = abs(ref_mean - curr_mean) / (ref_std + 1e-8)
                std_drift = abs(ref_std - curr_std) / (ref_std + 1e-8)

                drift_scores.append((mean_drift + std_drift) / 2)

            except Exception as e:
                logger.warning(
                    f"Error calculating distribution drift for column {col}: {e}"
                )

        return np.mean(drift_scores) if drift_scores else 0.0

    def _calculate_simple_drift(
        self, ref_data: pd.DataFrame, curr_data: pd.DataFrame
    ) -> float:
        """Simple drift calculation using correlation"""
        try:
            # Simple correlation-based drift
            common_cols = set(
                ref_data.select_dtypes(include=[np.number]).columns
            ) & set(curr_data.select_dtypes(include=[np.number]).columns)

            if not common_cols:
                return 0.0

            ref_means = ref_data[list(common_cols)].mean()
            curr_means = curr_data[list(common_cols)].mean()

            correlation = np.corrcoef(ref_means, curr_means)[0, 1]
            drift_score = 1 - abs(correlation) if not np.isnan(correlation) else 1.0

            return float(drift_score)

        except Exception as e:
            logger.warning(f"Error in simple drift calculation: {e}")
            return 0.0

    def monitor_system_health(self, models: Dict[str, Any]) -> SystemHealthMetrics:
        """Monitor overall system health"""
        try:
            total_models = len(models)
            trained_models = sum(
                1 for model in models.values() if getattr(model, "is_trained", False)
            )

            # Calculate average prediction latency from recent metrics
            recent_metrics = [
                m
                for m in self.performance_history
                if m.timestamp > datetime.now() - timedelta(hours=1)
            ]
            avg_latency = (
                np.mean(
                    [
                        m.prediction_latency_ms
                        for m in recent_metrics
                        if m.prediction_latency_ms is not None
                    ]
                )
                if recent_metrics
                else 0.0
            )

            # Estimate memory usage
            total_memory = sum(
                getattr(model, "memory_usage_mb", 0) for model in models.values()
            )

            # Calculate uptime
            uptime = (datetime.now() - self.system_start_time).total_seconds() / 3600

            # Find last training time
            last_training = None
            if self.performance_history:
                last_training = max(m.timestamp for m in self.performance_history)

            health_metrics = SystemHealthMetrics(
                timestamp=datetime.now(),
                total_models=total_models,
                trained_models=trained_models,
                healthy_models=trained_models,  # Simplified for now
                avg_prediction_latency=float(avg_latency),
                memory_usage_total_mb=float(total_memory),
                last_training_time=last_training,
                uptime_hours=float(uptime),
            )

            self.health_history.append(health_metrics)
            self.last_health_check = datetime.now()

            logger.info(
                f"System health: {trained_models}/{total_models} models trained, "
                f"avg latency: {avg_latency:.1f}ms, memory: {total_memory:.1f}MB"
            )

            return health_metrics

        except Exception as e:
            logger.error(f"Error monitoring system health: {e}")
            return SystemHealthMetrics(
                timestamp=datetime.now(),
                total_models=0,
                trained_models=0,
                healthy_models=0,
                avg_prediction_latency=0.0,
                memory_usage_total_mb=0.0,
                last_training_time=None,
                uptime_hours=0.0,
            )

    def add_performance_record(self, metrics: ModelPerformanceMetrics):
        """Add performance record to history"""
        self.performance_history.append(metrics)
        self._cleanup_old_records()

    def add_drift_record(self, drift_metrics: ModelDriftMetrics):
        """Add drift detection record to history"""
        self.drift_history.append(drift_metrics)
        self._cleanup_old_records()

    def _cleanup_old_records(self):
        """Clean up old records to prevent memory bloat"""
        cutoff_time = datetime.now() - timedelta(days=self.max_history_days)

        self.performance_history = [
            m for m in self.performance_history if m.timestamp > cutoff_time
        ]
        self.drift_history = [
            m for m in self.drift_history if m.timestamp > cutoff_time
        ]
        self.health_history = [
            m for m in self.health_history if m.timestamp > cutoff_time
        ]

    def get_performance_summary(
        self, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance summary for all models or specific model"""
        if model_name:
            relevant_metrics = [
                m for m in self.performance_history if m.model_name == model_name
            ]
        else:
            relevant_metrics = self.performance_history

        if not relevant_metrics:
            return {"error": "No performance data available"}

        # Group by model
        by_model = {}
        for metric in relevant_metrics:
            if metric.model_name not in by_model:
                by_model[metric.model_name] = []
            by_model[metric.model_name].append(metric)

        summary = {}
        for model, metrics in by_model.items():
            latest = max(metrics, key=lambda x: x.timestamp)
            summary[model] = {
                "latest_performance": asdict(latest),
                "total_evaluations": len(metrics),
                "avg_accuracy": np.mean(
                    [m.accuracy for m in metrics if m.accuracy is not None]
                ),
                "avg_latency_ms": np.mean(
                    [
                        m.prediction_latency_ms
                        for m in metrics
                        if m.prediction_latency_ms is not None
                    ]
                ),
            }

        return summary

    def save_diagnostics_report(self, filename: Optional[str] = None) -> str:
        """Save comprehensive diagnostics report to file"""
        if not filename:
            filename = f"ml_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.log_dir / filename

        report = {
            "timestamp": datetime.now().isoformat(),
            "performance_summary": self.get_performance_summary(),
            "recent_health": [asdict(h) for h in self.health_history[-10:]],
            "drift_alerts": [asdict(d) for d in self.drift_history if d.drift_detected][
                -10:
            ],
            "system_stats": {
                "total_performance_records": len(self.performance_history),
                "total_drift_records": len(self.drift_history),
                "monitoring_uptime_hours": (
                    datetime.now() - self.system_start_time
                ).total_seconds()
                / 3600,
            },
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Diagnostics report saved to {filepath}")
        return str(filepath)
