#!/usr/bin/env python3
"""
ML Model Manager - Production model lifecycle management
Handles model persistence, versioning, monitoring, and automated retraining
"""

import json
import logging
import pickle
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Memory-efficient processing import with fallback
try:
    from core.memory_processor import LazyDataLoader, MemoryConfig

    MEMORY_PROCESSOR_AVAILABLE = True
except ImportError:
    MEMORY_PROCESSOR_AVAILABLE = False
    LazyDataLoader = None
    MemoryConfig = None

# Local imports
from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine
from oracle_engine.ml_prediction_engine import PredictionType

# Import required dependencies for EnsemblePredictionEngine
try:
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
    from sentiment.sentiment_engine import AdvancedSentimentEngine

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Ensemble dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False
    DataFeedOrchestrator = None
    AdvancedSentimentEngine = None

logger = logging.getLogger(__name__)


class ModelMetrics:
    """Track model performance metrics over time"""

    def __init__(self):
        self.accuracy_scores: List[float] = []
        self.prediction_errors: List[float] = []
        self.confidence_scores: List[float] = []
        self.prediction_times: List[datetime] = []
        self.retrain_history: List[datetime] = []

    def add_prediction(self, actual: float, predicted: float, confidence: float):
        """Record a prediction and its outcome"""
        error = abs(actual - predicted) / actual if actual != 0 else abs(predicted)
        self.prediction_errors.append(error)
        self.confidence_scores.append(confidence)
        self.prediction_times.append(datetime.now())

        # Calculate accuracy based on direction prediction
        direction_correct = (actual > 0) == (predicted > 0)
        self.accuracy_scores.append(1.0 if direction_correct else 0.0)

    def get_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get performance metrics for recent period"""
        cutoff = datetime.now() - timedelta(days=days)
        recent_indices = [i for i, t in enumerate(self.prediction_times) if t >= cutoff]

        if not recent_indices:
            return {
                "accuracy": 0.0,
                "avg_error": 1.0,
                "avg_confidence": 0.0,
                "count": 0,
            }

        recent_accuracy = float(
            np.mean([self.accuracy_scores[i] for i in recent_indices])
        )
        recent_error = float(
            np.mean([self.prediction_errors[i] for i in recent_indices])
        )
        recent_confidence = float(
            np.mean([self.confidence_scores[i] for i in recent_indices])
        )

        return {
            "accuracy": recent_accuracy,
            "avg_error": recent_error,
            "avg_confidence": recent_confidence,
            "count": len(recent_indices),
        }


class ModelVersionManager:
    """Manage model versions and rollback capabilities with lazy loading"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.version_history: Dict[str, List[str]] = {}

        # Initialize lazy loader for memory efficiency
        if MEMORY_PROCESSOR_AVAILABLE and LazyDataLoader and MemoryConfig:
            self.lazy_loader = LazyDataLoader(
                MemoryConfig(
                    lazy_load_threshold=50000,  # Models larger than 50KB use lazy loading
                    cache_compression=True,
                    enable_gc_optimization=True,
                )
            )
        else:
            self.lazy_loader = None

    def save_model(
        self, model: Any, model_name: str, version: Optional[str] = None
    ) -> str:
        """Save model with version control"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = self.models_dir / f"{model_name}_v{version}.pkl"

        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Update version history
            if model_name not in self.version_history:
                self.version_history[model_name] = []
            self.version_history[model_name].append(version)

            # Keep only last 10 versions
            if len(self.version_history[model_name]) > 10:
                old_version = self.version_history[model_name].pop(0)
                old_path = self.models_dir / f"{model_name}_v{old_version}.pkl"
                if old_path.exists():
                    old_path.unlink()

            logger.info(f"Saved model {model_name} version {version}")
            return version

        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise

    def load_model(
        self, model_name: str, version: Optional[str] = None
    ) -> Optional[Any]:
        """Load model by name and version with lazy loading support"""
        if version is None:
            # Load latest version
            if (
                model_name not in self.version_history
                or not self.version_history[model_name]
            ):
                return None
            version = self.version_history[model_name][-1]

        model_path = self.models_dir / f"{model_name}_v{version}.pkl"
        cache_key = f"{model_name}_v{version}"

        # Use lazy loading if available and model file exists
        if (
            self.lazy_loader
            and model_path.exists()
            and model_path.stat().st_size > 50000
        ):  # Only for larger models

            def model_loader():
                with open(model_path, "rb") as f:
                    return pickle.load(f)

            try:
                with self.lazy_loader.lazy_load(cache_key, model_loader):
                    model = self.lazy_loader._cache[cache_key][1]  # Get from cache
                    logger.info(f"Lazy loaded model {model_name} version {version}")
                    return model
            except Exception as e:
                logger.warning(f"Lazy loading failed, falling back to direct load: {e}")

        # Standard loading
        try:
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                logger.info(f"Loaded model {model_name} version {version}")
                return model
            else:
                logger.warning(f"Model file not found: {model_path}")
                return None

        except Exception as e:
            logger.error(f"Failed to load model {model_name} v{version}: {e}")
            return None

    def list_versions(self, model_name: str) -> List[str]:
        """List all versions for a model"""
        return self.version_history.get(model_name, [])

    def rollback_model(self, model_name: str, target_version: str) -> bool:
        """Rollback to a previous model version"""
        if model_name not in self.version_history:
            logger.error(f"No version history for model {model_name}")
            return False

        if target_version not in self.version_history[model_name]:
            logger.error(f"Version {target_version} not found for model {model_name}")
            return False

        # Move target version to end of list (making it latest)
        versions = self.version_history[model_name]
        versions.remove(target_version)
        versions.append(target_version)

        logger.info(f"Rolled back model {model_name} to version {target_version}")
        return True


class ModelMonitor:
    """Monitor model performance and trigger retraining"""

    def __init__(self, db_path: str = "data/databases/model_monitoring.db"):
        self.db_path = db_path
        self.metrics: Dict[str, ModelMetrics] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._init_database()

    def _init_database(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                accuracy REAL,
                error_rate REAL,
                confidence REAL,
                prediction_count INTEGER
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS retraining_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                trigger_reason TEXT,
                old_accuracy REAL,
                new_accuracy REAL,
                success BOOLEAN
            )
        """
        )

        conn.commit()
        conn.close()

    def start_monitoring(self, check_interval: int = 3600):
        """Start continuous model monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(check_interval,), daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started model monitoring")

    def stop_monitoring(self):
        """Stop model monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped model monitoring")

    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_all_models()
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Brief pause before retrying

    def _check_all_models(self):
        """Check performance of all monitored models"""
        for model_name in self.metrics:
            self._check_model_performance(model_name)

    def _check_model_performance(self, model_name: str):
        """Check if a model needs retraining"""
        if model_name not in self.metrics:
            return

        recent_perf = self.metrics[model_name].get_recent_performance(days=7)

        # Store performance metrics
        self._store_performance_metrics(model_name, recent_perf)

        # Check retraining triggers
        needs_retrain = False
        trigger_reason = ""

        # Accuracy degradation
        if recent_perf["accuracy"] < 0.4 and recent_perf["count"] >= 10:
            needs_retrain = True
            trigger_reason = f"Low accuracy: {recent_perf['accuracy']:.3f}"

        # High error rate
        elif recent_perf["avg_error"] > 0.5 and recent_perf["count"] >= 10:
            needs_retrain = True
            trigger_reason = f"High error rate: {recent_perf['avg_error']:.3f}"

        # Low confidence scores
        elif recent_perf["avg_confidence"] < 0.3 and recent_perf["count"] >= 10:
            needs_retrain = True
            trigger_reason = f"Low confidence: {recent_perf['avg_confidence']:.3f}"

        if needs_retrain:
            logger.warning(f"Model {model_name} needs retraining: {trigger_reason}")
            self._trigger_retraining(
                model_name, trigger_reason, recent_perf["accuracy"]
            )

    def _store_performance_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Store performance metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO model_performance
            (model_name, timestamp, accuracy, error_rate, confidence, prediction_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                model_name,
                datetime.now(),
                metrics["accuracy"],
                metrics["avg_error"],
                metrics["avg_confidence"],
                metrics["count"],
            ),
        )

        conn.commit()
        conn.close()

    def _trigger_retraining(self, model_name: str, reason: str, old_accuracy: float):
        """Trigger model retraining"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO retraining_events
            (model_name, timestamp, trigger_reason, old_accuracy, success)
            VALUES (?, ?, ?, ?, ?)
        """,
            (model_name, datetime.now(), reason, old_accuracy, False),
        )

        conn.commit()
        conn.close()

        # Note: Actual retraining would be triggered here
        # This is a placeholder for retraining orchestration
        logger.info(f"Retraining triggered for {model_name}: {reason}")

    def record_prediction(
        self, model_name: str, actual: float, predicted: float, confidence: float
    ):
        """Record a prediction outcome for monitoring"""
        if model_name not in self.metrics:
            self.metrics[model_name] = ModelMetrics()

        self.metrics[model_name].add_prediction(actual, predicted, confidence)

    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get current status of a model"""
        if model_name not in self.metrics:
            return {"status": "unknown", "message": "Model not monitored"}

        recent_perf = self.metrics[model_name].get_recent_performance()

        status = "healthy"
        if recent_perf["accuracy"] < 0.4:
            status = "degraded"
        elif recent_perf["avg_error"] > 0.5:
            status = "poor"
        elif recent_perf["count"] < 5:
            status = "insufficient_data"

        return {
            "status": status,
            "performance": recent_perf,
            "last_retrain": (
                self.metrics[model_name].retrain_history[-1]
                if self.metrics[model_name].retrain_history
                else None
            ),
        }


class MLModelManager:
    """Production ML model lifecycle manager"""

    def __init__(
        self,
        models_dir: str = "models",
        monitoring_db: str = "data/databases/model_monitoring.db",
    ):
        self.ensemble_engine: Optional[EnsemblePredictionEngine] = None
        self.version_manager = ModelVersionManager(models_dir)
        self.monitor = ModelMonitor(monitoring_db)
        self.model_configs: Dict[str, Dict] = {}
        self.last_training_time: Optional[datetime] = None

        # Auto-retraining settings
        self.auto_retrain_enabled = True
        self.retrain_threshold_days = 7
        self.min_predictions_for_retrain = 50

    def initialize_ensemble(self, symbols: List[str]) -> bool:
        """Initialize the ensemble prediction engine"""
        try:
            logger.info("Initializing Ensemble Prediction Engine")

            if not DEPENDENCIES_AVAILABLE:
                logger.error("Required dependencies not available for ensemble engine")
                self.ensemble_engine = None
                return False

            # Initialize data orchestrator and sentiment engine
            try:
                data_orchestrator = DataFeedOrchestrator()
                sentiment_engine = AdvancedSentimentEngine()

                # Initialize the ensemble engine with proper dependencies
                self.ensemble_engine = EnsemblePredictionEngine(
                    data_orchestrator=data_orchestrator,
                    sentiment_engine=sentiment_engine,
                )

                logger.info("Ensemble Prediction Engine initialized successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to create ensemble engine dependencies: {e}")
                self.ensemble_engine = None
                return False

        except Exception as e:
            logger.error(f"Failed to initialize Ensemble Prediction Engine: {e}")
            self.ensemble_engine = None
            return False

    def train_models(
        self, symbols: List[str], days_back: int = 100, force_retrain: bool = False
    ) -> Dict[str, bool]:
        """Train models for given symbols"""
        if not self.ensemble_engine:
            logger.error("Ensemble engine not initialized")
            return {}

        # Check if retraining is needed
        if not force_retrain and not self._should_retrain():
            logger.info("Models are recent enough, skipping training")
            return {}

        training_results = {}

        try:
            # Train ensemble models
            trained_models = self.ensemble_engine.train_models(symbols, days_back)

            # Save trained models
            for model_name, model in trained_models.items():
                version = self.version_manager.save_model(model, model_name)
                training_results[model_name] = True
                logger.info(f"Saved trained model {model_name} version {version}")

            self.last_training_time = datetime.now()
            logger.info(f"Training completed for {len(trained_models)} models")

        except Exception as e:
            logger.error(f"Training failed: {e}")

        return training_results

    def _should_retrain(self) -> bool:
        """Check if models should be retrained"""
        if not self.auto_retrain_enabled:
            return False

        # Check time since last training
        if self.last_training_time is None:
            return True

        days_since_training = (datetime.now() - self.last_training_time).days
        if days_since_training >= self.retrain_threshold_days:
            logger.info(
                f"Models need retraining: {days_since_training} days since last training"
            )
            return True

        return False

    def predict(
        self, symbol: str, prediction_type: PredictionType, horizon_days: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Make prediction using ensemble engine"""
        if not self.ensemble_engine:
            logger.warning(
                "Ensemble engine not initialized, returning default prediction"
            )
            return {
                "symbol": symbol,
                "prediction_type": prediction_type.value,
                "prediction": 0.0,
                "confidence": 0.5,
                "source": "default",
            }

        try:
            # Use the ensemble engine for actual predictions
            prediction_result = self.ensemble_engine.predict(
                symbol, prediction_type, horizon_days
            )

            if prediction_result:
                # Convert PredictionResult to the expected dictionary format
                return {
                    "symbol": prediction_result.symbol,
                    "prediction_type": prediction_result.prediction_type.value,
                    "prediction": float(prediction_result.prediction),
                    "confidence": float(prediction_result.confidence),
                    "uncertainty": float(prediction_result.uncertainty),
                    "feature_importance": prediction_result.feature_importance,
                    "model_contributions": prediction_result.model_contributions,
                    "source": "ensemble_engine",
                }
            else:
                logger.warning(
                    f"Ensemble engine returned None for {symbol} {prediction_type}"
                )
                return {
                    "symbol": symbol,
                    "prediction_type": prediction_type.value,
                    "prediction": 0.0,
                    "confidence": 0.5,
                    "source": "fallback",
                }

        except Exception as e:
            logger.error(f"Prediction failed for {symbol} {prediction_type}: {e}")
            return None

    def start_monitoring(self):
        """Start model performance monitoring"""
        self.monitor.start_monitoring()
        logger.info("Started model performance monitoring")

    def stop_monitoring(self):
        """Stop model performance monitoring"""
        self.monitor.stop_monitoring()
        logger.info("Stopped model performance monitoring")

    def record_prediction_outcome(
        self, model_name: str, actual: float, predicted: float, confidence: float
    ):
        """Record prediction outcome for monitoring"""
        self.monitor.record_prediction(model_name, actual, predicted, confidence)

    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get status of a specific model"""
        return self.monitor.get_model_status(model_name)

    def get_all_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all monitored models"""
        status = {}
        for model_name in self.monitor.metrics:
            status[model_name] = self.get_model_status(model_name)
        return status

    def rollback_model(self, model_name: str, target_version: str) -> bool:
        """Rollback a model to a previous version"""
        success = self.version_manager.rollback_model(model_name, target_version)
        if success:
            # Reload ensemble engine with rollback version
            model = self.version_manager.load_model(model_name, target_version)
            if model and self.ensemble_engine:
                # Note: This would require updating the ensemble engine's internal models
                logger.info(
                    f"Rolled back model {model_name} to version {target_version}"
                )
        return success

    def cleanup_old_models(self, keep_versions: int = 5):
        """Clean up old model versions"""
        for model_name, versions in self.version_manager.version_history.items():
            if len(versions) > keep_versions:
                old_versions = versions[:-keep_versions]
                for version in old_versions:
                    model_path = (
                        self.version_manager.models_dir / f"{model_name}_v{version}.pkl"
                    )
                    if model_path.exists():
                        model_path.unlink()
                        logger.info(
                            f"Removed old model version: {model_name}_v{version}"
                        )

    def export_model_metrics(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """Export model performance metrics"""
        if model_name not in self.monitor.metrics:
            return {}

        metrics = self.monitor.metrics[model_name]
        recent_perf = metrics.get_recent_performance(days)

        return {
            "model_name": model_name,
            "period_days": days,
            "performance": recent_perf,
            "total_predictions": len(metrics.prediction_times),
            "retrain_count": len(metrics.retrain_history),
            "last_retrain": (
                metrics.retrain_history[-1] if metrics.retrain_history else None
            ),
        }

    def configure_auto_retraining(
        self, enabled: bool = True, threshold_days: int = 7, min_predictions: int = 50
    ):
        """Configure automatic retraining parameters"""
        self.auto_retrain_enabled = enabled
        self.retrain_threshold_days = threshold_days
        self.min_predictions_for_retrain = min_predictions

        logger.info(
            f"Auto-retraining configured: enabled={enabled}, threshold_days={threshold_days}, min_predictions={min_predictions}"
        )


# Factory function for easy initialization
def create_ml_model_manager(
    models_dir: str = "models",
    monitoring_db: str = "data/databases/model_monitoring.db",
) -> MLModelManager:
    """Factory function to create and initialize ML Model Manager"""
    manager = MLModelManager(models_dir, monitoring_db)
    return manager


if __name__ == "__main__":
    """Demonstration of ML Model Manager capabilities"""
    logging.basicConfig(level=logging.INFO)

    print("ML Model Manager Demonstration")
    print("=" * 50)

    # Create manager
    manager = create_ml_model_manager()

    # Initialize with sample symbols
    symbols = ["AAPL", "GOOGL", "TSLA"]
    success = manager.initialize_ensemble(symbols)
    print(f"✓ Ensemble initialized: {success}")

    # Start monitoring
    manager.start_monitoring()
    print("✓ Started model monitoring")

    # Configure auto-retraining
    manager.configure_auto_retraining(enabled=True, threshold_days=7)
    print("✓ Configured auto-retraining")

    # Simulate prediction outcomes
    for i in range(5):
        manager.record_prediction_outcome("AAPL_price_direction", 1.0, 0.8, 0.7)
        manager.record_prediction_outcome("GOOGL_price_direction", -1.0, -0.9, 0.6)

    # Check model status
    status = manager.get_all_model_status()
    print(f"✓ Model status checked: {len(status)} models monitored")

    # Export metrics
    for model_name in ["AAPL_price_direction", "GOOGL_price_direction"]:
        metrics = manager.export_model_metrics(model_name)
        if metrics:
            print(
                f"✓ {model_name}: {metrics['performance']['count']} predictions, {metrics['performance']['accuracy']:.3f} accuracy"
            )

    # Stop monitoring
    manager.stop_monitoring()
    print("✓ Stopped monitoring")

    print("\n🎉 ML Model Manager demonstration completed!")
