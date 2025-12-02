"""
Real-time Learning and Adaptation System
Phase 2B/2C Implementation: Online Learning, Dynamic Model Selection, Performance Optimization
"""

import logging
import numpy as np
import pandas as pd
import time
import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning system"""

    batch_size: int = 100
    learning_rate_decay: float = 0.95
    adaptation_threshold: float = 0.1
    performance_window: int = 1000
    retraining_frequency: int = 10000  # samples
    model_selection_frequency: int = 5000  # samples
    drift_detection_enabled: bool = True
    uncertainty_quantification: bool = True
    ensemble_update_strategy: str = "weighted"  # 'weighted', 'replacement', 'additive'


@dataclass
class ModelPerformanceTracker:
    """Track model performance over time"""

    model_name: str
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    total_predictions: int = 0
    correct_predictions: int = 0
    total_error: float = 0.0
    prediction_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_updated: datetime = field(default_factory=datetime.now)
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=1000))


@dataclass
class AdaptationEvent:
    """Record of system adaptations"""

    timestamp: datetime
    event_type: str  # 'model_update', 'ensemble_reweight', 'model_replacement', 'drift_detected'
    model_affected: str
    performance_before: float
    performance_after: float
    trigger_reason: str
    adaptation_details: Dict[str, Any]


class RealTimeLearningEngine:
    """
    Real-time learning engine with online adaptation, drift detection, and dynamic model selection
    """

    def __init__(self, config: Optional[OnlineLearningConfig] = None):
        self.config = config or OnlineLearningConfig()

        # Model management
        self.active_models = {}
        self.model_pool = {}
        self.performance_trackers = {}
        self.model_weights = {}

        # Data streams
        self.data_buffer = deque(maxlen=self.config.performance_window)
        self.prediction_buffer = deque(maxlen=self.config.performance_window)
        self.target_buffer = deque(maxlen=self.config.performance_window)

        # Adaptation tracking
        self.adaptation_history = []
        self.sample_count = 0
        self.last_retraining = 0
        self.last_model_selection = 0

        # Threading for real-time processing
        self.processing_lock = threading.Lock()
        self.adaptation_thread = None
        self.is_running = False

        logger.info("Real-time learning engine initialized")

    def register_model(
        self, model, model_name: str, model_type: str, initial_weight: float = 1.0
    ):
        """Register a model for real-time learning"""
        self.active_models[model_name] = {
            "model": model,
            "type": model_type,
            "is_online": hasattr(model, "partial_fit"),
            "last_training_time": datetime.now(),
        }

        self.performance_trackers[model_name] = ModelPerformanceTracker(
            model_name=model_name
        )
        self.model_weights[model_name] = initial_weight

        logger.info(f"Registered model for real-time learning: {model_name}")

    def start_real_time_learning(self):
        """Start real-time learning and adaptation"""
        self.is_running = True
        self.adaptation_thread = threading.Thread(
            target=self._adaptation_worker, daemon=True
        )
        self.adaptation_thread.start()
        logger.info("Real-time learning started")

    def stop_real_time_learning(self):
        """Stop real-time learning"""
        self.is_running = False
        if self.adaptation_thread and self.adaptation_thread.is_alive():
            self.adaptation_thread.join(timeout=5.0)
        logger.info("Real-time learning stopped")

    def process_new_sample(
        self, X: pd.Series, y: Any, sample_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a new data sample in real-time
        """
        with self.processing_lock:
            start_time = time.time()

            # Add to buffers
            self.data_buffer.append(X)
            self.target_buffer.append(y)
            self.sample_count += 1

            # Get predictions from all active models
            predictions = {}
            confidences = {}

            for model_name, model_info in self.active_models.items():
                try:
                    model = model_info["model"]

                    # Ensure X is properly formatted for prediction
                    if hasattr(X, "values"):
                        X_array = np.array(X.values).reshape(1, -1)
                    else:
                        X_array = np.array(X).reshape(1, -1)

                    pred_start = time.time()
                    if hasattr(model, "predict"):
                        prediction = model.predict(X_array)[0]
                        predictions[model_name] = prediction

                        # Get confidence/uncertainty if available
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(X_array)[0]
                            confidence = np.max(proba)
                        elif hasattr(model, "decision_function"):
                            decision = model.decision_function(X_array)[0]
                            confidence = abs(decision)
                        else:
                            confidence = 1.0  # Default confidence

                        confidences[model_name] = confidence

                    pred_time = (time.time() - pred_start) * 1000  # ms
                    self.performance_trackers[model_name].prediction_times.append(
                        pred_time
                    )

                except Exception as e:
                    logger.warning(f"Error getting prediction from {model_name}: {e}")
                    predictions[model_name] = 0
                    confidences[model_name] = 0.0

            # Calculate ensemble prediction
            ensemble_prediction = self._calculate_ensemble_prediction(
                predictions, confidences
            )

            self.prediction_buffer.append(
                {
                    "individual": predictions,
                    "ensemble": ensemble_prediction,
                    "confidences": confidences,
                    "timestamp": datetime.now(),
                }
            )

            processing_time = (time.time() - start_time) * 1000

            # Trigger adaptation if needed
            self._check_adaptation_triggers()

            return {
                "predictions": predictions,
                "ensemble_prediction": ensemble_prediction,
                "confidences": confidences,
                "processing_time_ms": processing_time,
                "sample_count": self.sample_count,
            }

    def _calculate_ensemble_prediction(
        self, predictions: Dict[str, Any], confidences: Dict[str, float]
    ) -> Any:
        """Calculate weighted ensemble prediction"""
        if not predictions:
            return 0

        try:
            if self.config.ensemble_update_strategy == "weighted":
                # Weight by both model weight and confidence
                total_weight = 0
                weighted_sum = 0

                for model_name, prediction in predictions.items():
                    model_weight = self.model_weights.get(model_name, 1.0)
                    confidence = confidences.get(model_name, 1.0)
                    combined_weight = model_weight * confidence

                    weighted_sum += prediction * combined_weight
                    total_weight += combined_weight

                return weighted_sum / total_weight if total_weight > 0 else 0

            else:
                # Simple average
                return np.mean(list(predictions.values()))

        except Exception as e:
            logger.warning(f"Error calculating ensemble prediction: {e}")
            return np.mean(list(predictions.values())) if predictions else 0

    def update_performance(
        self, true_values: List[Any], sample_ids: Optional[List[str]] = None
    ):
        """Update performance metrics with ground truth"""
        with self.processing_lock:
            try:
                # Match predictions with true values
                recent_predictions = list(self.prediction_buffer)[-len(true_values) :]

                for i, (true_value, pred_info) in enumerate(
                    zip(true_values, recent_predictions)
                ):
                    # Update individual model performance
                    for model_name, prediction in pred_info["individual"].items():
                        tracker = self.performance_trackers[model_name]

                        # Calculate score (accuracy for classification, negative error for regression)
                        if isinstance(true_value, (int, float)) and isinstance(
                            prediction, (int, float)
                        ):
                            if (
                                abs(true_value - prediction) < 0.5
                            ):  # Classification-like
                                score = (
                                    1.0 if abs(true_value - prediction) < 0.5 else 0.0
                                )
                                tracker.correct_predictions += 1 if score > 0.5 else 0
                            else:  # Regression-like
                                error = abs(true_value - prediction)
                                score = 1.0 / (1.0 + error)  # Convert error to score
                                tracker.total_error += error
                        else:
                            score = 1.0 if prediction == true_value else 0.0
                            tracker.correct_predictions += 1 if score > 0.5 else 0

                        tracker.recent_scores.append(score)
                        tracker.total_predictions += 1
                        tracker.last_updated = datetime.now()

                        # Update confidence scores
                        confidence = pred_info["confidences"].get(model_name, 1.0)
                        tracker.confidence_scores.append(confidence)

                logger.debug(f"Updated performance for {len(true_values)} samples")

            except Exception as e:
                logger.error(f"Error updating performance: {e}")

    def _check_adaptation_triggers(self):
        """Check if adaptation is needed"""
        try:
            # Check retraining trigger
            if (
                self.sample_count - self.last_retraining
                >= self.config.retraining_frequency
            ):
                self._trigger_retraining()
                self.last_retraining = self.sample_count

            # Check model selection trigger
            if (
                self.sample_count - self.last_model_selection
                >= self.config.model_selection_frequency
            ):
                self._trigger_model_selection()
                self.last_model_selection = self.sample_count

            # Check drift detection
            if self.config.drift_detection_enabled:
                self._check_concept_drift()

        except Exception as e:
            logger.error(f"Error checking adaptation triggers: {e}")

    def _trigger_retraining(self):
        """Trigger retraining of models with recent data"""
        if len(self.data_buffer) < self.config.batch_size:
            return

        try:
            # Prepare recent data
            recent_X = pd.DataFrame(list(self.data_buffer)[-self.config.batch_size :])
            recent_y = np.array(list(self.target_buffer)[-self.config.batch_size :])

            retrained_models = []

            for model_name, model_info in self.active_models.items():
                try:
                    model = model_info["model"]

                    if model_info["is_online"] and hasattr(model, "partial_fit"):
                        # Online learning
                        model.partial_fit(recent_X, recent_y)
                        retrained_models.append(model_name)

                    elif hasattr(model, "fit"):
                        # Batch retraining with recent data
                        model.fit(recent_X, recent_y)
                        retrained_models.append(model_name)

                    model_info["last_training_time"] = datetime.now()

                except Exception as e:
                    logger.warning(f"Failed to retrain model {model_name}: {e}")

            if retrained_models:
                event = AdaptationEvent(
                    timestamp=datetime.now(),
                    event_type="model_update",
                    model_affected=",".join(retrained_models),
                    performance_before=0.0,  # Would need to calculate
                    performance_after=0.0,  # Would need to calculate
                    trigger_reason="scheduled_retraining",
                    adaptation_details={
                        "batch_size": self.config.batch_size,
                        "models_updated": len(retrained_models),
                    },
                )
                self.adaptation_history.append(event)

                logger.info(
                    f"Retrained {len(retrained_models)} models with {self.config.batch_size} samples"
                )

        except Exception as e:
            logger.error(f"Error in retraining: {e}")

    def _trigger_model_selection(self):
        """Trigger dynamic model selection and weight adjustment"""
        try:
            # Calculate recent performance for each model
            performance_scores = {}

            for model_name, tracker in self.performance_trackers.items():
                if len(tracker.recent_scores) > 0:
                    recent_performance = np.mean(
                        list(tracker.recent_scores)[-100:]
                    )  # Last 100 predictions
                    performance_scores[model_name] = recent_performance
                else:
                    performance_scores[model_name] = 0.5  # Default neutral score

            # Update model weights based on performance
            total_performance = sum(performance_scores.values())
            if total_performance > 0:
                for model_name in self.model_weights:
                    new_weight = performance_scores[model_name] / total_performance
                    old_weight = self.model_weights[model_name]

                    # Smooth weight update
                    self.model_weights[model_name] = 0.7 * old_weight + 0.3 * new_weight

            event = AdaptationEvent(
                timestamp=datetime.now(),
                event_type="ensemble_reweight",
                model_affected="all",
                performance_before=0.0,
                performance_after=0.0,
                trigger_reason="scheduled_model_selection",
                adaptation_details={"new_weights": dict(self.model_weights)},
            )
            self.adaptation_history.append(event)

            logger.info(f"Updated model weights: {self.model_weights}")

        except Exception as e:
            logger.error(f"Error in model selection: {e}")

    def _check_concept_drift(self):
        """Check for concept drift and adapt accordingly"""
        if len(self.data_buffer) < 200:  # Need sufficient data
            return

        try:
            # Simple drift detection using performance degradation
            for model_name, tracker in self.performance_trackers.items():
                if len(tracker.recent_scores) < 100:
                    continue

                recent_scores = list(tracker.recent_scores)
                old_performance = (
                    np.mean(recent_scores[-200:-100])
                    if len(recent_scores) >= 200
                    else 0.5
                )
                new_performance = np.mean(recent_scores[-100:])

                performance_drop = old_performance - new_performance

                if performance_drop > self.config.adaptation_threshold:
                    logger.warning(
                        f"Concept drift detected for {model_name}: "
                        f"performance dropped by {performance_drop:.3f}"
                    )

                    # Trigger adaptation
                    self._adapt_to_drift(model_name, float(performance_drop))

        except Exception as e:
            logger.error(f"Error checking concept drift: {e}")

    def _adapt_to_drift(self, model_name: str, drift_magnitude: float):
        """Adapt to detected concept drift"""
        try:
            # Reduce model weight
            current_weight = self.model_weights.get(model_name, 1.0)
            new_weight = current_weight * (1.0 - drift_magnitude)
            self.model_weights[model_name] = max(0.1, new_weight)  # Minimum weight

            # Trigger immediate retraining with more recent data
            if len(self.data_buffer) >= self.config.batch_size:
                model_info = self.active_models[model_name]
                model = model_info["model"]

                recent_X = pd.DataFrame(
                    list(self.data_buffer)[-self.config.batch_size :]
                )
                recent_y = np.array(list(self.target_buffer)[-self.config.batch_size :])

                if hasattr(model, "fit"):
                    model.fit(recent_X, recent_y)
                    model_info["last_training_time"] = datetime.now()

            event = AdaptationEvent(
                timestamp=datetime.now(),
                event_type="drift_detected",
                model_affected=model_name,
                performance_before=0.0,
                performance_after=0.0,
                trigger_reason=f"drift_magnitude_{drift_magnitude:.3f}",
                adaptation_details={"weight_reduction": current_weight - new_weight},
            )
            self.adaptation_history.append(event)

            logger.info(
                f"Adapted to drift for {model_name}: weight {current_weight:.3f} -> {new_weight:.3f}"
            )

        except Exception as e:
            logger.error(f"Error adapting to drift for {model_name}: {e}")

    def _adaptation_worker(self):
        """Background worker for continuous adaptation"""
        while self.is_running:
            try:
                time.sleep(1.0)  # Check every second

                # Continuous monitoring and micro-adaptations
                if self.sample_count > 0 and self.sample_count % 100 == 0:
                    self._micro_adaptation()

            except Exception as e:
                logger.error(f"Error in adaptation worker: {e}")

    def _micro_adaptation(self):
        """Micro-adaptations performed frequently"""
        try:
            # Adjust learning rates, update confidence thresholds, etc.
            # This is a placeholder for more sophisticated adaptations
            pass

        except Exception as e:
            logger.error(f"Error in micro-adaptation: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                "timestamp": datetime.now(),
                "is_running": self.is_running,
                "total_samples_processed": self.sample_count,
                "active_models": len(self.active_models),
                "model_weights": dict(self.model_weights),
                "recent_adaptations": len(
                    [
                        e
                        for e in self.adaptation_history
                        if e.timestamp > datetime.now() - timedelta(hours=1)
                    ]
                ),
                "buffer_status": {
                    "data_buffer_size": len(self.data_buffer),
                    "prediction_buffer_size": len(self.prediction_buffer),
                    "target_buffer_size": len(self.target_buffer),
                },
                "model_performance": {},
            }

            # Add model performance summaries
            for model_name, tracker in self.performance_trackers.items():
                if tracker.total_predictions > 0:
                    accuracy = tracker.correct_predictions / tracker.total_predictions
                    recent_scores = list(tracker.recent_scores)
                    recent_avg = np.mean(recent_scores) if recent_scores else 0.0
                    avg_prediction_time = (
                        np.mean(list(tracker.prediction_times))
                        if tracker.prediction_times
                        else 0.0
                    )

                    status["model_performance"][model_name] = {
                        "total_predictions": tracker.total_predictions,
                        "accuracy": accuracy,
                        "recent_performance": recent_avg,
                        "avg_prediction_time_ms": avg_prediction_time,
                        "last_updated": tracker.last_updated,
                    }

            return status

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}

    def save_adaptation_history(self, filepath: Optional[str] = None) -> str:
        """Save adaptation history to file"""
        if not filepath:
            filepath = (
                f"adaptation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        try:
            history_data = {
                "system_config": {
                    "batch_size": self.config.batch_size,
                    "adaptation_threshold": self.config.adaptation_threshold,
                    "retraining_frequency": self.config.retraining_frequency,
                },
                "adaptation_events": [
                    {
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type,
                        "model_affected": event.model_affected,
                        "trigger_reason": event.trigger_reason,
                        "adaptation_details": event.adaptation_details,
                    }
                    for event in self.adaptation_history
                ],
                "final_system_status": self.get_system_status(),
            }

            with open(filepath, "w") as f:
                json.dump(history_data, f, indent=2, default=str)

            logger.info(f"Adaptation history saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving adaptation history: {e}")
            return ""
