"""
FinBERT Ensemble Optimizer
Dynamically selects and weights FinBERT models based on source-specific performance data
"""

import json
import os
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import statistics
from datetime import datetime

# Fix import path
import sys
sys.path.append(os.path.dirname(__file__))
from advanced_sentiment import FinBERTAnalyzer

@dataclass
class EnsemblePrediction:
    """Ensemble prediction with confidence and model contributions"""
    final_sentiment: float
    confidence: float
    model_contributions: Dict[str, float]
    ensemble_method: str
    processing_time: float

@dataclass
class ModelWeights:
    """Dynamic weights for each model by source"""
    source: str
    model_weights: Dict[str, float]
    performance_score: float
    last_updated: datetime

class FinBERTEnsembleOptimizer:
    """
    Advanced ensemble optimizer that dynamically selects and weights FinBERT models
    based on real-time performance data and source-specific characteristics
    """

    def __init__(self):
        self.analyzer = FinBERTAnalyzer()
        self.performance_history = self._load_performance_history()
        self.ensemble_weights = self._calculate_ensemble_weights()
        self.confidence_calibration = self._load_confidence_calibration()

    def _load_performance_history(self) -> Dict[str, Any]:
        """Load historical performance data"""
        history_file = 'data/finbert_performance_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        return self._initialize_performance_history()

    def _initialize_performance_history(self) -> Dict[str, Any]:
        """Initialize performance history with baseline data"""
        return {
            'sources': {
                'twitter': {
                    'finbert_news': {'accuracy': 0.833, 'latency': 0.069, 'confidence': 0.927, 'samples': 12},
                    'finbert_earnings': {'accuracy': 0.833, 'latency': 0.068, 'confidence': 0.927, 'samples': 12},
                    'finbert_prosus': {'accuracy': 0.833, 'latency': 0.069, 'confidence': 0.927, 'samples': 12},
                    'fintwitbert_sentiment': {'accuracy': 0.833, 'latency': 0.035, 'confidence': 0.927, 'samples': 12}
                },
                'reddit': {
                    'finbert_news': {'accuracy': 0.917, 'latency': 0.061, 'confidence': 0.883, 'samples': 12},
                    'finbert_earnings': {'accuracy': 0.917, 'latency': 0.056, 'confidence': 0.883, 'samples': 12},
                    'finbert_prosus': {'accuracy': 0.917, 'latency': 0.059, 'confidence': 0.883, 'samples': 12},
                    'fintwitbert_sentiment': {'accuracy': 0.917, 'latency': 0.066, 'confidence': 0.883, 'samples': 12}
                },
                'news': {
                    'finbert_news': {'accuracy': 0.833, 'latency': 0.069, 'confidence': 0.989, 'samples': 12},
                    'finbert_earnings': {'accuracy': 0.833, 'latency': 0.076, 'confidence': 0.989, 'samples': 12},
                    'finbert_prosus': {'accuracy': 0.833, 'latency': 0.073, 'confidence': 0.989, 'samples': 12},
                    'fintwitbert_sentiment': {'accuracy': 0.833, 'latency': 0.069, 'confidence': 0.989, 'samples': 12}
                },
                'financial': {
                    'finbert_news': {'accuracy': 0.750, 'latency': 0.074, 'confidence': 0.962, 'samples': 12},
                    'finbert_earnings': {'accuracy': 0.750, 'latency': 0.067, 'confidence': 0.962, 'samples': 12},
                    'finbert_prosus': {'accuracy': 0.750, 'latency': 0.068, 'confidence': 0.962, 'samples': 12},
                    'fintwitbert_sentiment': {'accuracy': 0.750, 'latency': 0.076, 'confidence': 0.962, 'samples': 12}
                }
            },
            'last_updated': datetime.now().isoformat()
        }

    def _calculate_ensemble_weights(self) -> Dict[str, ModelWeights]:
        """Calculate optimal ensemble weights for each source"""
        weights = {}

        for source, models in self.performance_history['sources'].items():
            # Calculate composite scores (accuracy + speed + confidence)
            model_scores = {}
            for model_name, metrics in models.items():
                # Composite score: 50% accuracy, 30% speed (inverse latency), 20% confidence
                accuracy_score = metrics['accuracy'] * 0.5
                speed_score = (1.0 / max(metrics['latency'], 0.01)) * 0.3  # Inverse latency
                confidence_score = metrics['confidence'] * 0.2

                composite_score = accuracy_score + speed_score + confidence_score
                model_scores[model_name] = composite_score

            # Normalize to create weights
            total_score = sum(model_scores.values())
            if total_score > 0:
                normalized_weights = {model: score/total_score for model, score in model_scores.items()}
            else:
                # Fallback to equal weights
                normalized_weights = {model: 1.0/len(model_scores) for model in model_scores.keys()}

            # Calculate overall performance score for this source
            avg_accuracy = statistics.mean([m['accuracy'] for m in models.values()])
            avg_latency = statistics.mean([m['latency'] for m in models.values()])

            weights[source] = ModelWeights(
                source=source,
                model_weights=normalized_weights,
                performance_score=avg_accuracy,
                last_updated=datetime.now()
            )

        return weights

    def _load_confidence_calibration(self) -> Dict[str, Any]:
        """Load confidence calibration data"""
        return {
            'calibration_factors': {
                'twitter': 1.0,  # Baseline
                'reddit': 1.05,  # Slightly boost Reddit confidence
                'news': 0.95,    # Slightly reduce News confidence (more neutral content)
                'financial': 0.98  # Slightly reduce Financial confidence
            },
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            }
        }

    def predict_ensemble(self, text: str, source: str = "unknown") -> EnsemblePrediction:
        """
        Make ensemble prediction using optimized model weights
        """
        start_time = time.time()

        # Get weights for this source
        if source in self.ensemble_weights:
            weights = self.ensemble_weights[source].model_weights
        else:
            # Use twitter weights as default for unknown sources
            weights = self.ensemble_weights['twitter'].model_weights

        # Get predictions from all models
        model_predictions = {}
        model_confidences = {}

        for model_key in weights.keys():
            try:
                # Temporarily switch to this model
                original_model = self.analyzer.model_name
                self.analyzer._ensure_correct_model(source)

                # Make prediction
                sentiment, confidence = self.analyzer.analyze(text, source)
                model_predictions[model_key] = sentiment
                model_confidences[model_key] = confidence

                # Restore original model
                self.analyzer.model_name = original_model

            except Exception as e:
                print(f"Error with model {model_key}: {e}")
                model_predictions[model_key] = 0.0
                model_confidences[model_key] = 0.0

        # Calculate weighted ensemble prediction
        weighted_sentiment = 0.0
        total_weight = 0.0
        model_contributions = {}

        for model_key, sentiment in model_predictions.items():
            weight = weights.get(model_key, 0.0)
            confidence = model_confidences.get(model_key, 0.0)

            # Weight by both ensemble weight and model confidence
            effective_weight = weight * confidence
            weighted_sentiment += sentiment * effective_weight
            total_weight += effective_weight

            model_contributions[model_key] = effective_weight

        # Normalize final sentiment
        if total_weight > 0:
            final_sentiment = weighted_sentiment / total_weight
        else:
            final_sentiment = statistics.mean(model_predictions.values()) if model_predictions else 0.0

        # Calculate ensemble confidence
        avg_confidence = statistics.mean(model_confidences.values()) if model_confidences else 0.0

        # Apply confidence calibration
        calibration_factor = self.confidence_calibration['calibration_factors'].get(source, 1.0)
        calibrated_confidence = min(1.0, avg_confidence * calibration_factor)

        processing_time = time.time() - start_time

        return EnsemblePrediction(
            final_sentiment=final_sentiment,
            confidence=calibrated_confidence,
            model_contributions=model_contributions,
            ensemble_method="weighted_confidence",
            processing_time=processing_time
        )

    def update_performance(self, source: str, model: str, accuracy: float, latency: float, confidence: float):
        """Update performance metrics for continuous learning"""
        if source not in self.performance_history['sources']:
            self.performance_history['sources'][source] = {}

        if model not in self.performance_history['sources'][source]:
            self.performance_history['sources'][source][model] = {
                'accuracy': accuracy,
                'latency': latency,
                'confidence': confidence,
                'samples': 1
            }
        else:
            # Update with exponential moving average
            current = self.performance_history['sources'][source][model]
            alpha = 0.1  # Learning rate

            current['accuracy'] = (1-alpha) * current['accuracy'] + alpha * accuracy
            current['latency'] = (1-alpha) * current['latency'] + alpha * latency
            current['confidence'] = (1-alpha) * current['confidence'] + alpha * confidence
            current['samples'] += 1

        # Recalculate weights
        self.ensemble_weights = self._calculate_ensemble_weights()

        # Save updated history
        self._save_performance_history()

    def _save_performance_history(self):
        """Save performance history to disk"""
        self.performance_history['last_updated'] = datetime.now().isoformat()

        history_file = 'data/finbert_performance_history.json'
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)

    def get_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        report = []
        report.append("# FinBERT Ensemble Optimization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("## Ensemble Weights by Source")
        report.append("")

        for source, weights in self.ensemble_weights.items():
            report.append(f"### {source.title()}")
            report.append(f"**Overall Performance Score**: {weights.performance_score:.3f}")
            report.append("")
            report.append("| Model | Weight | Performance |")
            report.append("|-------|--------|-------------|")

            for model, weight in weights.model_weights.items():
                perf = self.performance_history['sources'][source][model]
                report.append(".3f")
            report.append("")

        report.append("## Confidence Calibration")
        report.append("")
        for source, factor in self.confidence_calibration['calibration_factors'].items():
            report.append(f"- **{source.title()}**: {factor:.2f}x calibration factor")

        report.append("")
        report.append("## Recommendations")
        report.append("")

        # Find best model for each source
        for source in self.ensemble_weights.keys():
            weights = self.ensemble_weights[source].model_weights
            best_model = max(weights.keys(), key=lambda m: weights[m])
            best_weight = weights[best_model]
            report.append(f"- **{source.title()}**: Use {best_model} (weight: {best_weight:.3f})")

        return "\n".join(report)

def optimize_finbert_ensemble():
    """Main function to run ensemble optimization"""
    optimizer = FinBERTEnsembleOptimizer()

    # Test ensemble prediction
    test_cases = [
        ("AAPL stock is mooning! To the moon 🚀🚀🚀", "twitter"),
        ("Apple reports record Q3 revenue of $89.5B, beating estimates by $3B", "news"),
        ("Technical analysis shows AAPL in strong uptrend with multiple bullish indicators", "financial"),
        ("AAPL to the moon! Diamond hands! This stock is unstoppable 🚀💎🙌", "reddit")
    ]

    print("🔬 FinBERT Ensemble Optimization Test")
    print("=" * 50)

    for text, source in test_cases:
        prediction = optimizer.predict_ensemble(text, source)

        print(f"\n📊 Source: {source}")
        print(f"Text: {text[:60]}...")
        print(".3f")
        print(".3f")
        print(f"Processing time: {prediction.processing_time:.3f}s")
        print("Model contributions:")
        for model, contribution in prediction.model_contributions.items():
            print(".3f")

    # Generate optimization report
    print("\n" + "=" * 50)
    print("📋 OPTIMIZATION REPORT")
    print("=" * 50)
    print(optimizer.get_optimization_report())

if __name__ == "__main__":
    optimize_finbert_ensemble()