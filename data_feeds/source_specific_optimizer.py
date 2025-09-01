"""
Source-Specific FinBERT Optimizer
Implements fine-tuning and optimization for different content sources
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from data_feeds.advanced_sentiment import FinBERTAnalyzer

logger = logging.getLogger(__name__)

class SourceSpecificOptimizer:
    """
    Optimizes FinBERT models for specific content sources (Twitter, Reddit, News)
    """

    def __init__(self):
        self.source_configs = {
            'twitter': {
                'model': 'fintwitbert_sentiment',
                'calibration': {'factor': 0.95, 'offset': 0.02},
                'preprocessing': ['remove_urls', 'remove_mentions', 'expand_abbreviations'],
                'confidence_threshold': 0.6,
                'fallback_model': 'finbert_tone'
            },
            'reddit': {
                'model': 'fintwitbert_sentiment',
                'calibration': {'factor': 0.85, 'offset': -0.03},
                'preprocessing': ['remove_urls', 'expand_abbreviations', 'normalize_cashtags'],
                'confidence_threshold': 0.5,
                'fallback_model': 'finbert_tone'
            },
            'news': {
                'model': 'finbert_tone',
                'calibration': {'factor': 0.23, 'offset': -0.15},
                'preprocessing': ['expand_abbreviations', 'normalize_financial_terms'],
                'confidence_threshold': 0.7,
                'fallback_model': 'finbert_news'
            }
        }

        self.performance_history = {}
        self.load_performance_history()

    def load_performance_history(self):
        """Load historical performance data"""
        history_file = 'data/finbert_source_performance.json'
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"Loaded performance history for {len(self.performance_history)} sources")
            except Exception as e:
                logger.warning(f"Failed to load performance history: {e}")
                self.performance_history = {}

    def save_performance_history(self):
        """Save performance history"""
        history_file = 'data/finbert_source_performance.json'
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        try:
            with open(history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
            logger.info("Performance history saved")
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")

    def optimize_for_source(self, source: str, test_texts: List[str], ground_truth: List[float]) -> Dict[str, Any]:
        """
        Optimize model selection and calibration for a specific source
        """
        if source not in self.source_configs:
            logger.warning(f"Unknown source: {source}, using default optimization")
            source = 'news'  # Default fallback

        config = self.source_configs[source]
        results = {}

        # Test primary model
        primary_model = config['model']
        analyzer = FinBERTAnalyzer(primary_model)

        primary_scores = []
        primary_confidences = []

        for text in test_texts:
            try:
                score, confidence = analyzer.analyze(text, source)
                primary_scores.append(score)
                primary_confidences.append(confidence)
            except Exception as e:
                logger.warning(f"Primary model analysis failed for text: {e}")
                primary_scores.append(0.0)
                primary_confidences.append(0.0)

        # Calculate primary model performance
        primary_accuracy = self._calculate_accuracy(primary_scores, ground_truth)
        primary_calibration = self._calculate_calibration_error(primary_confidences, primary_scores)

        results['primary'] = {
            'model': primary_model,
            'accuracy': primary_accuracy,
            'calibration_error': primary_calibration,
            'avg_confidence': np.mean(primary_confidences),
            'config': config
        }

        # Test fallback model if primary performance is poor
        if primary_accuracy < config['confidence_threshold']:
            logger.info(f"Primary model performance ({primary_accuracy:.3f}) below threshold, testing fallback")

            fallback_model = config['fallback_model']
            fallback_analyzer = FinBERTAnalyzer(fallback_model)

            fallback_scores = []
            fallback_confidences = []

            for text in test_texts:
                try:
                    score, confidence = fallback_analyzer.analyze(text, source)
                    fallback_scores.append(score)
                    fallback_confidences.append(confidence)
                except Exception as e:
                    logger.warning(f"Fallback model analysis failed for text: {e}")
                    fallback_scores.append(0.0)
                    fallback_confidences.append(0.0)

            fallback_accuracy = self._calculate_accuracy(fallback_scores, ground_truth)
            fallback_calibration = self._calculate_calibration_error(fallback_confidences, fallback_scores)

            results['fallback'] = {
                'model': fallback_model,
                'accuracy': fallback_accuracy,
                'calibration_error': fallback_calibration,
                'avg_confidence': np.mean(fallback_confidences)
            }

            # Choose better model
            if fallback_accuracy > primary_accuracy:
                results['recommended'] = results['fallback']
                results['recommended']['reason'] = 'fallback_performed_better'
            else:
                results['recommended'] = results['primary']
                results['recommended']['reason'] = 'primary_still_better'
        else:
            results['recommended'] = results['primary']
            results['recommended']['reason'] = 'primary_above_threshold'

        # Update performance history
        if source not in self.performance_history:
            self.performance_history[source] = []

        self.performance_history[source].append({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'sample_size': len(test_texts)
        })

        # Keep only last 10 entries per source
        if len(self.performance_history[source]) > 10:
            self.performance_history[source] = self.performance_history[source][-10:]

        self.save_performance_history()

        return results

    def get_optimized_config(self, source: str) -> Dict[str, Any]:
        """
        Get the optimized configuration for a source based on historical performance
        """
        if source not in self.performance_history or not self.performance_history[source]:
            # Return default config
            return self.source_configs.get(source, self.source_configs['news'])

        # Analyze historical performance
        recent_results = self.performance_history[source][-5:]  # Last 5 tests

        best_accuracy = 0.0
        best_config = None

        for result_entry in recent_results:
            results = result_entry.get('results', {})
            recommended = results.get('recommended', {})

            if recommended and recommended.get('accuracy', 0) > best_accuracy:
                best_accuracy = recommended['accuracy']
                best_config = {
                    'model': recommended['model'],
                    'calibration': self.source_configs[source]['calibration'],
                    'source': source,
                    'performance_score': best_accuracy,
                    'last_updated': result_entry['timestamp']
                }

        if best_config:
            return best_config
        else:
            return self.source_configs.get(source, self.source_configs['news'])

    def _calculate_accuracy(self, predictions: List[float], ground_truth: List[float]) -> float:
        """Calculate accuracy based on prediction direction matching ground truth"""
        if len(predictions) != len(ground_truth):
            return 0.0

        correct = 0
        total = len(predictions)

        for pred, truth in zip(predictions, ground_truth):
            # Convert to direction: positive (>0.1), negative (<-0.1), neutral (otherwise)
            pred_dir = 1 if pred > 0.1 else -1 if pred < -0.1 else 0
            truth_dir = 1 if truth > 0.1 else -1 if truth < -0.1 else 0

            if pred_dir == truth_dir:
                correct += 1

        return correct / total if total > 0 else 0.0

    def _calculate_calibration_error(self, confidences: List[float], predictions: List[float]) -> float:
        """Calculate calibration error (Expected Calibration Error)"""
        if len(confidences) != len(predictions):
            return 1.0

        # Simple binning approach
        bins = np.linspace(0, 1, 11)  # 10 bins
        ece = 0.0

        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i + 1]

            # Find predictions in this confidence bin
            bin_mask = (np.array(confidences) >= bin_start) & (np.array(confidences) < bin_end)
            bin_predictions = np.array(predictions)[bin_mask]
            bin_confidences = np.array(confidences)[bin_mask]

            if len(bin_predictions) > 0:
                # Calculate accuracy in this bin
                bin_accuracy = np.mean(np.abs(bin_predictions) > 0.1)  # Simple accuracy measure
                bin_avg_confidence = np.mean(bin_confidences)

                # Add to ECE
                ece += len(bin_predictions) * abs(bin_accuracy - bin_avg_confidence)

        return ece / len(predictions) if len(predictions) > 0 else 1.0

    def generate_source_report(self) -> str:
        """Generate a report on source-specific optimizations"""
        report = []
        report.append("# Source-Specific FinBERT Optimization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        for source in ['twitter', 'reddit', 'news']:
            report.append(f"## {source.title()} Source Optimization")
            report.append("")

            config = self.get_optimized_config(source)
            report.append(f"- **Recommended Model**: {config.get('model', 'N/A')}")
            report.append(".3f")
            report.append(f"- **Last Updated**: {config.get('last_updated', 'N/A')}")
            report.append("")

            # Historical performance
            if source in self.performance_history:
                history = self.performance_history[source]
                if history:
                    recent = history[-1]
                    results = recent.get('results', {})
                    recommended = results.get('recommended', {})

                    report.append("### Recent Performance")
                    report.append(f"- **Accuracy**: {recommended.get('accuracy', 0):.3f}")
                    report.append(f"- **Calibration Error**: {recommended.get('calibration_error', 0):.3f}")
                    report.append(f"- **Average Confidence**: {recommended.get('avg_confidence', 0):.3f}")
                    report.append(f"- **Reason**: {recommended.get('reason', 'N/A')}")
                    report.append("")

        return "\n".join(report)

def main():
    """Test the source-specific optimizer"""
    logging.basicConfig(level=logging.INFO)

    optimizer = SourceSpecificOptimizer()

    # Test data for each source
    test_data = {
        'twitter': [
            ("AAPL stock is mooning! 🚀🚀🚀", 0.9),
            ("Just bought TSLA calls, this stock is going to explode!", 0.8),
            ("Selling all my GOOGL shares, this company is done", -0.8),
        ],
        'reddit': [
            ("r/stocks discussion about AAPL earnings", 0.6),
            ("TSLA delivery numbers are impressive", 0.8),
            ("Market crash incoming, bearish on everything", -0.9),
        ],
        'news': [
            ("Apple Inc. reported quarterly earnings that exceeded analyst expectations", 0.8),
            ("Tesla faces regulatory challenges in multiple markets", -0.6),
            ("Market volatility increased following economic indicators", -0.4),
        ]
    }

    print("🔧 Testing Source-Specific Optimizer...")
    print("=" * 50)

    for source, data in test_data.items():
        texts, ground_truth = zip(*data)
        print(f"\n📊 Optimizing for {source}...")
        results = optimizer.optimize_for_source(source, list(texts), list(ground_truth))

        recommended = results.get('recommended', {})
        print(f"  ✅ Recommended: {recommended.get('model', 'N/A')}")
        print(".3f")
        print(".3f")

    # Generate report
    print("
📄 Generating optimization report..."    report = optimizer.generate_source_report()
    print(report)

    print("\n✅ Source-specific optimization completed!")

if __name__ == "__main__":
    main()</content>