"""
FinBERT Model Accuracy Testing and Calibration Framework
Tests, optimizes, and calibrates multiple FinBERT models for financial sentiment analysis
"""

import logging
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_feeds.advanced_sentiment import FinBERTAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a single FinBERT model"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    calibration_error: float
    avg_confidence: float
    processing_time: float
    sample_size: int
    source_specific_accuracy: Dict[str, float]
    confidence_calibration_curve: List[Tuple[float, float]]

@dataclass
class TestSample:
    """Test sample with ground truth"""
    text: str
    true_sentiment: float  # -1 to 1 scale
    source: str  # 'twitter', 'news', 'reddit', etc.
    confidence: float  # Expected confidence level

class FinBERTAccuracyTester:
    """
    Comprehensive testing framework for FinBERT model accuracy and calibration
    """

    def __init__(self):
        self.models_to_test = [
            'finbert_tone',
            'finbert_pretrain',
            'finbert_news',
            'finbert_earnings',
            'finbert_prosus',
            'fintwitbert_sentiment'
        ]

        # Initialize test datasets
        self.test_datasets = {
            'twitter': self._load_twitter_test_data(),
            'news': self._load_news_test_data(),
            'reddit': self._load_reddit_test_data(),
        }
        
        # Add mixed dataset after individual datasets are loaded
        self.test_datasets['mixed'] = self._load_mixed_test_data()

        self.results_dir = 'data/finbert_accuracy_results'
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_twitter_test_data(self) -> List[TestSample]:
        """Load Twitter-specific test data"""
        return [
            TestSample("AAPL stock is mooning! To the moon 🚀🚀🚀", 0.9, "twitter", 0.8),
            TestSample("Just bought TSLA calls, this stock is going to explode!", 0.8, "twitter", 0.7),
            TestSample("NVDA earnings beat expectations, great quarter!", 0.7, "twitter", 0.8),
            TestSample("Selling all my GOOGL shares, this company is done", -0.8, "twitter", 0.7),
            TestSample("META is overvalued, time to short it", -0.6, "twitter", 0.6),
            TestSample("Just got TSLA dividends, reinvesting everything", 0.5, "twitter", 0.5),
            TestSample("Market is volatile today, staying in cash", -0.2, "twitter", 0.4),
            TestSample("Checking NVDA charts, looks like a breakout coming", 0.6, "twitter", 0.6),
            TestSample("AAPL just reported, missing estimates badly", -0.7, "twitter", 0.7),
            TestSample("TSLA delivery numbers are impressive this quarter", 0.8, "twitter", 0.8),
        ]

    def _load_news_test_data(self) -> List[TestSample]:
        """Load news-specific test data"""
        return [
            TestSample("Apple Inc. reported quarterly earnings that exceeded analyst expectations, with revenue growth of 15%.", 0.8, "news", 0.9),
            TestSample("Tesla faces regulatory challenges in multiple markets, shares decline 5%.", -0.6, "news", 0.8),
            TestSample("NVIDIA announces breakthrough in AI chip technology, stock surges.", 0.9, "news", 0.9),
            TestSample("Google parent company reports slower than expected growth in cloud services.", -0.4, "news", 0.7),
            TestSample("Meta Platforms achieves record quarterly profits, beating all estimates.", 0.8, "news", 0.9),
            TestSample("Oil prices surge following OPEC production cuts, energy stocks rally.", 0.7, "news", 0.8),
            TestSample("Federal Reserve signals potential interest rate hikes, market volatility increases.", -0.5, "news", 0.8),
            TestSample("Amazon reports strong holiday quarter, e-commerce growth accelerates.", 0.8, "news", 0.9),
            TestSample("Microsoft announces layoffs amid economic uncertainty.", -0.6, "news", 0.8),
            TestSample("Economic indicators show signs of recession, investor confidence wanes.", -0.7, "news", 0.8),
        ]

    def _load_reddit_test_data(self) -> List[TestSample]:
        """Load Reddit-specific test data"""
        return [
            TestSample("r/wallstreetbets is loving TSLA this week, calls are flying off the shelves", 0.8, "reddit", 0.7),
            TestSample("NVDA earnings discussion - everyone seems disappointed with the guidance", -0.6, "reddit", 0.7),
            TestSample("AAPL just crushed it, new product announcements are game-changing", 0.9, "reddit", 0.8),
            TestSample("GOOGL is getting hammered in after-hours, antitrust concerns mounting", -0.7, "reddit", 0.7),
            TestSample("META metaverse investments paying off, user growth accelerating", 0.6, "reddit", 0.6),
            TestSample("Short squeeze potential in GME, retail investors organizing", 0.5, "reddit", 0.5),
            TestSample("AMZN warehouse workers striking, supply chain disruptions feared", -0.5, "reddit", 0.6),
            TestSample("MSFT AI investments showing results, stock at all-time highs", 0.8, "reddit", 0.8),
            TestSample("Crypto market crash affecting altcoins, BTC holding steady", -0.4, "reddit", 0.5),
            TestSample("Discussion about inflation and its impact on tech stocks", -0.3, "reddit", 0.4),
        ]

    def _load_mixed_test_data(self) -> List[TestSample]:
        """Load mixed source test data"""
        mixed_data = []
        for source in ['twitter', 'news', 'reddit']:
            mixed_data.extend(self.test_datasets[source][:5])  # Take 5 from each
        return mixed_data

    def test_single_model(self, model_key: str, test_samples: List[TestSample]) -> ModelPerformanceMetrics:
        """
        Test a single FinBERT model on a dataset
        """
        logger.info(f"Testing model: {model_key} on {len(test_samples)} samples")

        # Create analyzer for this model
        analyzer = FinBERTAnalyzer(model_name=model_key)

        predictions = []
        processing_times = []
        source_accuracies = {}

        for sample in test_samples:
            start_time = time.time()

            # Get prediction
            predicted_sentiment, confidence = analyzer.analyze(sample.text, sample.source)

            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Calculate accuracy for this prediction
            prediction_error = abs(predicted_sentiment - sample.true_sentiment)
            accuracy = max(0, 1 - prediction_error)  # Convert error to accuracy

            predictions.append({
                'predicted': predicted_sentiment,
                'true': sample.true_sentiment,
                'confidence': confidence,
                'accuracy': accuracy,
                'source': sample.source
            })

            # Track source-specific accuracy
            if sample.source not in source_accuracies:
                source_accuracies[sample.source] = []
            source_accuracies[sample.source].append(accuracy)

        # Calculate overall metrics
        accuracies = [p['accuracy'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]

        # Calculate precision, recall, f1 for binary classification (positive vs negative)
        true_positives = sum(1 for p in predictions if p['predicted'] > 0.1 and p['true'] > 0.1)
        false_positives = sum(1 for p in predictions if p['predicted'] > 0.1 and p['true'] <= 0.1)
        false_negatives = sum(1 for p in predictions if p['predicted'] <= 0.1 and p['true'] > 0.1)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate calibration error (difference between confidence and actual accuracy)
        calibration_error = abs(statistics.mean(confidences) - statistics.mean(accuracies))

        # Create confidence calibration curve
        confidence_bins = np.linspace(0, 1, 11)
        calibration_curve = []
        for i in range(len(confidence_bins) - 1):
            bin_start = confidence_bins[i]
            bin_end = confidence_bins[i + 1]
            bin_predictions = [p for p in predictions if bin_start <= p['confidence'] < bin_end]
            if bin_predictions:
                avg_confidence = statistics.mean([p['confidence'] for p in bin_predictions])
                avg_accuracy = statistics.mean([p['accuracy'] for p in bin_predictions])
                calibration_curve.append((avg_confidence, avg_accuracy))

        # Calculate source-specific accuracies
        source_accuracy_dict = {}
        for source, acc_list in source_accuracies.items():
            source_accuracy_dict[source] = statistics.mean(acc_list) if acc_list else 0

        return ModelPerformanceMetrics(
            model_name=model_key,
            accuracy=statistics.mean(accuracies),
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            calibration_error=calibration_error,
            avg_confidence=statistics.mean(confidences),
            processing_time=statistics.mean(processing_times),
            sample_size=len(test_samples),
            source_specific_accuracy=source_accuracy_dict,
            confidence_calibration_curve=calibration_curve
        )

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive testing across all models and datasets
        """
        logger.info("Starting comprehensive FinBERT accuracy testing")

        results = {
            'timestamp': datetime.now().isoformat(),
            'models_tested': self.models_to_test,
            'datasets_tested': list(self.test_datasets.keys()),
            'model_performance': {},
            'dataset_performance': {},
            'recommendations': {}
        }

        # Test each model on each dataset
        for model_key in self.models_to_test:
            logger.info(f"Testing model: {model_key}")
            model_results = {}

            for dataset_name, test_samples in self.test_datasets.items():
                logger.info(f"  Testing on {dataset_name} dataset ({len(test_samples)} samples)")
                try:
                    performance = self.test_single_model(model_key, test_samples)
                    model_results[dataset_name] = performance
                except Exception as e:
                    logger.error(f"Error testing {model_key} on {dataset_name}: {e}")
                    model_results[dataset_name] = None

            results['model_performance'][model_key] = model_results

        # Analyze results and generate recommendations
        results['recommendations'] = self._generate_recommendations(results)

        # Save results
        self._save_results(results)

        logger.info("Comprehensive testing completed")
        return results

    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model selection and optimization recommendations"""
        recommendations = {
            'best_overall_model': None,
            'best_by_source': {},
            'calibration_issues': [],
            'performance_rankings': [],
            'optimization_suggestions': []
        }

        # Find best overall model
        model_scores = {}
        for model_key, datasets in results['model_performance'].items():
            total_score = 0
            valid_datasets = 0
            for dataset_name, performance in datasets.items():
                if performance and hasattr(performance, 'accuracy'):
                    total_score += performance.accuracy
                    valid_datasets += 1
            if valid_datasets > 0:
                model_scores[model_key] = total_score / valid_datasets

        if model_scores:
            best_model = max(model_scores.items(), key=lambda x: x[1])
            recommendations['best_overall_model'] = {
                'model': best_model[0],
                'average_accuracy': best_model[1]
            }

        # Find best model by source
        for dataset_name in self.test_datasets.keys():
            source_scores = {}
            for model_key, datasets in results['model_performance'].items():
                if dataset_name in datasets and datasets[dataset_name]:
                    performance = datasets[dataset_name]
                    if hasattr(performance, 'accuracy'):
                        source_scores[model_key] = performance.accuracy

            if source_scores:
                best_for_source = max(source_scores.items(), key=lambda x: x[1])
                recommendations['best_by_source'][dataset_name] = {
                    'model': best_for_source[0],
                    'accuracy': best_for_source[1]
                }

        # Identify calibration issues
        for model_key, datasets in results['model_performance'].items():
            for dataset_name, performance in datasets.items():
                if performance and hasattr(performance, 'calibration_error'):
                    if performance.calibration_error > 0.2:  # Significant calibration error
                        recommendations['calibration_issues'].append({
                            'model': model_key,
                            'dataset': dataset_name,
                            'calibration_error': performance.calibration_error
                        })

        # Performance rankings
        if model_scores:
            rankings = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations['performance_rankings'] = [
                {'rank': i+1, 'model': model, 'score': score}
                for i, (model, score) in enumerate(rankings)
            ]

        return recommendations

    def _save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/finbert_accuracy_test_{timestamp}.json"

        # Convert dataclasses to dictionaries for JSON serialization
        serializable_results = self._make_serializable(results)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Results saved to {filename}")

    def _make_serializable(self, obj):
        """Convert dataclasses and other non-serializable objects to dictionaries"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj

    def run_ab_test(self, model_a: str, model_b: str, test_samples: List[TestSample]) -> Dict[str, Any]:
        """
        Run A/B test between two models
        """
        logger.info(f"Running A/B test: {model_a} vs {model_b}")

        results_a = self.test_single_model(model_a, test_samples)
        results_b = self.test_single_model(model_b, test_samples)

        # Statistical significance test (simple t-test approximation)
        accuracies_a = []  # We'd need to collect per-sample accuracies
        accuracies_b = []

        # For now, compare overall metrics
        winner = "tie"
        if results_a.accuracy > results_b.accuracy:
            winner = model_a
        elif results_b.accuracy > results_a.accuracy:
            winner = model_b

        return {
            'model_a': {
                'name': model_a,
                'accuracy': results_a.accuracy,
                'confidence': results_a.avg_confidence,
                'calibration_error': results_a.calibration_error
            },
            'model_b': {
                'name': model_b,
                'accuracy': results_b.accuracy,
                'confidence': results_b.avg_confidence,
                'calibration_error': results_b.calibration_error
            },
            'winner': winner,
            'improvement': abs(results_a.accuracy - results_b.accuracy)
        }

def main():
    """Main function to run FinBERT accuracy testing"""
    logging.basicConfig(level=logging.INFO)

    tester = FinBERTAccuracyTester()
    results = tester.run_comprehensive_test()

    # Print summary
    print("\n" + "="*60)
    print("FINBERT ACCURACY TEST RESULTS")
    print("="*60)

    if 'best_overall_model' in results.get('recommendations', {}):
        best = results['recommendations']['best_overall_model']
        print(f"🏆 Best Overall Model: {best['model']} (Accuracy: {best['average_accuracy']:.3f})")

    print("\n📊 Model Performance Rankings:")
    rankings = results.get('recommendations', {}).get('performance_rankings', [])
    for rank_info in rankings[:5]:  # Top 5
        print(f"  {rank_info['rank']}. {rank_info['model']}: {rank_info['score']:.3f}")

    print("\n🎯 Best Model by Source:")
    best_by_source = results.get('recommendations', {}).get('best_by_source', {})
    for source, info in best_by_source.items():
        print(f"  {source.title()}: {info['model']} ({info['accuracy']:.3f})")

    calibration_issues = results.get('recommendations', {}).get('calibration_issues', [])
    if calibration_issues:
        print(f"\n⚠️  Calibration Issues Found: {len(calibration_issues)}")
        for issue in calibration_issues[:3]:  # Show top 3
            print(f"  {issue['model']} on {issue['dataset']}: {issue['calibration_error']:.3f}")

    print(f"\n📁 Results saved to: {tester.results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()