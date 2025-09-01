"""
Comprehensive Model Accuracy Testing Framework
Tests all FinBERT models across different sentiment sources to find optimal configurations
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set Hugging Face environment variables to reduce rate limiting
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '0')  # Allow online access but with retries
os.environ.setdefault('HF_HUB_TIMEOUT', '30')  # Increase timeout
os.environ.setdefault('HF_HUB_NUM_WORKERS', '1')  # Reduce parallel downloads

from data_feeds.advanced_sentiment import FinBERTAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Individual test result"""
    model: str
    source: str
    text: str
    expected_sentiment: float  # -1 to 1 scale
    predicted_sentiment: float
    confidence: float
    latency: float
    accuracy_score: float  # 1 if correct direction, 0 if wrong

@dataclass
class ModelPerformance:
    """Performance metrics for a model on a specific source"""
    model: str
    source: str
    accuracy: float
    avg_latency: float
    total_tests: int
    correct_predictions: int
    avg_confidence: float
    precision: float  # True positives / (True positives + False positives)
    recall: float     # True positives / (True positives + False negatives)

class ComprehensiveModelTester:
    """
    Comprehensive testing framework for FinBERT models across different sources
    """

    def __init__(self):
        self.analyzer = FinBERTAnalyzer()
        self.test_datasets = self._create_test_datasets()
        self.results_dir = 'data/model_accuracy_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Pre-load all models to avoid rate limiting
        self.loaded_models = {}
        self._preload_all_models()

    def _preload_all_models(self):
        """Pre-load all models to cache them and avoid repeated downloads"""
        print("🔄 Pre-loading all FinBERT models to avoid rate limiting...")
        
        for model_key, model_name in self.analyzer.model_options.items():
            print(f"  📥 Loading {model_key}...")
            try:
                # Create a temporary analyzer instance for this model
                temp_analyzer = FinBERTAnalyzer(model_name)
                self.loaded_models[model_key] = temp_analyzer
                print(f"    ✅ {model_key} loaded successfully")
                
                # Add delay between model loads to respect rate limits
                time.sleep(3)
                
            except Exception as e:
                print(f"    ❌ Failed to load {model_key}: {e}")
                self.loaded_models[model_key] = None
                
                # Add delay even on failure
                time.sleep(2)
        
        print(f"🔄 Pre-loading complete. Loaded {len([m for m in self.loaded_models.values() if m is not None])}/{len(self.loaded_models)} models")
    
    def cleanup_models(self):
        """Clean up loaded models to free memory"""
        print("🧹 Cleaning up loaded models...")
        for model_key, analyzer in self.loaded_models.items():
            if analyzer:
                # Clear model references
                analyzer.model = None
                analyzer.tokenizer = None
                analyzer.is_loaded = False
        self.loaded_models.clear()
        print("✅ Model cleanup complete")

    def _create_test_datasets(self) -> Dict[str, List[Tuple[str, float]]]:
        """Create comprehensive test datasets for different sources"""

        return {
            'twitter': [
                # Positive financial tweets
                ("Just bought $AAPL calls, this stock is mooning! 🚀", 0.9),
                ("$TSLA earnings beat expectations by $0.50! Bullish!", 0.95),
                ("$NVDA breaking out to new highs! This is just the beginning 📈", 0.85),
                ("Bought the dip on $SPY, expecting strong recovery", 0.8),

                # Negative financial tweets
                ("$AAPL stock crashing after disappointing iPhone sales 📉", -0.9),
                ("$TSLA delivery numbers missed estimates badly. Bearish!", -0.95),
                ("$NVDA guidance cut due to weak AI demand. Selling everything", -0.85),
                ("$SPY breaking major support levels. Market crash incoming", -0.8),

                # Neutral financial tweets
                ("$AAPL reports Q3 earnings after market close", 0.0),
                ("$TSLA announces new factory location in Mexico", 0.1),
                ("$NVDA board approves $25B stock buyback program", 0.2),
                ("$SPY dividend yield at 1.8%, above historical average", -0.1),
            ],

            'news': [
                # Positive financial news
                ("Apple reports record Q3 revenue of $89.5B, beating estimates by $3B", 0.95),
                ("Tesla delivers 400,000 vehicles in Q3, exceeding analyst expectations", 0.9),
                ("NVIDIA announces $10B AI partnership with major cloud providers", 0.85),
                ("S&P 500 reaches new all-time high as tech stocks rally", 0.8),

                # Negative financial news
                ("Apple warns of slowing iPhone demand in China, cuts production", -0.9),
                ("Tesla faces production delays at new Austin factory", -0.8),
                ("NVIDIA faces antitrust scrutiny over GPU market dominance", -0.75),
                ("Federal Reserve signals potential interest rate hikes", -0.7),

                # Neutral financial news
                ("Apple announces new iPhone 15 lineup with incremental improvements", 0.0),
                ("Tesla reports Q3 deliveries in line with guidance", 0.1),
                ("NVIDIA quarterly results to be released after market close", 0.0),
                ("S&P 500 futures trading slightly higher in pre-market", 0.2),
            ],

            'reddit': [
                # Positive Reddit/WallStreetBets style
                ("AAPL to the moon! Diamond hands! This stock is unstoppable 🚀💎🙌", 0.95),
                ("TSLA earnings crushed it! Bulls are taking over this market!", 0.9),
                ("NVDA calls paying off big time. This AI hype is real!", 0.85),
                ("Bought the dip on SPY, expecting 10% upside this week", 0.8),

                # Negative Reddit/WallStreetBets style
                ("AAPL overvalued garbage. Dumping all shares now 📉", -0.9),
                ("TSLA delivery numbers are a joke. Musk is lying to everyone", -0.85),
                ("NVDA is getting margin called. AI winter is here!", -0.8),
                ("SPY breaking down. Bears are winning this fight", -0.75),

                # Neutral Reddit/WallStreetBets style
                ("AAPL earnings call tonight. What do you think?", 0.0),
                ("TSLA new model announcement next week", 0.1),
                ("NVDA earnings after market close tomorrow", 0.0),
                ("SPY options expiring this Friday", -0.1),
            ],

            'financial': [
                # Positive financial analysis
                ("Technical analysis shows AAPL in strong uptrend with multiple bullish indicators", 0.8),
                ("TSLA fundamentals improving with positive free cash flow generation", 0.75),
                ("NVDA valuation attractive relative to growth prospects and competitive moat", 0.7),
                ("SPY showing accumulation patterns suggesting institutional buying", 0.65),

                # Negative financial analysis
                ("AAPL facing margin pressure from increased competition in premium segment", -0.75),
                ("TSLA balance sheet deterioration with negative operating cash flow", -0.8),
                ("NVDA facing supply chain constraints limiting production capacity", -0.7),
                ("SPY technical indicators suggest potential reversal at resistance levels", -0.65),

                # Neutral financial analysis
                ("AAPL trading at fair valuation given current growth trajectory", 0.0),
                ("TSLA valuation metrics in line with industry averages", 0.1),
                ("NVDA competitive positioning stable despite market share pressures", -0.1),
                ("SPY sector rotation analysis shows balanced market participation", 0.0),
            ]
        }

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive accuracy tests for all models across all sources"""

        print("🧪 Starting Comprehensive Model Accuracy Testing...")
        print("=" * 60)

        all_results = []
        model_performance = {}

        # Test each model individually
        models_to_test = [
            'finbert_news',
            'finbert_earnings',
            'finbert_prosus',
            'fintwitbert_sentiment'
        ]

        for model_key in models_to_test:
            print(f"\n🔬 Testing model: {model_key}")
            model_results = []

            for source, test_cases in self.test_datasets.items():
                print(f"  📊 Testing on {source} data ({len(test_cases)} samples)...")

                source_results = self._test_model_on_source(model_key, source, test_cases)
                model_results.extend(source_results)

                # Calculate performance metrics for this model-source combination
                performance = self._calculate_performance_metrics(source_results)
                key = f"{model_key}_{source}"
                model_performance[key] = performance

                print(".3f")
            all_results.extend(model_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(model_performance)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_data = {
            'timestamp': timestamp,
            'total_tests': len(all_results),
            'models_tested': len(models_to_test),
            'sources_tested': len(self.test_datasets),
            'model_performance': model_performance,
            'recommendations': recommendations,
            'detailed_results': [self._result_to_dict(r) for r in all_results]
        }

        results_file = f"{self.results_dir}/comprehensive_accuracy_test_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {results_file}")
        return results_data

    def _test_model_on_source(self, model_key: str, source: str, test_cases: List[Tuple[str, float]]) -> List[TestResult]:
        """Test a specific model on a specific source using pre-loaded model"""

        results = []
        
        # Get the pre-loaded model analyzer
        model_analyzer = self.loaded_models.get(model_key)
        if model_analyzer is None:
            print(f"  ⚠️  Skipping {model_key} on {source} - model failed to load")
            return results

        for text, expected_sentiment in test_cases:
            start_time = time.time()

            try:
                # Get sentiment prediction using the pre-loaded model
                # Pass the source to ensure proper model selection during accuracy testing
                sentiment_score, confidence = model_analyzer.analyze(text, source)
                
                latency = time.time() - start_time

                # Calculate accuracy (1 if correct direction, 0 if wrong)
                predicted_direction = 1 if sentiment_score > 0 else -1 if sentiment_score < 0 else 0
                expected_direction = 1 if expected_sentiment > 0 else -1 if expected_sentiment < 0 else 0
                accuracy_score = 1.0 if predicted_direction == expected_direction else 0.0

                test_result = TestResult(
                    model=model_key,
                    source=source,
                    text=text,
                    expected_sentiment=expected_sentiment,
                    predicted_sentiment=sentiment_score,
                    confidence=confidence,
                    latency=latency,
                    accuracy_score=accuracy_score
                )

                results.append(test_result)
                
            except Exception as e:
                print(f"    ❌ Error testing {model_key} on '{text[:50]}...': {e}")
                # Create a failed test result
                test_result = TestResult(
                    model=model_key,
                    source=source,
                    text=text,
                    expected_sentiment=expected_sentiment,
                    predicted_sentiment=0.0,
                    confidence=0.0,
                    latency=time.time() - start_time,
                    accuracy_score=0.0
                )
                results.append(test_result)

        return results

    def _calculate_performance_metrics(self, results: List[TestResult]) -> ModelPerformance:
        """Calculate comprehensive performance metrics"""

        if not results:
            return ModelPerformance(
                model=results[0].model if results else "unknown",
                source=results[0].source if results else "unknown",
                accuracy=0.0,
                avg_latency=0.0,
                total_tests=0,
                correct_predictions=0,
                avg_confidence=0.0,
                precision=0.0,
                recall=0.0
            )

        correct_predictions = sum(1 for r in results if r.accuracy_score == 1.0)
        accuracy = correct_predictions / len(results)

        avg_latency = statistics.mean(r.latency for r in results)
        avg_confidence = statistics.mean(r.confidence for r in results)

        # Calculate precision and recall for positive sentiment
        true_positives = sum(1 for r in results if r.expected_sentiment > 0 and r.predicted_sentiment > 0)
        false_positives = sum(1 for r in results if r.expected_sentiment <= 0 and r.predicted_sentiment > 0)
        false_negatives = sum(1 for r in results if r.expected_sentiment > 0 and r.predicted_sentiment <= 0)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        return ModelPerformance(
            model=results[0].model,
            source=results[0].source,
            accuracy=accuracy,
            avg_latency=avg_latency,
            total_tests=len(results),
            correct_predictions=correct_predictions,
            avg_confidence=avg_confidence,
            precision=precision,
            recall=recall
        )

    def _generate_recommendations(self, model_performance: Dict[str, ModelPerformance]) -> Dict[str, Any]:
        """Generate recommendations based on performance data"""

        recommendations = {
            'best_overall_model': {},
            'best_by_source': {},
            'performance_rankings': [],
            'calibration_issues': []
        }

        # Find best model for each source
        sources = set()
        for key, perf in model_performance.items():
            sources.add(perf.source)

        for source in sources:
            source_models = {k: v for k, v in model_performance.items() if v.source == source}
            if source_models:
                best_model = max(source_models.values(), key=lambda x: x.accuracy)
                recommendations['best_by_source'][source] = {
                    'model': best_model.model,
                    'accuracy': best_model.accuracy,
                    'avg_latency': best_model.avg_latency,
                    'confidence': best_model.avg_confidence
                }

        # Overall best model
        if model_performance:
            overall_best = max(model_performance.values(), key=lambda x: x.accuracy)
            recommendations['best_overall_model'] = {
                'model': overall_best.model,
                'accuracy': overall_best.accuracy,
                'source': overall_best.source
            }

        # Performance rankings
        rankings = []
        for i, (key, perf) in enumerate(sorted(model_performance.items(),
                                             key=lambda x: x[1].accuracy, reverse=True), 1):
            rankings.append({
                'rank': i,
                'model': perf.model,
                'source': perf.source,
                'score': perf.accuracy
            })
        recommendations['performance_rankings'] = rankings

        # Identify calibration issues (models with high confidence but low accuracy)
        for key, perf in model_performance.items():
            if perf.avg_confidence > 0.7 and perf.accuracy < 0.6:
                recommendations['calibration_issues'].append({
                    'model': perf.model,
                    'source': perf.source,
                    'confidence': perf.avg_confidence,
                    'accuracy': perf.accuracy,
                    'calibration_error': perf.avg_confidence - perf.accuracy
                })

        return recommendations

    def _result_to_dict(self, result: TestResult) -> Dict[str, Any]:
        """Convert TestResult to dictionary for JSON serialization"""
        return {
            'model': result.model,
            'source': result.source,
            'text': result.text,
            'expected_sentiment': result.expected_sentiment,
            'predicted_sentiment': result.predicted_sentiment,
            'confidence': result.confidence,
            'latency': result.latency,
            'accuracy_score': result.accuracy_score
        }

    def generate_optimization_report(self, test_results: Dict[str, Any]) -> str:
        """Generate human-readable optimization report"""

        report = []
        report.append("# Model Accuracy Optimization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive Summary
        recommendations = test_results.get('recommendations', {})
        report.append("## Executive Summary")
        report.append("")

        best_overall = recommendations.get('best_overall_model', {})
        if best_overall:
            report.append(f"- **Best Overall Model**: {best_overall.get('model', 'N/A')}")
            report.append(".3f")
            report.append(f"- **Total Models Tested**: {test_results.get('models_tested', 0)}")
            report.append(f"- **Sources Tested**: {test_results.get('sources_tested', 0)}")
            report.append(f"- **Total Test Cases**: {test_results.get('total_tests', 0)}")

        # Best Models by Source
        best_by_source = recommendations.get('best_by_source', {})
        if best_by_source:
            report.append("")
            report.append("### Recommended Models by Source")
            for source, info in best_by_source.items():
                report.append(f"- **{source.title()}**: `{info.get('model', 'N/A')}` "
                            ".3f")

        # Performance Rankings
        rankings = recommendations.get('performance_rankings', [])
        if rankings:
            report.append("")
            report.append("## Performance Rankings")
            report.append("")
            report.append("| Rank | Model | Source | Accuracy |")
            report.append("|------|-------|--------|----------|")
            for rank_info in rankings[:10]:  # Top 10
                report.append(f"| {rank_info['rank']} | {rank_info['model']} | {rank_info['source']} | {rank_info['score']:.3f} |")

        # Calibration Issues
        calibration_issues = recommendations.get('calibration_issues', [])
        if calibration_issues:
            report.append("")
            report.append("## Calibration Issues")
            report.append("")
            report.append(f"Found {len(calibration_issues)} models with confidence-accuracy mismatch:")
            report.append("")
            for issue in sorted(calibration_issues, key=lambda x: x.get('calibration_error', 0), reverse=True):
                report.append(f"- **{issue.get('model', 'N/A')}** on {issue.get('source', 'N/A')}:")
                report.append(".3f")
                report.append(".3f")

        return "\n".join(report)

def main():
    """Main function to run comprehensive model testing"""

    print("🚀 Starting Comprehensive Model Accuracy Testing Framework...")
    print("=" * 70)

    tester = ComprehensiveModelTester()

    # Run comprehensive tests
    print("📊 Running comprehensive accuracy tests...")
    test_results = tester.run_comprehensive_tests()

    # Generate optimization report
    print("\n📋 Generating optimization report...")
    report = tester.generate_optimization_report(test_results)

    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE MODEL ACCURACY TEST RESULTS")
    print("=" * 70)
    print(report)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{tester.results_dir}/optimization_report_{timestamp}.md"

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n💾 Optimization report saved to: {report_file}")
    print("\n✅ Comprehensive model testing completed!")
    print("🎯 Ready to update sentiment analysis pipeline with optimal models")
    
    # Clean up models
    tester.cleanup_models()

if __name__ == "__main__":
    main()
