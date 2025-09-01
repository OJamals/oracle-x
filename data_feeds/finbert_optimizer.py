"""
FinBERT Model Optimization and Calibration Framework
Optimizes model selection, calibration, and ensemble strategies based on accuracy testing results
"""

import json
import logging
import os
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statistics

logger = logging.getLogger(__name__)

class FinBERTOptimizer:
    """
    Optimizes FinBERT model selection and calibration based on performance data
    """

    def __init__(self, results_dir: str = 'data/finbert_accuracy_results'):
        self.results_dir = results_dir
        self.latest_results = self._load_latest_results()
        self.optimization_results = {}

    def _load_latest_results(self) -> Dict[str, Any]:
        """Load the most recent accuracy test results"""
        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory {self.results_dir} not found")
            return {}

        result_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
        if not result_files:
            logger.warning("No result files found")
            return {}

        # Get the latest file
        latest_file = max(result_files, key=lambda x: os.path.getctime(os.path.join(self.results_dir, x)))
        result_path = os.path.join(self.results_dir, latest_file)

        with open(result_path, 'r') as f:
            results = json.load(f)

        logger.info(f"Loaded results from {latest_file}")
        return results

    def optimize_model_selection(self) -> Dict[str, Any]:
        """Optimize model selection based on performance data"""
        logger.info("Optimizing model selection...")

        if not self.latest_results:
            return {"error": "No results data available"}

        recommendations = self.latest_results.get('recommendations', {})
        model_performance = self.latest_results.get('model_performance', {})

        optimization = {
            'optimized_model_mapping': {},
            'performance_based_weights': {},
            'calibration_corrections': {},
            'source_specific_optimizations': {}
        }

        # Optimize model selection by source
        best_by_source = recommendations.get('best_by_source', {})
        for source, info in best_by_source.items():
            optimization['optimized_model_mapping'][source.lower()] = {
                'recommended_model': info['model'],
                'expected_accuracy': info['accuracy'],
                'confidence_boost': self._calculate_confidence_boost(info['accuracy'])
            }

        # Calculate performance-based ensemble weights
        model_scores = {}
        for model_key, datasets in model_performance.items():
            total_score = 0
            valid_datasets = 0
            for dataset_name, performance in datasets.items():
                if performance and 'accuracy' in performance:
                    total_score += performance['accuracy']
                    valid_datasets += 1
            if valid_datasets > 0:
                model_scores[model_key] = total_score / valid_datasets

        if model_scores:
            # Normalize scores to create weights
            max_score = max(model_scores.values())
            min_score = min(model_scores.values())
            score_range = max_score - min_score

            if score_range > 0:
                for model, score in model_scores.items():
                    # Higher performing models get higher weights
                    normalized_weight = 0.3 + 0.7 * (score - min_score) / score_range
                    optimization['performance_based_weights'][model] = normalized_weight
            else:
                # All models have same performance
                for model in model_scores.keys():
                    optimization['performance_based_weights'][model] = 1.0 / len(model_scores)

        # Optimize calibration
        optimization['calibration_corrections'] = self._optimize_calibration()

        # Source-specific optimizations
        optimization['source_specific_optimizations'] = self._create_source_optimizations()

        self.optimization_results = optimization
        return optimization

    def _calculate_confidence_boost(self, accuracy: float) -> float:
        """Calculate confidence boost factor based on accuracy"""
        # Models with higher accuracy should have their confidence boosted
        if accuracy > 0.7:
            return 1.2  # 20% boost for high accuracy
        elif accuracy > 0.5:
            return 1.1  # 10% boost for medium accuracy
        elif accuracy > 0.3:
            return 1.0  # No boost for low accuracy
        else:
            return 0.9  # Slight reduction for very low accuracy

    def _optimize_calibration(self) -> Dict[str, Any]:
        """Optimize calibration based on test results"""
        calibration_corrections = {}

        model_performance = self.latest_results.get('model_performance', {})

        for model_key, datasets in model_performance.items():
            model_corrections = {}

            for dataset_name, performance in datasets.items():
                if not performance:
                    continue

                avg_confidence = performance.get('avg_confidence', 0)
                accuracy = performance.get('accuracy', 0)

                if avg_confidence > 0:
                    # Calculate calibration factor
                    calibration_factor = accuracy / avg_confidence if avg_confidence > 0 else 1.0

                    # Apply bounds to prevent extreme corrections
                    calibration_factor = max(0.5, min(2.0, calibration_factor))

                    model_corrections[dataset_name] = {
                        'calibration_factor': calibration_factor,
                        'original_confidence': avg_confidence,
                        'actual_accuracy': accuracy,
                        'calibration_error': abs(avg_confidence - accuracy)
                    }

            if model_corrections:
                calibration_corrections[model_key] = model_corrections

        return calibration_corrections

    def _create_source_optimizations(self) -> Dict[str, Any]:
        """Create source-specific optimizations"""
        source_optimizations = {}

        model_performance = self.latest_results.get('model_performance', {})

        # Analyze performance patterns by source
        for dataset_name in ['twitter', 'news', 'reddit', 'mixed']:
            source_performance = {}

            for model_key, datasets in model_performance.items():
                if dataset_name in datasets and datasets[dataset_name]:
                    performance = datasets[dataset_name]
                    if 'accuracy' in performance:
                        source_performance[model_key] = performance['accuracy']

            if source_performance:
                # Find best and worst performing models for this source
                best_model = max(source_performance.items(), key=lambda x: x[1])
                worst_model = min(source_performance.items(), key=lambda x: x[1])

                source_optimizations[dataset_name] = {
                    'best_model': best_model[0],
                    'best_accuracy': best_model[1],
                    'worst_model': worst_model[0],
                    'worst_accuracy': worst_model[1],
                    'performance_spread': best_model[1] - worst_model[1],
                    'recommendation': self._get_source_recommendation(dataset_name, best_model[1])
                }

        return source_optimizations

    def _get_source_recommendation(self, source: str, best_accuracy: float) -> str:
        """Get recommendation based on source performance"""
        if best_accuracy > 0.7:
            return f"Excellent performance on {source} - use specialized model"
        elif best_accuracy > 0.5:
            return f"Good performance on {source} - suitable for production"
        elif best_accuracy > 0.3:
            return f"Moderate performance on {source} - consider ensemble approach"
        else:
            return f"Poor performance on {source} - needs significant improvement"

    def create_optimized_config(self) -> Dict[str, Any]:
        """Create optimized configuration for production use"""
        logger.info("Creating optimized configuration...")

        if not self.optimization_results:
            self.optimize_model_selection()

        config = {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'model_selection': {
                'primary_model': 'finbert_tone',  # Default fallback
                'source_specific_models': self.optimization_results.get('optimized_model_mapping', {}),
                'ensemble_weights': self.optimization_results.get('performance_based_weights', {})
            },
            'calibration': {
                'enabled': True,
                'corrections': self.optimization_results.get('calibration_corrections', {}),
                'confidence_boosting': True
            },
            'performance_thresholds': {
                'min_accuracy': 0.3,
                'min_confidence': 0.5,
                'max_calibration_error': 0.2
            },
            'fallback_strategy': {
                'enable_model_switching': True,
                'fallback_models': ['finbert_tone', 'fintwitbert_sentiment'],
                'performance_monitoring': True
            }
        }

        return config

    def save_optimization_results(self, output_dir: str = 'data/finbert_optimizations'):
        """Save optimization results to file"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save optimization results
        opt_file = f"{output_dir}/finbert_optimization_{timestamp}.json"
        with open(opt_file, 'w') as f:
            json.dump(self.optimization_results, f, indent=2)

        # Save optimized config
        config = self.create_optimized_config()
        config_file = f"{output_dir}/finbert_optimized_config_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Optimization results saved to {opt_file}")
        logger.info(f"Optimized config saved to {config_file}")

        return opt_file, config_file

    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        if not self.latest_results:
            return "No performance data available"

        report = []
        report.append("# FinBERT Model Performance and Optimization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall performance summary
        recommendations = self.latest_results.get('recommendations', {})
        if 'best_overall_model' in recommendations:
            best = recommendations['best_overall_model']
            report.append("## Overall Performance Summary")
            report.append(f"- **Best Model**: {best['model']}")
            report.append(f"- **Average Accuracy**: {best['average_accuracy']:.3f}")
            report.append("")

        # Model rankings
        rankings = recommendations.get('performance_rankings', [])
        if rankings:
            report.append("## Model Performance Rankings")
            for rank_info in rankings:
                report.append(f"{rank_info['rank']}. {rank_info['model']}: {rank_info['score']:.3f}")
            report.append("")

        # Source-specific performance
        best_by_source = recommendations.get('best_by_source', {})
        if best_by_source:
            report.append("## Best Model by Source")
            for source, info in best_by_source.items():
                report.append(f"- **{source.title()}**: {info['model']} ({info['accuracy']:.3f})")
            report.append("")

        # Calibration issues
        calibration_issues = recommendations.get('calibration_issues', [])
        if calibration_issues:
            report.append("## Calibration Issues")
            report.append(f"Found {len(calibration_issues)} calibration issues:")
            for issue in calibration_issues[:10]:  # Show top 10
                report.append(f"- {issue['model']} on {issue['dataset']}: {issue['calibration_error']:.3f}")
            report.append("")

        # Optimization recommendations
        if self.optimization_results:
            report.append("## Optimization Recommendations")
            model_mapping = self.optimization_results.get('optimized_model_mapping', {})
            for source, config in model_mapping.items():
                report.append(f"- **{source.title()}**: Use {config['recommended_model']} (expected accuracy: {config['expected_accuracy']:.3f})")

            weights = self.optimization_results.get('performance_based_weights', {})
            if weights:
                report.append("")
                report.append("### Ensemble Weights")
                for model, weight in weights.items():
                    report.append(f"- {model}: {weight:.3f}")
            report.append("")

        return "\n".join(report)

def main():
    """Main function to run FinBERT optimization"""
    logging.basicConfig(level=logging.INFO)

    optimizer = FinBERTOptimizer()

    print("🔧 Running FinBERT Optimization...")
    print("=" * 50)

    # Run optimization
    optimization_results = optimizer.optimize_model_selection()

    # Generate and display report
    report = optimizer.generate_performance_report()
    print(report)

    # Save results
    opt_file, config_file = optimizer.save_optimization_results()

    print("\n📁 Files saved:")
    print(f"  - Optimization results: {opt_file}")
    print(f"  - Optimized config: {config_file}")

    print("\n✅ FinBERT optimization completed!")

if __name__ == "__main__":
    main()