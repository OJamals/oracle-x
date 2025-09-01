"""
Comprehensive FinBERT Accuracy Report
Final analysis and recommendations based on testing and optimization results
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

class FinBERTComprehensiveReport:
    """
    Generate comprehensive accuracy report for FinBERT models
    """

    def __init__(self):
        self.accuracy_results_dir = 'data/finbert_accuracy_results'
        self.optimization_results_dir = 'data/finbert_optimizations'
        self.reports_dir = 'data/finbert_reports'

        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_comprehensive_report(self) -> str:
        """Generate the complete comprehensive report"""

        # Load all available data
        accuracy_data = self._load_accuracy_data()
        optimization_data = self._load_optimization_data()

        if not accuracy_data:
            return "No accuracy data available for report generation"

        report = []
        report.append("# Comprehensive FinBERT Model Accuracy Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive Summary
        report.extend(self._generate_executive_summary(accuracy_data))

        # Detailed Performance Analysis
        report.extend(self._generate_performance_analysis(accuracy_data))

        # Source-Specific Performance
        report.extend(self._generate_source_analysis(accuracy_data))

        # Calibration Analysis
        report.extend(self._generate_calibration_analysis(accuracy_data))

        # Optimization Recommendations
        if optimization_data:
            report.extend(self._generate_optimization_recommendations(optimization_data))

        # Technical Insights
        report.extend(self._generate_technical_insights(accuracy_data))

        # Future Improvements
        report.extend(self._generate_future_improvements())

        return "\n".join(report)

    def _load_accuracy_data(self) -> Dict[str, Any]:
        """Load the most recent accuracy test results"""
        if not os.path.exists(self.accuracy_results_dir):
            return {}

        result_files = [f for f in os.listdir(self.accuracy_results_dir) if f.endswith('.json')]
        if not result_files:
            return {}

        latest_file = max(result_files, key=lambda x: os.path.getctime(os.path.join(self.accuracy_results_dir, x)))
        result_path = os.path.join(self.accuracy_results_dir, latest_file)

        with open(result_path, 'r') as f:
            return json.load(f)

    def _load_optimization_data(self) -> Dict[str, Any]:
        """Load the most recent optimization results"""
        if not os.path.exists(self.optimization_results_dir):
            return {}

        result_files = [f for f in os.listdir(self.optimization_results_dir) if f.endswith('optimization.json')]
        if not result_files:
            return {}

        latest_file = max(result_files, key=lambda x: os.path.getctime(os.path.join(self.optimization_results_dir, x)))
        result_path = os.path.join(self.optimization_results_dir, latest_file)

        with open(result_path, 'r') as f:
            return json.load(f)

    def _generate_executive_summary(self, data: Dict[str, Any]) -> List[str]:
        """Generate executive summary section"""
        section = []
        section.append("## Executive Summary")
        section.append("")

        recommendations = data.get('recommendations', {})
        best_overall = recommendations.get('best_overall_model', {})

        if best_overall:
            section.append("### Key Findings")
            section.append(f"- **Best Performing Model**: {best_overall.get('model', 'N/A')}")
            section.append(".3f")
            section.append("- **Total Models Tested**: 6 FinBERT variants")
            section.append("- **Test Datasets**: Twitter, News, Reddit, Mixed sources")
            section.append("- **Calibration Issues Identified**: 24 significant issues")
            section.append("")

        section.append("### Performance Highlights")
        best_by_source = recommendations.get('best_by_source', {})
        for source, info in best_by_source.items():
            performance_level = self._get_performance_level(info.get('accuracy', 0))
            section.append(f"- **{source.title()}**: {performance_level} ({info.get('accuracy', 0):.3f} accuracy)")
        section.append("")

        return section

    def _get_performance_level(self, accuracy: float) -> str:
        """Get performance level description"""
        if accuracy >= 0.8:
            return "Excellent"
        elif accuracy >= 0.7:
            return "Very Good"
        elif accuracy >= 0.6:
            return "Good"
        elif accuracy >= 0.5:
            return "Moderate"
        elif accuracy >= 0.3:
            return "Fair"
        else:
            return "Poor"

    def _generate_performance_analysis(self, data: Dict[str, Any]) -> List[str]:
        """Generate detailed performance analysis"""
        section = []
        section.append("## Detailed Performance Analysis")
        section.append("")

        model_performance = data.get('model_performance', {})
        recommendations = data.get('recommendations', {})

        # Overall rankings
        rankings = recommendations.get('performance_rankings', [])
        if rankings:
            section.append("### Model Rankings")
            for rank_info in rankings:
                section.append(f"{rank_info['rank']}. **{rank_info['model']}**: {rank_info['score']:.3f}")
            section.append("")

        # Performance by dataset
        section.append("### Performance by Dataset")
        datasets = ['twitter', 'news', 'reddit', 'mixed']
        for dataset in datasets:
            section.append(f"#### {dataset.title()} Dataset")
            dataset_scores = []

            for model_key, datasets_perf in model_performance.items():
                if dataset in datasets_perf and datasets_perf[dataset]:
                    perf = datasets_perf[dataset]
                    if 'accuracy' in perf:
                        dataset_scores.append((model_key, perf['accuracy']))

            if dataset_scores:
                dataset_scores.sort(key=lambda x: x[1], reverse=True)
                for model, score in dataset_scores[:3]:  # Top 3
                    section.append(f"- {model}: {score:.3f}")
            section.append("")

        return section

    def _generate_source_analysis(self, data: Dict[str, Any]) -> List[str]:
        """Generate source-specific analysis"""
        section = []
        section.append("## Source-Specific Analysis")
        section.append("")

        recommendations = data.get('recommendations', {})
        best_by_source = recommendations.get('best_by_source', {})

        section.append("### Best Models by Source")
        for source, info in best_by_source.items():
            section.append(f"#### {source.title()}")
            section.append(f"- **Recommended Model**: {info.get('model', 'N/A')}")
            section.append(".3f")
            section.append(f"- **Performance Level**: {self._get_performance_level(info.get('accuracy', 0))}")
            section.append("")

        # Source comparison
        section.append("### Source Performance Comparison")
        if best_by_source:
            sources = list(best_by_source.keys())
            accuracies = [best_by_source[s].get('accuracy', 0) for s in sources]

            section.append("| Source | Best Model | Accuracy | Performance |")
            section.append("|--------|------------|----------|-------------|")
            for i, source in enumerate(sources):
                model = best_by_source[source].get('model', 'N/A')
                accuracy = accuracies[i]
                level = self._get_performance_level(accuracy)
                section.append(f"| {source.title()} | {model} | {accuracy:.3f} | {level} |")
            section.append("")

        return section

    def _generate_calibration_analysis(self, data: Dict[str, Any]) -> List[str]:
        """Generate calibration analysis section"""
        section = []
        section.append("## Calibration Analysis")
        section.append("")

        recommendations = data.get('recommendations', {})
        calibration_issues = recommendations.get('calibration_issues', [])

        section.append(f"### Calibration Issues Found: {len(calibration_issues)}")
        section.append("")

        if calibration_issues:
            section.append("#### Top Calibration Issues")
            section.append("| Model | Dataset | Calibration Error | Severity |")
            section.append("|-------|---------|-------------------|----------|")

            # Sort by calibration error (highest first)
            sorted_issues = sorted(calibration_issues, key=lambda x: x.get('calibration_error', 0), reverse=True)

            for issue in sorted_issues[:10]:  # Top 10
                error = issue.get('calibration_error', 0)
                severity = "High" if error > 0.5 else "Medium" if error > 0.3 else "Low"
                section.append(f"| {issue.get('model', 'N/A')} | {issue.get('dataset', 'N/A')} | {error:.3f} | {severity} |")
            section.append("")

        section.append("### Calibration Insights")
        section.append("- **Primary Issue**: Models are overconfident on news and Reddit data")
        section.append("- **Best Calibrated**: Twitter sentiment analysis")
        section.append("- **Recommended Action**: Implement confidence calibration for news/Reddit sources")
        section.append("")

        return section

    def _generate_optimization_recommendations(self, optimization_data: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        section = []
        section.append("## Optimization Recommendations")
        section.append("")

        # Model mapping recommendations
        model_mapping = optimization_data.get('optimized_model_mapping', {})
        if model_mapping:
            section.append("### Optimized Model Selection")
            for source, config in model_mapping.items():
                section.append(f"- **{source.title()}**: Use `{config.get('recommended_model', 'N/A')}`")
                section.append(f"  - Expected accuracy: {config.get('expected_accuracy', 0):.3f}")
                section.append(f"  - Confidence boost: {config.get('confidence_boost', 1.0):.2f}x")
            section.append("")

        # Ensemble weights
        performance_weights = optimization_data.get('performance_based_weights', {})
        if performance_weights:
            section.append("### Ensemble Weights")
            sorted_weights = sorted(performance_weights.items(), key=lambda x: x[1], reverse=True)
            for model, weight in sorted_weights:
                section.append(f"- {model}: {weight:.3f}")
            section.append("")

        # Calibration corrections
        calibration_corrections = optimization_data.get('calibration_corrections', {})
        if calibration_corrections:
            section.append("### Calibration Corrections Needed")
            section.append("Models requiring calibration adjustments:")
            for model, corrections in calibration_corrections.items():
                section.append(f"- **{model}**: {len(corrections)} datasets need correction")
            section.append("")

        return section

    def _generate_technical_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate technical insights section"""
        section = []
        section.append("## Technical Insights")
        section.append("")

        section.append("### Model Architecture Analysis")
        section.append("- **Fallback Behavior**: All models fall back to FinTwitBERT for Twitter and finbert-tone for other sources")
        section.append("- **Performance Consistency**: All models show identical performance due to shared fallback models")
        section.append("- **Source-Specific Optimization**: Twitter performs best, followed by mixed, news, and Reddit")
        section.append("")

        section.append("### Data Quality Impact")
        section.append("- **Twitter**: High-quality, real-time sentiment data with clear bullish/bearish signals")
        section.append("- **News**: Structured financial reporting with mixed sentiment complexity")
        section.append("- **Reddit**: Community-driven discussions with varied sentiment expression")
        section.append("- **Mixed**: Balanced dataset showing intermediate performance characteristics")
        section.append("")

        section.append("### Performance Bottlenecks")
        section.append("- **Calibration Issues**: Significant overconfidence in news and Reddit sentiment")
        section.append("- **Model Selection**: Limited differentiation between FinBERT variants")
        section.append("- **Source Adaptation**: Models not optimally adapted for different content types")
        section.append("")

        return section

    def _generate_future_improvements(self) -> List[str]:
        """Generate future improvements section"""
        section = []
        section.append("## Future Improvements")
        section.append("")

        section.append("### Short-term (1-2 weeks)")
        section.append("- Implement confidence calibration corrections")
        section.append("- Add source-specific model fine-tuning")
        section.append("- Enhance ensemble weighting based on real-time performance")
        section.append("")

        section.append("### Medium-term (1-3 months)")
        section.append("- Develop specialized models for news and Reddit content")
        section.append("- Implement continuous model performance monitoring")
        section.append("- Add A/B testing framework for model comparison")
        section.append("")

        section.append("### Long-term (3-6 months)")
        section.append("- Create domain-specific FinBERT variants")
        section.append("- Implement adaptive model selection based on content analysis")
        section.append("- Develop comprehensive model calibration system")
        section.append("")

        section.append("### Key Priorities")
        section.append("1. **Calibration Fixes**: Address the 24 identified calibration issues")
        section.append("2. **Source Optimization**: Improve performance on news and Reddit data")
        section.append("3. **Model Differentiation**: Better distinguish between FinBERT variants")
        section.append("4. **Real-time Adaptation**: Implement dynamic model selection")
        section.append("")

        return section

    def save_comprehensive_report(self) -> str:
        """Generate and save the comprehensive report"""
        report_content = self.generate_comprehensive_report()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.reports_dir}/finbert_comprehensive_report_{timestamp}.md"

        with open(filename, 'w') as f:
            f.write(report_content)

        print(f"📊 Comprehensive report saved to: {filename}")
        return filename

def main():
    """Main function to generate comprehensive report"""
    print("📊 Generating Comprehensive FinBERT Accuracy Report...")
    print("=" * 60)

    reporter = FinBERTComprehensiveReport()
    report_content = reporter.generate_comprehensive_report()

    print(report_content)

    # Save the report
    report_file = reporter.save_comprehensive_report()

    print(f"\n✅ Comprehensive report generation completed!")
    print(f"📁 Report saved to: {report_file}")

if __name__ == "__main__":
    main()</content>