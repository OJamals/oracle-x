"""
Financial Sentiment Accuracy Optimizer
Specialized improvements for financial text sentiment analysis
"""

import json
import os
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import statistics
from datetime import datetime

# Fix import path
import sys
sys.path.append(os.path.dirname(__file__))
from advanced_sentiment import FinBERTAnalyzer

@dataclass
class FinancialTextAnalysis:
    """Analysis of financial text characteristics"""
    text: str
    technical_terms: List[str]
    sentiment_indicators: List[str]
    neutral_indicators: List[str]
    complexity_score: float
    domain_confidence: float

@dataclass
class FinancialAccuracyMetrics:
    """Metrics specific to financial sentiment accuracy"""
    neutral_detection_accuracy: float
    technical_analysis_accuracy: float
    fundamental_analysis_accuracy: float
    overall_financial_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    neutral_misclassification_rate: float

class FinancialAccuracyOptimizer:
    """
    Specialized optimizer for improving financial sentiment analysis accuracy
    Focuses on technical analysis, fundamental analysis, and neutral sentiment detection
    """

    def __init__(self):
        self.analyzer = FinBERTAnalyzer()
        self.financial_lexicon = self._load_financial_lexicon()
        self.technical_indicators = self._load_technical_indicators()
        self.neutral_patterns = self._load_neutral_patterns()

    def _load_financial_lexicon(self) -> Dict[str, float]:
        """Load specialized financial sentiment lexicon"""
        return {
            # Technical Analysis Terms
            'breakout': 0.3, 'breakdown': -0.3, 'support': 0.1, 'resistance': -0.1,
            'accumulation': 0.2, 'distribution': -0.2, 'momentum': 0.2, 'reversal': -0.2,
            'bullish': 0.4, 'bearish': -0.4, 'overbought': -0.2, 'oversold': 0.2,
            'divergence': -0.1, 'convergence': 0.1, 'consolidation': 0.0, 'volatility': -0.1,

            # Fundamental Analysis Terms
            'earnings': 0.0, 'revenue': 0.0, 'profit': 0.0, 'margin': 0.0,
            'valuation': 0.0, 'growth': 0.2, 'decline': -0.2, 'expansion': 0.3,
            'contraction': -0.3, 'acquisition': 0.2, 'divestiture': -0.1,

            # Market Structure Terms
            'institutional': 0.1, 'retail': 0.0, 'flow': 0.0, 'volume': 0.0,
            'liquidity': 0.1, 'illiquid': -0.1, 'bid': 0.0, 'ask': 0.0,

            # Risk Terms
            'risk': -0.1, 'exposure': -0.1, 'hedge': 0.0, 'leverage': -0.1,
            'volatility': -0.1, 'uncertainty': -0.2, 'stability': 0.1,

            # Neutral/Analytical Terms
            'analysis': 0.0, 'assessment': 0.0, 'evaluation': 0.0, 'review': 0.0,
            'positioning': 0.0, 'strategy': 0.0, 'approach': 0.0, 'perspective': 0.0,
            'consideration': 0.0, 'assessment': 0.0, 'outlook': 0.0
        }

    def _load_technical_indicators(self) -> List[str]:
        """Load technical analysis indicator terms"""
        return [
            'moving average', 'rsi', 'macd', 'bollinger', 'fibonacci', 'pivot',
            'trendline', 'channel', 'wedge', 'triangle', 'flag', 'pennant',
            'head and shoulders', 'double top', 'double bottom', 'cup and handle',
            'volume profile', 'order flow', 'level 2', 'time and sales',
            'vwap', 'support level', 'resistance level', 'fibonacci retracement',
            'stochastic', 'williams %r', 'cci', 'adx', 'parabolic sar'
        ]

    def _load_neutral_patterns(self) -> List[str]:
        """Load patterns that typically indicate neutral sentiment"""
        return [
            'trading at', 'valued at', 'priced at', 'trading around',
            'currently at', 'sitting at', 'hovering around', 'ranging between',
            'consolidating around', 'stabilizing at', 'holding steady',
            'maintaining position', 'unchanged', 'stable', 'steady',
            'neutral', 'balanced', 'mixed', 'moderate', 'reasonable',
            'fair value', 'appropriate valuation', 'proper pricing',
            'industry average', 'sector median', 'market standard'
        ]

    def analyze_financial_text(self, text: str) -> FinancialTextAnalysis:
        """Analyze financial text for specialized characteristics"""
        text_lower = text.lower()

        # Extract technical terms
        technical_terms = []
        for indicator in self.technical_indicators:
            if indicator in text_lower:
                technical_terms.append(indicator)

        # Extract sentiment indicators
        sentiment_indicators = []
        for term, score in self.financial_lexicon.items():
            if abs(score) > 0.1 and term in text_lower:
                sentiment_indicators.append(term)

        # Extract neutral indicators
        neutral_indicators = []
        for pattern in self.neutral_patterns:
            if pattern in text_lower:
                neutral_indicators.append(pattern)

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(text, technical_terms, sentiment_indicators)

        # Calculate domain confidence
        domain_confidence = self._calculate_domain_confidence(text, technical_terms, sentiment_indicators, neutral_indicators)

        return FinancialTextAnalysis(
            text=text,
            technical_terms=technical_terms,
            sentiment_indicators=sentiment_indicators,
            neutral_indicators=neutral_indicators,
            complexity_score=complexity_score,
            domain_confidence=domain_confidence
        )

    def _calculate_complexity_score(self, text: str, technical_terms: List[str], sentiment_indicators: List[str]) -> float:
        """Calculate complexity score based on financial terminology usage"""
        text_length = len(text.split())
        technical_density = len(technical_terms) / max(text_length, 1)
        sentiment_density = len(sentiment_indicators) / max(text_length, 1)

        # Complexity increases with technical and sentiment term density
        complexity = (technical_density * 0.6) + (sentiment_density * 0.4)

        # Cap at 1.0
        return min(1.0, complexity * 2.0)

    def _calculate_domain_confidence(self, text: str, technical_terms: List[str],
                                   sentiment_indicators: List[str], neutral_indicators: List[str]) -> float:
        """Calculate confidence that text is financial domain"""
        text_lower = text.lower()
        financial_keywords = [
            'stock', 'shares', 'market', 'trading', 'investment', 'portfolio',
            'earnings', 'revenue', 'profit', 'valuation', 'analysis', 'technical',
            'fundamental', 'bullish', 'bearish', 'momentum', 'volatility'
        ]

        keyword_count = sum(1 for keyword in financial_keywords if keyword in text_lower)
        total_indicators = len(technical_terms) + len(sentiment_indicators) + len(neutral_indicators)

        # Domain confidence based on financial keywords and indicators
        confidence = (keyword_count * 0.4) + (total_indicators * 0.6)

        return min(1.0, confidence / 5.0)  # Normalize

    def predict_financial_sentiment(self, text: str) -> Tuple[float, float, Dict[str, Any]]:
        """
        Predict sentiment for financial text with specialized processing
        Returns: (sentiment, confidence, metadata)
        """
        # Analyze text characteristics
        analysis = self.analyze_financial_text(text)

        # Get base FinBERT prediction
        base_sentiment, base_confidence = self.analyzer.analyze(text, source="financial")

        # Apply financial-specific adjustments
        adjusted_sentiment = self._apply_financial_adjustments(base_sentiment, analysis)

        # Calculate adjusted confidence
        adjusted_confidence = self._calculate_adjusted_confidence(base_confidence, analysis)

        # Create metadata
        metadata = {
            'analysis': analysis,
            'base_sentiment': base_sentiment,
            'base_confidence': base_confidence,
            'adjustments_applied': self._get_adjustments_summary(analysis),
            'domain_confidence': analysis.domain_confidence,
            'complexity_score': analysis.complexity_score
        }

        return adjusted_sentiment, adjusted_confidence, metadata

    def _apply_financial_adjustments(self, base_sentiment: float, analysis: FinancialTextAnalysis) -> float:
        """Apply financial-specific sentiment adjustments"""
        adjusted_sentiment = base_sentiment

        # Neutral indicator adjustment
        if analysis.neutral_indicators:
            # Strong neutral indicators should push sentiment toward neutral
            neutral_strength = len(analysis.neutral_indicators) / 3.0  # Max 3 indicators
            neutral_adjustment = (0.0 - base_sentiment) * neutral_strength * 0.5
            adjusted_sentiment += neutral_adjustment

        # Technical analysis adjustment
        if analysis.technical_terms:
            # Technical analysis often has more neutral/analytical tone
            technical_adjustment = (0.0 - base_sentiment) * 0.2
            adjusted_sentiment += technical_adjustment

        # Complexity adjustment
        if analysis.complexity_score > 0.7:
            # High complexity often indicates analytical rather than emotional content
            complexity_adjustment = (0.0 - base_sentiment) * 0.3
            adjusted_sentiment += complexity_adjustment

        # Financial lexicon adjustment
        lexicon_sentiment = 0.0
        lexicon_count = 0
        for term in analysis.sentiment_indicators:
            if term in self.financial_lexicon:
                lexicon_sentiment += self.financial_lexicon[term]
                lexicon_count += 1

        if lexicon_count > 0:
            avg_lexicon_sentiment = lexicon_sentiment / lexicon_count
            lexicon_adjustment = (avg_lexicon_sentiment - base_sentiment) * 0.4
            adjusted_sentiment += lexicon_adjustment

        # Clamp to valid range
        return max(-1.0, min(1.0, adjusted_sentiment))

    def _calculate_adjusted_confidence(self, base_confidence: float, analysis: FinancialTextAnalysis) -> float:
        """Calculate adjusted confidence based on financial analysis"""
        adjusted_confidence = base_confidence

        # Domain confidence boost
        if analysis.domain_confidence > 0.7:
            adjusted_confidence = min(1.0, adjusted_confidence * 1.2)

        # Neutral indicator confidence adjustment
        if analysis.neutral_indicators:
            # Neutral indicators can increase confidence in neutral predictions
            neutral_boost = len(analysis.neutral_indicators) * 0.1
            adjusted_confidence = min(1.0, adjusted_confidence + neutral_boost)

        # Complexity penalty
        if analysis.complexity_score > 0.8:
            # Very complex financial text may be harder to analyze accurately
            complexity_penalty = (analysis.complexity_score - 0.8) * 0.2
            adjusted_confidence = max(0.1, adjusted_confidence - complexity_penalty)

        return adjusted_confidence

    def _get_adjustments_summary(self, analysis: FinancialTextAnalysis) -> List[str]:
        """Get summary of adjustments applied"""
        adjustments = []

        if analysis.neutral_indicators:
            adjustments.append(f"Neutral indicators: {len(analysis.neutral_indicators)}")
        if analysis.technical_terms:
            adjustments.append(f"Technical terms: {len(analysis.technical_terms)}")
        if analysis.sentiment_indicators:
            adjustments.append(f"Sentiment indicators: {len(analysis.sentiment_indicators)}")
        if analysis.complexity_score > 0.5:
            adjustments.append(f"Complexity: .2f")
        if analysis.domain_confidence > 0.7:
            adjustments.append(f"Domain confidence: .2f")

        return adjustments

    def evaluate_financial_accuracy(self, test_cases: List[Dict[str, Any]]) -> FinancialAccuracyMetrics:
        """Evaluate accuracy on financial test cases"""
        results = []

        for test_case in test_cases:
            text = test_case['text']
            expected_sentiment = test_case['expected_sentiment']

            predicted_sentiment, confidence, metadata = self.predict_financial_sentiment(text)

            # Calculate accuracy (simplified binary classification for now)
            if abs(expected_sentiment) < 0.1:  # Neutral
                predicted_neutral = abs(predicted_sentiment) < 0.2
                accuracy = 1.0 if predicted_neutral else 0.0
            else:  # Positive or negative
                predicted_direction = 1 if predicted_sentiment > 0 else -1
                expected_direction = 1 if expected_sentiment > 0 else -1
                accuracy = 1.0 if predicted_direction == expected_direction else 0.0

            results.append({
                'text': text,
                'expected': expected_sentiment,
                'predicted': predicted_sentiment,
                'accuracy': accuracy,
                'confidence': confidence,
                'metadata': metadata
            })

        # Calculate metrics
        accuracies = [r['accuracy'] for r in results]
        overall_accuracy = statistics.mean(accuracies)

        # Neutral detection accuracy
        neutral_cases = [r for r in results if abs(r['expected']) < 0.1]
        neutral_accuracies = [r['accuracy'] for r in neutral_cases]
        neutral_detection_accuracy = statistics.mean(neutral_accuracies) if neutral_accuracies else 0.0

        # Technical analysis accuracy
        technical_cases = [r for r in results if any(term in r['text'].lower() for term in self.technical_indicators)]
        technical_accuracies = [r['accuracy'] for r in technical_cases]
        technical_analysis_accuracy = statistics.mean(technical_accuracies) if technical_accuracies else 0.0

        # Fundamental analysis accuracy
        fundamental_cases = [r for r in results if any(term in r['text'].lower() for term in ['earnings', 'revenue', 'profit', 'valuation', 'growth'])]
        fundamental_accuracies = [r['accuracy'] for r in fundamental_cases]
        fundamental_analysis_accuracy = statistics.mean(fundamental_accuracies) if fundamental_accuracies else 0.0

        # Error rates
        false_positives = sum(1 for r in results if r['expected'] < -0.1 and r['predicted'] > 0.1)
        false_negatives = sum(1 for r in results if r['expected'] > 0.1 and r['predicted'] < -0.1)
        neutral_misclassifications = sum(1 for r in results if abs(r['expected']) < 0.1 and abs(r['predicted']) > 0.2)

        total_cases = len(results)
        false_positive_rate = false_positives / total_cases if total_cases > 0 else 0.0
        false_negative_rate = false_negatives / total_cases if total_cases > 0 else 0.0
        neutral_misclassification_rate = neutral_misclassifications / total_cases if total_cases > 0 else 0.0

        return FinancialAccuracyMetrics(
            neutral_detection_accuracy=neutral_detection_accuracy,
            technical_analysis_accuracy=technical_analysis_accuracy,
            fundamental_analysis_accuracy=fundamental_analysis_accuracy,
            overall_financial_accuracy=overall_accuracy,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            neutral_misclassification_rate=neutral_misclassification_rate
        )

    def generate_improvement_report(self, metrics: FinancialAccuracyMetrics) -> str:
        """Generate detailed improvement report"""
        report = []
        report.append("# Financial Sentiment Accuracy Improvement Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("## Current Performance Metrics")
        report.append("")
        report.append(".3f")
        report.append(".3f")
        report.append(".3f")
        report.append(".3f")
        report.append(".3f")
        report.append(".3f")
        report.append(".3f")
        report.append("")

        report.append("## Key Issues Identified")
        report.append("")

        if metrics.neutral_detection_accuracy < 0.7:
            report.append("### ⚠️ Neutral Detection Issues")
            report.append(".3f")
            report.append("- **Impact**: Financial analysis often contains neutral/analytical content")
            report.append("- **Solution**: Enhanced neutral pattern recognition and context understanding")
            report.append("")

        if metrics.technical_analysis_accuracy < 0.8:
            report.append("### ⚠️ Technical Analysis Issues")
            report.append(".3f")
            report.append("- **Impact**: Technical indicators and chart patterns are misclassified")
            report.append("- **Solution**: Specialized technical analysis terminology handling")
            report.append("")

        if metrics.false_positive_rate > 0.2 or metrics.false_negative_rate > 0.2:
            report.append("### ⚠️ Sentiment Polarity Issues")
            report.append(".3f")
            report.append(".3f")
            report.append("- **Impact**: Incorrect bullish/bearish classifications")
            report.append("- **Solution**: Improved context-aware sentiment analysis")
            report.append("")

        report.append("## Recommended Improvements")
        report.append("")
        report.append("### 1. Enhanced Neutral Detection")
        report.append("- Expand neutral pattern recognition")
        report.append("- Add context-aware neutral classification")
        report.append("- Implement confidence thresholds for neutral predictions")
        report.append("")

        report.append("### 2. Technical Analysis Specialization")
        report.append("- Create technical indicator lexicon")
        report.append("- Add chart pattern recognition")
        report.append("- Implement technical analysis context understanding")
        report.append("")

        report.append("### 3. Financial Domain Adaptation")
        report.append("- Fine-tune models on financial analysis text")
        report.append("- Add financial news preprocessing")
        report.append("- Implement domain-specific confidence calibration")
        report.append("")

        report.append("### 4. Error Analysis and Correction")
        report.append("- Implement error pattern recognition")
        report.append("- Add post-processing correction rules")
        report.append("- Create feedback loop for continuous improvement")
        report.append("")

        return "\n".join(report)

def test_financial_accuracy_improvement():
    """Test the financial accuracy improvements"""
    optimizer = FinancialAccuracyOptimizer()

    # Test cases based on the actual failing financial examples
    test_cases = [
        {
            'text': 'SPY showing accumulation patterns suggesting institutional buying',
            'expected_sentiment': 0.65
        },
        {
            'text': 'NVDA competitive positioning stable despite market share pressures',
            'expected_sentiment': -0.1
        },
        {
            'text': 'SPY technical indicators suggest potential reversal at resistance levels',
            'expected_sentiment': -0.65
        },
        {
            'text': 'Technical analysis shows AAPL in strong uptrend with multiple bullish indicators',
            'expected_sentiment': 0.8
        },
        {
            'text': 'TSLA fundamentals improving with positive free cash flow generation',
            'expected_sentiment': 0.75
        },
        {
            'text': 'AAPL trading at fair valuation given current growth trajectory',
            'expected_sentiment': 0.0
        },
        {
            'text': 'NVDA valuation attractive relative to growth prospects and competitive moat',
            'expected_sentiment': 0.7
        },
        {
            'text': 'TSLA balance sheet deterioration with negative operating cash flow',
            'expected_sentiment': -0.8
        }
    ]

    print("🔬 Financial Accuracy Improvement Test")
    print("=" * 50)

    # Test individual predictions
    for i, test_case in enumerate(test_cases, 1):
        text = test_case['text']
        expected = test_case['expected_sentiment']

        sentiment, confidence, metadata = optimizer.predict_financial_sentiment(text)

        print(f"\n{i}. {text[:60]}...")
        print(".3f")
        print(".3f")
        print(f"   Adjustments: {metadata['adjustments_applied']}")

    # Evaluate overall accuracy
    print("\n" + "=" * 50)
    print("📊 ACCURACY EVALUATION")
    print("=" * 50)

    metrics = optimizer.evaluate_financial_accuracy(test_cases)

    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    # Generate improvement report
    print("\n" + "=" * 50)
    print("📋 IMPROVEMENT REPORT")
    print("=" * 50)
    print(optimizer.generate_improvement_report(metrics))

if __name__ == "__main__":
    test_financial_accuracy_improvement()