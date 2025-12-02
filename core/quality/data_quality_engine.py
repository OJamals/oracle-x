"""
Data Quality Assessment Engine for Oracle-X financial trading system.
Provides comprehensive data quality scoring, anomaly detection, and quality metrics.
"""

import re
import statistics
from datetime import datetime

from typing import Dict, List, Optional, Any
from enum import Enum, auto
from dataclasses import dataclass, field


from core.types import MarketData, OptionContract, DataSource
from core.validation.advanced_validators import ValidationResult


class QualityMetric(Enum):
    """Data quality metrics for comprehensive assessment."""

    COMPLETENESS = auto()  # Field completeness
    CONSISTENCY = auto()  # Data consistency
    ACCURACY = auto()  # Data accuracy
    TIMELINESS = auto()  # Data freshness
    VALIDITY = auto()  # Format validity
    UNIQUENESS = auto()  # Data uniqueness
    RELIABILITY = auto()  # Source reliability


class AnomalyType(Enum):
    """Types of data anomalies."""

    PRICE_OUTLIER = auto()  # Price outlier
    VOLUME_SPIKE = auto()  # Volume anomaly
    SPREAD_ANOMALY = auto()  # Bid-ask spread anomaly
    TIMESTAMP_GAP = auto()  # Timestamp gap
    SOURCE_DISCREPANCY = auto()  # Source data discrepancy


@dataclass
class QualityScore:
    """Comprehensive quality score with breakdown by metric."""

    overall: float = 1.0
    completeness: float = 1.0
    consistency: float = 1.0
    accuracy: float = 1.0
    timeliness: float = 1.0
    validity: float = 1.0
    uniqueness: float = 1.0
    reliability: float = 1.0


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""

    anomaly_type: AnomalyType
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0.0 to 1.0
    description: str
    suggested_action: Optional[str] = None


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment result."""

    quality_score: QualityScore
    anomalies: List[AnomalyDetection] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    recommendations: List[str] = field(default_factory=list)


class DataQualityAssessor:
    """
    Comprehensive data quality assessment engine with statistical analysis,
    pattern recognition, and historical comparison capabilities.
    """

    def __init__(self, historical_window: int = 100, confidence_level: float = 0.95):
        self.historical_window = historical_window
        self.confidence_level = confidence_level
        self.historical_data: Dict[str, List[MarketData]] = {}
        self.historical_options: Dict[str, List[OptionContract]] = {}
        self.source_performance: Dict[DataSource, List[bool]] = (
            {}
        )  # Track source success/failure

    def assess_market_data_quality(self, data: MarketData) -> QualityAssessment:
        """Comprehensive quality assessment for MarketData."""
        assessment = QualityAssessment(quality_score=QualityScore())

        # Calculate individual metric scores
        assessment.quality_score.completeness = self._calculate_completeness(data)
        assessment.quality_score.consistency = self._calculate_consistency(data)
        assessment.quality_score.accuracy = self._calculate_accuracy(data)
        assessment.quality_score.timeliness = self._calculate_timeliness(data)
        assessment.quality_score.validity = self._calculate_validity(data)
        assessment.quality_score.reliability = self._calculate_reliability(data)

        # Detect anomalies BEFORE updating historical data
        assessment.anomalies = self._detect_anomalies(data)

        # Calculate overall score (weighted average)
        weights = {
            "completeness": 0.15,
            "consistency": 0.20,
            "accuracy": 0.25,
            "timeliness": 0.15,
            "validity": 0.15,
            "reliability": 0.10,
        }

        overall_score = (
            assessment.quality_score.completeness * weights["completeness"]
            + assessment.quality_score.consistency * weights["consistency"]
            + assessment.quality_score.accuracy * weights["accuracy"]
            + assessment.quality_score.timeliness * weights["timeliness"]
            + assessment.quality_score.validity * weights["validity"]
            + assessment.quality_score.reliability * weights["reliability"]
        )

        assessment.quality_score.overall = overall_score

        # Generate recommendations
        assessment.recommendations = self._generate_recommendations(assessment)

        # Update historical data for future analysis (after anomaly detection)
        self._update_historical_data(data)

        return assessment

    def assess_option_quality(self, data: OptionContract) -> QualityAssessment:
        """Comprehensive quality assessment for OptionContract."""
        assessment = QualityAssessment(quality_score=QualityScore())

        # Calculate individual metric scores
        assessment.quality_score.completeness = self._calculate_option_completeness(
            data
        )
        assessment.quality_score.consistency = self._calculate_option_consistency(data)
        assessment.quality_score.accuracy = self._calculate_option_accuracy(data)
        assessment.quality_score.validity = self._calculate_option_validity(data)

        # Detect anomalies
        assessment.anomalies = self._detect_option_anomalies(data)

        # Calculate overall score
        weights = {
            "completeness": 0.25,
            "consistency": 0.25,
            "accuracy": 0.30,
            "validity": 0.20,
        }

        overall_score = (
            assessment.quality_score.completeness * weights["completeness"]
            + assessment.quality_score.consistency * weights["consistency"]
            + assessment.quality_score.accuracy * weights["accuracy"]
            + assessment.quality_score.validity * weights["validity"]
        )

        assessment.quality_score.overall = overall_score

        # Generate recommendations
        assessment.recommendations = self._generate_option_recommendations(assessment)

        # Update historical data for future analysis
        self._update_historical_options(data)

        return assessment

    def _calculate_completeness(self, data: MarketData) -> float:
        """Calculate completeness score for MarketData."""
        required_fields = [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
        ]
        present_fields = sum(
            1 for field in required_fields if getattr(data, field) is not None
        )
        return present_fields / len(required_fields)

    def _calculate_consistency(self, data: MarketData) -> float:
        """Calculate consistency score for MarketData."""
        score = 1.0

        # Price consistency checks
        if data.low > data.high:
            score *= 0.0  # Critical error
        elif data.open > data.high or data.open < data.low:
            score *= 0.7
        elif data.close > data.high or data.close < data.low:
            score *= 0.7

        # Volume consistency (check if volume is reasonable)
        if data.volume == 0:
            score *= 0.8

        return score

    def _calculate_accuracy(self, data: MarketData) -> float:
        """Calculate accuracy score using statistical methods."""
        score = 1.0

        # Use historical data for statistical accuracy assessment
        if (
            data.symbol in self.historical_data
            and len(self.historical_data[data.symbol]) >= 10
        ):
            historical = self.historical_data[data.symbol][-10:]

            # Price accuracy (z-score based)
            prices = [d.close for d in historical]
            mean_price = statistics.mean(prices)
            std_price = statistics.stdev(prices) if len(prices) > 1 else 0

            if std_price > 0:
                z_score = abs((data.close - mean_price) / std_price)
                if z_score > 3.0:
                    score *= 0.7
                elif z_score > 2.0:
                    score *= 0.9

            # Volume accuracy
            volumes = [d.volume for d in historical]
            mean_volume = statistics.mean(volumes)
            if mean_volume > 0:
                volume_ratio = data.volume / mean_volume
                if volume_ratio > 10.0 or volume_ratio < 0.1:
                    score *= 0.8

        return score

    def _calculate_timeliness(self, data: MarketData) -> float:
        """Calculate timeliness score based on data freshness."""
        now = (
            datetime.now(data.timestamp.tzinfo)
            if data.timestamp.tzinfo
            else datetime.utcnow()
        )
        age_hours = (now - data.timestamp).total_seconds() / 3600

        if age_hours > 24:
            return 0.3
        elif age_hours > 12:
            return 0.6
        elif age_hours > 6:
            return 0.8
        elif age_hours > 1:
            return 0.9
        else:
            return 1.0

    def _calculate_validity(self, data: MarketData) -> float:
        """Calculate validity score based on format validation."""
        score = 1.0

        # Symbol format validation
        if not re.match(r"^[A-Z]+$", data.symbol):
            score *= 0.5

        # Timestamp validation
        if data.timestamp.tzinfo is None:
            score *= 0.7

        return score

    def _calculate_reliability(self, data: MarketData) -> float:
        """Calculate reliability score based on source performance."""
        # Simple reliability scoring based on source type
        reliability_scores = {
            DataSource.YFINANCE: 0.9,
            DataSource.TWELVE_DATA: 0.8,
            DataSource.FMP: 0.7,
            DataSource.FINNHUB: 0.6,
            DataSource.ALPHA_VANTAGE: 0.5,
        }

        return reliability_scores.get(data.source, 0.5)

    def _calculate_option_completeness(self, data: OptionContract) -> float:
        """Calculate completeness score for OptionContract."""
        required_fields = ["symbol", "strike", "expiry", "option_type"]
        optional_fields = [
            "bid",
            "ask",
            "last",
            "volume",
            "open_interest",
            "implied_volatility",
            "underlying_price",
        ]

        present_required = sum(
            1 for field in required_fields if getattr(data, field) is not None
        )
        present_optional = sum(
            1 for field in optional_fields if getattr(data, field) is not None
        )

        required_score = present_required / len(required_fields)
        optional_score = (
            present_optional / len(optional_fields) if optional_fields else 1.0
        )

        return (required_score * 0.7) + (optional_score * 0.3)

    def _calculate_option_consistency(self, data: OptionContract) -> float:
        """Calculate consistency score for OptionContract."""
        score = 1.0

        # Bid-ask spread consistency
        if data.bid is not None and data.ask is not None:
            if data.bid >= data.ask:
                score *= 0.0  # Critical error
            elif data.underlying_price is not None:
                spread_ratio = (data.ask - data.bid) / data.underlying_price
                if spread_ratio > 0.1:  # 10% spread
                    score *= 0.7

        # Implied volatility consistency
        if data.implied_volatility is not None:
            if data.implied_volatility < 0 or data.implied_volatility > 5.0:
                score *= 0.5

        return score

    def _calculate_option_accuracy(self, data: OptionContract) -> float:
        """Calculate accuracy score for OptionContract."""
        # Placeholder - would use option pricing models and historical data
        return 0.8  # Default accuracy score

    def _calculate_option_validity(self, data: OptionContract) -> float:
        """Calculate validity score for OptionContract."""
        score = 1.0

        # Option symbol format validation
        if not re.match(r"^[A-Z]+\d{6}[CP]\d{8}$", data.symbol):
            score *= 0.5

        # Expiry date validation
        if data.expiry.year < 2000:
            score *= 0.3

        return score

    def _detect_anomalies(self, data: MarketData) -> List[AnomalyDetection]:
        """Detect anomalies in MarketData."""
        anomalies = []

        # Price outlier detection - use historical data excluding current data point
        if (
            data.symbol in self.historical_data
            and len(self.historical_data[data.symbol]) >= 9
        ):
            # Use the last 9 data points (excluding current data)
            historical_prices = [
                d.close for d in self.historical_data[data.symbol][-9:]
            ]
            mean = statistics.mean(historical_prices)
            std = (
                statistics.stdev(historical_prices) if len(historical_prices) > 1 else 0
            )

            if std > 0:
                z_score = abs((data.close - mean) / std)
                if z_score > 3.0:
                    # Convert z_score to float for confidence calculation
                    z_score_float = float(z_score)
                    anomalies.append(
                        AnomalyDetection(
                            anomaly_type=AnomalyType.PRICE_OUTLIER,
                            severity="high",
                            confidence=min(0.95, z_score_float / 5.0),
                            description=f"Price outlier detected (z-score: {z_score_float:.2f})",
                            suggested_action="Verify with alternative data source",
                        )
                    )

        # Volume spike detection - use historical data excluding current data point
        if (
            data.symbol in self.historical_data
            and len(self.historical_data[data.symbol]) >= 9
        ):
            # Use the last 9 data points (excluding current data)
            historical_volumes = [
                d.volume for d in self.historical_data[data.symbol][-9:]
            ]
            mean_volume = statistics.mean(historical_volumes)

            if mean_volume > 0 and data.volume > mean_volume * 5:
                anomalies.append(
                    AnomalyDetection(
                        anomaly_type=AnomalyType.VOLUME_SPIKE,
                        severity="medium",
                        confidence=0.8,
                        description=f"Volume spike detected: {data.volume} vs avg {mean_volume:.0f}",
                        suggested_action="Check for news events or data errors",
                    )
                )

        return anomalies

    def _detect_option_anomalies(self, data: OptionContract) -> List[AnomalyDetection]:
        """Detect anomalies in OptionContract."""
        anomalies = []

        # Bid-ask spread anomaly
        if (
            data.bid is not None
            and data.ask is not None
            and data.underlying_price is not None
        ):
            spread_ratio = (data.ask - data.bid) / data.underlying_price
            if spread_ratio > 0.2:  # 20% spread
                anomalies.append(
                    AnomalyDetection(
                        anomaly_type=AnomalyType.SPREAD_ANOMALY,
                        severity="medium",
                        confidence=0.7,
                        description=f"Large bid-ask spread: {spread_ratio:.1%}",
                        suggested_action="Verify liquidity and market conditions",
                    )
                )

        return anomalies

    def _generate_recommendations(self, assessment: QualityAssessment) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        score = assessment.quality_score

        if score.completeness < 0.8:
            recommendations.append(
                "Consider fetching missing data fields from alternative sources"
            )

        if score.consistency < 0.7:
            recommendations.append("Verify data consistency with source provider")

        if score.timeliness < 0.6:
            recommendations.append(
                "Consider using more recent data or real-time sources"
            )

        if score.reliability < 0.7:
            recommendations.append("Consider using higher reliability data sources")

        if assessment.anomalies:
            recommendations.append(
                "Review detected anomalies before using data for trading decisions"
            )

        return recommendations

    def _generate_option_recommendations(
        self, assessment: QualityAssessment
    ) -> List[str]:
        """Generate quality improvement recommendations for options."""
        recommendations = []
        score = assessment.quality_score

        if score.completeness < 0.7:
            recommendations.append(
                "Fetch missing option data fields (bid, ask, IV, etc.)"
            )

        if score.consistency < 0.7:
            recommendations.append("Verify option pricing consistency")

        return recommendations

    def _update_historical_data(self, data: MarketData):
        """Update historical data store for statistical analysis."""
        if data.symbol not in self.historical_data:
            self.historical_data[data.symbol] = []

        self.historical_data[data.symbol].append(data)

        # Maintain window size
        if len(self.historical_data[data.symbol]) > self.historical_window:
            self.historical_data[data.symbol] = self.historical_data[data.symbol][
                -self.historical_window :
            ]

    def _update_historical_options(self, data: OptionContract):
        """Update historical options store for statistical analysis."""
        # Use underlying symbol for grouping
        underlying_symbol = data.symbol[:-15]  # Extract underlying from option symbol
        if underlying_symbol not in self.historical_options:
            self.historical_options[underlying_symbol] = []

        self.historical_options[underlying_symbol].append(data)

        # Maintain window size
        if len(self.historical_options[underlying_symbol]) > self.historical_window:
            self.historical_options[underlying_symbol] = self.historical_options[
                underlying_symbol
            ][-self.historical_window :]


# Factory function for easy access
def get_data_quality_assessor(
    historical_window: int = 100, confidence_level: float = 0.95
) -> DataQualityAssessor:
    """Get an instance of DataQualityAssessor with specified configuration."""
    return DataQualityAssessor(historical_window, confidence_level)


# Utility functions for common quality assessment tasks
def calculate_data_quality_score(data: Any) -> QualityAssessment:
    """Convenience function for comprehensive data quality assessment."""
    assessor = get_data_quality_assessor()

    if isinstance(data, MarketData):
        return assessor.assess_market_data_quality(data)
    elif isinstance(data, OptionContract):
        return assessor.assess_option_quality(data)
    else:
        # Default assessment for unknown types
        return QualityAssessment(quality_score=QualityScore(overall=0.5))


def detect_data_anomalies(data: Any) -> List[AnomalyDetection]:
    """Convenience function for anomaly detection."""
    assessor = get_data_quality_assessor()

    if isinstance(data, MarketData):
        return assessor._detect_anomalies(data)
    elif isinstance(data, OptionContract):
        return assessor._detect_option_anomalies(data)
    else:
        return []
