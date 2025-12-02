"""
Advanced validation system for Oracle-X financial trading system.
Provides comprehensive validation rules, statistical validation, and cross-field validation.
"""

import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Callable
from enum import Enum, auto
import statistics
from dataclasses import dataclass, field


from core.types import MarketData, OptionContract, DataSource


class ValidationLevel(Enum):
    """Validation strictness levels."""

    BASIC = auto()  # Basic format validation only
    STANDARD = auto()  # Standard validation with basic sanity checks
    STRICT = auto()  # Strict validation with statistical checks
    PARANOID = auto()  # Paranoid validation with full statistical analysis


class ValidationRuleType(Enum):
    """Types of validation rules."""

    FORMAT = auto()  # Format/pattern validation
    RANGE = auto()  # Value range validation
    CROSS_FIELD = auto()  # Cross-field validation
    STATISTICAL = auto()  # Statistical validation
    BUSINESS = auto()  # Business rule validation


@dataclass
class ValidationRule:
    """A validation rule with metadata and validation function."""

    rule_type: ValidationRuleType
    name: str
    description: str
    severity: str  # "error", "warning", "info"
    validator_func: Callable
    enabled: bool = True
    threshold: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    quality_score: float = 1.0


class AdvancedValidators:
    """Advanced validation system with comprehensive validation rules."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.rules = self._initialize_rules()
        self.historical_data: Dict[str, List[Any]] = {}

    def _initialize_rules(self) -> Dict[ValidationRuleType, List[ValidationRule]]:
        """Initialize all validation rules."""
        rules = {}

        # Format validation rules
        rules[ValidationRuleType.FORMAT] = [
            ValidationRule(
                rule_type=ValidationRuleType.FORMAT,
                name="symbol_format",
                description="Validate stock symbol format (uppercase letters only)",
                severity="error",
                validator_func=self._validate_symbol_format,
            ),
            ValidationRule(
                rule_type=ValidationRuleType.FORMAT,
                name="option_symbol_format",
                description="Validate option symbol format",
                severity="error",
                validator_func=self._validate_option_symbol_format,
            ),
            ValidationRule(
                rule_type=ValidationRuleType.FORMAT,
                name="timestamp_format",
                description="Validate timestamp format and timezone",
                severity="error",
                validator_func=self._validate_timestamp_format,
            ),
        ]

        # Range validation rules
        rules[ValidationRuleType.RANGE] = [
            ValidationRule(
                rule_type=ValidationRuleType.RANGE,
                name="price_positive",
                description="Validate prices are positive",
                severity="error",
                validator_func=self._validate_price_positive,
            ),
            ValidationRule(
                rule_type=ValidationRuleType.RANGE,
                name="volume_non_negative",
                description="Validate volume is non-negative",
                severity="error",
                validator_func=self._validate_volume_non_negative,
            ),
            ValidationRule(
                rule_type=ValidationRuleType.RANGE,
                name="iv_range",
                description="Validate implied volatility range (0-5.0)",
                severity="error",
                validator_func=self._validate_iv_range,
            ),
        ]

        # Cross-field validation rules
        rules[ValidationRuleType.CROSS_FIELD] = [
            ValidationRule(
                rule_type=ValidationRuleType.CROSS_FIELD,
                name="price_consistency",
                description="Validate price consistency (low <= high, open/close within range)",
                severity="error",
                validator_func=self._validate_price_consistency,
            ),
            ValidationRule(
                rule_type=ValidationRuleType.CROSS_FIELD,
                name="bid_ask_spread",
                description="Validate bid-ask spread sanity",
                severity="warning",
                validator_func=self._validate_bid_ask_spread,
            ),
            ValidationRule(
                rule_type=ValidationRuleType.CROSS_FIELD,
                name="expiry_reasonableness",
                description="Validate expiry date is reasonable",
                severity="error",
                validator_func=self._validate_expiry_reasonableness,
            ),
        ]

        # Statistical validation rules (only enabled in STRICT or PARANOID modes)
        rules[ValidationRuleType.STATISTICAL] = [
            ValidationRule(
                rule_type=ValidationRuleType.STATISTICAL,
                name="price_outlier_detection",
                description="Detect price outliers using statistical methods",
                severity="warning",
                validator_func=self._validate_price_outliers,
                enabled=self.validation_level
                in [ValidationLevel.STRICT, ValidationLevel.PARANOID],
            ),
            ValidationRule(
                rule_type=ValidationRuleType.STATISTICAL,
                name="volume_anomaly_detection",
                description="Detect volume anomalies using statistical methods",
                severity="warning",
                validator_func=self._validate_volume_anomalies,
                enabled=self.validation_level
                in [ValidationLevel.STRICT, ValidationLevel.PARANOID],
            ),
        ]

        # Business rule validation
        rules[ValidationRuleType.BUSINESS] = [
            ValidationRule(
                rule_type=ValidationRuleType.BUSINESS,
                name="data_freshness",
                description="Validate data freshness based on timestamp",
                severity="warning",
                validator_func=self._validate_data_freshness,
            ),
            ValidationRule(
                rule_type=ValidationRuleType.BUSINESS,
                name="source_reliability",
                description="Validate data source reliability",
                severity="info",
                validator_func=self._validate_source_reliability,
            ),
        ]

        return rules

    def validate_market_data(self, data: MarketData) -> ValidationResult:
        """Comprehensive validation of MarketData."""
        result = ValidationResult(is_valid=True)

        # Apply all enabled rules
        for rule_type, rules in self.rules.items():
            for rule in rules:
                if rule.enabled:
                    try:
                        rule.validator_func(data, result)
                    except Exception as e:
                        result.errors.append(
                            f"Validation error in {rule.name}: {str(e)}"
                        )
                        result.is_valid = False

        # Calculate final quality score
        result.quality_score = self._calculate_quality_score(result)

        return result

    def validate_option_contract(self, data: OptionContract) -> ValidationResult:
        """Comprehensive validation of OptionContract."""
        result = ValidationResult(is_valid=True)

        # Apply all enabled rules
        for rule_type, rules in self.rules.items():
            for rule in rules:
                if rule.enabled:
                    try:
                        rule.validator_func(data, result)
                    except Exception as e:
                        result.errors.append(
                            f"Validation error in {rule.name}: {str(e)}"
                        )
                        result.is_valid = False

        # Calculate final quality score
        result.quality_score = self._calculate_quality_score(result)

        return result

    def _calculate_quality_score(self, result: ValidationResult) -> float:
        """Calculate overall quality score based on validation results."""
        score = 1.0

        # Penalize for errors
        if result.errors:
            score *= 0.5 ** min(
                len(result.errors), 5
            )  # Exponential decay for multiple errors

        # Penalize for warnings
        if result.warnings:
            score *= 0.9 ** min(len(result.warnings), 10)

        return max(0.0, min(1.0, score))

    # Format validation methods
    def _validate_symbol_format(self, data: Any, result: ValidationResult):
        if isinstance(data, MarketData):
            if not re.match(r"^[A-Z]+$", data.symbol):
                result.errors.append(f"Invalid symbol format: {data.symbol}")
                result.is_valid = False

    def _validate_option_symbol_format(self, data: Any, result: ValidationResult):
        if isinstance(data, OptionContract):
            if not re.match(r"^[A-Z]+\d{6}[CP]\d{8}$", data.symbol):
                result.errors.append(f"Invalid option symbol format: {data.symbol}")
                result.is_valid = False

    def _validate_timestamp_format(self, data: Any, result: ValidationResult):
        if hasattr(data, "timestamp"):
            timestamp = data.timestamp
            if timestamp.tzinfo is None:
                result.errors.append("Timestamp must have timezone information")
                result.is_valid = False
            elif timestamp.tzinfo.utcoffset(timestamp) != timestamp.utcoffset():
                result.errors.append("Timestamp must be in UTC timezone")
                result.is_valid = False

    # Range validation methods
    def _validate_price_positive(self, data: Any, result: ValidationResult):
        price_fields = [
            "open",
            "high",
            "low",
            "close",
            "bid",
            "ask",
            "last",
            "strike",
            "underlying_price",
        ]
        for price_field in price_fields:
            if hasattr(data, price_field) and getattr(data, price_field) is not None:
                price = getattr(data, price_field)
                if price <= Decimal("0"):
                    result.errors.append(f"{price_field} must be positive: {price}")
                    result.is_valid = False

    def _validate_volume_non_negative(self, data: Any, result: ValidationResult):
        volume_fields = ["volume", "open_interest"]
        for volume_field in volume_fields:
            if hasattr(data, volume_field) and getattr(data, volume_field) is not None:
                volume = getattr(data, volume_field)
                if volume < 0:
                    result.errors.append(
                        f"{volume_field} must be non-negative: {volume}"
                    )
                    result.is_valid = False

    def _validate_iv_range(self, data: Any, result: ValidationResult):
        if isinstance(data, OptionContract) and data.implied_volatility is not None:
            iv = data.implied_volatility
            if iv < Decimal("0") or iv > Decimal("5.0"):
                result.errors.append(
                    f"Implied volatility must be between 0 and 5.0: {iv}"
                )
                result.is_valid = False

    # Cross-field validation methods
    def _validate_price_consistency(self, data: Any, result: ValidationResult):
        if isinstance(data, MarketData):
            if data.low > data.high:
                result.errors.append(
                    f"Low price ({data.low}) cannot be greater than high price ({data.high})"
                )
                result.is_valid = False

            if data.open > data.high or data.open < data.low:
                result.warnings.append(
                    f"Open price ({data.open}) outside daily range ({data.low}-{data.high})"
                )

            if data.close > data.high or data.close < data.low:
                result.warnings.append(
                    f"Close price ({data.close}) outside daily range ({data.low}-{data.high})"
                )

    def _validate_bid_ask_spread(self, data: Any, result: ValidationResult):
        if (
            isinstance(data, OptionContract)
            and data.bid is not None
            and data.ask is not None
        ):
            spread = data.ask - data.bid
            if spread <= Decimal("0"):
                result.errors.append(
                    f"Bid-ask spread must be positive: bid={data.bid}, ask={data.ask}"
                )
                result.is_valid = False
            elif data.underlying_price is not None:
                spread_ratio = (spread / data.underlying_price) * Decimal("100")
                if spread_ratio > Decimal("10"):  # 10% spread threshold
                    result.warnings.append(f"Large bid-ask spread: {spread_ratio:.2f}%")

    def _validate_expiry_reasonableness(self, data: Any, result: ValidationResult):
        if isinstance(data, OptionContract):
            if data.expiry.year < 2000:
                result.errors.append("Expiry date is too far in the past")
                result.is_valid = False
            elif data.expiry > datetime.now(data.expiry.tzinfo) + timedelta(
                days=365 * 3
            ):
                result.warnings.append("Expiry date is more than 3 years in the future")

    # Statistical validation methods
    def _validate_price_outliers(self, data: Any, result: ValidationResult):
        # This would use historical data for statistical analysis
        # Placeholder implementation
        if isinstance(data, MarketData) and self.historical_data.get(data.symbol):
            historical_prices = self.historical_data[data.symbol]
            if len(historical_prices) >= 10:  # Need sufficient data
                prices = [p.close for p in historical_prices[-10:]]
                mean = statistics.mean(prices)
                std = statistics.stdev(prices) if len(prices) > 1 else 0

                if std > 0:
                    z_score = abs((data.close - mean) / std)
                    if z_score > 3.0:
                        result.warnings.append(
                            f"Price outlier detected (z-score: {z_score:.2f})"
                        )

    def _validate_volume_anomalies(self, data: Any, result: ValidationResult):
        # This would use historical data for statistical analysis
        # Placeholder implementation
        if isinstance(data, MarketData) and self.historical_data.get(data.symbol):
            historical_volumes = [p.volume for p in self.historical_data[data.symbol]]
            if len(historical_volumes) >= 10:
                mean_volume = statistics.mean(historical_volumes)
                if data.volume > mean_volume * 5:
                    result.warnings.append(
                        f"Volume anomaly detected: {data.volume} vs avg {mean_volume:.0f}"
                    )

    # Business rule validation methods
    def _validate_data_freshness(self, data: Any, result: ValidationResult):
        if hasattr(data, "timestamp"):
            age_hours = (
                datetime.now(data.timestamp.tzinfo) - data.timestamp
            ).total_seconds() / 3600
            if age_hours > 24:
                result.warnings.append(f"Data is stale: {age_hours:.1f} hours old")
            elif age_hours > 1:
                result.info.append(f"Data age: {age_hours:.1f} hours")

    def _validate_source_reliability(self, data: Any, result: ValidationResult):
        if hasattr(data, "source"):
            # Simple reliability scoring based on source type
            reliability_scores = {
                DataSource.YFINANCE: 0.9,
                DataSource.TWELVE_DATA: 0.8,
                DataSource.FMP: 0.7,
                DataSource.FINNHUB: 0.6,
                DataSource.ALPHA_VANTAGE: 0.5,
            }
            reliability = reliability_scores.get(data.source, 0.5)
            if reliability < 0.7:
                result.info.append(
                    f"Data source reliability: {reliability:.1f} ({data.source.name})"
                )


# Factory function for easy access
def get_advanced_validator(
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
) -> AdvancedValidators:
    """Get an instance of AdvancedValidators with specified validation level."""
    return AdvancedValidators(validation_level)


# Utility functions for common validation tasks
def validate_market_data_advanced(
    data: MarketData, validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationResult:
    """Convenience function for advanced market data validation."""
    validator = get_advanced_validator(validation_level)
    return validator.validate_market_data(data)


def validate_option_contract_advanced(
    data: OptionContract, validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationResult:
    """Convenience function for advanced option contract validation."""
    validator = get_advanced_validator(validation_level)
    return validator.validate_option_contract(data)
