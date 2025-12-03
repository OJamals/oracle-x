"""
Test script to demonstrate Phase 2.1 Enhanced Validation Foundation.
Shows the new advanced validation and data quality assessment capabilities.
"""

import sys
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# Add the project root to Python path
sys.path.insert(0, "/Users/omar/Documents/Projects/oracle-x")

from core.types_internal import (
    MarketData,
    OptionContract,
    DataSource,
    OptionType,
    OptionStyle,
)
from core.validation.advanced_validators import (
    AdvancedValidators,
    ValidationLevel,
    validate_market_data_advanced,
    validate_option_contract_advanced,
)
from core.quality.data_quality_engine import DataQualityAssessor


def test_enhanced_validation_demonstration():
    """Demonstrate the enhanced validation capabilities."""
    print("=" * 60)
    print("Oracle-X Phase 2.1: Enhanced Validation Foundation")
    print("=" * 60)

    # Create test data with various quality issues
    now = datetime.now(timezone.utc)

    # Test case 1: Valid market data
    print("\n1. Testing Valid Market Data:")
    print("-" * 40)
    valid_market_data = MarketData(
        symbol="AAPL",
        timestamp=now,
        open=Decimal("150.00"),
        high=Decimal("155.00"),
        low=Decimal("149.50"),
        close=Decimal("154.50"),
        volume=1000000,
        source=DataSource.YFINANCE,
    )

    result = validate_market_data_advanced(valid_market_data)
    print(f"✓ Valid data: Quality Score = {result.quality_score:.3f}")
    print(f"  Warnings: {len(result.warnings)}, Errors: {len(result.errors)}")

    # Test case 2: Market data with issues
    print("\n2. Testing Market Data with Quality Issues:")
    print("-" * 40)
    problematic_market_data = MarketData(
        symbol="AAPL",
        timestamp=now - timedelta(hours=25),  # Stale data
        open=Decimal("160.00"),  # Outside range
        high=Decimal("155.00"),
        low=Decimal("149.50"),
        close=Decimal("154.50"),
        volume=0,  # Zero volume
        source=DataSource.ALPHA_VANTAGE,  # Less reliable source
    )

    result = validate_market_data_advanced(problematic_market_data)
    print(f"✓ Problematic data: Quality Score = {result.quality_score:.3f}")
    print(f"  Warnings: {len(result.warnings)}, Errors: {len(result.errors)}")
    for warning in result.warnings:
        print(f"  ⚠️  {warning}")

    # Test case 3: Option contract validation
    print("\n3. Testing Option Contract Validation:")
    print("-" * 40)
    option_contract = OptionContract(
        symbol="AAPL240621C00150000",
        strike=Decimal("150.00"),
        expiry=now + timedelta(days=30),
        option_type=OptionType.CALL,
        style=OptionStyle.AMERICAN,
        bid=Decimal("3.00"),  # Higher than ask (issue)
        ask=Decimal("2.50"),
        underlying_price=Decimal("154.50"),
    )

    result = validate_option_contract_advanced(option_contract)
    print(f"✓ Option contract: Quality Score = {result.quality_score:.3f}")
    print(f"  Warnings: {len(result.warnings)}, Errors: {len(result.errors)}")
    for error in result.errors:
        print(f"  ❌ {error}")

    # Test case 4: Different validation levels
    print("\n4. Testing Different Validation Levels:")
    print("-" * 40)

    # BASIC level
    basic_validator = AdvancedValidators(ValidationLevel.BASIC)
    basic_result = basic_validator.validate_market_data(problematic_market_data)
    print(f"✓ BASIC level: Quality Score = {basic_result.quality_score:.3f}")
    print(
        f"  Warnings: {len(basic_result.warnings)}, Errors: {len(basic_result.errors)}"
    )

    # STRICT level
    strict_validator = AdvancedValidators(ValidationLevel.STRICT)
    strict_result = strict_validator.validate_market_data(problematic_market_data)
    print(f"✓ STRICT level: Quality Score = {strict_result.quality_score:.3f}")
    print(
        f"  Warnings: {len(strict_result.warnings)}, Errors: {len(strict_result.errors)}"
    )

    # Test case 5: Data Quality Assessment
    print("\n5. Testing Data Quality Assessment Engine:")
    print("-" * 40)
    quality_assessor = DataQualityAssessor()

    # Assess market data quality
    quality_result = quality_assessor.assess_market_data_quality(
        problematic_market_data
    )
    print(f"✓ Data Quality Score: {quality_result.quality_score.overall:.3f}")
    print(f"  Completeness: {quality_result.quality_score.completeness:.3f}")
    print(f"  Consistency: {quality_result.quality_score.consistency:.3f}")
    print(f"  Timeliness: {quality_result.quality_score.timeliness:.3f}")
    print(f"  Reliability: {quality_result.quality_score.reliability:.3f}")

    # Show recommendations from the quality result
    print(f"✓ Recommendations: {len(quality_result.recommendations)}")
    for i, rec in enumerate(quality_result.recommendations[:3], 1):  # Show first 3
        print(f"  {i}. {rec}")

    print("\n" + "=" * 60)
    print("Phase 2.1 Enhanced Validation Foundation - COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    test_enhanced_validation_demonstration()
