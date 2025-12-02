#!/usr/bin/env python3
"""
Test script for Phase 1 implementation of Oracle-X financial trading system.
Tests core types, caching, rate limiting, and performance monitoring.
"""

import time
from datetime import datetime, timezone
from decimal import Decimal
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our Phase 1 components
from core.types import (
    OptionContract,
    DataSource,
    validate_market_data,
    calculate_data_quality,
)
from core.cache.redis_manager import get_cache_manager, cache_decorator
from core.cache.sqlite_manager import (
    get_sqlite_cache_manager,
    get_unified_cache_manager,
)
from core.rate_limiter import (
    get_rate_limiter,
    rate_limit_decorator,
    RateLimitExceededError,
)
from utils.performance_monitor import (
    get_performance_monitor,
    monitor_decorator,
    monitor_performance,
)


def test_core_types():
    """Test core type definitions and validation."""
    print("=== Testing Core Types ===")

    # Test MarketData validation
    market_data = {
        "symbol": "AAPL",
        "timestamp": datetime.now(timezone.utc),
        "open": Decimal("150.25"),
        "high": Decimal("152.50"),
        "low": Decimal("149.75"),
        "close": Decimal("151.80"),
        "volume": 1000000,
        "source": DataSource.YFINANCE,
        "cache_ttl": 300,
        "quality_score": 0.95,
    }

    try:
        validated_data = validate_market_data(market_data)
        print(
            f"✓ MarketData validation passed: {validated_data.symbol} at {validated_data.close}"
        )

        # Test quality calculation
        quality = calculate_data_quality(validated_data)
        print(f"✓ Data quality calculation: {quality:.2f}")

    except Exception as e:
        print(f"✗ MarketData validation failed: {e}")
        return False

    # Test OptionContract validation
    option_data = {
        "symbol": "AAPL240621C00150000",
        "strike": Decimal("150.00"),
        "expiry": datetime(2024, 6, 21, tzinfo=timezone.utc),
        "option_type": "CALL",
        "bid": Decimal("2.50"),
        "ask": Decimal("2.60"),
        "last": Decimal("2.55"),
        "volume": 150,
        "open_interest": 1000,
        "implied_volatility": Decimal("0.25"),
        "underlying_price": Decimal("151.80"),
    }

    try:
        validated_option = OptionContract(**option_data)
        print(f"✓ OptionContract validation passed: {validated_option.symbol}")

        # Test quality calculation
        quality = calculate_data_quality(validated_option)
        print(f"✓ Option quality calculation: {quality:.2f}")

    except Exception as e:
        print(f"✗ OptionContract validation failed: {e}")
        return False

    return True


def test_caching():
    """Test caching functionality."""
    print("\n=== Testing Caching ===")

    try:
        # Test Redis cache (if available)
        try:
            cache = get_cache_manager()
            print("✓ Redis cache manager initialized")

            # Test basic caching
            test_data = {"test": "data", "number": 42}
            cache.set_cached_data("test_key", test_data, ttl=10)
            cached = cache.get_cached_data("test_key")

            if cached and cached["test"] == "data":
                print("✓ Redis caching working correctly")
            else:
                print("✗ Redis caching test failed")
                return False

        except Exception as e:
            print(f"Redis not available, testing SQLite fallback: {e}")

        # Test SQLite cache
        sqlite_cache = get_sqlite_cache_manager(":memory:")
        sqlite_cache.set_cached_data("sqlite_test", {"sqlite": "working"}, ttl=10)
        sqlite_cached = sqlite_cache.get_cached_data("sqlite_test")

        if sqlite_cached and sqlite_cached["sqlite"] == "working":
            print("✓ SQLite caching working correctly")
        else:
            print("✗ SQLite caching test failed")
            return False

        # Test unified cache manager
        unified_cache = get_unified_cache_manager(prefer_redis=False)
        unified_cache.set_cached_data("unified_test", {"unified": "working"}, ttl=10)
        unified_cached = unified_cache.get_cached_data("unified_test")

        if unified_cached and unified_cached["unified"] == "working":
            print("✓ Unified cache manager working correctly")
        else:
            print("✗ Unified cache manager test failed")
            return False

    except Exception as e:
        print(f"✗ Caching test failed: {e}")
        return False

    return True


def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\n=== Testing Rate Limiting ===")

    try:
        # Use Redis for rate limiting since it has the required hash methods
        limiter = get_rate_limiter(use_redis=True)

        # Test rate limiting calls
        allowed_calls = 0
        for i in range(10):
            if limiter.rate_limit_call(DataSource.TWELVE_DATA, "test_call"):
                allowed_calls += 1

        print(f"✓ Rate limiting allowed {allowed_calls}/10 calls for TWELVE_DATA")

        # Test circuit breaker (simulate failures - need more than default threshold of 5)
        for i in range(6):  # Default threshold is 5 failures
            limiter.record_failure(DataSource.FMP)

        # Should be limited due to circuit breaker
        if not limiter.rate_limit_call(DataSource.FMP, "test_call"):
            print("✓ Circuit breaker working correctly")
        else:
            print("✗ Circuit breaker test failed")
            return False

    except Exception as e:
        print(f"✗ Rate limiting test failed: {e}")
        return False

    return True


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("\n=== Testing Performance Monitoring ===")

    try:
        monitor = get_performance_monitor()

        # Test monitoring a function call
        start_time = time.time()
        time.sleep(0.1)  # Simulate work
        metrics = monitor.monitor_performance("test_component", start_time)

        if metrics["component"] == "test_component" and metrics["duration"] > 0:
            print(f"✓ Performance monitoring working: {metrics['duration']:.3f}s")
        else:
            print("✗ Performance monitoring test failed")
            return False

        # Test decorator
        @monitor_decorator("decorated_component")
        def test_function():
            time.sleep(0.05)
            return "success"

        result = test_function()
        if result == "success":
            print("✓ Performance monitoring decorator working")
        else:
            print("✗ Performance monitoring decorator test failed")
            return False

    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        return False

    return True


def test_integration():
    """Test integration of all Phase 1 components."""
    print("\n=== Testing Integration ===")

    try:
        # Create a function that uses all components
        @monitor_decorator("integrated_component")
        @rate_limit_decorator(DataSource.YFINANCE, "price_lookup")
        @cache_decorator(ttl=30, key_prefix="price")
        def get_stock_price(symbol):
            start_time = time.time()

            # Simulate API call
            time.sleep(0.02)

            # Create market data
            price_data = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc),
                "open": Decimal("150.25"),
                "high": Decimal("152.50"),
                "low": Decimal("149.75"),
                "close": Decimal("151.80"),
                "volume": 1000000,
                "source": DataSource.YFINANCE,
            }

            # Validate and return
            validated = validate_market_data(price_data)
            monitor_performance("data_validation", start_time)
            return validated

        # Test the integrated function
        result = get_stock_price("AAPL")
        if result.symbol == "AAPL" and result.close == Decimal("151.80"):
            print("✓ Integrated function working correctly")

            # Check performance metrics
            monitor = get_performance_monitor()
            metrics = monitor.get_metrics("integrated_component")
            if metrics["total_calls"] > 0:
                print(
                    f"✓ Performance metrics collected: {metrics['total_calls']} calls"
                )
            else:
                print("✗ Performance metrics not collected")
                return False

        else:
            print("✗ Integrated function test failed")
            return False

    except RateLimitExceededError:
        print("✓ Rate limiting working in integration test")
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

    return True


def main():
    """Run all Phase 1 tests."""
    print("Oracle-X Phase 1 Implementation Test")
    print("=" * 50)

    tests = [
        test_core_types,
        test_caching,
        test_rate_limiting,
        test_performance_monitoring,
        test_integration,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "PASS" if result else "FAIL"
        print(f"{i:2d}. {test.__name__:25s} [{status}]")

    print("=" * 50)
    print(f"Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Phase 1 implementation completed successfully!")
        return 0
    else:
        print("❌ Phase 1 implementation has issues that need to be addressed.")
        return 1


if __name__ == "__main__":
    exit(main())
