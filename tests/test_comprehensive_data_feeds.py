#!/usr/bin/env python3
"""
Comprehensive test suite for all Oracle-X data feeds.
Tests real data quality, error handling, and integration.
"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_data_feed_integration():
    """Test integration with oracle_engine pipeline"""
    print("Testing Oracle Engine Integration...")
    try:
        from oracle_engine.chains.prompt_chain import get_signals_from_scrapers

        start_time = time.time()
        signals = get_signals_from_scrapers(
            "Analyze AAPL for trading opportunities", chart_image_b64=None
        )
        execution_time = time.time() - start_time

        print(f"✅ Pipeline integration: {execution_time:.2f}s execution time")

        # Check for our updated data feeds
        feed_checks = {
            "dark_pools": "dark_pools" in signals,
            "earnings_calendar": "earnings_calendar" in signals,
            "market_internals": "market_internals" in signals,
            "options_flow": "options_flow" in signals,
        }

        for feed, present in feed_checks.items():
            status = "✅" if present else "❌"
            print(f"   {status} {feed}: {'Present' if present else 'Missing'}")

        return all(feed_checks.values())

    except Exception as e:
        print(f"❌ Pipeline integration failed: {e}")
        return False


def test_data_quality():
    """Test data quality and realistic values"""
    print("\nTesting Data Quality...")

    from data_feeds.dark_pools import fetch_dark_pool_data
    from data_feeds.earnings_calendar import fetch_earnings_calendar
    from data_feeds.market_internals import fetch_market_internals
    from data_feeds.options_flow import fetch_options_flow

    quality_score = 0
    total_tests = 0

    # Test dark pools
    try:
        dark_pools = fetch_dark_pool_data(["AAPL", "TSLA", "MSFT"])
        total_tests += 1
        if dark_pools.get("data_source") == "volume_analysis_proxy":
            quality_score += 1
            print("   ✅ Dark pools: Real volume analysis")
        else:
            print("   ❌ Dark pools: Unexpected data source")
    except Exception as e:
        print(f"   ❌ Dark pools: {e}")
        total_tests += 1

    # Test earnings calendar
    try:
        earnings = fetch_earnings_calendar(["AAPL", "TSLA"])
        total_tests += 1
        if earnings and earnings[0].get("data_source") == "yfinance_income_stmt":
            quality_score += 1
            print(f"   ✅ Earnings: Real EPS data (${earnings[0]['estimate']})")
        else:
            print("   ❌ Earnings: Missing or invalid data")
    except Exception as e:
        print(f"   ❌ Earnings: {e}")
        total_tests += 1

    # Test market internals
    try:
        internals = fetch_market_internals()
        total_tests += 1
        vix = internals.get("vix", 0)
        if 10 < vix < 50 and internals.get("data_source") == "yfinance_calculated":
            quality_score += 1
            print(f"   ✅ Market internals: Realistic VIX ({vix})")
        else:
            print(f"   ❌ Market internals: Unrealistic VIX ({vix})")
    except Exception as e:
        print(f"   ❌ Market internals: {e}")
        total_tests += 1

    # Test options flow
    try:
        options = fetch_options_flow(["AAPL", "TSLA"])
        total_tests += 1
        sweeps = options.get("unusual_sweeps", [])
        if options.get("data_source") == "yfinance_options":
            quality_score += 1
            print(f"   ✅ Options flow: {len(sweeps)} real unusual sweeps")
        else:
            print("   ❌ Options flow: Missing or invalid data")
    except Exception as e:
        print(f"   ❌ Options flow: {e}")
        total_tests += 1

    return quality_score, total_tests


def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\nTesting Error Handling...")

    from data_feeds.dark_pools import fetch_dark_pool_data
    from data_feeds.earnings_calendar import fetch_earnings_calendar
    from data_feeds.market_internals import fetch_market_internals
    from data_feeds.options_flow import fetch_options_flow

    error_tests = 0
    passed_tests = 0

    # Test with invalid tickers
    invalid_tickers = ["INVALID123", "FAKE456"]

    try:
        result = fetch_dark_pool_data(invalid_tickers)
        error_tests += 1
        if isinstance(result, dict) and "dark_pools" in result:
            passed_tests += 1
            print("   ✅ Dark pools: Graceful handling of invalid tickers")
        else:
            print("   ❌ Dark pools: Poor error handling")
    except Exception:
        print("   ❌ Dark pools: Exception on invalid tickers")
        error_tests += 1

    try:
        result = fetch_earnings_calendar(invalid_tickers)
        error_tests += 1
        if isinstance(result, list):
            passed_tests += 1
            print("   ✅ Earnings: Graceful handling of invalid tickers")
        else:
            print("   ❌ Earnings: Poor error handling")
    except Exception:
        print("   ❌ Earnings: Exception on invalid tickers")
        error_tests += 1

    try:
        result = fetch_market_internals()
        error_tests += 1
        if isinstance(result, dict) and "data_source" in result:
            passed_tests += 1
            print("   ✅ Market internals: Robust error handling")
        else:
            print("   ❌ Market internals: Poor error handling")
    except Exception:
        print("   ❌ Market internals: Exception in error handling")
        error_tests += 1

    try:
        result = fetch_options_flow(invalid_tickers)
        error_tests += 1
        if isinstance(result, dict) and "unusual_sweeps" in result:
            passed_tests += 1
            print("   ✅ Options flow: Graceful handling of invalid tickers")
        else:
            print("   ❌ Options flow: Poor error handling")
    except Exception:
        print("   ❌ Options flow: Exception on invalid tickers")
        error_tests += 1

    return passed_tests, error_tests


def test_performance_benchmark():
    """Benchmark performance of data feeds"""
    print("\nTesting Performance...")

    from data_feeds.dark_pools import fetch_dark_pool_data
    from data_feeds.earnings_calendar import fetch_earnings_calendar
    from data_feeds.market_internals import fetch_market_internals
    from data_feeds.options_flow import fetch_options_flow

    tickers = ["AAPL", "TSLA", "MSFT"]

    benchmarks = {}

    # Benchmark each feed
    feeds = [
        ("Dark Pools", lambda: fetch_dark_pool_data(tickers)),
        ("Earnings", lambda: fetch_earnings_calendar(tickers)),
        ("Market Internals", lambda: fetch_market_internals()),
        ("Options Flow", lambda: fetch_options_flow(tickers)),
    ]

    for name, func in feeds:
        try:
            start_time = time.time()
            result = func()
            execution_time = time.time() - start_time
            benchmarks[name] = execution_time

            status = (
                "✅" if execution_time < 10 else "⚠️" if execution_time < 30 else "❌"
            )
            print(f"   {status} {name}: {execution_time:.2f}s")

        except Exception as e:
            print(f"   ❌ {name}: Failed ({e})")
            benchmarks[name] = float("inf")

    return benchmarks


def main():
    """Run comprehensive test suite"""
    print("=" * 70)
    print("COMPREHENSIVE ORACLE-X DATA FEEDS TEST SUITE")
    print("=" * 70)

    # Test results tracking
    results = {"timestamp": datetime.now().isoformat(), "tests": {}}

    # 1. Integration test
    integration_passed = test_data_feed_integration()
    results["tests"]["integration"] = integration_passed

    # 2. Data quality test
    quality_score, quality_total = test_data_quality()
    quality_passed = quality_score == quality_total
    results["tests"]["data_quality"] = {
        "passed": quality_passed,
        "score": f"{quality_score}/{quality_total}",
    }

    # 3. Error handling test
    error_passed, error_total = test_error_handling()
    error_handling_ok = error_passed == error_total
    results["tests"]["error_handling"] = {
        "passed": error_handling_ok,
        "score": f"{error_passed}/{error_total}",
    }

    # 4. Performance benchmark
    benchmarks = test_performance_benchmark()
    avg_time = sum(t for t in benchmarks.values() if t != float("inf")) / len(
        [t for t in benchmarks.values() if t != float("inf")]
    )
    performance_ok = avg_time < 15  # Average under 15 seconds
    results["tests"]["performance"] = {
        "passed": performance_ok,
        "average_time": f"{avg_time:.2f}s",
        "benchmarks": benchmarks,
    }

    # Overall results
    all_tests = [integration_passed, quality_passed, error_handling_ok, performance_ok]
    total_passed = sum(all_tests)
    total_tests = len(all_tests)

    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    print(f"Integration Test: {'✅ PASS' if integration_passed else '❌ FAIL'}")
    print(
        f"Data Quality: {'✅ PASS' if quality_passed else '❌ FAIL'} ({quality_score}/{quality_total})"
    )
    print(
        f"Error Handling: {'✅ PASS' if error_handling_ok else '❌ FAIL'} ({error_passed}/{error_total})"
    )
    print(
        f"Performance: {'✅ PASS' if performance_ok else '❌ FAIL'} (avg: {avg_time:.2f}s)"
    )
    print("=" * 70)
    print(
        f"OVERALL: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)"
    )

    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED - Oracle-X data feeds are production ready!")
    else:
        print("⚠️  Some tests failed - review results above")

    # Save detailed results
    with open("test_results_comprehensive.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: test_results_comprehensive.json")

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
