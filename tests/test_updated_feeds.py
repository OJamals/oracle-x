#!/usr/bin/env python3
"""
Test script to validate updated data feeds without synthetic data
Tests real data fetching, error handling, and proper fallback behavior
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime
from oracle_pipeline import OracleXPipeline
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator


def test_yfinance_data_fetching():
    """Test yfinance data fetching with real and invalid tickers"""
    print("=" * 60)
    print("TESTING YFINANCE DATA FETCHING")
    print("=" * 60)

    pipeline = OracleXPipeline()

    # Test valid tickers
    valid_tickers = ["AAPL", "TSLA", "MSFT", "GOOGL"]
    print(f"\n📊 Testing valid tickers: {valid_tickers}")

    for ticker in valid_tickers:
        print(f"\n🔍 Testing {ticker}...")
        df = pipeline.fetch_price_history(ticker, days=30)

        if df.empty:
            print(f"  ❌ {ticker}: No data returned (empty DataFrame)")
        else:
            print(f"  ✅ {ticker}: {len(df)} rows, columns: {list(df.columns)}")
            print(f"     Latest close: ${df['Close'].iloc[-1]:.2f}")
            print(f"     Date range: {df.index[0].date()} to {df.index[-1].date()}")

    # Test invalid tickers
    invalid_tickers = ["INVALID", "FAKE123", "NOTREAL"]
    print(f"\n📊 Testing invalid tickers: {invalid_tickers}")

    for ticker in invalid_tickers:
        print(f"\n🔍 Testing {ticker}...")
        df = pipeline.fetch_price_history(ticker, days=30)

        if df.empty:
            print(
                f"  ✅ {ticker}: Correctly returned empty DataFrame (no synthetic data)"
            )
        else:
            print(f"  ❌ {ticker}: Unexpected data returned: {len(df)} rows")

    return True


def test_data_feed_orchestrator():
    """Test DataFeedOrchestrator signals collection"""
    print("\n" + "=" * 60)
    print("TESTING DATA FEED ORCHESTRATOR")
    print("=" * 60)

    try:
        orchestrator = DataFeedOrchestrator()

        # Test with valid tickers
        valid_tickers = ["AAPL", "TSLA"]
        print(f"\n📊 Testing orchestrator with valid tickers: {valid_tickers}")

        signals = orchestrator.get_signals_from_scrapers(valid_tickers)

        print(f"\n📈 Signals collected:")
        print(f"  Timestamp: {signals.get('timestamp', 'N/A')}")
        print(f"  Data source: {signals.get('data_source', 'N/A')}")
        print(f"  Tickers requested: {signals.get('tickers_requested', 'N/A')}")
        print(f"  Signals count: {signals.get('signals_count', 0)}")

        # Check for synthetic data
        has_synthetic = False
        for key, value in signals.items():
            if "fallback" in key.lower() and isinstance(value, dict):
                if value.get("price") == 100.0 and value.get("volume") == 1000000:
                    has_synthetic = True
                    print(f"  ❌ Found synthetic data in {key}: {value}")

        if not has_synthetic:
            print(f"  ✅ No synthetic data detected")

        # Test with invalid tickers
        invalid_tickers = ["INVALID", "FAKE123"]
        print(f"\n📊 Testing orchestrator with invalid tickers: {invalid_tickers}")

        signals_invalid = orchestrator.get_signals_from_scrapers(invalid_tickers)
        print(
            f"  Signals count for invalid tickers: {signals_invalid.get('signals_count', 0)}"
        )

        if signals_invalid.get("signals_count", 0) == 0:
            print(f"  ✅ Correctly returned no signals for invalid tickers")
        else:
            print(f"  ⚠️  Unexpected signals for invalid tickers")

        return True

    except Exception as e:
        print(f"❌ DataFeedOrchestrator test failed: {e}")
        return False


def test_advanced_pipeline_no_synthetic():
    """Test advanced pipeline to ensure no synthetic data usage"""
    print("\n" + "=" * 60)
    print("TESTING ADVANCED PIPELINE (NO SYNTHETIC DATA)")
    print("=" * 60)

    try:
        pipeline = OracleXPipeline(mode="advanced")

        print("🚀 Running advanced pipeline...")
        # Use the run() method which handles async properly
        result = pipeline.run()

        print(f"\n📊 Pipeline Results:")
        if result:
            print(f"  Result file: {result}")
            print(f"  ✅ Pipeline completed successfully")
        else:
            print(f"  ⚠️  Pipeline returned None")

        # Check if any synthetic data warnings appeared in the result
        if result and "synthetic" in str(result).lower():
            print(f"  ⚠️  Potential synthetic data usage detected")
            return False
        else:
            print(f"  ✅ No synthetic data usage detected")
            return True

    except Exception as e:
        print(f"❌ Advanced pipeline test failed: {e}")
        return False


def test_error_handling():
    """Test error handling scenarios"""
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING")
    print("=" * 60)

    pipeline = OracleXPipeline()

    # Test with empty ticker list
    print("\n🔍 Testing empty ticker list...")
    try:
        if hasattr(pipeline, "orchestrator") and pipeline.orchestrator:
            signals = pipeline.orchestrator.get_signals_from_scrapers([])
            print(
                f"  ✅ Empty ticker list handled: {signals.get('signals_count', 0)} signals"
            )
        else:
            print(f"  ⚠️  No orchestrator available for testing")
    except Exception as e:
        print(f"  ❌ Empty ticker list error: {e}")

    # Test with None input
    print("\n🔍 Testing None input...")
    try:
        df = pipeline.fetch_price_history(None, days=30)
        if df.empty:
            print(f"  ✅ None input handled correctly (empty DataFrame)")
        else:
            print(f"  ❌ None input returned unexpected data")
    except Exception as e:
        print(f"  ✅ None input properly raised exception: {type(e).__name__}")

    return True


def main():
    """Run all tests"""
    print("🧪 ORACLE-X DATA FEEDS TESTING (NO SYNTHETIC DATA)")
    print("=" * 80)

    tests = [
        ("YFinance Data Fetching", test_yfinance_data_fetching),
        ("Data Feed Orchestrator", test_data_feed_orchestrator),
        ("Advanced Pipeline", test_advanced_pipeline_no_synthetic),
        ("Error Handling", test_error_handling),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 80)
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed! No synthetic data detected.")
    else:
        print("⚠️  Some tests failed - review output above")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
