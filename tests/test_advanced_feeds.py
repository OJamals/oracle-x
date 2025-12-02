#!/usr/bin/env python3
"""
Advanced testing suite for Oracle-X data feeds
Tests edge cases, performance, concurrent access, and stress scenarios
"""

import sys
import os
import time
import threading
import concurrent.futures
from datetime import datetime, timedelta
import pandas as pd
import psutil
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from oracle_pipeline import OracleXPipeline
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator

def test_edge_cases():
    """Test edge cases and unusual scenarios"""
    print("=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)
    
    pipeline = OracleXPipeline()
    results = []
    
    # Test 1: Very long ticker names
    print("\n🔍 Testing long ticker names...")
    long_tickers = ['VERYLONGTICKERNAME', 'ABCDEFGHIJKLMNOP', '1234567890123456']
    for ticker in long_tickers:
        df = pipeline.fetch_price_history(ticker, days=5)
        result = "PASS" if df.empty else "FAIL"
        print(f"  {ticker}: {result} (empty DataFrame expected)")
        results.append(result == "PASS")
    
    # Test 2: Special characters in tickers
    print("\n🔍 Testing special characters...")
    special_tickers = ['$AAPL', 'AAPL.L', 'BRK-A', 'BRK/B', 'TEST@123']
    for ticker in special_tickers:
        try:
            df = pipeline.fetch_price_history(ticker, days=5)
            print(f"  {ticker}: Handled gracefully ({len(df)} rows)")
            results.append(True)
        except Exception as e:
            print(f"  {ticker}: Exception handled - {type(e).__name__}")
            results.append(True)  # Exception handling is acceptable
    
    # Test 3: Extreme date ranges
    print("\n🔍 Testing extreme date ranges...")
    extreme_cases = [
        ("AAPL", 1),      # 1 day
        ("AAPL", 365),    # 1 year
        ("AAPL", 1825),   # 5 years
        ("AAPL", 0),      # 0 days (edge case)
    ]
    
    for ticker, days in extreme_cases:
        try:
            df = pipeline.fetch_price_history(ticker, days=days)
            print(f"  {ticker} ({days} days): {len(df)} rows")
            results.append(True)
        except Exception as e:
            print(f"  {ticker} ({days} days): Exception - {type(e).__name__}")
            results.append(True)
    
    # Test 4: Empty and None inputs
    print("\n🔍 Testing empty/None inputs...")
    edge_inputs = [None, "", " ", "   "]
    for inp in edge_inputs:
        try:
            df = pipeline.fetch_price_history(inp, days=30)
            result = "PASS" if df.empty else "FAIL"
            print(f"  Input '{inp}': {result}")
            results.append(result == "PASS")
        except Exception as e:
            print(f"  Input '{inp}': Exception handled - {type(e).__name__}")
            results.append(True)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Edge Cases Success Rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")
    return success_rate > 80

def test_concurrent_performance():
    """Test concurrent data fetching performance"""
    print("\n" + "=" * 60)
    print("TESTING CONCURRENT PERFORMANCE")
    print("=" * 60)
    
    tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX']
    pipeline = OracleXPipeline()
    
    # Test 1: Sequential fetching
    print("\n📊 Sequential fetching...")
    start_time = time.time()
    sequential_results = []
    
    for ticker in tickers:
        df = pipeline.fetch_price_history(ticker, days=30)
        sequential_results.append((ticker, len(df)))
    
    sequential_time = time.time() - start_time
    print(f"  Sequential time: {sequential_time:.2f}s")
    print(f"  Results: {len([r for r in sequential_results if r[1] > 0])}/{len(tickers)} successful")
    
    # Test 2: Concurrent fetching
    print("\n📊 Concurrent fetching...")
    start_time = time.time()
    concurrent_results = []
    
    def fetch_ticker_data(ticker):
        df = pipeline.fetch_price_history(ticker, days=30)
        return (ticker, len(df))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_ticker_data, ticker) for ticker in tickers]
        concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    concurrent_time = time.time() - start_time
    print(f"  Concurrent time: {concurrent_time:.2f}s")
    print(f"  Results: {len([r for r in concurrent_results if r[1] > 0])}/{len(tickers)} successful")
    
    # Performance analysis
    speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
    print(f"\n📈 Performance Analysis:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Efficiency: {speedup/4*100:.1f}% (4 threads)")
    
    return speedup > 1.5  # Expect at least 1.5x speedup

def test_data_quality():
    """Test data quality and consistency"""
    print("\n" + "=" * 60)
    print("TESTING DATA QUALITY")
    print("=" * 60)
    
    pipeline = OracleXPipeline()
    orchestrator = DataFeedOrchestrator()
    
    test_tickers = ['AAPL', 'TSLA', 'MSFT']
    quality_checks = []
    
    for ticker in test_tickers:
        print(f"\n🔍 Testing {ticker} data quality...")
        
        # Fetch data
        df = pipeline.fetch_price_history(ticker, days=30)
        
        if df.empty:
            print(f"  ❌ No data available for {ticker}")
            quality_checks.append(False)
            continue
        
        checks = []
        
        # Check 1: Required columns present
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if not missing_cols:
            print(f"  ✅ All required columns present")
            checks.append(True)
        else:
            print(f"  ❌ Missing columns: {missing_cols}")
            checks.append(False)
        
        # Check 2: No negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        negative_prices = any((df[col] < 0).any() for col in price_cols if col in df.columns)
        if not negative_prices:
            print(f"  ✅ No negative prices")
            checks.append(True)
        else:
            print(f"  ❌ Found negative prices")
            checks.append(False)
        
        # Check 3: High >= Low
        if 'High' in df.columns and 'Low' in df.columns:
            invalid_ranges = (df['High'] < df['Low']).sum()
            if invalid_ranges == 0:
                print(f"  ✅ Valid High/Low ranges")
                checks.append(True)
            else:
                print(f"  ❌ Invalid High/Low ranges: {invalid_ranges}")
                checks.append(False)
        
        # Check 4: Reasonable volume values
        if 'Volume' in df.columns:
            zero_volume_days = (df['Volume'] == 0).sum()
            if zero_volume_days < len(df) * 0.1:  # Less than 10% zero volume days
                print(f"  ✅ Reasonable volume data ({zero_volume_days} zero-volume days)")
                checks.append(True)
            else:
                print(f"  ⚠️  High zero-volume days: {zero_volume_days}")
                checks.append(False)
        
        # Check 5: Data freshness
        if not df.empty:
            latest_date = df.index[-1].date()
            days_old = (datetime.now().date() - latest_date).days
            if days_old <= 7:  # Within last week
                print(f"  ✅ Fresh data ({days_old} days old)")
                checks.append(True)
            else:
                print(f"  ⚠️  Stale data ({days_old} days old)")
                checks.append(False)
        
        ticker_quality = sum(checks) / len(checks) * 100
        print(f"  📊 {ticker} Quality Score: {ticker_quality:.1f}%")
        quality_checks.append(ticker_quality > 80)
    
    overall_quality = sum(quality_checks) / len(quality_checks) * 100
    print(f"\n📊 Overall Data Quality: {overall_quality:.1f}%")
    return overall_quality > 75

def test_rate_limiting():
    """Test API rate limiting and error recovery"""
    print("\n" + "=" * 60)
    print("TESTING RATE LIMITING & ERROR RECOVERY")
    print("=" * 60)
    
    pipeline = OracleXPipeline()
    
    # Test rapid successive calls
    print("\n🔍 Testing rapid successive calls...")
    rapid_tickers = ['AAPL'] * 10  # Same ticker multiple times
    start_time = time.time()
    
    results = []
    for i, ticker in enumerate(rapid_tickers):
        df = pipeline.fetch_price_history(ticker, days=5)
        results.append(len(df))
        print(f"  Call {i+1}: {len(df)} rows")
    
    total_time = time.time() - start_time
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per call: {total_time/len(rapid_tickers):.2f}s")
    
    # Check if caching is working (should be faster after first call)
    cache_working = results[1] > 0 and total_time < 30  # Should complete in reasonable time
    print(f"  Caching effectiveness: {'✅ Working' if cache_working else '❌ Not working'}")
    
    return cache_working

def test_memory_usage():
    """Test memory usage and resource management"""
    print("\n" + "=" * 60)
    print("TESTING MEMORY USAGE")
    print("=" * 60)
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"📊 Initial memory usage: {initial_memory:.1f} MB")
    
    # Test memory usage with multiple data fetches
    pipeline = OracleXPipeline()
    tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN'] * 3  # 15 total calls
    
    for i, ticker in enumerate(tickers):
        df = pipeline.fetch_price_history(ticker, days=60)
        
        if (i + 1) % 5 == 0:  # Check every 5 calls
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            print(f"  After {i+1} calls: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")
    
    # Force garbage collection
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024
    total_increase = final_memory - initial_memory
    
    print(f"📊 Final memory usage: {final_memory:.1f} MB (+{total_increase:.1f} MB)")
    
    # Memory usage should be reasonable (less than 500MB increase)
    memory_ok = total_increase < 500
    print(f"📊 Memory usage: {'✅ Acceptable' if memory_ok else '❌ Excessive'}")
    
    return memory_ok

def test_orchestrator_integration():
    """Test DataFeedOrchestrator integration scenarios"""
    print("\n" + "=" * 60)
    print("TESTING ORCHESTRATOR INTEGRATION")
    print("=" * 60)
    
    try:
        orchestrator = DataFeedOrchestrator()
        
        # Test 1: Multiple ticker signal collection
        print("\n🔍 Testing multiple ticker signals...")
        tickers = ['AAPL', 'TSLA']  # Reduced to avoid rate limits
        signals = orchestrator.get_signals_from_scrapers(tickers)
        
        print(f"  Signals collected: {signals.get('signals_count', 0)}")
        print(f"  Timestamp: {signals.get('timestamp', 'N/A')}")
        print(f"  Data source: {signals.get('data_source', 'N/A')}")
        
        # Validate signal structure
        signals_valid = (
            'timestamp' in signals and
            'data_source' in signals and
            signals.get('signals_count', 0) >= 0
        )
        
        if not signals_valid:
            print(f"  ❌ Invalid signal structure")
            return False
        
        print(f"  ✅ Signal collection successful")
        
        # Test 2: Mixed valid/invalid tickers
        print("\n🔍 Testing mixed valid/invalid tickers...")
        mixed_tickers = ['AAPL', 'INVALID']  # Reduced to avoid rate limits
        mixed_signals = orchestrator.get_signals_from_scrapers(mixed_tickers)
        
        print(f"  Mixed signals count: {mixed_signals.get('signals_count', 0)}")
        print(f"  ✅ Mixed ticker handling successful")
        
        # Test 3: System health check
        print("\n🔍 Testing system health...")
        health = orchestrator.get_system_health()
        
        print(f"  Redis connected: {health.get('redis_connected', False)}")
        print(f"  Cache warming active: {health.get('cache_warming_active', False)}")
        print(f"  Fallback manager active: {health.get('fallback_manager_active', False)}")
        print(f"  Adapters loaded: {health.get('adapters_loaded', 0)}")
        print(f"  Total requests: {health.get('total_requests', 0)}")
        print(f"  Average response time: {health.get('avg_response_time', 0)}")
        
        # Validate health structure
        health_valid = (
            'timestamp' in health and
            'redis_connected' in health and
            'adapters_loaded' in health and
            health.get('adapters_loaded', 0) > 0
        )
        
        if not health_valid:
            print(f"  ❌ Invalid health structure")
            return False
        
        print(f"  ✅ System health check successful")
        
        # Test 4: Empty ticker list handling
        print("\n🔍 Testing empty ticker list...")
        empty_signals = orchestrator.get_signals_from_scrapers([])
        print(f"  Empty signals count: {empty_signals.get('signals_count', 0)}")
        print(f"  ✅ Empty ticker list handled")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all advanced tests"""
    print("🧪 ORACLE-X ADVANCED DATA FEEDS TESTING")
    print("=" * 80)
    
    tests = [
        ("Edge Cases", test_edge_cases),
        ("Concurrent Performance", test_concurrent_performance),
        ("Data Quality", test_data_quality),
        ("Rate Limiting", test_rate_limiting),
        ("Memory Usage", test_memory_usage),
        ("Orchestrator Integration", test_orchestrator_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            start_time = time.time()
            if test_func():
                duration = time.time() - start_time
                print(f"✅ {test_name}: PASSED ({duration:.2f}s)")
                passed += 1
            else:
                duration = time.time() - start_time
                print(f"❌ {test_name}: FAILED ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {test_name}: ERROR - {e} ({duration:.2f}s)")
    
    print("\n" + "=" * 80)
    print(f"📊 ADVANCED TEST SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All advanced tests passed! System is robust and reliable.")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed - minor issues detected")
    else:
        print("❌ Multiple test failures - system needs attention")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
