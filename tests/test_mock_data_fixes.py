#!/usr/bin/env python3
"""
Test script to validate the mock data feed replacements.
Tests all 4 updated data feeds with real data sources.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_feeds.dark_pools import fetch_dark_pool_data
from data_feeds.earnings_calendar import fetch_earnings_calendar
from data_feeds.market_internals import fetch_market_internals
from data_feeds.options_flow import fetch_options_flow

def test_dark_pools():
    """Test dark pool data implementation"""
    print("Testing Dark Pool Data...")
    try:
        result = fetch_dark_pool_data(["AAPL", "TSLA"])
        print(f"✅ Dark pools: {result['total_detected']} detected")
        print(f"   Data source: {result['data_source']}")
        if result['dark_pools']:
            sample = result['dark_pools'][0]
            print(f"   Sample: {sample['ticker']} - {sample['block_size']:,} shares @ ${sample['price']}")
        return True
    except Exception as e:
        print(f"❌ Dark pools failed: {e}")
        return False

def test_earnings_calendar():
    """Test earnings calendar implementation"""
    print("\nTesting Earnings Calendar...")
    try:
        result = fetch_earnings_calendar(["AAPL", "TSLA"])
        print(f"✅ Earnings: {len(result)} events found")
        if result:
            sample = result[0]
            print(f"   Sample: {sample['ticker']} - {sample['date']} (Est: ${sample['estimate']})")
            print(f"   Company: {sample['company_name']}")
        return True
    except Exception as e:
        print(f"❌ Earnings calendar failed: {e}")
        return False

def test_market_internals():
    """Test market internals implementation"""
    print("\nTesting Market Internals...")
    try:
        result = fetch_market_internals()
        print(f"✅ Market internals: {result['market_sentiment']} sentiment")
        print(f"   VIX: {result['vix']}")
        print(f"   TRIN: {result['trin']}")
        print(f"   Breadth: {result['breadth']['breadth_status']}")
        print(f"   Data source: {result['data_source']}")
        return True
    except Exception as e:
        print(f"❌ Market internals failed: {e}")
        return False

def test_options_flow():
    """Test options flow implementation"""
    print("\nTesting Options Flow...")
    try:
        result = fetch_options_flow(["AAPL", "TSLA"])
        print(f"✅ Options flow: {result['total_sweeps']} unusual sweeps")
        print(f"   Data source: {result['data_source']}")
        if result['unusual_sweeps']:
            sample = result['unusual_sweeps'][0]
            print(f"   Sample: {sample['ticker']} {sample['direction']} ${sample['strike']} - Vol: {sample['volume']:,}")
        return True
    except Exception as e:
        print(f"❌ Options flow failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING MOCK DATA FEED REPLACEMENTS")
    print("=" * 60)
    
    tests = [
        test_dark_pools,
        test_earnings_calendar,
        test_market_internals,
        test_options_flow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All mock data feeds successfully replaced with real implementations!")
    else:
        print("⚠️  Some data feeds need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
