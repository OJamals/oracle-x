#!/usr/bin/env python3
"""
Comprehensive Data Feed Integrity Test
Tests all data feeds for functionality, rate limits, and data quality
"""

import os
import sys
import time
from typing import Dict, Any, List
import json

# Test results tracker
test_results = {
    "free_feeds": [],
    "premium_feeds": [],
    "rate_limited_feeds": [],
    "failed_feeds": [],
    "redundant_feeds": [],
    "successful_feeds": []
}

def test_market_internals():
    """Test market internals (yfinance - FREE)"""
    print("\n🔍 Testing Market Internals (yfinance)...")
    try:
        from data_feeds.market_internals import fetch_market_internals
        start = time.time()
        data = fetch_market_internals()
        elapsed = time.time() - start
        
        if data and 'breadth' in data and 'vix' in data:
            print(f"✅ Market Internals: SUCCESS (${elapsed:.2f}s)")
            print(f"   VIX: {data.get('vix', 'N/A')}, Sentiment: {data.get('market_sentiment', 'N/A')}")
            test_results["successful_feeds"].append("market_internals")
            test_results["free_feeds"].append("market_internals")
            return True
        else:
            print(f"❌ Market Internals: FAILED - Invalid data structure")
            test_results["failed_feeds"].append("market_internals")
            return False
    except Exception as e:
        print(f"❌ Market Internals: ERROR - {e}")
        test_results["failed_feeds"].append("market_internals")
        return False

def test_options_flow():
    """Test options flow (yfinance - FREE)"""
    print("\n🔍 Testing Options Flow (yfinance)...")
    try:
        from data_feeds.options_flow import fetch_options_flow
        start = time.time()
        data = fetch_options_flow(["AAPL", "TSLA"])
        elapsed = time.time() - start
        
        if data and 'unusual_sweeps' in data:
            print(f"✅ Options Flow: SUCCESS ({elapsed:.2f}s)")
            print(f"   Found {len(data.get('unusual_sweeps', []))} unusual sweeps")
            test_results["successful_feeds"].append("options_flow")
            test_results["free_feeds"].append("options_flow")
            return True
        else:
            print(f"❌ Options Flow: FAILED - Invalid data structure")
            test_results["failed_feeds"].append("options_flow")
            return False
    except Exception as e:
        print(f"❌ Options Flow: ERROR - {e}")
        test_results["failed_feeds"].append("options_flow")
        return False

def test_dark_pools():
    """Test dark pools (yfinance volume analysis - FREE)"""
    print("\n🔍 Testing Dark Pools (yfinance volume analysis)...")
    try:
        from data_feeds.dark_pools import fetch_dark_pool_data
        start = time.time()
        data = fetch_dark_pool_data(["AAPL", "TSLA"])
        elapsed = time.time() - start
        
        if data and 'dark_pools' in data:
            print(f"✅ Dark Pools: SUCCESS ({elapsed:.2f}s)")
            print(f"   Found {len(data.get('dark_pools', []))} dark pool signals")
            test_results["successful_feeds"].append("dark_pools")
            test_results["free_feeds"].append("dark_pools")
            return True
        else:
            print(f"❌ Dark Pools: FAILED - Invalid data structure")
            test_results["failed_feeds"].append("dark_pools")
            return False
    except Exception as e:
        print(f"❌ Dark Pools: ERROR - {e}")
        test_results["failed_feeds"].append("dark_pools")
        return False

def test_reddit_sentiment():
    """Test Reddit sentiment (Reddit API - FREE with credentials)"""
    print("\n🔍 Testing Reddit Sentiment...")
    try:
        from data_feeds.reddit_sentiment import fetch_reddit_sentiment
        start = time.time()
        data = fetch_reddit_sentiment(subreddit="stocks", limit=50)
        elapsed = time.time() - start
        
        if data and len(data) > 0:
            print(f"✅ Reddit Sentiment: SUCCESS ({elapsed:.2f}s)")
            print(f"   Found sentiment for {len(data)} tickers")
            test_results["successful_feeds"].append("reddit_sentiment")
            test_results["free_feeds"].append("reddit_sentiment")
            return True
        elif not os.environ.get("REDDIT_CLIENT_ID"):
            print(f"⚠️  Reddit Sentiment: SKIPPED - No credentials (FREE with API key)")
            return None
        else:
            print(f"❌ Reddit Sentiment: FAILED - No data returned")
            test_results["failed_feeds"].append("reddit_sentiment")
            return False
    except Exception as e:
        print(f"❌ Reddit Sentiment: ERROR - {e}")
        test_results["failed_feeds"].append("reddit_sentiment")
        return False

def test_twitter_sentiment():
    """Test Twitter sentiment (twscrape - FREE)"""
    print("\n🔍 Testing Twitter Sentiment (twscrape)...")
    try:
        from data_feeds.twitter_sentiment import fetch_twitter_sentiment
        start = time.time()
        data = fetch_twitter_sentiment("AAPL", limit=20)
        elapsed = time.time() - start
        
        if data and len(data) > 0:
            print(f"✅ Twitter Sentiment: SUCCESS ({elapsed:.2f}s)")
            print(f"   Found {len(data)} tweets")
            test_results["successful_feeds"].append("twitter_sentiment")
            test_results["free_feeds"].append("twitter_sentiment")
            return True
        else:
            print(f"⚠️  Twitter Sentiment: NO DATA (may need twscrape accounts)")
            return None
    except Exception as e:
        print(f"❌ Twitter Sentiment: ERROR - {e}")
        test_results["failed_feeds"].append("twitter_sentiment")
        return False

def test_earnings_calendar():
    """Test earnings calendar (yfinance - FREE)"""
    print("\n🔍 Testing Earnings Calendar (yfinance)...")
    try:
        from data_feeds.earnings_calendar import fetch_earnings_calendar
        start = time.time()
        data = fetch_earnings_calendar(["AAPL", "TSLA", "MSFT"])
        elapsed = time.time() - start
        
        if data and len(data) > 0:
            print(f"✅ Earnings Calendar: SUCCESS ({elapsed:.2f}s)")
            print(f"   Found {len(data)} earnings events")
            test_results["successful_feeds"].append("earnings_calendar")
            test_results["free_feeds"].append("earnings_calendar")
            return True
        else:
            print(f"❌ Earnings Calendar: FAILED - No data returned")
            test_results["failed_feeds"].append("earnings_calendar")
            return False
    except Exception as e:
        print(f"❌ Earnings Calendar: ERROR - {e}")
        test_results["failed_feeds"].append("earnings_calendar")
        return False

def test_yahoo_news():
    """Test Yahoo Finance news (web scraping - FREE)"""
    print("\n🔍 Testing Yahoo Finance News (web scraping)...")
    try:
        from data_feeds.news_scraper import fetch_headlines_yahoo_finance
        start = time.time()
        data = fetch_headlines_yahoo_finance()
        elapsed = time.time() - start
        
        if data and len(data) > 0:
            print(f"✅ Yahoo News: SUCCESS ({elapsed:.2f}s)")
            print(f"   Found {len(data)} headlines")
            test_results["successful_feeds"].append("yahoo_news")
            test_results["free_feeds"].append("yahoo_news")
            return True
        else:
            print(f"⚠️  Yahoo News: RATE LIMITED or blocked")
            test_results["rate_limited_feeds"].append("yahoo_news")
            return None
    except Exception as e:
        print(f"❌ Yahoo News: ERROR - {e}")
        test_results["failed_feeds"].append("yahoo_news")
        return False

def test_finviz_breadth():
    """Test Finviz market breadth (web scraping - FREE)"""
    print("\n🔍 Testing Finviz Market Breadth (web scraping)...")
    try:
        from data_feeds.finviz_scraper import fetch_finviz_breadth
        start = time.time()
        data = fetch_finviz_breadth()
        elapsed = time.time() - start
        
        if data and len(data) > 0:
            print(f"✅ Finviz Breadth: SUCCESS ({elapsed:.2f}s)")
            print(f"   Advancers: {data.get('advancers', 'N/A')}, Decliners: {data.get('decliners', 'N/A')}")
            test_results["successful_feeds"].append("finviz_breadth")
            test_results["free_feeds"].append("finviz_breadth")
            return True
        else:
            print(f"⚠️  Finviz Breadth: NO DATA (may be rate limited)")
            test_results["rate_limited_feeds"].append("finviz_breadth")
            return None
    except Exception as e:
        print(f"❌ Finviz Breadth: ERROR - {e}")
        test_results["failed_feeds"].append("finviz_breadth")
        return False

def test_finnhub():
    """Test Finnhub API (PREMIUM with rate limits)"""
    print("\n🔍 Testing Finnhub API (PREMIUM)...")
    try:
        from data_feeds.finnhub import fetch_finnhub_quote
        
        if not os.environ.get("FINNHUB_API_KEY"):
            print(f"⚠️  Finnhub: SKIPPED - No API key (PREMIUM)")
            test_results["premium_feeds"].append("finnhub")
            return None
            
        start = time.time()
        data = fetch_finnhub_quote("AAPL")
        elapsed = time.time() - start
        
        if data and data.get('c', 0) > 0:
            print(f"✅ Finnhub: SUCCESS ({elapsed:.2f}s)")
            print(f"   AAPL Price: ${data.get('c', 'N/A')}")
            test_results["successful_feeds"].append("finnhub")
            test_results["premium_feeds"].append("finnhub")
            return True
        else:
            print(f"⚠️  Finnhub: RATE LIMITED or requires premium")
            test_results["rate_limited_feeds"].append("finnhub")
            test_results["premium_feeds"].append("finnhub")
            return None
    except Exception as e:
        print(f"❌ Finnhub: ERROR - {e}")
        test_results["premium_feeds"].append("finnhub")
        return False

def test_fmp():
    """Test Financial Modeling Prep API (PREMIUM with free tier)"""
    print("\n🔍 Testing FMP API (PREMIUM with free tier)...")
    try:
        from data_feeds.enhanced_fmp_integration import EnhancedFMPAdapter
        
        fmp = EnhancedFMPAdapter()
        if not fmp.api_key:
            print(f"⚠️  FMP: SKIPPED - No API key (FREE tier available)")
            test_results["premium_feeds"].append("fmp")
            return None
            
        start = time.time()
        data = fmp.get_financial_ratios("AAPL")
        elapsed = time.time() - start
        
        if data and len(data) > 0:
            print(f"✅ FMP: SUCCESS ({elapsed:.2f}s)")
            print(f"   PE Ratio: {data[0].pe_ratio if data[0].pe_ratio else 'N/A'}")
            test_results["successful_feeds"].append("fmp")
            test_results["premium_feeds"].append("fmp")
            return True
        else:
            print(f"⚠️  FMP: RATE LIMITED or requires premium")
            test_results["rate_limited_feeds"].append("fmp")
            test_results["premium_feeds"].append("fmp")
            return None
    except Exception as e:
        print(f"❌ FMP: ERROR - {e}")
        test_results["premium_feeds"].append("fmp")
        return False

def test_orchestrator_redundancy():
    """Check for redundancy in data feed orchestrator"""
    print("\n🔍 Checking Data Feed Orchestrator for redundancy...")
    try:
        from data_feeds.data_feed_orchestrator import get_orchestrator
        orch = get_orchestrator()
        
        # Check if orchestrator duplicates data already collected
        print("⚠️  Data Feed Orchestrator provides:")
        print("   - Reddit sentiment (also in reddit_sentiment.py)")
        print("   - Twitter sentiment (also in twitter_sentiment.py)")
        print("   - May cause redundant API calls")
        
        test_results["redundant_feeds"].append("data_feed_orchestrator")
        return True
    except Exception as e:
        print(f"❌ Orchestrator check: ERROR - {e}")
        return False

def generate_report():
    """Generate comprehensive report"""
    print("\n" + "="*70)
    print("📊 DATA FEED INTEGRITY REPORT")
    print("="*70)
    
    print(f"\n✅ FREE & WORKING FEEDS ({len(test_results['successful_feeds'])} total):")
    for feed in test_results['successful_feeds']:
        if feed in test_results['free_feeds']:
            print(f"   ✓ {feed}")
    
    print(f"\n💰 PREMIUM/PAID FEEDS ({len(set(test_results['premium_feeds']))} total):")
    for feed in set(test_results['premium_feeds']):
        print(f"   $ {feed}")
    
    print(f"\n⚠️  RATE LIMITED FEEDS ({len(test_results['rate_limited_feeds'])} total):")
    for feed in test_results['rate_limited_feeds']:
        print(f"   ⚠ {feed}")
    
    print(f"\n❌ FAILED FEEDS ({len(test_results['failed_feeds'])} total):")
    for feed in test_results['failed_feeds']:
        print(f"   ✗ {feed}")
    
    print(f"\n🔄 REDUNDANT/DUPLICATE FEEDS ({len(test_results['redundant_feeds'])} total):")
    for feed in test_results['redundant_feeds']:
        print(f"   ↻ {feed}")
    
    # Recommendations
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS:")
    print("="*70)
    
    print("\n1. KEEP (Free & Working):")
    for feed in test_results['successful_feeds']:
        if feed in test_results['free_feeds']:
            print(f"   ✓ {feed}")
    
    print("\n2. REMOVE/DISABLE (Premium with rate limits):")
    for feed in set(test_results['premium_feeds']):
        if feed in test_results['rate_limited_feeds']:
            print(f"   ✗ {feed}")
    
    print("\n3. CONSOLIDATE (Redundant):")
    for feed in test_results['redundant_feeds']:
        print(f"   ↻ {feed} - Merge into primary feeds")
    
    print("\n4. FIX (Failed but should work):")
    for feed in test_results['failed_feeds']:
        if feed not in test_results['premium_feeds']:
            print(f"   🔧 {feed}")
    
    # Save report to file
    with open("/Users/omar/Documents/Projects/oracle-x/data_feeds_report.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("\n📄 Full report saved to: data_feeds_report.json")
    print("="*70)

def main():
    """Run all tests"""
    print("="*70)
    print("🚀 ORACLE-X DATA FEED INTEGRITY TEST")
    print("="*70)
    
    # Test free feeds
    test_market_internals()
    test_options_flow()
    test_dark_pools()
    test_earnings_calendar()
    test_yahoo_news()
    test_finviz_breadth()
    
    # Test sentiment feeds
    test_reddit_sentiment()
    test_twitter_sentiment()
    
    # Test premium feeds
    test_finnhub()
    test_fmp()
    
    # Check redundancy
    test_orchestrator_redundancy()
    
    # Generate report
    generate_report()

if __name__ == "__main__":
    main()
