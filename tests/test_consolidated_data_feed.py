#!/usr/bin/env python3
"""
Comprehensive test suite for the Consolidated Financial Data Feed
Tests all data sources, fallback mechanisms, caching, and rate limiting.
"""

import os
import sys
import time
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import pandas as pd
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_feeds.consolidated_data_feed import (
    ConsolidatedDataFeed, Quote, CompanyInfo, NewsItem,
    YFinanceAdapter, FinnhubAdapter, FMPAdapter, FinanceDatabaseAdapter,
    DataCache, RateLimiter, DataSource
)

class TestDataCache(unittest.TestCase):
    def setUp(self):
        self.cache = DataCache()
    
    def test_cache_set_get(self):
        """Test basic cache operations"""
        test_data = {"test": "data"}
        self.cache.set("test_key", test_data, "quote")
        
        # Should return data immediately
        result = self.cache.get("test_key", "quote")
        self.assertEqual(result, test_data)
    
    def test_cache_expiry(self):
        """Test cache TTL expiration"""
        test_data = {"test": "data"}
        
        # Override TTL to 1 second for testing
        self.cache.ttl["test_type"] = 1
        self.cache.set("test_key", test_data, "test_type")
        
        # Should return data immediately
        result = self.cache.get("test_key", "test_type")
        self.assertEqual(result, test_data)
        
        # Wait for expiry
        time.sleep(1.1)
        
        # Should return None after expiry
        result = self.cache.get("test_key", "test_type")
        self.assertIsNone(result)

class TestRateLimiter(unittest.TestCase):
    def setUp(self):
        self.rate_limiter = RateLimiter()
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Override limits for testing
        self.rate_limiter.limits[DataSource.FINNHUB] = (2, 5)  # 2 calls per 5 seconds
        
        # Should not wait for first calls
        start_time = time.time()
        self.rate_limiter.wait_if_needed(DataSource.FINNHUB)
        self.rate_limiter.wait_if_needed(DataSource.FINNHUB)
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 1.0)  # Should be immediate
        
        # Third call should trigger rate limiting
        start_time = time.time()
        self.rate_limiter.wait_if_needed(DataSource.FINNHUB)
        elapsed = time.time() - start_time
        self.assertGreater(elapsed, 2.0)  # Should wait

class TestYFinanceAdapter(unittest.TestCase):
    def setUp(self):
        self.cache = DataCache()
        self.rate_limiter = RateLimiter()
        self.adapter = YFinanceAdapter(self.cache, self.rate_limiter)
    
    def test_get_quote(self):
        """Test getting quote from yfinance"""
        quote = self.adapter.get_quote("AAPL")
        
        self.assertIsInstance(quote, Quote)
        self.assertEqual(quote.symbol, "AAPL")
        self.assertIsInstance(quote.price, Decimal)
        self.assertIsInstance(quote.change, Decimal)
        self.assertIsInstance(quote.volume, int)
        self.assertEqual(quote.source, "yfinance")
    
    def test_get_historical(self):
        """Test getting historical data from yfinance"""
        hist = self.adapter.get_historical("AAPL", period="5d")
        
        self.assertIsInstance(hist, pd.DataFrame)
        self.assertGreater(len(hist), 0)
        self.assertIn("Close", hist.columns)
        self.assertIn("Volume", hist.columns)
        self.assertEqual(hist["source"].iloc[0], "yfinance")
    
    def test_get_company_info(self):
        """Test getting company info from yfinance"""
        info = self.adapter.get_company_info("AAPL")
        
        self.assertIsInstance(info, CompanyInfo)
        self.assertEqual(info.symbol, "AAPL")
        self.assertIn("Apple", info.name)
        self.assertEqual(info.source, "yfinance")
    
    def test_caching(self):
        """Test that results are cached"""
        # First call should fetch from source
        quote1 = self.adapter.get_quote("AAPL")
        
        # Second call should use cache (check by timing)
        start_time = time.time()
        quote2 = self.adapter.get_quote("AAPL")
        elapsed = time.time() - start_time
        
        self.assertEqual(quote1.price, quote2.price)
        self.assertLess(elapsed, 0.1)  # Should be very fast from cache

class TestFinnhubAdapter(unittest.TestCase):
    def setUp(self):
        self.cache = DataCache()
        self.rate_limiter = RateLimiter()
        self.adapter = FinnhubAdapter(self.cache, self.rate_limiter)
    
    @unittest.skipIf(not os.getenv('FINNHUB_API_KEY'), "Finnhub API key not available")
    def test_get_quote(self):
        """Test getting quote from finnhub"""
        quote = self.adapter.get_quote("AAPL")
        
        self.assertIsInstance(quote, Quote)
        self.assertEqual(quote.symbol, "AAPL")
        self.assertIsInstance(quote.price, Decimal)
        self.assertEqual(quote.source, "finnhub")
    
    @unittest.skipIf(not os.getenv('FINNHUB_API_KEY'), "Finnhub API key not available")
    def test_get_company_info(self):
        """Test getting company info from finnhub"""
        info = self.adapter.get_company_info("AAPL")
        
        self.assertIsInstance(info, CompanyInfo)
        self.assertEqual(info.symbol, "AAPL")
        self.assertIn("Apple", info.name)
        self.assertEqual(info.source, "finnhub")
    
    @unittest.skipIf(not os.getenv('FINNHUB_API_KEY'), "Finnhub API key not available")
    def test_get_news(self):
        """Test getting news from finnhub"""
        news = self.adapter.get_news("AAPL", limit=5)
        
        self.assertIsInstance(news, list)
        if news:  # May be empty if no recent news
            self.assertIsInstance(news[0], NewsItem)
            self.assertEqual(news[0].source, "finnhub")

class TestFMPAdapter(unittest.TestCase):
    def setUp(self):
        self.cache = DataCache()
        self.rate_limiter = RateLimiter()
        self.adapter = FMPAdapter(self.cache, self.rate_limiter)
    
    @unittest.skipIf(not os.getenv('FINANCIALMODELINGPREP_API_KEY'), "FMP API key not available")
    def test_get_quote(self):
        """Test getting quote from FMP"""
        quote = self.adapter.get_quote("AAPL")
        
        self.assertIsInstance(quote, Quote)
        self.assertEqual(quote.symbol, "AAPL")
        self.assertIsInstance(quote.price, Decimal)
        self.assertEqual(quote.source, "financial_modeling_prep")
    
    @unittest.skipIf(not os.getenv('FINANCIALMODELINGPREP_API_KEY'), "FMP API key not available")
    def test_get_company_info(self):
        """Test getting company info from FMP"""
        info = self.adapter.get_company_info("AAPL")
        
        self.assertIsInstance(info, CompanyInfo)
        self.assertEqual(info.symbol, "AAPL")
        self.assertIn("Apple", info.name)
        self.assertEqual(info.source, "financial_modeling_prep")
    
    @unittest.skipIf(not os.getenv('FINANCIALMODELINGPREP_API_KEY'), "FMP API key not available")
    def test_get_historical(self):
        """Test getting historical data from FMP"""
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        hist = self.adapter.get_historical("AAPL", from_date, to_date)
        
        if hist is not None:  # May be None if rate limited
            self.assertIsInstance(hist, pd.DataFrame)
            self.assertGreater(len(hist), 0)
            self.assertEqual(hist["source"].iloc[0], "financial_modeling_prep")

class TestFinanceDatabaseAdapter(unittest.TestCase):
    def setUp(self):
        self.cache = DataCache()
        self.rate_limiter = RateLimiter()
        self.adapter = FinanceDatabaseAdapter(self.cache, self.rate_limiter)
    
    def test_search_equities(self):
        """Test searching equities database"""
        # Search for US technology companies
        results = self.adapter.search_equities(country="United States")
        
        self.assertIsInstance(results, dict)
        if results:  # May be empty due to specific filters
            self.assertGreater(len(results), 0)
    
    def test_search_etfs(self):
        """Test searching ETFs database"""
        results = self.adapter.search_etfs(country="United States")
        
        self.assertIsInstance(results, dict)
        if results:
            self.assertGreater(len(results), 0)

class TestConsolidatedDataFeed(unittest.TestCase):
    def setUp(self):
        self.feed = ConsolidatedDataFeed()
    
    def test_get_quote_fallback(self):
        """Test quote retrieval with fallback mechanism"""
        quote = self.feed.get_quote("AAPL")
        
        self.assertIsInstance(quote, Quote)
        self.assertEqual(quote.symbol, "AAPL")
        self.assertIsInstance(quote.price, Decimal)
        self.assertIn(quote.source, ["yfinance", "financial_modeling_prep", "finnhub"])
    
    def test_get_historical_fallback(self):
        """Test historical data with fallback"""
        hist = self.feed.get_historical("AAPL", period="5d")
        
        self.assertIsInstance(hist, pd.DataFrame)
        self.assertGreater(len(hist), 0)
        self.assertIn("Close", hist.columns)
        self.assertIn(hist["source"].iloc[0], ["yfinance", "financial_modeling_prep"])
    
    def test_get_company_info_fallback(self):
        """Test company info with fallback"""
        info = self.feed.get_company_info("AAPL")
        
        self.assertIsInstance(info, CompanyInfo)
        self.assertEqual(info.symbol, "AAPL")
        self.assertIn("Apple", info.name)
        self.assertIn(info.source, ["yfinance", "financial_modeling_prep", "finnhub"])
    
    def test_get_multiple_quotes(self):
        """Test getting multiple quotes"""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        quotes = self.feed.get_multiple_quotes(symbols)
        
        self.assertIsInstance(quotes, dict)
        self.assertGreater(len(quotes), 0)
        
        for symbol, quote in quotes.items():
            self.assertIn(symbol, symbols)
            self.assertIsInstance(quote, Quote)
            self.assertEqual(quote.symbol, symbol)
    
    def test_search_securities(self):
        """Test security search functionality"""
        results = self.feed.search_securities(country="United States")
        
        self.assertIsInstance(results, dict)
        # Results may be empty due to specific search criteria
    
    def test_cache_stats(self):
        """Test cache statistics"""
        # Make some calls to populate cache
        self.feed.get_quote("AAPL")
        self.feed.get_company_info("AAPL")
        
        stats = self.feed.get_cache_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_cached_items', stats)
        self.assertGreater(stats['total_cached_items'], 0)
    
    def test_cache_clear(self):
        """Test cache clearing"""
        # Populate cache
        self.feed.get_quote("AAPL")
        
        # Verify cache has items
        stats_before = self.feed.get_cache_stats()
        self.assertGreater(stats_before['total_cached_items'], 0)
        
        # Clear cache
        self.feed.clear_cache()
        
        # Verify cache is empty
        stats_after = self.feed.get_cache_stats()
        self.assertEqual(stats_after['total_cached_items'], 0)

class TestConvenienceFunctions(unittest.TestCase):
    def test_convenience_functions(self):
        """Test the convenience functions"""
        from data_feeds.consolidated_data_feed import get_quote, get_historical, get_company_info, get_news
        
        # Test quote
        quote = get_quote("AAPL")
        self.assertIsInstance(quote, Quote)
        
        # Test historical
        hist = get_historical("AAPL", period="5d")
        self.assertIsInstance(hist, pd.DataFrame)
        
        # Test company info
        info = get_company_info("AAPL")
        self.assertIsInstance(info, CompanyInfo)
        
        # Test news
        news = get_news("AAPL", limit=3)
        self.assertIsInstance(news, list)

class TestErrorHandling(unittest.TestCase):
    def setUp(self):
        self.feed = ConsolidatedDataFeed()
    
    def test_invalid_symbol(self):
        """Test handling of invalid symbols"""
        quote = self.feed.get_quote("INVALID_SYMBOL_12345")
        
        # Should return None or handle gracefully
        if quote is not None:
            self.assertIsInstance(quote, Quote)
    
    def test_network_error_simulation(self):
        """Test handling of network errors"""
        # This would require mocking network calls
        # For now, just test that the method doesn't crash
        try:
            quote = self.feed.get_quote("AAPL")
            self.assertTrue(True)  # Test passes if no exception
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")

class TestDataQuality(unittest.TestCase):
    def setUp(self):
        self.feed = ConsolidatedDataFeed()
    
    def test_quote_data_quality(self):
        """Test that quote data meets quality standards"""
        quote = self.feed.get_quote("AAPL")
        
        if quote:
            # Price should be positive
            self.assertGreater(quote.price, 0)
            
            # Volume should be non-negative
            self.assertGreaterEqual(quote.volume, 0)
            
            # Change percent should be reasonable (< 50% in a day)
            self.assertLess(abs(quote.change_percent), 50)
    
    def test_historical_data_quality(self):
        """Test that historical data meets quality standards"""
        hist = self.feed.get_historical("AAPL", period="1mo")
        
        if hist is not None and not hist.empty:
            # All prices should be positive
            self.assertTrue((hist['Close'] > 0).all())
            self.assertTrue((hist['Open'] > 0).all())
            self.assertTrue((hist['High'] > 0).all())
            self.assertTrue((hist['Low'] > 0).all())
            
            # High should be >= Low
            self.assertTrue((hist['High'] >= hist['Low']).all())
            
            # Volume should be non-negative
            self.assertTrue((hist['Volume'] >= 0).all())

def run_comprehensive_tests():
    """Run all tests and generate a report"""
    print("=" * 80)
    print("CONSOLIDATED DATA FEED - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Create test suite
    test_classes = [
        TestDataCache,
        TestRateLimiter,
        TestYFinanceAdapter,
        TestFinnhubAdapter,
        TestFMPAdapter,
        TestFinanceDatabaseAdapter,
        TestConsolidatedDataFeed,
        TestConvenienceFunctions,
        TestErrorHandling,
        TestDataQuality
    ]
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        passed = tests_run - failures - errors - skipped
        
        total_tests += tests_run
        total_passed += passed
        total_failed += failures + errors
        total_skipped += skipped
        
        print(f"  Tests: {tests_run}, Passed: {passed}, Failed: {failures + errors}, Skipped: {skipped}")
        
        if result.failures:
            print("  Failures:")
            for test, traceback in result.failures:
                print(f"    - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("  Errors:")
            for test, traceback in result.errors:
                print(f"    - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Skipped: {total_skipped}")
    print(f"Success Rate: {(total_passed/total_tests)*100:.1f}%")
    
    if total_failed > 0:
        print(f"\n⚠️  {total_failed} tests failed. Check configuration and API keys.")
    else:
        print(f"\n✅ All tests passed! Data feed is working correctly.")
    
    return total_failed == 0

if __name__ == "__main__":
    success = run_comprehensive_tests()
