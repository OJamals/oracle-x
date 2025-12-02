#!/usr/bin/env python3
"""
Comprehensive Data Feed Testing and Optimization Script

This script systematically tests each data feed adapter and the orchestrator,
validating data quality, error handling, and performance.
"""

import os
import sys
import json
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Global list to store test results
test_results = []

# Mock services for testing
class MockRateLimiter:
    def wait_if_needed(self, source): pass
    
class MockPerformanceTracker:
    def record_success(self, *args, **kwargs): pass
    def record_error(self, *args, **kwargs): pass

def log_test(test_name: str, status: str, details: Optional[str] = None, timing: Optional[float] = None):
    """Log test results in a structured format"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "timestamp": timestamp,
        "test": test_name,
        "status": status,
        "details": details,
        "timing_ms": round(timing * 1000, 2) if timing else None
    }
    test_results.append(result)
    print(json.dumps(result))

def test_reddit_adapter():
    """Test Reddit sentiment adapter"""
    try:
        start_time = time.time()
        from data_feeds.reddit_sentiment import fetch_reddit_sentiment
        
        # Test with a small limit to be fast
        result = fetch_reddit_sentiment('stocks', 20)
        timing = time.time() - start_time
        
        if result:
            tickers_found = len(result)
            sample_ticker = list(result.keys())[0] if result else None
            sample_data = result[sample_ticker] if sample_ticker else None
            
            details = {
                "tickers_found": tickers_found,
                "sample_ticker": sample_ticker,
                "sample_mentions": sample_data.get('mentions', 0) if sample_data else 0,
                "sample_sentiment": sample_data.get('sentiment_score', 0) if sample_data else 0,
                "has_sample_texts": bool(sample_data.get('sample_texts', [])) if sample_data else False
            }
            log_test("reddit_adapter", "PASS", json.dumps(details), timing)
        else:
            log_test("reddit_adapter", "FAIL", "No data returned", timing)
            
    except Exception as e:
        log_test("reddit_adapter", "ERROR", str(e))

def test_finviz_adapter():
    """Test FinViz adapter"""
    try:
        from data_feeds.finviz_adapter import FinVizAdapter
        adapter = FinVizAdapter()
        
        # Test market breadth
        start_time = time.time()
        breadth = adapter.get_market_breadth()
        breadth_timing = time.time() - start_time
        
        if breadth:
            details = {
                "advancers": breadth.advancers,
                "decliners": breadth.decliners,
                "new_highs": breadth.new_highs,
                "new_lows": breadth.new_lows
            }
            log_test("finviz_market_breadth", "PASS", json.dumps(details), breadth_timing)
        else:
            log_test("finviz_market_breadth", "FAIL", "No breadth data", breadth_timing)
        
        # Test sector performance
        start_time = time.time()
        sectors = adapter.get_sector_performance()
        sector_timing = time.time() - start_time
        
        if sectors and len(sectors) > 0:
            details = {
                "sectors_count": len(sectors),
                "sample_sector": sectors[0].group_name,
                "sample_1d_perf": float(sectors[0].perf_1d) if sectors[0].perf_1d else None
            }
            log_test("finviz_sector_performance", "PASS", json.dumps(details), sector_timing)
        else:
            log_test("finviz_sector_performance", "FAIL", "No sector data", sector_timing)
            
    except Exception as e:
        log_test("finviz_adapter", "ERROR", str(e))

def test_twitter_adapter():
    """Test Twitter adapter (uses twscrape, no API keys required)"""
    # Twitter adapter uses twscrape which doesn't require API credentials
    # Skip only if twscrape is not available
    try:
        from twscrape import API
        from data_feeds.twitter_feed import TwitterSentimentFeed
        has_twscrape = True
    except ImportError:
        has_twscrape = False
    
    if not has_twscrape:
        log_test("twitter_adapter", "SKIP", "twscrape not available")
        return
        
    try:
        from data_feeds.twitter_adapter import TwitterAdapter
        
        # TwitterAdapter doesn't require cache/rate_limiter parameters
        adapter = TwitterAdapter()
        
        start_time = time.time()
        sentiment = adapter.get_sentiment('AAPL')
        timing = time.time() - start_time
        
        if sentiment and hasattr(sentiment, 'sentiment_score'):
            details = {
                "symbol": sentiment.symbol,
                "sentiment_score": float(sentiment.sentiment_score),
                "confidence": float(sentiment.confidence),
                "sample_size": getattr(sentiment, 'sample_size', None)
            }
            log_test("twitter_sentiment", "PASS", json.dumps(details), timing)
        else:
            log_test("twitter_sentiment", "FAIL", "No sentiment data", timing)
            
    except Exception as e:
        log_test("twitter_adapter", "ERROR", str(e))

def test_investiny_adapter():
    """Test Investiny adapter"""
    try:
        from data_feeds.investiny_adapter import get_history, search_investing_id
        
        start_time = time.time()
        
        # Test searching for a symbol with debug
        print("DEBUG: Testing Investiny search for AAPL...")
        investing_id = search_investing_id('AAPL')
        print(f"DEBUG: Investiny search result: {investing_id}")
        timing = time.time() - start_time
        
        if investing_id:
            log_test("investiny_search", "PASS", json.dumps({"investing_id": investing_id}), timing)
            
            # Try to get some historical data
            start_time = time.time()
            df = get_history('AAPL', start_date='2024-01-01', end_date='2024-01-31')
            timing = time.time() - start_time
            
            if df is not None and len(df) > 0:
                log_test("investiny_history", "PASS", json.dumps({"rows": len(df), "columns": list(df.columns)}), timing)
            else:
                log_test("investiny_history", "FAIL", "No historical data", timing)
        else:
            log_test("investiny_search", "FAIL", "Could not find investing_id for AAPL", timing)
            
    except ImportError:
        log_test("investiny_adapter", "SKIP", "Investiny adapter dependencies not available")
    except Exception as e:
        log_test("investiny_adapter", "ERROR", str(e))

def test_adapter_wrappers():
    """Test the adapter wrapper classes"""
    try:
        from data_feeds.adapter_wrappers import (
            YFinanceAdapterWrapper, 
            FMPAdapterWrapper, 
            FinnhubAdapterWrapper,
            FinanceDatabaseAdapterWrapper
        )
        from data_feeds.cache.cache_service import CacheService
        
        # Local mock classes
        class LocalMockRateLimiter:
            def wait_if_needed(self, source): pass
            
        class LocalMockPerformanceTracker:
            def record_success(self, *args, **kwargs): pass
            def record_error(self, *args, **kwargs): pass
        
        cache = CacheService()
        rate_limiter = LocalMockRateLimiter()
        performance_tracker = LocalMockPerformanceTracker()
        
        # Test YFinance wrapper
        try:
            # Create a compatibility wrapper for the cache since the consolidated adapter expects a different interface
            class CacheAdapter:
                def __init__(self, cache_service):
                    self.cache_service = cache_service
                    self.ttl = {
                        'quote': 30,
                        'historical_daily': 3600,
                        'historical_intraday': 300,
                        'financials': 86400,
                        'company_info': 604800,
                        'news': 1800,
                    }
                    
                def get(self, key: str, data_type: str):
                    # For testing, just return None to bypass caching
                    return None
                    
                def set(self, key: str, data: Any, data_type: str):
                    # For testing, just ignore set operations to avoid serialization issues
                    pass
            
            cache_adapter = CacheAdapter(cache)
            yf_adapter = YFinanceAdapterWrapper(cache_adapter, rate_limiter, performance_tracker)
            
            start_time = time.time()
            capabilities = yf_adapter.capabilities()
            timing = time.time() - start_time
            
            log_test("yfinance_wrapper_capabilities", "PASS", json.dumps({"capabilities": list(capabilities)}), timing)
            
            # Test fetch_quote
            start_time = time.time()
            quote = yf_adapter.fetch_quote('AAPL')
            timing = time.time() - start_time
            
            if quote and hasattr(quote, 'symbol'):
                details = {
                    "symbol": quote.symbol,
                    "has_price": hasattr(quote, 'price') and quote.price is not None,
                    "has_volume": hasattr(quote, 'volume') and quote.volume is not None
                }
                log_test("yfinance_wrapper_quote", "PASS", json.dumps(details), timing)
            else:
                log_test("yfinance_wrapper_quote", "FAIL", "No quote data", timing)
                
        except Exception as e:
            log_test("yfinance_wrapper", "ERROR", str(e))
        
        # Test FMP wrapper (may need API key)
        try:
            if os.getenv('FINANCIALMODELINGPREP_API_KEY'):
                fmp_adapter = FMPAdapterWrapper(cache, rate_limiter, performance_tracker)
                
                start_time = time.time()
                capabilities = fmp_adapter.capabilities()
                timing = time.time() - start_time
                
                log_test("fmp_wrapper_capabilities", "PASS", json.dumps({"capabilities": list(capabilities)}), timing)
                
                # Test a quote fetch
                start_time = time.time()
                quote = fmp_adapter.fetch_quote('AAPL')
                timing = time.time() - start_time
                
                if quote:
                    details = {
                        "symbol": quote.symbol,
                        "has_price": quote.price is not None and quote.price > 0,
                        "source": getattr(quote, 'source', 'fmp')
                    }
                    log_test("fmp_wrapper_quote", "PASS", json.dumps(details), timing)
                else:
                    log_test("fmp_wrapper_quote", "FAIL", "No quote data", timing)
            else:
                log_test("fmp_wrapper", "SKIP", "FINANCIALMODELINGPREP_API_KEY not configured")
        except Exception as e:
            log_test("fmp_wrapper", "ERROR", str(e))
        
        # Test Finnhub wrapper (may need API key)
        try:
            if os.getenv('FINNHUB_API_KEY'):
                finnhub_adapter = FinnhubAdapterWrapper(cache, rate_limiter, performance_tracker)
                
                start_time = time.time()
                capabilities = finnhub_adapter.capabilities()
                timing = time.time() - start_time
                
                log_test("finnhub_wrapper_capabilities", "PASS", json.dumps({"capabilities": list(capabilities)}), timing)
            else:
                log_test("finnhub_wrapper", "SKIP", "FINNHUB_API_KEY not configured")
        except Exception as e:
            log_test("finnhub_wrapper", "ERROR", str(e))
        
        # Test FinanceDatabase wrapper (should work without API key)
        try:
            findb_adapter = FinanceDatabaseAdapterWrapper(cache, rate_limiter, performance_tracker)
            
            start_time = time.time()
            capabilities = findb_adapter.capabilities()
            timing = time.time() - start_time
            
            log_test("financedb_wrapper_capabilities", "PASS", json.dumps({"capabilities": list(capabilities)}), timing)
            
        except Exception as e:
            log_test("financedb_wrapper", "ERROR", str(e))
            
    except ImportError as e:
        log_test("adapter_wrappers", "SKIP", f"Adapter wrappers not available: {e}")
    except Exception as e:
        log_test("adapter_wrappers", "ERROR", str(e))
    """Test Investiny adapter"""
    try:
        from data_feeds.investiny_adapter import get_history, search_investing_id
        
        start_time = time.time()
        
        # Test searching for a symbol
        investing_id = search_investing_id('AAPL')
        timing = time.time() - start_time
        
        if investing_id:
            log_test("investiny_search", "PASS", json.dumps({"investing_id": investing_id}), timing)
            
            # Try to get some historical data
            start_time = time.time()
            df = get_history('AAPL', start_date='2024-01-01', end_date='2024-01-31')
            timing = time.time() - start_time
            
            if df is not None and len(df) > 0:
                log_test("investiny_history", "PASS", json.dumps({"rows": len(df), "columns": list(df.columns)}), timing)
            else:
                log_test("investiny_history", "FAIL", "No historical data", timing)
        else:
            log_test("investiny_search", "FAIL", "Could not find investing_id for AAPL", timing)
            
    except ImportError:
        log_test("investiny_adapter", "SKIP", "Investiny adapter dependencies not available")
    except Exception as e:
        log_test("investiny_adapter", "ERROR", str(e))
    """Test Twitter adapter (using twscrape)"""
    # Twitter adapter uses twscrape which doesn't require API credentials
    # Check if twscrape is available
    try:
        import twscrape
        has_twscrape = True
    except ImportError:
        has_twscrape = False
    
    if not has_twscrape:
        log_test("twitter_adapter", "SKIP", "twscrape library not available")
        return
        
    try:
        from data_feeds.twitter_adapter import TwitterAdapter
        from data_feeds.cache.cache_service import CacheService
        
        # Mock cache and rate limiter for testing
        cache = CacheService()
        
        class MockRateLimiter:
            def wait_if_needed(self, source): pass
            
        class MockPerformanceTracker:
            def record_success(self, *args, **kwargs): pass
            def record_error(self, *args, **kwargs): pass
        
        adapter = TwitterAdapter()
        
        start_time = time.time()
        sentiment = adapter.get_sentiment('AAPL')
        timing = time.time() - start_time
        
        if sentiment and hasattr(sentiment, 'sentiment_score'):
            details = {
                "symbol": sentiment.symbol,
                "sentiment_score": float(sentiment.sentiment_score),
                "confidence": float(sentiment.confidence),
                "sample_size": getattr(sentiment, 'sample_size', None)
            }
            log_test("twitter_sentiment", "PASS", json.dumps(details), timing)
        else:
            log_test("twitter_sentiment", "FAIL", "No sentiment data", timing)
            
    except Exception as e:
        log_test("twitter_adapter", "ERROR", str(e))
def test_orchestrator_basic():
    """Test basic orchestrator functionality"""
    try:
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        
        # Initialize orchestrator
        start_time = time.time()
        orchestrator = DataFeedOrchestrator()
        init_timing = time.time() - start_time
        log_test("orchestrator_init", "PASS", f"Initialized in {init_timing:.2f}s", init_timing)
        
        # Test quote
        start_time = time.time()
        quote = orchestrator.get_quote('AAPL')
        quote_timing = time.time() - start_time
        
        if quote and quote.price:
            details = {
                "symbol": quote.symbol,
                "price": float(quote.price),
                "source": quote.source,
                "has_timestamp": quote.timestamp is not None
            }
            log_test("orchestrator_quote", "PASS", json.dumps(details), quote_timing)
        else:
            log_test("orchestrator_quote", "FAIL", "No quote data", quote_timing)
            
    except Exception as e:
        log_test("orchestrator_basic", "ERROR", str(e))

def test_orchestrator_sentiment():
    """Test orchestrator sentiment functionality"""
    try:
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        orchestrator = DataFeedOrchestrator()
        
        start_time = time.time()
        sentiment_data = orchestrator.get_sentiment_data('AAPL')
        timing = time.time() - start_time
        
        if sentiment_data:
            sources = list(sentiment_data.keys())
            sample_source = sources[0] if sources else None
            sample_data = sentiment_data[sample_source] if sample_source else None
            
            details = {
                "sources_count": len(sources),
                "sources": sources,
                "sample_source": sample_source,
                "sample_score": getattr(sample_data, 'sentiment_score', None) if sample_data else None,
                "sample_confidence": getattr(sample_data, 'confidence', None) if sample_data else None
            }
            log_test("orchestrator_sentiment", "PASS", json.dumps(details), timing)
        else:
            log_test("orchestrator_sentiment", "FAIL", "No sentiment data", timing)
            
    except Exception as e:
        log_test("orchestrator_sentiment", "ERROR", str(e))

def test_orchestrator_market_data():
    """Test orchestrator market data functionality"""
    try:
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        orchestrator = DataFeedOrchestrator()
        
        start_time = time.time()
        market_data = orchestrator.get_market_data('AAPL', period='1mo', interval='1d')
        timing = time.time() - start_time
        
        if market_data and market_data.data is not None:
            details = {
                "symbol": market_data.symbol,
                "source": market_data.source,
                "rows": len(market_data.data),
                "columns": list(market_data.data.columns) if hasattr(market_data.data, 'columns') else [],
                "timeframe": market_data.timeframe,
                "quality_score": market_data.quality_score
            }
            log_test("orchestrator_market_data", "PASS", json.dumps(details), timing)
        else:
            log_test("orchestrator_market_data", "FAIL", "No market data", timing)
            
    except Exception as e:
        log_test("orchestrator_market_data", "ERROR", str(e))

def test_cli_validate_integration():
    """Test CLI validate tool integration"""
    import subprocess
    
    test_commands = [
        ["python", "cli_validate.py", "--json", "quote", "--symbol", "AAPL"],
        ["python", "cli_validate.py", "--json", "market_breadth"],
        ["python", "cli_validate.py", "--json", "sector_performance"]
    ]
    
    for cmd in test_commands:
        cmd_name = f"cli_validate_{cmd[3]}" if len(cmd) > 3 else "cli_validate"
        try:
            start_time = time.time()
            env = os.environ.copy()
            env['PYTHONPATH'] = project_root
            
            result = subprocess.run(
                cmd, 
                cwd=project_root,
                env=env,
                capture_output=True, 
                text=True, 
                timeout=30
            )
            timing = time.time() - start_time
            
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    if data.get('ok'):
                        log_test(cmd_name, "PASS", f"CLI command succeeded", timing)
                    else:
                        log_test(cmd_name, "FAIL", data.get('error', 'Unknown error'), timing)
                except json.JSONDecodeError:
                    log_test(cmd_name, "FAIL", "Invalid JSON output", timing)
            else:
                log_test(cmd_name, "ERROR", result.stderr[:200], timing)
                
        except subprocess.TimeoutExpired:
            log_test(cmd_name, "ERROR", "Command timeout")
        except Exception as e:
            log_test(cmd_name, "ERROR", str(e))

def main():
    """Run all tests"""
    print("=== Oracle-X Data Feed Testing Report ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test individual adapters
    print("# Individual Adapter Tests")
    test_reddit_adapter()
    test_finviz_adapter()
    test_twitter_adapter()
    test_investiny_adapter()
    test_adapter_wrappers()
    
    print()
    print("# Orchestrator Tests")
    test_orchestrator_basic()
    test_orchestrator_sentiment()
    test_orchestrator_market_data()
    
    print()
    print("# CLI Integration Tests")
    test_cli_validate_integration()
    
    print()
    print(f"Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print summary
    print("\n=== TEST RESULTS SUMMARY ===")
    total = len(test_results)
    passed = len([r for r in test_results if r["status"] == "PASS"])
    failed = len([r for r in test_results if r["status"] == "FAIL"])
    skipped = len([r for r in test_results if r["status"] == "SKIP"])
    errors = len([r for r in test_results if r["status"] == "ERROR"])
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️ Skipped: {skipped}")
    print(f"🚨 Errors: {errors}")
    
    # Show individual results with status symbols
    for result in test_results:
        status_symbol = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️" if result["status"] == "SKIP" else "🚨"
        timing_str = f" ({result['timing_ms']:.0f}ms)" if result["timing_ms"] else ""
        print(f"{status_symbol} {result['test']}: {result['status']}{timing_str}")
        if result["details"]:
            print(f"   Details: {result['details']}")
    
    if failed > 0 or errors > 0:
        print("\n⚠️ Some tests failed or had errors. Review the details above.")
    else:
        print("\n🎉 All non-skipped tests passed!")
    
    # Save detailed results to JSON
    import json
    with open("test_results_comprehensive.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nDetailed results saved to test_results_comprehensive.json")

if __name__ == "__main__":
    main()
