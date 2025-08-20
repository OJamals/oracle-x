#!/usr/bin/env python3
"""
Comprehensive test suite for TwelveData adapter implementation.
Tests all implemented and planned endpoints according to GAP_ANALYSIS.md.
"""

import os
import sys
import logging
from datetime import datetime
from decimal import Decimal
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_feeds.twelvedata_adapter import TwelveDataAdapter
from data_feeds.data_feed_orchestrator import get_orchestrator, DataSource

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwelveDataComprehensiveTest:
    def __init__(self):
        self.api_key = os.getenv('TWELVEDATA_API_KEY')
        if not self.api_key:
            raise ValueError("TWELVEDATA_API_KEY environment variable is required")
        
        self.adapter = TwelveDataAdapter(api_key=self.api_key)
        self.orchestrator = get_orchestrator()
        self.test_symbol = "AAPL"
        self.test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
    def test_quote_data(self):
        """Test quote data retrieval - PRIORITY 1"""
        print("\n" + "="*60)
        print("TESTING QUOTE DATA (PRIORITY 1)")
        print("="*60)
        
        try:
            # Test direct adapter
            print("1. Testing direct adapter quote...")
            quote = self.adapter.get_quote(self.test_symbol)
            if quote:
                print(f"   ✅ Quote retrieved successfully")
                print(f"   Symbol: {quote.symbol}")
                print(f"   Price: {quote.price}")
                print(f"   Change: {quote.change}")
                print(f"   Change %: {quote.change_percent}")
                print(f"   Volume: {quote.volume}")
                print(f"   Day Low: {quote.day_low}")
                print(f"   Day High: {quote.day_high}")
                print(f"   Year Low: {quote.year_low}")
                print(f"   Year High: {quote.year_high}")
                print(f"   Source: {quote.source}")
                print(f"   Quality Score: {quote.quality_score}")
            else:
                print("   ❌ Quote retrieval failed")
                return False
                
            # Test through orchestrator
            print("\n2. Testing orchestrator quote...")
            orch_quote = self.orchestrator.get_quote(self.test_symbol, preferred_sources=[DataSource.TWELVE_DATA])
            if orch_quote:
                print(f"   ✅ Orchestrator quote retrieved successfully")
                print(f"   Price: {orch_quote.price}")
                print(f"   Source: {orch_quote.source}")
                print(f"   Quality Score: {orch_quote.quality_score}")
            else:
                print("   ⚠️  Orchestrator quote retrieval returned None")
                
            return True
            
        except Exception as e:
            print(f"   ❌ Quote test failed: {e}")
            logger.error(f"Quote test failed: {e}")
            return False

    def test_market_data(self):
        """Test market data/time series - PRIORITY 1"""
        print("\n" + "="*60)
        print("TESTING MARKET DATA (PRIORITY 1)")
        print("="*60)
        
        try:
            # Test different timeframes
            test_cases = [
                ("1mo", "1d", "Monthly daily data"),
                ("3mo", "1d", "Quarterly daily data"),
                ("1y", "1d", "Yearly daily data"),
                ("5d", "1h", "5-day hourly data"),
                ("1mo", "1wk", "Monthly weekly data")
            ]
            
            for period, interval, description in test_cases:
                print(f"\n1. Testing {description} ({period}, {interval})...")
                market_data = self.adapter.get_market_data(self.test_symbol, period=period, interval=interval)
                if market_data and market_data.data is not None and not market_data.data.empty:
                    print(f"   ✅ {description} retrieved successfully")
                    print(f"   Data shape: {market_data.data.shape}")
                    print(f"   First few rows:")
                    print(market_data.data.head(2))
                    print(f"   Source: {market_data.source}")
                    print(f"   Quality Score: {market_data.quality_score}")
                else:
                    print(f"   ⚠️  {description} retrieval returned no data")
                    
            # Test through orchestrator
            print("\n2. Testing orchestrator market data...")
            orch_data = self.orchestrator.get_market_data(
                self.test_symbol, 
                period="1mo", 
                interval="1d", 
                preferred_sources=[DataSource.TWELVE_DATA]
            )
            if orch_data and orch_data.data is not None and not orch_data.data.empty:
                print(f"   ✅ Orchestrator market data retrieved successfully")
                print(f"   Data shape: {orch_data.data.shape}")
                print(f"   Source: {orch_data.source}")
            else:
                print("   ⚠️  Orchestrator market data retrieval returned None")
                
            return True
            
        except Exception as e:
            print(f"   ❌ Market data test failed: {e}")
            logger.error(f"Market data test failed: {e}")
            return False

    def test_fundamental_endpoints(self):
        """Test fundamental data endpoints - PRIORITY 1"""
        print("\n" + "="*60)
        print("TESTING FUNDAMENTAL ENDPOINTS (PRIORITY 1)")
        print("="*60)
        print("Note: Fundamental endpoints need to be implemented based on GAP_ANALYSIS.md")
        
        # This is a placeholder for future implementation
        # According to GAP_ANALYSIS.md, we need to implement:
        # - /income_statement
        # - /balance_sheet  
        # - /cash_flow
        # - /earnings
        # - /dividends
        # - /splits
        # - /statistics
        
        print("\n1. Checking if fundamental methods exist...")
        fundamental_methods = [
            'get_income_statement',
            'get_balance_sheet',
            'get_cash_flow',
            'get_earnings',
            'get_dividends',
            'get_splits',
            'get_statistics'
        ]
        
        implemented_methods = []
        missing_methods = []
        
        for method_name in fundamental_methods:
            if hasattr(self.adapter, method_name):
                implemented_methods.append(method_name)
            else:
                missing_methods.append(method_name)
        
        print(f"   Implemented methods: {implemented_methods}")
        print(f"   Missing methods: {missing_methods}")
        
        if missing_methods:
            print(f"\n   ⚠️  {len(missing_methods)} fundamental endpoints need implementation:")
            for method in missing_methods:
                print(f"      - {method}")
        
        return len(implemented_methods) > 0

    def test_technical_indicators(self):
        """Test technical indicators - PRIORITY 1"""
        print("\n" + "="*60)
        print("TESTING TECHNICAL INDICATORS (PRIORITY 1)")
        print("="*60)
        print("Note: Technical indicators need to be implemented based on GAP_ANALYSIS.md")
        
        # This is a placeholder for future implementation
        # According to GAP_ANALYSIS.md, we need to implement:
        # - 80+ technical indicators via /technical_indicators endpoint family
        # - Moving Averages: SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA
        # - Momentum: RSI, STOCH, CCI, ADX, AROON, MOM, ROC, Williams %R
        # - Volatility: BBANDS, ATR, NATR, AD, OBV
        # - Volume: MFI, PVT, EMV, FORCE, NVI, PVI
        
        print("\n1. Checking if technical indicator methods exist...")
        tech_methods = [
            'get_sma',
            'get_ema',
            'get_rsi',
            'get_macd',
            'get_bbands',
            'get_adx',
            'get_stoch',
            'get_atr',
            'get_obv',
            'get_cci'
        ]
        
        implemented_methods = []
        missing_methods = []
        
        for method_name in tech_methods:
            if hasattr(self.adapter, method_name):
                implemented_methods.append(method_name)
            else:
                missing_methods.append(method_name)
        
        print(f"   Implemented methods: {implemented_methods}")
        print(f"   Missing methods: {missing_methods}")
        
        if missing_methods:
            print(f"\n   ⚠️  {len(missing_methods)} technical indicators need implementation:")
            for method in missing_methods[:5]:  # Show first 5
                print(f"      - {method}")
            if len(missing_methods) > 5:
                print(f"      ... and {len(missing_methods) - 5} more")
        
        return len(implemented_methods) > 0

    def test_options_data(self):
        """Test options data endpoints - PRIORITY 2"""
        print("\n" + "="*60)
        print("TESTING OPTIONS DATA (PRIORITY 2)")
        print("="*60)
        print("Note: Options data needs to be implemented based on GAP_ANALYSIS.md")
        
        # According to GAP_ANALYSIS.md, we need to implement:
        # - /options - Options chains and Greeks
        # - Options historical data
        # - Volatility surfaces
        # - Options analytics
        
        print("\n1. Checking if options methods exist...")
        options_methods = [
            'get_options_chain',
            'get_options_greeks',
            'get_volatility_surface'
        ]
        
        implemented_methods = []
        missing_methods = []
        
        for method_name in options_methods:
            if hasattr(self.adapter, method_name):
                implemented_methods.append(method_name)
            else:
                missing_methods.append(method_name)
        
        print(f"   Implemented methods: {implemented_methods}")
        print(f"   Missing methods: {missing_methods}")
        
        return len(implemented_methods) > 0

    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n" + "="*60)
        print("TESTING ERROR HANDLING")
        print("="*60)
        
        try:
            # Test invalid symbol
            print("1. Testing invalid symbol...")
            invalid_quote = self.adapter.get_quote("INVALIDXYZ123")
            if invalid_quote is None:
                print("   ✅ Invalid symbol handled correctly (returned None)")
            else:
                print(f"   ⚠️  Invalid symbol returned data: {invalid_quote}")
                
            # Test invalid timeframe
            print("\n2. Testing invalid timeframe...")
            try:
                invalid_market_data = self.adapter.get_market_data(self.test_symbol, period="invalid", interval="1d")
                if invalid_market_data is None:
                    print("   ✅ Invalid timeframe handled correctly (returned None)")
                else:
                    print(f"   ⚠️  Invalid timeframe returned data")
            except Exception as e:
                print(f"   ✅ Invalid timeframe raised exception as expected: {type(e).__name__}")
                
            return True
            
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            logger.error(f"Error handling test failed: {e}")
            return False

    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 TWELVEDATA ADAPTER COMPREHENSIVE TEST SUITE")
        print("="*60)
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Symbol: {self.test_symbol}")
        print(f"API Key Available: {self.api_key is not None}")
        print("="*60)
        
        results = {}
        
        # Run tests in order of priority
        results['quote_data'] = self.test_quote_data()
        results['market_data'] = self.test_market_data()
        results['fundamental_endpoints'] = self.test_fundamental_endpoints()
        results['technical_indicators'] = self.test_technical_indicators()
        results['options_data'] = self.test_options_data()
        results['error_handling'] = self.test_error_handling()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
            
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        if not results.get('fundamental_endpoints'):
            print("⚠️  Implement fundamental endpoints:")
            print("   - get_income_statement()")
            print("   - get_balance_sheet()")
            print("   - get_cash_flow()")
            print("   - get_earnings()")
            print("   - get_dividends()")
            print("   - get_splits()")
            print("   - get_statistics()")
            
        if not results.get('technical_indicators'):
            print("\n⚠️  Implement technical indicators:")
            print("   - get_sma(), get_ema(), get_rsi(), get_macd()")
            print("   - get_bbands(), get_adx(), get_stoch(), get_atr()")
            print("   - get_obv(), get_cci()")
            
        if not results.get('options_data'):
            print("\n⚠️  Consider implementing options data:")
            print("   - get_options_chain()")
            print("   - get_options_greeks()")
            print("   - get_volatility_surface()")
            
        return passed == total

def main():
    """Main test execution"""
    try:
        tester = TwelveDataComprehensiveTest()
        success = tester.run_all_tests()
        
        if success:
            print("\n🎉 ALL TESTS PASSED - TwelveData adapter is functioning correctly!")
            return 0
        else:
            print("\n⚠️  SOME TESTS FAILED - Review the results above")
            return 1
            
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        logger.error(f"Fatal error in test execution: {e}")
        return 1

if __name__ == "__main__":
    exit(main())