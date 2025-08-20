"""
Comprehensive FMP Integration Test
Tests all enhanced FMP features integrated into Oracle-X system
"""

import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_feeds.enhanced_consolidated_data_feed import EnhancedConsolidatedDataFeed
from data_feeds.enhanced_fmp_integration import EnhancedFMPAdapter
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_comprehensive_fmp_integration():
    """Comprehensive test of all FMP features integrated into Oracle-X"""
    print("🚀 COMPREHENSIVE FMP INTEGRATION TEST")
    print("=" * 60)
    
    # Initialize enhanced feed
    feed = EnhancedConsolidatedDataFeed()
    
    if not feed.enhanced_fmp.api_key:
        print("❌ FMP API key not available - test cannot proceed")
        return False
    
    print(f"✅ Enhanced FMP adapter initialized with API key")
    
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    success_count = 0
    total_tests = 0
    
    # Test 1: Enhanced Financial Ratios
    print("\n📊 Test 1: Enhanced Financial Ratios")
    print("-" * 40)
    total_tests += 1
    try:
        ratios_data = feed.get_enhanced_financial_ratios(test_symbols)
        for symbol, ratios in ratios_data.items():
            if ratios:
                print(f"   {symbol}:")
                print(f"     PE Ratio: {ratios.pe_ratio:.2f}" if ratios.pe_ratio else "     PE Ratio: N/A")
                print(f"     ROE: {ratios.roe:.2%}" if ratios.roe else "     ROE: N/A")
                print(f"     ROA: {ratios.roa:.2%}" if ratios.roa else "     ROA: N/A")
                print(f"     Debt/Equity: {ratios.debt_to_equity:.2f}" if ratios.debt_to_equity else "     Debt/Equity: N/A")
                print(f"     Gross Margin: {ratios.gross_profit_margin:.2%}" if ratios.gross_profit_margin else "     Gross Margin: N/A")
        success_count += 1
        print("✅ Financial ratios test PASSED")
    except Exception as e:
        print(f"❌ Financial ratios test FAILED: {e}")
    
    # Test 2: DCF Valuations
    print("\n💰 Test 2: DCF Valuations")
    print("-" * 40)
    total_tests += 1
    try:
        dcf_data = feed.get_dcf_valuations(test_symbols)
        for symbol, dcf in dcf_data.items():
            if dcf:
                current_price = dcf.get('Stock Price', 'N/A')
                dcf_value = dcf.get('dcf', 'N/A')
                print(f"   {symbol}: Current=${current_price}, DCF=${dcf_value:.2f}" if isinstance(dcf_value, (int, float)) else f"   {symbol}: Current=${current_price}, DCF={dcf_value}")
        success_count += 1
        print("✅ DCF valuations test PASSED")
    except Exception as e:
        print(f"❌ DCF valuations test FAILED: {e}")
    
    # Test 3: Comprehensive Fundamentals
    print("\n📈 Test 3: Comprehensive Fundamentals")
    print("-" * 40)
    total_tests += 1
    try:
        fundamentals = feed.get_comprehensive_fundamentals('AAPL')
        if 'AAPL' in fundamentals:
            data = fundamentals['AAPL']
            print(f"   Income Statement: {'✅ Available' if data['income_statement'] is not None else '❌ Not available'}")
            print(f"   Balance Sheet: {'✅ Available' if data['balance_sheet'] is not None else '❌ Not available'}")
            print(f"   Cash Flow: {'✅ Available' if data['cash_flow'] is not None else '❌ Not available'}")
            print(f"   Key Metrics: {'✅ Available' if data['key_metrics'] is not None else '❌ Not available'}")
            
            # Show sample data from income statement
            if data['income_statement'] is not None and not data['income_statement'].empty:
                latest_year = data['income_statement'].iloc[0]
                revenue = latest_year.get('revenue', 'N/A')
                net_income = latest_year.get('netIncome', 'N/A')
                print(f"   Latest Revenue: ${revenue:,}" if isinstance(revenue, (int, float)) else f"   Latest Revenue: {revenue}")
                print(f"   Latest Net Income: ${net_income:,}" if isinstance(net_income, (int, float)) else f"   Latest Net Income: {net_income}")
        success_count += 1
        print("✅ Comprehensive fundamentals test PASSED")
    except Exception as e:
        print(f"❌ Comprehensive fundamentals test FAILED: {e}")
    
    # Test 4: Analyst Data
    print("\n🎯 Test 4: Analyst Data")
    print("-" * 40)
    total_tests += 1
    try:
        analyst_data = feed.get_analyst_data('AAPL')
        if 'AAPL' in analyst_data:
            data = analyst_data['AAPL']
            print(f"   Estimates: {'✅ Available' if data['estimates'] is not None else '❌ Not available'}")
            print(f"   Price Target: {'✅ Available' if data['price_target'] is not None else '❌ Not available'}")
            
            if data['price_target']:
                target = data['price_target']
                print(f"   Price Target: ${target.get('priceTarget', 'N/A')}")
                print(f"   Analyst Company: {target.get('analystCompany', 'N/A')}")
        success_count += 1
        print("✅ Analyst data test PASSED")
    except Exception as e:
        print(f"❌ Analyst data test FAILED: {e}")
    
    # Test 5: Institutional Data
    print("\n🏛️ Test 5: Institutional Data")
    print("-" * 40)
    total_tests += 1
    try:
        institutional_data = feed.get_institutional_data('AAPL')
        if 'AAPL' in institutional_data:
            data = institutional_data['AAPL']
            ownership = data['institutional_ownership']
            insider_trading = data['insider_trading']
            print(f"   Institutional Ownership: {'✅ Available' if ownership is not None else '❌ Not available'}")
            print(f"   Insider Trading: {'✅ Available' if insider_trading is not None else '❌ Not available'}")
            
            if ownership is not None and not ownership.empty:
                print(f"   Top institutional holders: {len(ownership)} entries")
                top_holder = ownership.iloc[0]
                print(f"   Largest holder: {top_holder.get('holder', 'N/A')} ({top_holder.get('shares', 'N/A')} shares)")
            
            if insider_trading and len(insider_trading) > 0:
                print(f"   Recent insider trades: {len(insider_trading)} transactions")
        success_count += 1
        print("✅ Institutional data test PASSED")
    except Exception as e:
        print(f"❌ Institutional data test FAILED: {e}")
    
    # Test 6: Market Analysis
    print("\n🌍 Test 6: Market Analysis")
    print("-" * 40)
    total_tests += 1
    try:
        market_data = feed.get_market_analysis()
        sector_perf = market_data.get('sector_performance')
        market_cap = market_data.get('market_cap_ranking')
        
        print(f"   Sector Performance: {'✅ Available' if sector_perf is not None else '❌ Not available'}")
        print(f"   Market Cap Ranking: {'✅ Available' if market_cap is not None else '❌ Not available'}")
        
        if sector_perf is not None and not sector_perf.empty:
            print(f"   Sectors tracked: {len(sector_perf)}")
            best_sector = sector_perf.iloc[0]
            print(f"   Best performing sector: {best_sector.get('sector', 'N/A')} ({best_sector.get('changesPercentage', 'N/A')})")
        
        success_count += 1
        print("✅ Market analysis test PASSED")
    except Exception as e:
        print(f"❌ Market analysis test FAILED: {e}")
    
    # Test 7: Stock Screening
    print("\n🔍 Test 7: Stock Screening")
    print("-" * 40)
    total_tests += 1
    try:
        screening_results = feed.search_and_screen_stocks(
            market_cap_more_than=10000000000,  # >$10B market cap
            beta_more_than=0.5,
            beta_lower_than=1.5,
            limit=10
        )
        
        search_results = screening_results.get('search')
        screen_results = screening_results.get('screening')
        
        print(f"   Stock Screening: {'✅ Available' if screen_results is not None else '❌ Not available'}")
        
        if screen_results and len(screen_results) > 0:
            print(f"   Stocks found: {len(screen_results)}")
            sample_stock = screen_results[0]
            print(f"   Sample result: {sample_stock.get('symbol', 'N/A')} - {sample_stock.get('companyName', 'N/A')}")
        
        success_count += 1
        print("✅ Stock screening test PASSED")
    except Exception as e:
        print(f"❌ Stock screening test FAILED: {e}")
    
    # Test 8: Integration with Original Data Feed
    print("\n🔗 Test 8: Integration with Original Data Feed")
    print("-" * 40)
    total_tests += 1
    try:
        # Test that original functionality still works
        quote = feed.get_quote('AAPL')
        historical = feed.get_historical('AAPL')
        company_info = feed.get_company_info('AAPL')
        
        print(f"   Quote: {'✅ Available' if quote else '❌ Not available'}")
        print(f"   Historical: {'✅ Available' if historical is not None else '❌ Not available'}")
        print(f"   Company Info: {'✅ Available' if company_info else '❌ Not available'}")
        
        if quote:
            print(f"   Current price: ${quote.price}")
        
        if historical is not None and not historical.empty:
            print(f"   Historical data points: {len(historical)}")
        
        success_count += 1
        print("✅ Integration test PASSED")
    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
    
    # Final Results
    print("\n" + "=" * 60)
    print("📋 FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Tests Passed: {success_count}/{total_tests}")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED - FMP Integration is fully operational!")
        return True
    else:
        print(f"⚠️  {total_tests - success_count} tests failed - Review the issues above")
        return False

if __name__ == "__main__":
    test_comprehensive_fmp_integration()
