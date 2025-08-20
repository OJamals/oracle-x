#!/usr/bin/env python3
"""
Comprehensive test for enhanced consolidated data feed with FinanceToolkit and FinanceDatabase integration
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_feeds.finance_integration import (
    FinanceToolkitAdapter,
    EnhancedFinanceDatabaseAdapter,
    FinancialRatios,
    SecurityInfo,
    FINANCE_TOOLKIT_AVAILABLE,
    FINANCE_DATABASE_AVAILABLE
)

def test_enhanced_finance_database():
    """Test enhanced FinanceDatabase functionality"""
    print("=== Testing Enhanced FinanceDatabase ===")
    
    if not FINANCE_DATABASE_AVAILABLE:
        print("❌ FinanceDatabase not available")
        return False
    
    try:
        adapter = EnhancedFinanceDatabaseAdapter(None, None)
        
        if not adapter.available:
            print("❌ Enhanced FinanceDatabase adapter not available")
            return False
        
        # Test security search with filters
        print("\n1. Testing search for US Technology companies...")
        results = adapter.search_securities(
            asset_class='equity',
            country='United States',
            sector='Technology'
        )
        
        if 'equities' in results and results['equities']:
            print(f"✅ Found {len(results['equities'])} technology companies")
            # Show first few results
            for i, security in enumerate(results['equities'][:3]):
                print(f"   {i+1}. {security.symbol}: {security.name} ({security.sector})")
        else:
            print("⚠️  No technology companies found")
        
        # Test ETF search with corrected category
        print("\n2. Testing search for Equity ETFs...")
        etf_results = adapter.search_securities(
            asset_class='etf',
            category_group='Equities'  # Using correct category
        )
        
        if 'etfs' in etf_results and etf_results['etfs']:
            print(f"✅ Found {len(etf_results['etfs'])} equity ETFs")
            for i, etf in enumerate(etf_results['etfs'][:3]):
                print(f"   {i+1}. {etf.symbol}: {etf.name} ({etf.category_group})")
        else:
            print("⚠️  No equity ETFs found")
        
        # Test specific security lookup
        print("\n3. Testing specific security lookup...")
        security_info = adapter.get_security_info('AAPL')
        if security_info:
            print(f"✅ Found AAPL: {security_info.name}")
            print(f"   Sector: {security_info.sector}")
            print(f"   Industry: {security_info.industry}")
            print(f"   Exchange: {security_info.exchange}")
        else:
            print("⚠️  AAPL not found in database")
        
        # Test search by name
        print("\n4. Testing search by company name...")
        apple_results = adapter.search_securities(query='Apple')
        if apple_results:
            total_found = sum(len(securities) for securities in apple_results.values())
            print(f"✅ Found {total_found} securities matching 'Apple'")
            for asset_class, securities in apple_results.items():
                if securities:
                    print(f"   {asset_class.title()}: {len(securities)} results")
        else:
            print("⚠️  No Apple-related securities found")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced FinanceDatabase test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_finance_toolkit_integration():
    """Test FinanceToolkit functionality"""
    print("\n=== Testing FinanceToolkit Integration ===")
    
    if not FINANCE_TOOLKIT_AVAILABLE:
        print("❌ FinanceToolkit not available")
        return False
    
    try:
        adapter = FinanceToolkitAdapter(None, None)
        
        if not adapter.available:
            print("❌ FinanceToolkit adapter not available")
            return False
        
        # Test basic toolkit creation
        print("\n1. Testing Toolkit creation...")
        symbols = ['AAPL', 'MSFT']
        toolkit = adapter.create_toolkit(symbols, start_date='2023-01-01', end_date='2023-12-31')
        
        if toolkit:
            print(f"✅ Successfully created toolkit for {symbols}")
        else:
            print("❌ Failed to create toolkit")
            return False
        
        # Test historical analysis
        print("\n2. Testing historical data analysis...")
        historical = adapter.get_historical_analysis(symbols, start_date='2023-01-01', end_date='2023-12-31')
        
        if historical is not None and not historical.empty:
            print(f"✅ Retrieved historical data: {historical.shape}")
            print("Available columns:")
            cols = [col for col in historical.columns if isinstance(col, tuple)][:10]  # Show first 10
            for col in cols:
                print(f"   {col}")
        else:
            print("⚠️  No historical data retrieved")
        
        # Test financial ratios calculation
        print("\n3. Testing financial ratios calculation...")
        ratios_dict = adapter.get_financial_ratios(['AAPL'])
        
        if ratios_dict and 'AAPL' in ratios_dict:
            ratios = ratios_dict['AAPL']
            print(f"✅ Retrieved financial ratios for AAPL")
            
            # Display some key ratios
            ratio_items = [
                ('Gross Margin', ratios.gross_margin),
                ('Operating Margin', ratios.operating_margin),
                ('Net Profit Margin', ratios.net_profit_margin),
                ('Return on Assets', ratios.return_on_assets),
                ('Return on Equity', ratios.return_on_equity),
                ('Current Ratio', ratios.current_ratio),
                ('Quick Ratio', ratios.quick_ratio),
                ('Debt to Equity', ratios.debt_to_equity),
                ('Price to Earnings', ratios.price_to_earnings),
                ('Price to Book', ratios.price_to_book)
            ]
            
            print("   Key Financial Ratios:")
            for name, value in ratio_items:
                if value is not None:
                    if 'Margin' in name or 'Return' in name:
                        print(f"   {name}: {value:.2%}")
                    else:
                        print(f"   {name}: {value:.2f}")
                else:
                    print(f"   {name}: N/A")
        else:
            print("⚠️  No financial ratios calculated (may need API key for full functionality)")
        
        return True
        
    except Exception as e:
        print(f"❌ FinanceToolkit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_combined_workflow():
    """Test a combined workflow using both libraries"""
    print("\n=== Testing Combined Workflow ===")
    
    if not (FINANCE_DATABASE_AVAILABLE and FINANCE_TOOLKIT_AVAILABLE):
        print("❌ Both libraries required for combined workflow")
        return False
    
    try:
        # Step 1: Use FinanceDatabase to find tech companies
        db_adapter = EnhancedFinanceDatabaseAdapter(None, None)
        
        print("1. Finding large-cap US technology companies...")
        results = db_adapter.search_securities(
            asset_class='equity',
            country='United States',
            sector='Technology',
            market_cap='Large Cap'
        )
        
        if not results.get('equities'):
            print("❌ No technology companies found")
            return False
        
        # Get top 3 companies for analysis
        tech_companies = results['equities'][:3]
        symbols = [company.symbol for company in tech_companies]
        
        print(f"✅ Found {len(tech_companies)} companies for analysis:")
        for company in tech_companies:
            print(f"   {company.symbol}: {company.name}")
        
        # Step 2: Use FinanceToolkit for detailed analysis
        ft_adapter = FinanceToolkitAdapter(None, None)
        
        print(f"\n2. Performing financial analysis on {symbols}...")
        
        # Get historical data
        historical = ft_adapter.get_historical_analysis(symbols, start_date='2023-01-01')
        if historical is not None and not historical.empty:
            print(f"✅ Retrieved historical data: {historical.shape}")
            
            # Calculate basic statistics
            for symbol in symbols:
                try:
                    close_col = ('Close', symbol)
                    if close_col in historical.columns:
                        prices = historical[close_col].dropna()
                        if not prices.empty:
                            start_price = prices.iloc[0]
                            end_price = prices.iloc[-1]
                            total_return = (end_price / start_price - 1) * 100
                            volatility = prices.pct_change().std() * (252 ** 0.5) * 100  # Annualized
                            
                            print(f"   {symbol}:")
                            print(f"     Total Return (2023): {total_return:.1f}%")
                            print(f"     Annualized Volatility: {volatility:.1f}%")
                except Exception as e:
                    print(f"   {symbol}: Analysis failed - {e}")
        
        # Get financial ratios
        print(f"\n3. Calculating financial ratios...")
        ratios_dict = ft_adapter.get_financial_ratios(symbols)
        
        for symbol in symbols:
            if symbol in ratios_dict:
                ratios = ratios_dict[symbol]
                print(f"   {symbol} Key Ratios:")
                if ratios.net_profit_margin is not None:
                    print(f"     Net Profit Margin: {ratios.net_profit_margin:.1%}")
                if ratios.return_on_equity is not None:
                    print(f"     Return on Equity: {ratios.return_on_equity:.1%}")
                if ratios.current_ratio is not None:
                    print(f"     Current Ratio: {ratios.current_ratio:.2f}")
            else:
                print(f"   {symbol}: No ratios available")
        
        print("\n✅ Combined workflow completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Combined workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Testing Enhanced Financial Data Integration")
    print("=" * 60)
    
    # Test results
    results = {
        'enhanced_finance_database': False,
        'finance_toolkit_integration': False,
        'combined_workflow': False
    }
    
    # Run tests
    results['enhanced_finance_database'] = test_enhanced_finance_database()
    results['finance_toolkit_integration'] = test_finance_toolkit_integration()
    results['combined_workflow'] = test_combined_workflow()
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    # Provide next steps
    if all_passed:
        print("\n🎉 Integration successful! Next steps:")
        print("1. Add these adapters to consolidated_data_feed.py")
        print("2. Integrate financial ratios into the main data feed")
        print("3. Add enhanced security search capabilities")
        print("4. Consider adding portfolio analysis features")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
        print("Consider installing missing dependencies or checking API keys.")
    
    return results

if __name__ == "__main__":
    main()
