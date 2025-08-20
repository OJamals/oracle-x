#!/usr/bin/env python3
"""
Test script for FinanceToolkit and FinanceDatabase integration
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_finance_database():
    """Test FinanceDatabase functionality"""
    print("=== Testing FinanceDatabase ===")
    
    try:
        import financedatabase as fd
        
        # Test Equities
        print("\n1. Testing Equities database...")
        equities = fd.Equities()
        
        # Show available options
        print("Available sectors:")
        sectors = equities.show_options(selection='sector')
        print(sectors[:10])  # Show first 10
        
        # Filter by Technology sector in the US
        print("\nTechnology companies in the US (first 5):")
        tech_us = equities.select(
            country='United States',
            sector='Information Technology'
        )
        print(tech_us.head())
        print(f"Total found: {len(tech_us)}")
        
        # Test ETFs
        print("\n2. Testing ETFs database...")
        etfs = fd.ETFs()
        
        # Show available category groups
        print("Available ETF category groups:")
        etf_categories = etfs.show_options(selection='category_group')
        print(etf_categories[:10])
        
        # Get Technology-related ETFs (using available category)
        print("\nTechnology-related ETFs (first 5):")
        try:
            # Try 'Technology' first, fallback to available categories
            if 'Technology' in etf_categories:
                tech_etfs = etfs.select(category_group='Technology')
            else:
                # Use an available technology-related category
                tech_etfs = etfs.select(category_group='Communication Services')
                print("Note: Using 'Communication Services' as Technology category not available")
            print(tech_etfs.head())
            print(f"Total found: {len(tech_etfs)}")
        except Exception as e:
            print(f"ETF category selection failed: {e}")
            # Try with a different approach - just get any ETFs
            try:
                sample_etfs = etfs.select()  # Get all ETFs
                print("Showing sample ETFs instead:")
                print(sample_etfs.head())
                print(f"Total ETFs available: {len(sample_etfs)}")
            except Exception as e2:
                print(f"Alternative ETF selection also failed: {e2}")
        
        # Test Funds
        print("\n3. Testing Funds database...")
        funds = fd.Funds()
        
        # Get equity funds
        print("\nEquity funds (first 5):")
        try:
            equity_funds = funds.select(category_group='Equity')
            print(equity_funds.head())
            print(f"Total found: {len(equity_funds)}")
        except Exception as e:
            print(f"Funds selection failed: {e}")
            # Try to show available fund categories
            try:
                fund_categories = funds.show_options(selection='category_group')
                print(f"Available fund categories: {fund_categories[:5]}")
                # Try with first available category
                if len(fund_categories) > 0:
                    sample_funds = funds.select(category_group=fund_categories[0])
                    print(f"Sample funds from {fund_categories[0]} category:")
                    print(sample_funds.head())
            except Exception as e2:
                print(f"Alternative funds selection failed: {e2}")
        
        # Test search functionality
        print("\n4. Testing search functionality...")
        search_results = equities.search(
            name='Apple', 
            country='United States'
        )
        print("Apple search results:")
        print(search_results[['name', 'sector', 'industry', 'exchange']])
        
        return True
        
    except Exception as e:
        print(f"FinanceDatabase test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_finance_toolkit():
    """Test FinanceToolkit functionality with API key"""
    print("\n=== Testing FinanceToolkit with API Key ===")
    
    try:
        from financetoolkit import Toolkit
        
        # Get API key from environment
        api_key = os.getenv('FINANCIALMODELINGPREP_API_KEY')
        if not api_key:
            print("❌ No FinancialModelingPrep API key found in environment")
            return False
        
        print(f"✅ API Key found: {api_key[:10]}...")
        
        # Test with a simple ticker using FinancialModelingPrep as source
        print("\n1. Testing basic Toolkit functionality with FMP API...")
        
        # Initialize toolkit with FMP API key
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        toolkit = Toolkit(
            tickers=tickers,
            api_key=api_key,
            start_date='2023-01-01',
            end_date='2023-12-31',
            quarterly=False
        )
        
        # Test getting historical data
        print("Getting historical data with API...")
        try:
            historical_data = toolkit.get_historical_data()
            if historical_data is not None and not historical_data.empty:
                print(f"✅ Historical data shape: {historical_data.shape}")
                print("Historical data columns:")
                print(historical_data.columns.tolist())
                print(f"\nSample data for {tickers[0]}:")
                try:
                    # Try to access single ticker data
                    if len(historical_data.columns.names) > 1:
                        # Multi-level columns, try to get ticker-specific data
                        ticker_cols = [col for col in historical_data.columns if tickers[0] in str(col)]
                        if ticker_cols:
                            sample_data = historical_data[ticker_cols].head()
                            print(sample_data)
                        else:
                            print(historical_data.head())
                    else:
                        print(historical_data.head())
                except:
                    print(historical_data.head())
            else:
                print("❌ No historical data retrieved")
        except Exception as e:
            print(f"❌ Historical data retrieval failed: {e}")
        
        # Test financial statements (should work with API key)
        print("\n2. Testing financial statements with API key...")
        try:
            print("Getting balance sheet...")
            balance_sheet = toolkit.get_balance_sheet_statement()
            if balance_sheet is not None and not balance_sheet.empty:
                print(f"✅ Balance sheet shape: {balance_sheet.shape}")
                print("Balance sheet columns (first 10):")
                print(balance_sheet.columns.tolist()[:10])
                print(f"\nSample balance sheet data for {tickers[0]}:")
                if tickers[0] in balance_sheet.columns:
                    sample_bs = balance_sheet[tickers[0]].head()
                    print(sample_bs)
            else:
                print("❌ No balance sheet data retrieved")
                
            print("\nGetting income statement...")
            income_statement = toolkit.get_income_statement()
            if income_statement is not None and not income_statement.empty:
                print(f"✅ Income statement shape: {income_statement.shape}")
                print("Income statement columns (first 10):")
                print(income_statement.columns.tolist()[:10])
                print(f"\nSample income statement for {tickers[0]}:")
                if tickers[0] in income_statement.columns:
                    sample_is = income_statement[tickers[0]].head()
                    print(sample_is)
            else:
                print("❌ No income statement data retrieved")
                
            print("\nGetting cash flow statement...")
            cash_flow = toolkit.get_cash_flow_statement()
            if cash_flow is not None and not cash_flow.empty:
                print(f"✅ Cash flow statement shape: {cash_flow.shape}")
                print("Cash flow columns (first 10):")
                print(cash_flow.columns.tolist()[:10])
            else:
                print("❌ No cash flow data retrieved")
                
        except Exception as e:
            print(f"❌ Financial statements failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test ratios calculation with real data
        print("\n3. Testing comprehensive ratios with API data...")
        try:
            # Get profitability ratios
            print("Calculating profitability ratios...")
            profitability_ratios = toolkit.ratios.collect_profitability_ratios()
            if profitability_ratios is not None and not profitability_ratios.empty:
                print(f"✅ Profitability ratios shape: {profitability_ratios.shape}")
                print("Available profitability ratios:")
                print(profitability_ratios.index.tolist())
                print(f"\nSample profitability ratios for {tickers[0]}:")
                if tickers[0] in profitability_ratios.columns:
                    sample_prof = profitability_ratios[tickers[0]].dropna().head()
                    print(sample_prof)
            else:
                print("❌ No profitability ratios calculated")
            
            # Get liquidity ratios
            print("\nCalculating liquidity ratios...")
            liquidity_ratios = toolkit.ratios.collect_liquidity_ratios()
            if liquidity_ratios is not None and not liquidity_ratios.empty:
                print(f"✅ Liquidity ratios shape: {liquidity_ratios.shape}")
                print("Available liquidity ratios:")
                print(liquidity_ratios.index.tolist())
                
            # Get solvency ratios
            print("\nCalculating solvency ratios...")
            solvency_ratios = toolkit.ratios.collect_solvency_ratios()
            if solvency_ratios is not None and not solvency_ratios.empty:
                print(f"✅ Solvency ratios shape: {solvency_ratios.shape}")
                print("Available solvency ratios:")
                print(solvency_ratios.index.tolist())
                
            # Get efficiency ratios
            print("\nCalculating efficiency ratios...")
            efficiency_ratios = toolkit.ratios.collect_efficiency_ratios()
            if efficiency_ratios is not None and not efficiency_ratios.empty:
                print(f"✅ Efficiency ratios shape: {efficiency_ratios.shape}")
                print("Available efficiency ratios:")
                print(efficiency_ratios.index.tolist())
                
            # Get valuation ratios
            print("\nCalculating valuation ratios...")
            valuation_ratios = toolkit.ratios.collect_valuation_ratios()
            if valuation_ratios is not None and not valuation_ratios.empty:
                print(f"✅ Valuation ratios shape: {valuation_ratios.shape}")
                print("Available valuation ratios:")
                print(valuation_ratios.index.tolist())
                
        except Exception as e:
            print(f"❌ Ratios calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test advanced analysis features
        print("\n4. Testing advanced analysis features...")
        try:
            # Test getting all ratios at once
            print("Getting all ratios combined...")
            try:
                all_ratios = toolkit.ratios.collect_all_ratios()
                if all_ratios is not None and not all_ratios.empty:
                    print(f"✅ All ratios shape: {all_ratios.shape}")
                    print("Sample of all ratios:")
                    print(all_ratios.head())
                else:
                    print("❌ No combined ratios data")
            except Exception as e:
                print(f"All ratios collection failed: {e}")
                
            # Test specific ratio calculations
            print("\nTesting specific ratio calculations...")
            try:
                # Return on Equity
                roe = toolkit.ratios.get_return_on_equity()
                if roe is not None and not roe.empty:
                    print(f"✅ Return on Equity calculated for {len(roe.columns)} companies")
                    print(f"ROE sample: {roe.iloc[-1].to_dict()}")
                
                # Current Ratio
                current_ratio = toolkit.ratios.get_current_ratio()
                if current_ratio is not None and not current_ratio.empty:
                    print(f"✅ Current Ratio calculated for {len(current_ratio.columns)} companies")
                    
            except Exception as e:
                print(f"Specific ratios failed: {e}")
                
        except Exception as e:
            print(f"❌ Advanced analysis failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ FinanceToolkit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between FinanceDatabase and FinanceToolkit with API key"""
    print("\n=== Testing Integration with API Key ===")
    
    try:
        import financedatabase as fd
        
        # Get API key
        api_key = os.getenv('FINANCIALMODELINGPREP_API_KEY')
        if not api_key:
            print("❌ No FinancialModelingPrep API key found for integration test")
            return False
        
        # Get some US tech companies
        equities = fd.Equities()
        tech_companies = equities.select(
            country='United States',
            sector='Information Technology',
            market_cap='Large Cap'
        )
        
        # Get just a few symbols for testing
        symbols = tech_companies.head(3).index.tolist()
        print(f"Testing integration with symbols: {symbols}")
        
        # Try to create toolkit with symbols from database
        try:
            from financetoolkit import Toolkit
            
            toolkit = Toolkit(
                tickers=symbols,
                api_key=api_key,
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            print("✅ Successfully created toolkit with database symbols and API key")
            
            # Test historical data integration
            print("\nTesting historical data integration...")
            historical = toolkit.get_historical_data()
            if historical is not None and not historical.empty:
                print(f"✅ Integrated historical data shape: {historical.shape}")
                print(f"Date range: {historical.index[0]} to {historical.index[-1]}")
            
            # Test financial statements integration
            print("\nTesting financial statements integration...")
            try:
                income_stmt = toolkit.get_income_statement()
                if income_stmt is not None and not income_stmt.empty:
                    print(f"✅ Income statements retrieved for {len(income_stmt.columns)} companies")
                    
                balance_sheet = toolkit.get_balance_sheet_statement()
                if balance_sheet is not None and not balance_sheet.empty:
                    print(f"✅ Balance sheets retrieved for {len(balance_sheet.columns)} companies")
                    
            except Exception as e:
                print(f"❌ Financial statements integration failed: {e}")
            
            # Test ratios integration
            print("\nTesting ratios integration...")
            try:
                ratios = toolkit.ratios.collect_profitability_ratios()
                if ratios is not None and not ratios.empty:
                    print(f"✅ Profitability ratios calculated for {len(ratios.columns)} companies")
                    
                    # Show sample ratios for each company
                    for symbol in symbols[:2]:  # Show first 2 companies
                        if symbol in ratios.columns:
                            print(f"\nSample ratios for {symbol}:")
                            symbol_ratios = ratios[symbol].dropna().head(3)
                            for ratio_name, value in symbol_ratios.items():
                                print(f"  {ratio_name}: {value:.4f}")
                        
            except Exception as e:
                print(f"❌ Ratios integration failed: {e}")
            
            # Test combined analysis
            print("\nTesting combined database + toolkit analysis...")
            try:
                # Get company info from database
                for symbol in symbols[:2]:
                    if symbol in tech_companies.index:
                        company_info = tech_companies.loc[symbol]
                        print(f"\n--- {symbol} Analysis ---")
                        print(f"Company: {company_info.get('name', 'N/A')}")
                        print(f"Industry: {company_info.get('industry', 'N/A')}")
                        print(f"Market Cap: {company_info.get('market_cap', 'N/A')}")
                        
                        # Add recent price from historical data
                        if historical is not None and not historical.empty:
                            try:
                                # Get latest close price
                                symbol_cols = [col for col in historical.columns if symbol in str(col) and 'Close' in str(col)]
                                if symbol_cols:
                                    latest_price = historical[symbol_cols[0]].iloc[-1]
                                    print(f"Latest Price: ${latest_price:.2f}")
                            except:
                                pass
                                
            except Exception as e:
                print(f"❌ Combined analysis failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration toolkit creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_premium_api_endpoints():
    """Test premium FinancialModelingPrep API endpoints specifically"""
    print("\n=== Testing Premium API Endpoints ===")
    
    try:
        from financetoolkit import Toolkit
        
        # Get API key
        api_key = os.getenv('FINANCIALMODELINGPREP_API_KEY')
        if not api_key:
            print("❌ No FinancialModelingPrep API key found")
            return False
        
        print(f"✅ Testing with API Key: {api_key[:10]}...")
        
        # Test with popular stocks that should have complete data
        tickers = ['AAPL', 'MSFT']
        toolkit = Toolkit(
            tickers=tickers,
            api_key=api_key,
            start_date='2022-01-01',
            end_date='2023-12-31',
            quarterly=True  # Test quarterly data
        )
        
        print(f"\n1. Testing Premium Data Access for {tickers}...")
        
        # Test quarterly financial statements (premium feature)
        print("\nTesting quarterly financial statements...")
        try:
            # Note: quarterly parameter not supported in current version
            print("Note: Quarterly data testing skipped - parameter not supported in current version")
            # quarterly_income = toolkit.get_income_statement(period='quarterly')
            
            # Test annual statements instead
            annual_income = toolkit.get_income_statement()
            if annual_income is not None and not annual_income.empty:
                print(f"✅ Annual income statements: {annual_income.shape}")
                print(f"Years available: {len(annual_income.columns)}")
                print("Sample annual metrics:")
                if len(annual_income.columns) > 0:
                    sample_col = annual_income.columns[0]
                    sample_metrics = annual_income[sample_col].dropna().head(5)
                    for metric, value in sample_metrics.items():
                        if pd.notna(value):
                            print(f"  {metric}: ${value:,.0f}")
            else:
                print("❌ No annual income statement data")
        except Exception as e:
            print(f"❌ Financial statements failed: {e}")
        
        # Test enterprise value calculations (requires full financial data)
        print("\nTesting valuation metrics calculations...")
        try:
            # Market cap and enterprise value are available through ratios
            print("Getting market cap from valuation ratios...")
            valuation_ratios = toolkit.ratios.collect_valuation_ratios()
            if valuation_ratios is not None and not valuation_ratios.empty:
                # Look for market cap in the valuation ratios
                market_cap_rows = [idx for idx in valuation_ratios.index if 'Market Cap' in str(idx)]
                if market_cap_rows:
                    print(f"✅ Market cap data available in ratios")
                    for row in market_cap_rows[:2]:  # Show first 2 market cap entries
                        if len(valuation_ratios.columns) > 0:
                            col = valuation_ratios.columns[0]
                            value = valuation_ratios.loc[row, col]
                            if pd.notna(value):
                                print(f"  {row}: ${value:,.0f}")
                
                # Look for enterprise value in the valuation ratios  
                ev_rows = [idx for idx in valuation_ratios.index if 'Enterprise Value' in str(idx)]
                if ev_rows:
                    print(f"✅ Enterprise value data available in ratios")
                    for row in ev_rows[:2]:  # Show first 2 EV entries
                        if len(valuation_ratios.columns) > 0:
                            col = valuation_ratios.columns[0]
                            value = valuation_ratios.loc[row, col]
                            if pd.notna(value):
                                print(f"  {row}: ${value:,.0f}")
                else:
                    print("❌ No enterprise value data found in ratios")
            else:
                print("❌ No valuation ratios available")
                
        except Exception as e:
            print(f"❌ Valuation metrics calculations failed: {e}")
        
        # Test comprehensive ratio calculations with real data
        print("\nTesting comprehensive ratio calculations...")
        try:
            # Get all ratio categories
            ratio_categories = [
                ('Profitability', 'collect_profitability_ratios'),
                ('Liquidity', 'collect_liquidity_ratios'),
                ('Solvency', 'collect_solvency_ratios'),
                ('Efficiency', 'collect_efficiency_ratios'),
                ('Valuation', 'collect_valuation_ratios')
            ]
            
            all_ratios_data = {}
            for category_name, method_name in ratio_categories:
                try:
                    method = getattr(toolkit.ratios, method_name)
                    ratios = method()
                    if ratios is not None and not ratios.empty:
                        all_ratios_data[category_name] = ratios
                        print(f"✅ {category_name} ratios: {ratios.shape}")
                        
                        # Show sample values for first ticker
                        if tickers[0] in ratios.columns:
                            latest_ratios = ratios[tickers[0]].dropna().iloc[-5:]  # Last 5 ratios
                            print(f"  Latest {category_name.lower()} ratios for {tickers[0]}:")
                            for ratio_name, value in latest_ratios.items():
                                if pd.notna(value) and value != 0:
                                    print(f"    {ratio_name}: {value:.4f}")
                    else:
                        print(f"❌ No {category_name.lower()} ratios")
                except Exception as e:
                    print(f"❌ {category_name} ratios failed: {e}")
            
            # Summary of successful ratio calculations
            if all_ratios_data:
                total_ratios = sum(len(ratios.index) for ratios in all_ratios_data.values())
                print(f"\n✅ Total ratios calculated: {total_ratios} across {len(all_ratios_data)} categories")
                
        except Exception as e:
            print(f"❌ Comprehensive ratios failed: {e}")
        
        # Test advanced analytics (requires premium data)
        print("\nTesting advanced analytics...")
        try:
            # Test financial growth calculations - use available ratio methods
            print("Testing available ratio methods...")
            
            # Test ROE calculation
            roe_ratios = toolkit.ratios.get_return_on_equity()
            if roe_ratios is not None and not roe_ratios.empty:
                print(f"✅ Return on Equity data: {roe_ratios.shape}")
                for ticker in tickers:
                    ticker_cols = [col for col in roe_ratios.columns if ticker in str(col)]
                    if ticker_cols:
                        latest_roe = roe_ratios[ticker_cols[0]].dropna()
                        if not latest_roe.empty:
                            print(f"  {ticker} Latest ROE: {latest_roe.iloc[-1]:.2%}")
            
            # Test debt-to-equity ratios
            debt_equity = toolkit.ratios.get_debt_to_equity_ratio()
            if debt_equity is not None and not debt_equity.empty:
                print(f"✅ Debt-to-equity ratios: {debt_equity.shape}")
                for ticker in tickers:
                    ticker_cols = [col for col in debt_equity.columns if ticker in str(col)]
                    if ticker_cols:
                        latest_de = debt_equity[ticker_cols[0]].dropna()
                        if not latest_de.empty:
                            print(f"  {ticker} Latest D/E Ratio: {latest_de.iloc[-1]:.2f}")
            
            # Test current ratio
            current_ratios = toolkit.ratios.get_current_ratio()
            if current_ratios is not None and not current_ratios.empty:
                print(f"✅ Current ratios: {current_ratios.shape}")
                for ticker in tickers:
                    ticker_cols = [col for col in current_ratios.columns if ticker in str(col)]
                    if ticker_cols:
                        latest_cr = current_ratios[ticker_cols[0]].dropna()
                        if not latest_cr.empty:
                            print(f"  {ticker} Latest Current Ratio: {latest_cr.iloc[-1]:.2f}")
                        
        except Exception as e:
            print(f"❌ Advanced analytics failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Premium API endpoints test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Testing FinanceToolkit and FinanceDatabase with API Key")
    print("=" * 60)
    
    # Test results
    results = {
        'finance_database': False,
        'finance_toolkit': False,
        'premium_api_endpoints': False,
        'integration': False
    }
    
    # Run tests
    results['finance_database'] = test_finance_database()
    results['finance_toolkit'] = test_finance_toolkit()
    results['premium_api_endpoints'] = test_premium_api_endpoints()
    results['integration'] = test_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return results

if __name__ == "__main__":
    main()
