#!/usr/bin/env python3
"""
Comprehensive testing script for investiny, quantsumore, and stockdex data sources.
This will thoroughly evaluate their capabilities, limitations, and available endpoints.
CORRECTED VERSION with proper API usage.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_investiny_comprehensive():
    """Comprehensive test of investiny capabilities"""
    logger.info("Starting comprehensive investiny testing...")
    results = {}
    
    try:
        from investiny import historical_data, search_assets, info
        
        # Test 1: Asset search functionality
        logger.info("Testing asset search...")
        try:
            # Search for stocks
            stock_results = search_assets("Apple", limit=5)
            results['search_stocks'] = bool(stock_results and len(stock_results) > 0)
            if results['search_stocks']:
                logger.info(f"Found {len(stock_results)} stock results for 'Apple'")
                logger.info(f"Sample stock result: {stock_results[0]}")
                
            # Search for crypto
            crypto_results = search_assets("Bitcoin", limit=3)
            results['search_crypto'] = bool(crypto_results and len(crypto_results) > 0)
            if results['search_crypto']:
                logger.info(f"Found {len(crypto_results)} crypto results for 'Bitcoin'")
                
            # Search for forex
            forex_results = search_assets("EUR/USD", limit=3)
            results['search_forex'] = bool(forex_results and len(forex_results) > 0)
            if results['search_forex']:
                logger.info(f"Found {len(forex_results)} forex results for 'EUR/USD'")
                
        except Exception as e:
            logger.error(f"Asset search error: {e}")
            results['search_stocks'] = False
            results['search_crypto'] = False
            results['search_forex'] = False
        
        # Test 2: Historical data for stocks
        logger.info("Testing stock historical data...")
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Get Apple ticker ID from search results
            apple_results = search_assets("Apple", limit=1)
            if apple_results:
                apple_ticker_id = int(apple_results[0]['ticker'])
                hist_data = historical_data(
                    apple_ticker_id, 
                    start_date.strftime('%m/%d/%Y'), 
                    end_date.strftime('%m/%d/%Y')
                )
                results['historical_stocks'] = bool(hist_data and len(hist_data) > 0)
                if results['historical_stocks']:
                    logger.info(f"Retrieved {len(hist_data)} days of historical data for Apple")
                    logger.info(f"Sample data keys: {list(hist_data.keys())}")
                    # Check first date's data structure
                    first_date = list(hist_data.keys())[0] if hist_data else None
                    if first_date:
                        logger.info(f"Sample data point: {hist_data[first_date]}")
            else:
                results['historical_stocks'] = False
                
        except Exception as e:
            logger.error(f"Stock historical data error: {e}")
            results['historical_stocks'] = False
        
        # Test 3: Historical data for different timeframes
        logger.info("Testing different timeframes...")
        try:
            if apple_results:
                apple_ticker_id = int(apple_results[0]['ticker'])
                
                # 1 week data
                week_start = end_date - timedelta(days=7)
                week_data = historical_data(
                    apple_ticker_id,
                    week_start.strftime('%m/%d/%Y'),
                    end_date.strftime('%m/%d/%Y')
                )
                results['historical_1week'] = bool(week_data and len(week_data) > 0)
                
                # 3 months data
                month3_start = end_date - timedelta(days=90)
                month3_data = historical_data(
                    apple_ticker_id,
                    month3_start.strftime('%m/%d/%Y'),
                    end_date.strftime('%m/%d/%Y')
                )
                results['historical_3months'] = bool(month3_data and len(month3_data) > 0)
                
        except Exception as e:
            logger.error(f"Timeframe testing error: {e}")
            results['historical_1week'] = False
            results['historical_3months'] = False
        
        # Test 4: Asset info functionality
        logger.info("Testing asset info...")
        try:
            if apple_results:
                apple_ticker_id = int(apple_results[0]['ticker'])
                asset_info = info(apple_ticker_id)
                results['asset_info'] = bool(asset_info)
                if results['asset_info']:
                    logger.info(f"Asset info type: {type(asset_info)}")
                    logger.info(f"Asset info sample: {str(asset_info)[:200]}...")
        except Exception as e:
            logger.error(f"Asset info error: {e}")
            results['asset_info'] = False
        
        # Test 5: Different asset types
        logger.info("Testing different asset types...")
        try:
            # Test crypto historical data
            crypto_results = search_assets("Bitcoin", limit=1)
            if crypto_results:
                crypto_ticker_id = int(crypto_results[0]['ticker'])
                crypto_hist = historical_data(
                    crypto_ticker_id,
                    start_date.strftime('%m/%d/%Y'),
                    end_date.strftime('%m/%d/%Y')
                )
                results['historical_crypto'] = bool(crypto_hist and len(crypto_hist) > 0)
            else:
                results['historical_crypto'] = False
                
            # Test forex historical data  
            forex_results = search_assets("EUR/USD", limit=1)
            if forex_results:
                forex_ticker_id = int(forex_results[0]['ticker'])
                forex_hist = historical_data(
                    forex_ticker_id,
                    start_date.strftime('%m/%d/%Y'),
                    end_date.strftime('%m/%d/%Y')
                )
                results['historical_forex'] = bool(forex_hist and len(forex_hist) > 0)
            else:
                results['historical_forex'] = False
                
        except Exception as e:
            logger.error(f"Different asset types error: {e}")
            results['historical_crypto'] = False
            results['historical_forex'] = False
        
        results['status'] = 'SUCCESS'
        results['rate_limit'] = 'Free with web scraping - no official rate limits'
        results['data_coverage'] = 'Stocks, crypto, forex, indices, commodities'
        
    except Exception as e:
        results['status'] = f'ERROR: {str(e)}'
        logger.error(f"Investiny critical error: {e}")
        
    logger.info(f"Investiny test results: {results}")
    return results

def test_quantsumore_comprehensive():
    """Comprehensive test of quantsumore capabilities"""
    logger.info("Starting comprehensive quantsumore testing...")
    results = {}
    
    try:
        # Note: quantsumore has disrupted equity endpoints due to Yahoo Finance protections
        # Let's test what's still available
        
        # Test 1: CPI data
        logger.info("Testing CPI data...")
        try:
            from quantsumore.econ.country.usa import CPI
            cpi = CPI()
            
            # Test latest CPI data
            latest_cpi = cpi.latest()
            results['cpi_latest'] = bool(latest_cpi)
            if results['cpi_latest']:
                logger.info(f"Latest CPI data: {latest_cpi}")
                
            # Test historical CPI data
            hist_cpi = cpi.last_n(n=12)  # Last 12 months
            results['cpi_historical'] = bool(hist_cpi and len(hist_cpi) > 0)
            if results['cpi_historical']:
                logger.info(f"Retrieved {len(hist_cpi)} CPI data points")
                
        except Exception as e:
            logger.error(f"CPI data error: {e}")
            results['cpi_latest'] = False
            results['cpi_historical'] = False
        
        # Test 2: Treasury data
        logger.info("Testing Treasury data...")
        try:
            from quantsumore.econ.country.usa import Treasury
            treasury = Treasury()
            
            # Test latest treasury rates
            latest_treasury = treasury.latest()
            results['treasury_latest'] = bool(latest_treasury)
            if results['treasury_latest']:
                logger.info(f"Latest Treasury data type: {type(latest_treasury)}")
                
            # Test historical treasury data
            hist_treasury = treasury.last_n(n=10)
            results['treasury_historical'] = bool(hist_treasury and len(hist_treasury) > 0)
            if results['treasury_historical']:
                logger.info(f"Retrieved {len(hist_treasury)} Treasury data points")
                
        except Exception as e:
            logger.error(f"Treasury data error: {e}")
            results['treasury_latest'] = False
            results['treasury_historical'] = False
        
        # Test 3: Technical analysis tools
        logger.info("Testing technical analysis...")
        try:
            from quantsumore.analysis.technical import RSI, SMA, EMA
            import pandas as pd
            import numpy as np
            
            # Create sample price data for testing
            dates = pd.date_range('2024-01-01', periods=50, freq='D')
            prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5), index=dates)
            
            # Test RSI calculation
            rsi_values = RSI(prices, period=14)
            results['technical_rsi'] = bool(rsi_values is not None and len(rsi_values) > 0)
            
            # Test SMA calculation
            sma_values = SMA(prices, period=20)
            results['technical_sma'] = bool(sma_values is not None and len(sma_values) > 0)
            
            # Test EMA calculation
            ema_values = EMA(prices, period=20)
            results['technical_ema'] = bool(ema_values is not None and len(ema_values) > 0)
            
            if results['technical_rsi']:
                logger.info(f"RSI calculation successful, latest value: {rsi_values.iloc[-1]:.2f}")
                
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            results['technical_rsi'] = False
            results['technical_sma'] = False
            results['technical_ema'] = False
        
        # Test 4: Fundamental analysis (if available)
        logger.info("Testing fundamental analysis...")
        try:
            from quantsumore.analysis.fundamental import financial_ratios
            # This may not work due to the Yahoo Finance disruption
            results['fundamental_analysis'] = False  # Assume unavailable due to notice
            logger.info("Fundamental analysis unavailable due to Yahoo Finance disruption")
            
        except Exception as e:
            logger.error(f"Fundamental analysis error: {e}")
            results['fundamental_analysis'] = False
        
        # Test 5: Check for forex capabilities
        logger.info("Testing forex capabilities...")
        try:
            # Try to access forex data if available
            from quantsumore.econ.country.usa import Forex
            forex = Forex()
            forex_data = forex.latest()
            results['forex_data'] = bool(forex_data)
            
        except Exception as e:
            logger.error(f"Forex data error: {e}")
            results['forex_data'] = False
        
        results['status'] = 'PARTIAL_SUCCESS'
        results['rate_limit'] = 'Free with limitations - equity endpoints disrupted'
        results['data_coverage'] = 'Economic data (CPI, Treasury), technical analysis tools'
        results['limitations'] = 'Yahoo Finance disruption affects equity endpoints'
        
    except Exception as e:
        results['status'] = f'ERROR: {str(e)}'
        logger.error(f"Quantsumore critical error: {e}")
        
    logger.info(f"Quantsumore test results: {results}")
    return results

def test_stockdex_comprehensive():
    """Comprehensive test of stockdex capabilities"""
    logger.info("Starting comprehensive stockdex testing...")
    results = {}
    
    try:
        from stockdex import Ticker
        
        # Test 1: Basic ticker functionality
        logger.info("Testing basic ticker functionality...")
        try:
            ticker = Ticker("AAPL")
            results['ticker_creation'] = True
            logger.info("Ticker creation successful")
            
        except Exception as e:
            logger.error(f"Ticker creation error: {e}")
            results['ticker_creation'] = False
            return results
        
        # Test 2: Company information
        logger.info("Testing company information...")
        try:
            company_info = ticker.company_info()
            results['company_info'] = bool(company_info)
            if results['company_info']:
                logger.info(f"Company info type: {type(company_info)}")
                logger.info(f"Company info keys: {list(company_info.keys()) if isinstance(company_info, dict) else 'Not a dict'}")
                
        except Exception as e:
            logger.error(f"Company info error: {e}")
            results['company_info'] = False
        
        # Test 3: Stock price data
        logger.info("Testing stock price data...")
        try:
            price_data = ticker.price()
            results['price_data'] = bool(price_data)
            if results['price_data']:
                logger.info(f"Price data: {price_data}")
                
        except Exception as e:
            logger.error(f"Price data error: {e}")
            results['price_data'] = False
        
        # Test 4: Historical data
        logger.info("Testing historical data...")
        try:
            # Test different periods
            hist_1m = ticker.historical_data(period="1mo")
            results['historical_1month'] = bool(hist_1m is not None)
            
            hist_3m = ticker.historical_data(period="3mo")
            results['historical_3months'] = bool(hist_3m is not None)
            
            hist_1y = ticker.historical_data(period="1y")
            results['historical_1year'] = bool(hist_1y is not None)
            
            if results['historical_1month']:
                logger.info(f"1-month historical data type: {type(hist_1m)}")
                
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            results['historical_1month'] = False
            results['historical_3months'] = False
            results['historical_1year'] = False
        
        # Test 5: Financial statements
        logger.info("Testing financial statements...")
        try:
            # Income statement
            income_stmt = ticker.income_statement()
            results['income_statement'] = bool(income_stmt is not None)
            
            # Balance sheet
            balance_sheet = ticker.balance_sheet()
            results['balance_sheet'] = bool(balance_sheet is not None)
            
            # Cash flow statement
            cash_flow = ticker.cash_flow_statement()
            results['cash_flow'] = bool(cash_flow is not None)
            
            if results['income_statement']:
                logger.info(f"Income statement type: {type(income_stmt)}")
                
        except Exception as e:
            logger.error(f"Financial statements error: {e}")
            results['income_statement'] = False
            results['balance_sheet'] = False
            results['cash_flow'] = False
        
        # Test 6: Key statistics
        logger.info("Testing key statistics...")
        try:
            key_stats = ticker.key_statistics()
            results['key_statistics'] = bool(key_stats)
            if results['key_statistics']:
                logger.info(f"Key statistics type: {type(key_stats)}")
                
        except Exception as e:
            logger.error(f"Key statistics error: {e}")
            results['key_statistics'] = False
        
        # Test 7: Analyst recommendations
        logger.info("Testing analyst recommendations...")
        try:
            recommendations = ticker.analyst_recommendations()
            results['analyst_recommendations'] = bool(recommendations)
            if results['analyst_recommendations']:
                logger.info(f"Recommendations type: {type(recommendations)}")
                
        except Exception as e:
            logger.error(f"Analyst recommendations error: {e}")
            results['analyst_recommendations'] = False
        
        # Test 8: Insider transactions
        logger.info("Testing insider transactions...")
        try:
            insider_tx = ticker.insider_transactions()
            results['insider_transactions'] = bool(insider_tx)
            if results['insider_transactions']:
                logger.info(f"Insider transactions type: {type(insider_tx)}")
                
        except Exception as e:
            logger.error(f"Insider transactions error: {e}")
            results['insider_transactions'] = False
        
        # Test 9: Options data
        logger.info("Testing options data...")
        try:
            options = ticker.options()
            results['options_data'] = bool(options)
            if results['options_data']:
                logger.info(f"Options data type: {type(options)}")
                
        except Exception as e:
            logger.error(f"Options data error: {e}")
            results['options_data'] = False
        
        # Test 10: Different ticker symbols
        logger.info("Testing different ticker symbols...")
        try:
            # Test with a different stock
            msft_ticker = Ticker("MSFT")
            msft_price = msft_ticker.price()
            results['multi_ticker_support'] = bool(msft_price)
            
        except Exception as e:
            logger.error(f"Multi-ticker error: {e}")
            results['multi_ticker_support'] = False
        
        results['status'] = 'SUCCESS'
        results['rate_limit'] = 'Free with web scraping limitations'
        results['data_coverage'] = 'Comprehensive stock data including financials, options, insider trades'
        
    except Exception as e:
        results['status'] = f'ERROR: {str(e)}'
        logger.error(f"Stockdex critical error: {e}")
        
    logger.info(f"Stockdex test results: {results}")
    return results

def generate_comprehensive_report(all_results: Dict[str, Any]):
    """Generate a comprehensive report of all test results"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE DATA SOURCE ANALYSIS REPORT")
    print("="*100)
    
    for source_name, results in all_results.items():
        print(f"\n📊 {source_name.upper()}")
        print("-" * 80)
        
        status = results.get('status', 'UNKNOWN')
        print(f"Overall Status: {status}")
        
        if 'rate_limit' in results:
            print(f"Rate Limits: {results['rate_limit']}")
            
        if 'data_coverage' in results:
            print(f"Data Coverage: {results['data_coverage']}")
            
        if 'limitations' in results:
            print(f"Limitations: {results['limitations']}")
        
        # Count successful endpoints
        test_results = {k: v for k, v in results.items() 
                       if k not in ['status', 'rate_limit', 'data_coverage', 'limitations']}
        
        if test_results:
            successful = sum(1 for v in test_results.values() if v is True)
            total = len(test_results)
            success_rate = (successful / total) * 100 if total > 0 else 0
            
            print(f"Success Rate: {successful}/{total} ({success_rate:.1f}%)")
            
            # Working endpoints
            working_endpoints = [k for k, v in test_results.items() if v is True]
            if working_endpoints:
                print(f"✅ Working Endpoints: {', '.join(working_endpoints)}")
            
            # Failed endpoints
            failed_endpoints = [k for k, v in test_results.items() if v is False]
            if failed_endpoints:
                print(f"❌ Failed Endpoints: {', '.join(failed_endpoints)}")
    
    print("\n" + "="*100)
    print("INTEGRATION RECOMMENDATIONS")
    print("="*100)
    
    # Generate recommendations based on results
    for source_name, results in all_results.items():
        status = results.get('status', 'UNKNOWN')
        if 'SUCCESS' in status:
            test_results = {k: v for k, v in results.items() 
                           if k not in ['status', 'rate_limit', 'data_coverage', 'limitations']}
            successful = sum(1 for v in test_results.values() if v is True)
            
            if successful > 0:
                print(f"\n✅ {source_name.upper()}: RECOMMENDED FOR INTEGRATION")
                print(f"   Best for: {results.get('data_coverage', 'Various data types')}")
                print(f"   Rate limits: {results.get('rate_limit', 'Unknown')}")
                
                working_endpoints = [k for k, v in test_results.items() if v is True]
                print(f"   Priority endpoints: {', '.join(working_endpoints[:3])}...")
            else:
                print(f"\n⚠️  {source_name.upper()}: NOT RECOMMENDED")
                print(f"   Reason: No working endpoints found")
        else:
            print(f"\n❌ {source_name.upper()}: NOT AVAILABLE")
            print(f"   Reason: {status}")

def main():
    """Run comprehensive tests for all remaining data sources"""
    logger.info("Starting comprehensive testing of remaining data sources...")
    
    # Run all tests
    all_results = {}
    
    # Test investiny
    try:
        all_results['investiny'] = test_investiny_comprehensive()
        time.sleep(2)  # Rate limiting precaution
    except Exception as e:
        logger.error(f"Failed to test investiny: {e}")
        all_results['investiny'] = {'status': f'CRITICAL_ERROR: {str(e)}'}
    
    # Test quantsumore  
    try:
        all_results['quantsumore'] = test_quantsumore_comprehensive()
        time.sleep(2)  # Rate limiting precaution
    except Exception as e:
        logger.error(f"Failed to test quantsumore: {e}")
        all_results['quantsumore'] = {'status': f'CRITICAL_ERROR: {str(e)}'}
    
    # Test stockdex
    try:
        all_results['stockdex'] = test_stockdex_comprehensive()
        time.sleep(2)  # Rate limiting precaution
    except Exception as e:
        logger.error(f"Failed to test stockdex: {e}")
        all_results['stockdex'] = {'status': f'CRITICAL_ERROR: {str(e)}'}
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/Users/omar/Documents/Projects/oracle-x/remaining_sources_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Detailed test results saved to: {results_file}")
    
    # Generate comprehensive report
    generate_comprehensive_report(all_results)
    
    return all_results

if __name__ == "__main__":
    results = main()
