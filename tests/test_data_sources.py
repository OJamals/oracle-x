#!/usr/bin/env python3
"""
Test script to evaluate all financial data sources and their capabilities.
This will help us understand what endpoints work and their limitations.
"""

import os
import sys
import time
from datetime import datetime, timedelta
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_yfinance():
    """Test yfinance capabilities"""
    logger.info("Testing yfinance...")
    results = {}

    try:
        import yfinance as yf

        # Test basic stock data
        ticker = yf.Ticker("AAPL")

        # Get info
        info = ticker.info
        results["info"] = bool(info and len(info) > 0)

        # Get historical data
        hist = ticker.history(period="1mo")
        results["historical_data"] = not hist.empty

        # Get financials
        try:
            financials = ticker.financials
            results["financials"] = not financials.empty
        except:
            results["financials"] = False

        # Get recommendations
        try:
            recommendations = ticker.recommendations
            results["recommendations"] = (
                recommendations is not None and len(recommendations) > 0
            )
        except:
            results["recommendations"] = False

        # Get news
        try:
            news = ticker.news
            results["news"] = bool(news and len(news) > 0)
        except:
            results["news"] = False

        results["status"] = "SUCCESS"
        results["rate_limit"] = "None - Free unlimited for most endpoints"

    except Exception as e:
        results["status"] = f"ERROR: {str(e)}"

    logger.info(f"yfinance results: {results}")
    return results


def test_investiny():
    """Test investiny capabilities comprehensively"""
    logger.info("Testing investiny...")
    results = {}

    try:
        from investiny import historical_data, search_assets, info

        # Test asset search with different queries
        try:
            search_results = search_assets(query="Apple", limit=5)
            results["search_assets"] = bool(search_results and len(search_results) > 0)
            if results["search_assets"]:
                logger.info(f"Found {len(search_results)} assets for 'Apple'")
                # Log first result for debugging
                logger.info(
                    f"First result: {search_results[0] if search_results else 'None'}"
                )
        except Exception as e:
            logger.warning(f"Investiny asset search error: {e}")
            results["search_assets"] = False

        # Test crypto search
        try:
            crypto_results = search_assets(query="Bitcoin", limit=3)
            results["search_crypto"] = bool(crypto_results and len(crypto_results) > 0)
        except Exception as e:
            logger.warning(f"Investiny crypto search error: {e}")
            results["search_crypto"] = False

        # Test historical data for stocks
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            hist_data = historical_data("AAPL", start_date, end_date)
            results["historical_stocks"] = bool(hist_data and len(hist_data) > 0)
            if results["historical_stocks"]:
                logger.info(
                    f"Retrieved {len(hist_data)} historical data points for AAPL"
                )
        except Exception as e:
            logger.warning(f"Investiny historical stocks error: {e}")
            results["historical_stocks"] = False

        # Test historical data for different assets
        try:
            # Test with EUR/USD
            forex_data = historical_data("EUR/USD", start_date, end_date)
            results["historical_forex"] = bool(forex_data and len(forex_data) > 0)
        except Exception as e:
            logger.warning(f"Investiny forex historical error: {e}")
            results["historical_forex"] = False

        # Test crypto historical data
        try:
            crypto_data = historical_data("BTC/USD", start_date, end_date)
            results["historical_crypto"] = bool(crypto_data and len(crypto_data) > 0)
        except Exception as e:
            logger.warning(f"Investiny crypto historical error: {e}")
            results["historical_crypto"] = False

        # Test asset info functionality
        try:
            asset_info = info("AAPL")
            results["asset_info"] = bool(asset_info)
            if results["asset_info"]:
                logger.info(f"Retrieved info for AAPL: {type(asset_info)}")
        except Exception as e:
            logger.warning(f"Investiny asset info error: {e}")
            results["asset_info"] = False

        # Test different time periods
        try:
            # Test 1 year data
            year_start = end_date - timedelta(days=365)
            year_data = historical_data("AAPL", year_start, end_date)
            results["historical_1year"] = bool(year_data and len(year_data) > 0)
        except Exception as e:
            logger.warning(f"Investiny 1-year historical error: {e}")
            results["historical_1year"] = False

        results["status"] = "SUCCESS"
        results["rate_limit"] = "Free with unknown limits - appears to use web scraping"

    except Exception as e:
        results["status"] = f"ERROR: {str(e)}"

    logger.info(f"investiny results: {results}")
    return results


def test_financetoolkit():
    """Test financetoolkit capabilities"""
    logger.info("Testing financetoolkit...")
    results = {}

    try:
        from financetoolkit import Toolkit

        # Initialize toolkit
        toolkit = Toolkit(
            tickers=["AAPL"],
            api_key=os.getenv("FINANCIALMODELINGPREP_API_KEY"),
            start_date="2023-01-01",
            end_date="2024-01-01",
        )

        # Test historical data
        try:
            historical = toolkit.get_historical_data()
            results["historical_data"] = not historical.empty
        except:
            results["historical_data"] = False

        # Test balance sheet
        try:
            balance_sheet = toolkit.get_balance_sheet_statement()
            results["balance_sheet"] = not balance_sheet.empty
        except:
            results["balance_sheet"] = False

        # Test income statement
        try:
            income = toolkit.get_income_statement()
            results["income_statement"] = not income.empty
        except:
            results["income_statement"] = False

        # Test cash flow
        try:
            cash_flow = toolkit.get_cash_flow_statement()
            results["cash_flow"] = not cash_flow.empty
        except:
            results["cash_flow"] = False

        # Test ratios
        try:
            ratios = toolkit.ratios.collect_efficiency_ratios()
            results["ratios"] = not ratios.empty
        except:
            results["ratios"] = False

        results["status"] = "SUCCESS"
        results["rate_limit"] = "Depends on FMP API (250 calls/day free)"

    except Exception as e:
        results["status"] = f"ERROR: {str(e)}"

    logger.info(f"financetoolkit results: {results}")
    return results


def test_financedatabase():
    """Test financedatabase capabilities"""
    logger.info("Testing financedatabase...")
    results = {}

    try:
        import financedatabase as fd

        # Test equities database
        try:
            equities = fd.Equities()
            apple_data = equities.select(symbol="AAPL")
            results["equities"] = bool(apple_data and len(apple_data) > 0)
        except:
            results["equities"] = False

        # Test ETFs database
        try:
            etfs = fd.ETFs()
            spy_data = etfs.select(symbol="SPY")
            results["etfs"] = bool(spy_data and len(spy_data) > 0)
        except:
            results["etfs"] = False

        # Test funds database
        try:
            funds = fd.Funds()
            results["funds"] = True  # Just test instantiation
        except:
            results["funds"] = False

        # Test indices
        try:
            indices = fd.Indices()
            results["indices"] = True  # Just test instantiation
        except:
            results["indices"] = False

        # Test currencies
        try:
            currencies = fd.Currencies()
            results["currencies"] = True
        except:
            results["currencies"] = False

        # Test cryptocurrencies
        try:
            crypto = fd.Cryptocurrencies()
            results["cryptocurrencies"] = True
        except:
            results["cryptocurrencies"] = False

        results["status"] = "SUCCESS"
        results["rate_limit"] = "Free - local database"

    except Exception as e:
        results["status"] = f"ERROR: {str(e)}"

    logger.info(f"financedatabase results: {results}")
    return results


def test_quantsumore():
    """Test quantsumore capabilities"""
    logger.info("Testing quantsumore...")
    results = {}

    try:
        from quantsumore import fetch_data

        # Test basic data fetch
        try:
            data = fetch_data("AAPL", period="1mo")
            results["basic_fetch"] = bool(data is not None)
        except Exception as e:
            logger.warning(f"Quantsumore basic fetch error: {e}")
            results["basic_fetch"] = False

        results["status"] = "SUCCESS"
        results["rate_limit"] = "Free with unknown limits"

    except Exception as e:
        results["status"] = f"ERROR: {str(e)}"

    logger.info(f"quantsumore results: {results}")
    return results


def test_stockdex():
    """Test stockdex capabilities"""
    logger.info("Testing stockdex...")
    results = {}

    try:
        from stockdex import Ticker

        # Test ticker creation and data retrieval
        try:
            ticker = Ticker("AAPL")

            # Test basic info
            try:
                info = ticker.get_info()
                results["info"] = bool(info)
            except:
                results["info"] = False

            # Test historical data
            try:
                hist = ticker.get_historical_data(period="1mo")
                results["historical_data"] = bool(hist)
            except:
                results["historical_data"] = False

            # Test financial data
            try:
                financials = ticker.get_financials()
                results["financials"] = bool(financials)
            except:
                results["financials"] = False

        except Exception as e:
            logger.warning(f"Stockdex ticker error: {e}")
            results["info"] = False
            results["historical_data"] = False
            results["financials"] = False

        results["status"] = "SUCCESS"
        results["rate_limit"] = "Free with scraping limitations"

    except Exception as e:
        results["status"] = f"ERROR: {str(e)}"

    logger.info(f"stockdex results: {results}")
    return results


def test_finnhub():
    """Test finnhub capabilities"""
    logger.info("Testing finnhub...")
    results = {}

    try:
        import finnhub

        # Initialize client
        finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

        # Test quote
        try:
            quote = finnhub_client.quote("AAPL")
            results["quote"] = bool(quote and "c" in quote)
        except:
            results["quote"] = False

        # Test company profile
        try:
            profile = finnhub_client.company_profile2(symbol="AAPL")
            results["company_profile"] = bool(profile and "name" in profile)
        except:
            results["company_profile"] = False

        # Test candles (historical data)
        try:
            from datetime import datetime

            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=30)).timestamp())
            candles = finnhub_client.stock_candles("AAPL", "D", start_time, end_time)
            results["candles"] = bool(
                candles and "s" in candles and candles["s"] == "ok"
            )
        except:
            results["candles"] = False

        # Test news
        try:
            news = finnhub_client.company_news(
                "AAPL", _from="2024-01-01", to="2024-01-31"
            )
            results["news"] = bool(news and len(news) > 0)
        except:
            results["news"] = False

        # Test financials
        try:
            financials = finnhub_client.financials("AAPL", "annual")
            results["financials"] = bool(financials and "metric" in financials)
        except:
            results["financials"] = False

        # Test earnings
        try:
            earnings = finnhub_client.earnings("AAPL")
            results["earnings"] = bool(earnings and len(earnings) > 0)
        except:
            results["earnings"] = False

        results["status"] = "SUCCESS"
        results["rate_limit"] = "60 calls/minute free tier"

    except Exception as e:
        results["status"] = f"ERROR: {str(e)}"

    logger.info(f"finnhub results: {results}")
    return results


def test_alpha_vantage():
    """Test Alpha Vantage via existing module"""
    logger.info("Testing Alpha Vantage...")
    results = {}

    try:
        # Try to use existing alpha vantage module
        sys.path.append("/Users/omar/Documents/Projects/oracle-x/data_feeds")
        from alpha_vantage import AlphaVantageAPI

        av = AlphaVantageAPI()

        # Test daily data
        try:
            daily_data = av.get_daily_prices("AAPL")
            results["daily_prices"] = bool(daily_data and len(daily_data) > 0)
        except:
            results["daily_prices"] = False

        # Test intraday data
        try:
            intraday_data = av.get_intraday_prices("AAPL", interval="5min")
            results["intraday_prices"] = bool(intraday_data and len(intraday_data) > 0)
        except:
            results["intraday_prices"] = False

        # Test company overview
        try:
            overview = av.get_company_overview("AAPL")
            results["company_overview"] = bool(overview and len(overview) > 0)
        except:
            results["company_overview"] = False

        results["status"] = "SUCCESS"
        results["rate_limit"] = "5 calls/minute, 500 calls/day free"

    except Exception as e:
        results["status"] = f"ERROR: {str(e)}"

    logger.info(f"alpha_vantage results: {results}")
    return results


def test_financial_modeling_prep():
    """Test Financial Modeling Prep API directly"""
    logger.info("Testing Financial Modeling Prep...")
    results = {}

    try:
        import requests

        api_key = os.getenv("FINANCIALMODELINGPREP_API_KEY")
        base_url = "https://financialmodelingprep.com/api/v3"

        # Test quote
        try:
            response = requests.get(f"{base_url}/quote/AAPL?apikey={api_key}")
            if response.status_code == 200:
                data = response.json()
                results["quote"] = bool(data and len(data) > 0)
            else:
                results["quote"] = False
        except:
            results["quote"] = False

        # Test profile
        try:
            response = requests.get(f"{base_url}/profile/AAPL?apikey={api_key}")
            if response.status_code == 200:
                data = response.json()
                results["profile"] = bool(data and len(data) > 0)
            else:
                results["profile"] = False
        except:
            results["profile"] = False

        # Test historical data
        try:
            response = requests.get(
                f"{base_url}/historical-price-full/AAPL?from=2024-01-01&to=2024-01-31&apikey={api_key}"
            )
            if response.status_code == 200:
                data = response.json()
                results["historical"] = bool(data and "historical" in data)
            else:
                results["historical"] = False
        except:
            results["historical"] = False

        # Test income statement
        try:
            response = requests.get(
                f"{base_url}/income-statement/AAPL?period=annual&apikey={api_key}"
            )
            if response.status_code == 200:
                data = response.json()
                results["income_statement"] = bool(data and len(data) > 0)
            else:
                results["income_statement"] = False
        except:
            results["income_statement"] = False

        # Test balance sheet
        try:
            response = requests.get(
                f"{base_url}/balance-sheet-statement/AAPL?period=annual&apikey={api_key}"
            )
            if response.status_code == 200:
                data = response.json()
                results["balance_sheet"] = bool(data and len(data) > 0)
            else:
                results["balance_sheet"] = False
        except:
            results["balance_sheet"] = False

        # Test cash flow
        try:
            response = requests.get(
                f"{base_url}/cash-flow-statement/AAPL?period=annual&apikey={api_key}"
            )
            if response.status_code == 200:
                data = response.json()
                results["cash_flow"] = bool(data and len(data) > 0)
            else:
                results["cash_flow"] = False
        except:
            results["cash_flow"] = False

        results["status"] = "SUCCESS"
        results["rate_limit"] = "250 calls/day free, 500MB bandwidth/month"

    except Exception as e:
        results["status"] = f"ERROR: {str(e)}"

    logger.info(f"financial_modeling_prep results: {results}")
    return results


def run_all_tests():
    """Run all data source tests and compile results"""
    logger.info("Starting comprehensive data source testing...")

    all_results = {}

    # Test each data source
    test_functions = [
        ("yfinance", test_yfinance),
        ("investiny", test_investiny),
        ("financetoolkit", test_financetoolkit),
        ("financedatabase", test_financedatabase),
        ("quantsumore", test_quantsumore),
        ("stockdex", test_stockdex),
        ("finnhub", test_finnhub),
        ("alpha_vantage", test_alpha_vantage),
        ("financial_modeling_prep", test_financial_modeling_prep),
    ]

    for name, test_func in test_functions:
        try:
            logger.info(f"Testing {name}...")
            results = test_func()
            all_results[name] = results

            # Add delay to respect rate limits
            time.sleep(2)

        except Exception as e:
            logger.error(f"Failed to test {name}: {e}")
            all_results[name] = {"status": f"CRITICAL_ERROR: {str(e)}"}

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/Users/omar/Documents/Projects/oracle-x/data_source_test_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Test results saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("DATA SOURCE TEST SUMMARY")
    print("=" * 80)

    for source, results in all_results.items():
        status = results.get("status", "UNKNOWN")
        rate_limit = results.get("rate_limit", "Unknown")

        print(f"\n{source.upper()}:")
        print(f"  Status: {status}")
        print(f"  Rate Limit: {rate_limit}")

        if status == "SUCCESS":
            # Count successful endpoints
            successful_endpoints = sum(
                1
                for k, v in results.items()
                if k not in ["status", "rate_limit"] and v is True
            )
            total_endpoints = len(
                [k for k in results.keys() if k not in ["status", "rate_limit"]]
            )
            print(f"  Working Endpoints: {successful_endpoints}/{total_endpoints}")

            # List working endpoints
            working = [
                k
                for k, v in results.items()
                if k not in ["status", "rate_limit"] and v is True
            ]
            if working:
                print(f"  Available: {', '.join(working)}")

    print("\n" + "=" * 80)
    return all_results


if __name__ == "__main__":
    results = run_all_tests()
