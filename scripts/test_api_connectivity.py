#!/usr/bin/env python3
"""
🔗 API Connectivity Test Script
Tests connectivity to all configured API endpoints to ensure data sources are accessible.
This script should be run after API key configuration to verify functionality.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_feeds.api_key_validator import validate_all_api_keys, execute_with_fallback
from data_feeds.finnhub import fetch_finnhub_quote
from data_feeds.enhanced_fmp_integration import EnhancedFMPAdapter
from data_feeds.twelvedata_adapter import TwelveDataAdapter
import requests

def test_finnhub_connectivity() -> Dict[str, Any]:
    """Test Finnhub API connectivity"""
    print("📈 Testing Finnhub API connectivity...")

    try:
        result = fetch_finnhub_quote("AAPL")
        if result and isinstance(result, dict) and 'c' in result:
            return {
                'status': 'success',
                'message': f"Successfully fetched AAPL quote: ${result.get('c', 'N/A')}",
                'data': result
            }
        else:
            return {
                'status': 'error',
                'message': 'Failed to fetch valid quote data from Finnhub',
                'data': result
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Finnhub API error: {str(e)}',
            'data': None
        }

def test_fmp_connectivity() -> Dict[str, Any]:
    """Test Financial Modeling Prep API connectivity"""
    print("💰 Testing Financial Modeling Prep API connectivity...")

    try:
        fmp = EnhancedFMPAdapter()
        if not fmp.api_key:
            return {
                'status': 'error',
                'message': 'FMP API key not configured',
                'data': None
            }

        # Test with a simple quote endpoint
        endpoint = f"quote/AAPL"
        data = fmp._make_request(endpoint)

        if data and isinstance(data, list) and len(data) > 0:
            quote = data[0]
            price = quote.get('price', 'N/A')
            return {
                'status': 'success',
                'message': f"Successfully fetched AAPL quote: ${price}",
                'data': quote
            }
        else:
            return {
                'status': 'error',
                'message': 'Failed to fetch valid quote data from FMP',
                'data': data
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'FMP API error: {str(e)}',
            'data': None
        }

def test_twelvedata_connectivity() -> Dict[str, Any]:
    """Test TwelveData API connectivity"""
    print("📊 Testing TwelveData API connectivity...")

    try:
        adapter = TwelveDataAdapter()
        quote = adapter.get_quote("AAPL")

        if quote and hasattr(quote, 'price') and quote.price:
            return {
                'status': 'success',
                'message': f"Successfully fetched AAPL quote: ${quote.price}",
                'data': {
                    'price': quote.price,
                    'change': quote.change,
                    'volume': quote.volume
                }
            }
        else:
            return {
                'status': 'error',
                'message': 'Failed to fetch valid quote data from TwelveData',
                'data': quote
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'TwelveData API error: {str(e)}',
            'data': None
        }

def test_alphavantage_connectivity() -> Dict[str, Any]:
    """Test Alpha Vantage API connectivity"""
    print("📈 Testing Alpha Vantage API connectivity...")

    try:
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if not api_key:
            return {
                'status': 'error',
                'message': 'Alpha Vantage API key not configured',
                'data': None
            }

        # Test with a simple quote endpoint
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data and data['Global Quote']:
                price = data['Global Quote'].get('05. price', 'N/A')
                return {
                    'status': 'success',
                    'message': f"Successfully fetched AAPL quote: ${price}",
                    'data': data['Global Quote']
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Invalid response format from Alpha Vantage',
                    'data': data
                }
        else:
            return {
                'status': 'error',
                'message': f'Alpha Vantage API error: HTTP {response.status_code}',
                'data': None
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Alpha Vantage API error: {str(e)}',
            'data': None
        }

def test_polygon_connectivity() -> Dict[str, Any]:
    """Test Polygon.io API connectivity"""
    print("📊 Testing Polygon.io API connectivity...")

    try:
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            return {
                'status': 'error',
                'message': 'Polygon.io API key not configured',
                'data': None
            }

        # Test with a simple ticker endpoint
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apikey={api_key}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                price = data['results'][0].get('c', 'N/A')  # Close price
                return {
                    'status': 'success',
                    'message': f"Successfully fetched AAPL data: ${price}",
                    'data': data['results'][0]
                }
            else:
                return {
                    'status': 'error',
                    'message': 'No results returned from Polygon.io',
                    'data': data
                }
        elif response.status_code == 403:
            return {
                'status': 'error',
                'message': 'Invalid Polygon.io API key or insufficient permissions',
                'data': None
            }
        else:
            return {
                'status': 'error',
                'message': f'Polygon.io API error: HTTP {response.status_code}',
                'data': None
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Polygon.io API error: {str(e)}',
            'data': None
        }

def test_twitter_connectivity() -> Dict[str, Any]:
    """Test Twitter connectivity (using twscrape, no API key needed)"""
    print("🐦 Testing Twitter connectivity...")

    try:
        from data_feeds.twitter_feed import TwitterSentimentFeed

        feed = TwitterSentimentFeed()
        # Test with a small limit to avoid rate limits
        tweets = feed.fetch("AAPL", limit=3)

        if tweets and len(tweets) > 0:
            return {
                'status': 'success',
                'message': f"Successfully fetched {len(tweets)} tweets about AAPL",
                'data': {
                    'tweet_count': len(tweets),
                    'sample_text': tweets[0].get('text', '')[:100] + '...' if tweets[0].get('text') else 'No text'
                }
            }
        else:
            return {
                'status': 'warning',
                'message': 'No tweets returned from Twitter (might be rate limited or no recent tweets)',
                'data': tweets
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Twitter connectivity error: {str(e)}',
            'data': None
        }

def run_connectivity_tests() -> Dict[str, Any]:
    """Run all connectivity tests"""
    print("🔗 Starting API Connectivity Tests")
    print("=" * 50)

    # First validate API keys
    print("🔑 Validating API keys...")
    validation = validate_all_api_keys()

    if validation['summary']['required_missing'] > 0:
        print("❌ Cannot run connectivity tests - missing required API keys")
        return {
            'timestamp': datetime.now(),
            'status': 'failed',
            'reason': 'missing_api_keys',
            'validation': validation
        }

    # Define test functions
    test_functions = {
        'finnhub': test_finnhub_connectivity,
        'fmp': test_fmp_connectivity,
        'twelvedata': test_twelvedata_connectivity,
        'alphavantage': test_alphavantage_connectivity,
        'polygon': test_polygon_connectivity,
        'twitter': test_twitter_connectivity
    }

    results = {
        'timestamp': datetime.now(),
        'tests': {},
        'summary': {
            'total': len(test_functions),
            'passed': 0,
            'failed': 0,
            'warnings': 0
        }
    }

    # Run each test
    for service_name, test_func in test_functions.items():
        print(f"\n{'-'*30}")
        result = test_func()
        results['tests'][service_name] = result

        if result['status'] == 'success':
            results['summary']['passed'] += 1
            print(f"✅ {service_name.upper()}: {result['message']}")
        elif result['status'] == 'warning':
            results['summary']['warnings'] += 1
            print(f"⚠️  {service_name.upper()}: {result['message']}")
        else:
            results['summary']['failed'] += 1
            print(f"❌ {service_name.upper()}: {result['message']}")

    # Overall status
    summary = results['summary']
    if summary['failed'] > 0:
        overall_status = 'failed'
        overall_message = f"{summary['failed']} API(s) failed connectivity tests"
    elif summary['warnings'] > 0:
        overall_status = 'warning'
        overall_message = f"{summary['warnings']} API(s) have warnings"
    else:
        overall_status = 'success'
        overall_message = "All APIs passed connectivity tests"

    results['summary']['overall_status'] = overall_status
    results['summary']['overall_message'] = overall_message

    print(f"\n{'='*50}")
    print("📋 CONNECTIVITY TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Total APIs tested: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"\nOverall Status: {'✅' if overall_status == 'success' else '⚠️' if overall_status == 'warning' else '❌'} {overall_message}")

    return results

def main():
    """Main function"""
    try:
        results = run_connectivity_tests()

        # Exit with appropriate code
        if results['summary']['overall_status'] == 'failed':
            print("\n❌ Some APIs failed connectivity tests")
            sys.exit(1)
        elif results['summary']['overall_status'] == 'warning':
            print("\n⚠️  All APIs are accessible but some have warnings")
            sys.exit(0)
        else:
            print("\n✅ All APIs passed connectivity tests")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Connectivity tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Connectivity tests failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()