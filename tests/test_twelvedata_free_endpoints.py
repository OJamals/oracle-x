import os
import requests
import json

def test_free_endpoints():
    api_key = os.getenv('TWELVEDATA_API_KEY')
    if not api_key:
        print("TWELVEDATA_API_KEY not found in environment")
        return
    
    endpoints = [
        'quote',
        'time_series', 
        'symbol_search',
        'exchange',
        'cryptocurrencies',
        'forex_pairs',
        'indices',
        'economy'
    ]
    
    print("Testing TwelveData free endpoints...")
    print("=" * 50)
    
    for endpoint in endpoints:
        try:
            print(f"\nTesting {endpoint}...")
            url = f'https://api.twelvedata.com/{endpoint}'
            
            # Set parameters based on endpoint
            if endpoint == 'quote':
                params = {'symbol': 'AAPL', 'apikey': api_key}
            elif endpoint == 'time_series':
                params = {'symbol': 'AAPL', 'interval': '1day', 'apikey': api_key}
            elif endpoint == 'symbol_search':
                params = {'symbol': 'AAPL', 'apikey': api_key}
            else:
                params = {'apikey': api_key}
            
            response = requests.get(url, params=params, timeout=10)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'status' in data and data['status'] == 'error':
                    print(f"  Error: {data.get('message', 'Unknown error')}")
                else:
                    print(f"  Success! Sample data keys: {list(data.keys())[:5] if isinstance(data, dict) else 'Not a dict'}")
                    if isinstance(data, dict):
                        print(f"  Data type: {type(data)}")
                        if len(str(data)) < 200:
                            print(f"  Sample: {json.dumps(data, indent=2)[:100]}...")
                        else:
                            print(f"  Data size: {len(str(data))} characters")
            else:
                print(f"  Response: {response.text[:100]}")
                
        except Exception as e:
            print(f"  Exception: {e}")

if __name__ == "__main__":
    test_free_endpoints()