import os
from dotenv import load_dotenv
import requests
from data_feeds.twelvedata_adapter import TwelveDataAdapter

# Load environment variables
load_dotenv()

def main():
    api_key = os.getenv("TWELVEDATA_API_KEY")
    print(f"API Key: {api_key}")
    
    # Test raw API call
    print("\n=== Raw API Call for Quote ===")
    url = "https://api.twelvedata.com/quote"
    params = {"symbol": "AAPL", "apikey": api_key}
    response = requests.get(url, params=params)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test the adapter
    print("\n=== Adapter Call ===")
    adapter = TwelveDataAdapter(api_key=api_key)
    try:
        quote = adapter.get_quote("AAPL")
        print(f"Adapter Quote: {quote}")
        if quote:
            print(f"Price: {quote.price}")
            print(f"Change: {quote.change}")
            print(f"Change %: {quote.change_percent}")
            print(f"Volume: {quote.volume}")
    except Exception as e:
        print(f"Adapter error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()