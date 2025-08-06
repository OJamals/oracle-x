import os
import logging
from data_feeds.twelvedata_adapter import TwelveDataAdapter

# Set up logging to see detailed errors
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    api_key = os.getenv("TWELVEDATA_API_KEY")
    print(f"API Key present: {api_key is not None}")
    if api_key:
        print(f"API Key length: {len(api_key)}")
    
    adapter = TwelveDataAdapter()
    print(f"Adapter API key: {adapter.api_key}")
    
    try:
        print("Testing quote for AAPL...")
        quote = adapter.get_quote("AAPL")
        if quote:
            print(f"Quote: {quote}")
        else:
            print("Quote is None")
    except Exception as e:
        print(f"Error getting quote: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("Testing market data for AAPL...")
        market_data = adapter.get_market_data("AAPL", period="1mo", interval="1d")
        if market_data:
            print(f"Market data retrieved successfully")
            print(f"Data shape: {market_data.data.shape}")
        else:
            print("Market data is None")
    except Exception as e:
        print(f"Error getting market data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()