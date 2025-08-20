import os
import logging
from data_feeds.twelvedata_adapter import TwelveDataAdapter
from data_feeds.data_feed_orchestrator import get_orchestrator, DataSource

# Set up logging to see detailed errors
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    # Test direct adapter
    print("=== Testing TwelveData Adapter Directly ===")
    api_key = os.getenv("TWELVEDATA_API_KEY")
    print(f"API Key present: {api_key is not None}")
    if api_key:
        print(f"API Key length: {len(api_key)}")
    
    adapter = TwelveDataAdapter(api_key=api_key)
    print(f"Adapter API key: {adapter.api_key}")
    
    try:
        print("Testing quote for AAPL...")
        quote = adapter.get_quote("AAPL")
        if quote:
            print(f"Quote: {quote}")
            print(f"Price: {quote.price}")
            print(f"Source: {quote.source}")
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
            print(f"Source: {market_data.source}")
        else:
            print("Market data is None")
    except Exception as e:
        print(f"Error getting market data: {e}")
        import traceback
        traceback.print_exc()
    
    # Test through orchestrator
    print("\n=== Testing Through Orchestrator ===")
    orch = get_orchestrator()
    print(f"TwelveData adapter in orchestrator: {DataSource.TWELVE_DATA in orch.adapters}")
    if DataSource.TWELVE_DATA in orch.adapters:
        td_adapter = orch.adapters[DataSource.TWELVE_DATA]
        print(f"Orchestrator adapter API key: {td_adapter.api_key}")
    
    try:
        print("Testing orchestrator quote for AAPL...")
        quote = orch.get_quote("AAPL", preferred_sources=[DataSource.TWELVE_DATA])
        if quote:
            print(f"Orchestrator Quote: {quote}")
            print(f"Price: {quote.price}")
            print(f"Source: {quote.source}")
        else:
            print("Orchestrator Quote is None")
    except Exception as e:
        print(f"Error getting orchestrator quote: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()