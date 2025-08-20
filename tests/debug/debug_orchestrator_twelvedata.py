import os
from dotenv import load_dotenv
from data_feeds.data_feed_orchestrator import get_orchestrator, DataSource

# Load environment variables
load_dotenv()

def main():
    orch = get_orchestrator()
    
    print("=== Orchestrator TwelveData Test ===")
    print(f"TWELVEDATA_API_KEY in env: {os.getenv('TWELVEDATA_API_KEY') is not None}")
    
    # Check if TwelveData adapter exists
    print(f"TwelveData adapter exists: {DataSource.TWELVE_DATA in orch.adapters}")
    if DataSource.TWELVE_DATA in orch.adapters:
        adapter = orch.adapters[DataSource.TWELVE_DATA]
        print(f"Adapter API key: {adapter.api_key}")
        
        # Test direct adapter call
        try:
            print("Testing direct adapter call...")
            quote = adapter.get_quote("AAPL")
            print(f"Direct quote result: {quote}")
            if quote:
                print(f"Direct quote price: {quote.price}")
        except Exception as e:
            print(f"Direct adapter error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test orchestrator call
    print("\nTesting orchestrator call...")
    try:
        quote = orch.get_quote("AAPL", preferred_sources=[DataSource.TWELVE_DATA])
        print(f"Orchestrator quote result: {quote}")
        if quote:
            print(f"Orchestrator quote price: {quote.price}")
            print(f"Orchestrator quote source: {quote.source}")
            print(f"Orchestrator quote quality: {quote.quality_score}")
    except Exception as e:
        print(f"Orchestrator error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()