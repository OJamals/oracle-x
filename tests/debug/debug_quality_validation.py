import os
from dotenv import load_dotenv
from data_feeds.data_feed_orchestrator import get_orchestrator, DataSource, DataValidator

# Load environment variables
load_dotenv()

def main():
    orch = get_orchestrator()
    
    print("=== Testing Quality Validation ===")
    
    # Get a quote from TwelveData adapter directly
    if DataSource.TWELVE_DATA in orch.adapters:
        adapter = orch.adapters[DataSource.TWELVE_DATA]
        quote = adapter.get_quote("AAPL")
        print(f"Raw TwelveData quote: {quote}")
        if quote:
            print(f"Price: {quote.price}")
            print(f"Quality score: {quote.quality_score}")
            
            # Test validation
            validator = DataValidator()
            quality_score, issues = validator.validate_quote(quote)
            print(f"Validation quality score: {quality_score}")
            print(f"Validation issues: {issues}")
            
            # Test the quality comparison logic
            best_quality = 0
            quote_quality = quote.quality_score or 0
            print(f"Quote quality (or 0): {quote_quality}")
            print(f"Is quote quality > best quality? {quote_quality > best_quality}")

if __name__ == "__main__":
    main()