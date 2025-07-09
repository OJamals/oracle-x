import pprint
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_feeds.reddit_sentiment import fetch_reddit_sentiment

if __name__ == "__main__":
    print("[TEST] Running fetch_reddit_sentiment()...")
    result = fetch_reddit_sentiment(limit=10)
    pprint.pprint(result)
