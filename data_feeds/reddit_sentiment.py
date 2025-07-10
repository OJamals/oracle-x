
# Load environment variables from .env for local testing (global scope)
import os
try:
    from dotenv import load_dotenv
    import pathlib
    env_path = pathlib.Path(__file__).parent.parent / '.env'
    print(f"[DEBUG] Loading .env from: {env_path}")
    load_dotenv(dotenv_path=env_path)
    print(f"[DEBUG] REDDIT_CLIENT_ID after load: {os.environ.get('REDDIT_CLIENT_ID')}")
except ImportError:
    print("[WARN] python-dotenv not installed; .env loading skipped.")
import praw
from collections import Counter

def fetch_reddit_sentiment(subreddit="stocks", limit=100) -> dict:
    """
    Fetch sentiment from a curated list of top finance/market subreddits using PRAW (free, open-source).
    Returns a dict of ticker mentions and simple sentiment counts, with debug output.
    """
    import pprint
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from data_feeds.ticker_universe import fetch_ticker_universe
    import re
    analyzer = SentimentIntensityAnalyzer()
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "oracle-x-bot")
    if not client_id or not client_secret:
        print("[ERROR] Reddit API credentials not set.")
        return {}
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    # Curated list of top finance/market subreddits
    subreddits = [
        "stocks", "investing", "wallstreetbets", "StockMarket", "options", "pennystocks",
        "SPACs", "RobinHood", "Daytrading", "CryptoCurrency", "personalfinance"
    ]
    # Get a universe of valid tickers for filtering
    valid_tickers = set(fetch_ticker_universe(source="finviz", sample_size=200))
    ticker_pattern = re.compile(r'\b([A-Z]{2,5})\b')
    sentiment_by_ticker = {}
    min_upvotes = 2
    for sub in subreddits:
        try:
            posts = list(reddit.subreddit(sub).hot(limit=limit))
            print(f"[DEBUG] Fetched {len(posts)} posts from r/{sub}")
            for post in posts:
                # Filter: ignore low-upvote posts
                if hasattr(post, 'score') and post.score < min_upvotes:
                    continue
                # Combine title and selftext for analysis
                text = f"{post.title} {getattr(post, 'selftext', '')}".upper()
                # Extract tickers using regex and filter by valid tickers
                found = [t for t in set(ticker_pattern.findall(text)) if t in valid_tickers]
                if found:
                    print(f"[DEBUG] r/{sub} post: '{post.title[:80]}' | Tickers: {found}")
                # Sentiment scoring
                sentiment = analyzer.polarity_scores(
                    f"{post.title} " + getattr(post, 'selftext', '')
                )
                for ticker in found:
                    _extracted_from_fetch_reddit_sentiment_45(
                        ticker, sentiment_by_ticker, sentiment
                    )
        except Exception as e:
            print(f"[ERROR] Failed to fetch from r/{sub}: {e}")
    # Average sentiment scores per ticker
    for stats in sentiment_by_ticker.values():
        if stats["mentions"] > 0:
            stats["compound"] /= stats["mentions"]
            stats["positive"] /= stats["mentions"]
            stats["neutral"] /= stats["mentions"]
            stats["negative"] /= stats["mentions"]
    print("[DEBUG] Aggregated ticker sentiment:")
    pprint.pprint(sentiment_by_ticker)
    return sentiment_by_ticker


# TODO Rename this here and in `fetch_reddit_sentiment`
def _extracted_from_fetch_reddit_sentiment_45(ticker, sentiment_by_ticker, sentiment):
    if ticker not in sentiment_by_ticker:
        sentiment_by_ticker[ticker] = {"mentions": 0, "compound": 0.0, "positive": 0.0, "neutral": 0.0, "negative": 0.0}
    sentiment_by_ticker[ticker]["mentions"] += 1
    sentiment_by_ticker[ticker]["compound"] += sentiment["compound"]
    sentiment_by_ticker[ticker]["positive"] += sentiment["pos"]
    sentiment_by_ticker[ticker]["neutral"] += sentiment["neu"]
    sentiment_by_ticker[ticker]["negative"] += sentiment["neg"]
