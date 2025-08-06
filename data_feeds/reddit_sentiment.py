
# Load environment variables from .env for local testing (global scope)
import os
try:
    from dotenv import load_dotenv
    import pathlib
    env_path = pathlib.Path(__file__).parent.parent / '.env'
    # print(f"[DEBUG] Loading .env from: {env_path}")
    load_dotenv(dotenv_path=env_path)
    # print(f"[DEBUG] REDDIT_CLIENT_ID after load: {os.environ.get('REDDIT_CLIENT_ID')}")
except ImportError:
    print("[WARN] python-dotenv not installed; .env loading skipped.")
import praw
from collections import Counter

def fetch_reddit_sentiment(subreddit="stocks", limit=100) -> dict:
    """
    Fetch sentiment from a curated list of top finance/market subreddits using PRAW (free, open-source).
    Returns a dict of ticker mentions and simple sentiment counts.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import re
    
    # Safely import ticker universe with fallback
    try:
        from data_feeds.ticker_universe import fetch_ticker_universe
        valid_tickers = set(fetch_ticker_universe(source="finviz", sample_size=200))
    except Exception:
        # Fallback to common tickers if ticker universe fails
        valid_tickers = {
            "AAPL", "TSLA", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "AMD",
            "META", "NFLX", "SPY", "QQQ", "IWM", "VTI", "SHOP", "ROKU",
            "PLTR", "COIN", "HOOD", "SQ", "PYPL", "DIS", "BABA", "TSM"
        }
    
    analyzer = SentimentIntensityAnalyzer()
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "oracle-x-bot")
    
    if not client_id or not client_secret:
        print("[ERROR] Reddit API credentials not set.")
        return {}
    
    try:
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    except Exception as e:
        print(f"[ERROR] Failed to initialize Reddit client: {e}")
        return {}
    
    # Curated list of top finance/market subreddits
    subreddits = [
        "stocks", "investing", "wallstreetbets", "StockMarket", "options", "pennystocks",
        "SPACs", "RobinHood", "Daytrading", "CryptoCurrency", "personalfinance"
    ]
    
    ticker_pattern = re.compile(r'\b([A-Z]{2,5})\b')
    # Aggregation structures
    sentiment_by_ticker = {}
    post_texts_by_ticker = {}  # Store actual post texts for advanced analysis
    sums_by_ticker = {}        # Track sums for averaging
    # NEW: store unique post IDs per ticker to dedupe at ticker-level across subreddits
    ids_by_ticker = {}
    
    # Filtering and debug flags
    min_upvotes = 2
    try:
        DEBUG_REDDIT = os.environ.get("DEBUG_REDDIT", "0") == "1"
    except Exception:
        DEBUG_REDDIT = False
    DEBUG_REDDIT = os.environ.get("DEBUG_REDDIT", "0") == "1"
    
    # Reduce limit to improve performance
    reduced_limit = min(limit, 50)  # Cap at 50 posts for faster processing
    total_posts_seen = 0
    # Lower thresholds to avoid over-filtering in validation runs
    min_upvotes = 0
    
    for sub in subreddits:
        try:
            posts = list(reddit.subreddit(sub).hot(limit=reduced_limit))
            # Some PRAW listings may yield the same object multiple times via iterator buffering;
            # convert to list of unique IDs to avoid duplicate prints/processing.
            unique_posts = []
            seen_local = set()
            for p in posts:
                pid = getattr(p, "id", None)
                key = pid if pid is not None else id(p)
                if key in seen_local:
                    continue
                seen_local.add(key)
                unique_posts.append(p)
            total_posts_seen += len(unique_posts)
            if DEBUG_REDDIT:
                print(f"[DEBUG][reddit] fetched {len(unique_posts)} unique posts from r/{sub} (limit={reduced_limit})")
            posts = unique_posts
            # Secondary safety: ensure loop doesn't double-handle same object
            _seen_ids_loop = set()
            for post in posts:
                _pid = getattr(post, "id", None)
                if _pid is not None:
                    if _pid in _seen_ids_loop:
                        continue
                    _seen_ids_loop.add(_pid)
                # Filter: ignore low-upvote posts
                if hasattr(post, 'score') and post.score < min_upvotes:
                    if DEBUG_REDDIT:
                        try:
                            _t = post.title[:80] if hasattr(post, 'title') and isinstance(post.title, str) else ''
                        except Exception:
                            _t = ''
                        print(f"[DEBUG][reddit] skip low-score post in r/{sub}: score={getattr(post,'score',None)} title='{_t}'")
                    continue
                
                # Combine title and selftext for analysis
                text = f"{post.title} {getattr(post, 'selftext', '')}".upper()
                
                # Extract tickers using regex and filter by valid tickers
                found = [t for t in set(ticker_pattern.findall(text)) if t in valid_tickers]
                if DEBUG_REDDIT and found:
                    try:
                        title_snip = post.title[:80] if hasattr(post, 'title') and isinstance(post.title, str) else ''
                    except Exception:
                        title_snip = ''
                    print(f"[DEBUG][reddit] r/{sub} found tickers: {found} | title='{title_snip}'")
                # no-op body removed (indentation fix already applied earlier)
                
                # Sentiment scoring
                try:
                    sentiment = analyzer.polarity_scores(
                        f"{post.title} " + getattr(post, 'selftext', '')
                    )
                except Exception as e:
                    print(f"[ERROR] Sentiment analysis failed for post: {e}")
                    continue

                # Only accumulate if we actually found valid tickers in this post
                if not found:
                    continue

                # Dedupe at ticker-level across subreddits: don't count the same post twice for a ticker
                post_id_key = getattr(post, "id", None) or (hash(getattr(post, "title", "")) ^ hash(getattr(post, "selftext", "")))
                for ticker in found:
                    if ticker not in ids_by_ticker:
                        ids_by_ticker[ticker] = set()
                    if post_id_key in ids_by_ticker[ticker]:
                        continue
                    ids_by_ticker[ticker].add(post_id_key)

                    # Initialize ticker entry if not exists
                    if ticker not in sentiment_by_ticker:
                        sentiment_by_ticker[ticker] = {
                            "mentions": 0,
                            "compound": 0.0,
                            "positive": 0.0,
                            "neutral": 0.0,
                            "negative": 0.0
                        }
                    # Update sentiment counts
                    sentiment_by_ticker[ticker]["mentions"] += 1
                    sentiment_by_ticker[ticker]["compound"] += sentiment["compound"]
                    sentiment_by_ticker[ticker]["positive"] += sentiment["pos"]
                    sentiment_by_ticker[ticker]["neutral"] += sentiment["neu"]
                    sentiment_by_ticker[ticker]["negative"] += sentiment["neg"]
                    
                    # Store the post text for advanced sentiment analysis
                    if ticker not in post_texts_by_ticker:
                        post_texts_by_ticker[ticker] = []
                    
                    # Store truncated post text to reduce memory usage
                    full_text = f"{post.title} " + getattr(post, 'selftext', '')
                    truncated_text = full_text[:200] + "..." if len(full_text) > 200 else full_text
                    post_texts_by_ticker[ticker].append(truncated_text.strip())
                    
        except Exception as e:
            print(f"[ERROR] Failed to fetch from r/{sub}: {e}")
            continue
    
    # Average sentiment scores per ticker and add sample texts + orchestrator-expected fields
    for ticker, stats in sentiment_by_ticker.items():
        mentions = int(stats.get("mentions", 0) or 0)
        if mentions > 0:
            # Normalize averages
            stats["compound"] /= mentions
            stats["positive"] /= mentions
            stats["neutral"] /= mentions
            stats["negative"] /= mentions

            # Add sample texts for this ticker (limit to reduce data size)
            sample_texts = post_texts_by_ticker.get(ticker, [])
            stats["sample_texts"] = sample_texts[:5]  # Only keep first 5 samples

            # Provide fields expected by orchestrator's RedditAdapter fallback
            stats["sentiment_score"] = float(stats["compound"])
            # Simple confidence heuristic bounded [0.2, 0.95] increasing with sample size
            stats["confidence"] = float(max(0.2, min(0.95, 0.5 + 0.02 * mentions)))
            stats["sample_size"] = mentions
    
    if DEBUG_REDDIT:
        total_mentions = sum(v.get("mentions",0) for v in sentiment_by_ticker.values())
        print(f"[DEBUG][reddit] aggregated tickers={len(sentiment_by_ticker)} total_mentions={total_mentions} total_posts_seen={total_posts_seen}")
    if 'DEBUG_REDDIT' in locals() and DEBUG_REDDIT:
        total_mentions = sum(v.get("mentions",0) for v in sentiment_by_ticker.values())
        print(f"[DEBUG][reddit] aggregated tickers={len(sentiment_by_ticker)} total_mentions={total_mentions}")
    if DEBUG_REDDIT:
        total_mentions = sum(v.get("mentions",0) for v in sentiment_by_ticker.values())
        print(f"[DEBUG][reddit] aggregated tickers={len(sentiment_by_ticker)} total_mentions={total_mentions}")
    # print("[DEBUG] Aggregated ticker sentiment:")
    # pprint.pprint(sentiment_by_ticker)
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
