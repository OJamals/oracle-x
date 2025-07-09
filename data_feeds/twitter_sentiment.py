# Load environment variables from .env for local testing (global scope)
import os
try:
    from dotenv import load_dotenv
    import pathlib
    env_path = pathlib.Path(__file__).parent.parent / '.env'
    print(f"[DEBUG] Loading .env from: {env_path}")
    load_dotenv(dotenv_path=env_path)
    print(f"[DEBUG] TWITTER_BEARER_TOKEN after load: {os.environ.get('TWITTER_BEARER_TOKEN')}")
except ImportError:
    print("[WARN] python-dotenv not installed; .env loading skipped.")
# Patch httpx globally to disable SSL verification for snscrape
try:
    import httpx
    orig_client_init = httpx.Client.__init__
    def noverify_client_init(self, *args, **kwargs):
        kwargs["verify"] = False
        orig_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = noverify_client_init

    orig_asyncclient_init = httpx.AsyncClient.__init__
    def noverify_asyncclient_init(self, *args, **kwargs):
        kwargs["verify"] = False
        orig_asyncclient_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = noverify_asyncclient_init
    print("[WARN] Global SSL verification disabled for httpx (insecure, for snscrape workaround, always on).")
except Exception as e:
    print(f"[ERROR] Could not patch httpx for SSL globally: {e}")
import subprocess
import json
import os


# Permanently patch requests globally to disable SSL verification for snscrape workaround
try:
    import requests
    try:
        from urllib3.exceptions import InsecureRequestWarning
        import urllib3
        urllib3.disable_warnings(InsecureRequestWarning)
    except ImportError:
        InsecureRequestWarning = None
    # If urllib3 is not available, skip disabling warnings
    orig_request = requests.Session.request
    def noverify_request(self, *args, **kwargs):
        kwargs["verify"] = False
        return orig_request(self, *args, **kwargs)
    requests.Session.request = noverify_request
    print("[WARN] Global SSL verification disabled for requests (insecure, for snscrape workaround, always on).")
except Exception as ssl_patch_e:
    print(f"[ERROR] Could not patch requests for SSL globally: {ssl_patch_e}")

def fetch_twitter_sentiment(query="AAPL", limit=100) -> list:
    """
    Fetch tweets using Tweepy (Twitter API v2, bearer token required).
    Returns a list of tweet texts for further sentiment analysis.
    """
    import os
    try:
        import tweepy
    except ImportError:
        print("[ERROR] Tweepy not installed. Please install tweepy.")
        return []
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        print("[ERROR] TWITTER_BEARER_TOKEN not set in environment.")
        return []
    try:
        client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
        # Twitter API v2 requires max_results between 10 and 100
        max_results = min(max(limit, 10), 100)
        tweets = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["text"])
        tweet_texts = [tweet.text for tweet in tweets.data] if tweets.data else []
        print(f"[DEBUG] Retrieved {len(tweet_texts)} tweets for query '{query}'")
        return tweet_texts
    except Exception as e:
        print(f"[ERROR] Tweepy Twitter API fetch failed for {query}: {e}")
        return []
