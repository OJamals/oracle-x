import pprint
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_feeds.reddit_sentiment import fetch_reddit_sentiment


def test_default():
    print("[TEST] Default scenario (limit=5)...")
    result = fetch_reddit_sentiment(limit=5)
    pprint.pprint(result)
    assert isinstance(result, dict)
    print("[PASS] Default scenario")


def test_high_limit():
    print("[TEST] High limit scenario (limit=20)...")
    result = fetch_reddit_sentiment(limit=20)
    pprint.pprint(result)
    assert isinstance(result, dict)
    print("[PASS] High limit scenario")


def test_no_credentials():
    print("[TEST] No credentials scenario...")
    # Temporarily unset env vars
    old_id = os.environ.get("REDDIT_CLIENT_ID")
    old_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    os.environ.pop("REDDIT_CLIENT_ID", None)
    os.environ.pop("REDDIT_CLIENT_SECRET", None)
    try:
        result = fetch_reddit_sentiment(limit=5)
        assert result == {}
        print("[PASS] No credentials scenario")
    finally:
        os.environ["REDDIT_CLIENT_ID"] = old_id if old_id is not None else ""
        os.environ["REDDIT_CLIENT_SECRET"] = (
            old_secret if old_secret is not None else ""
        )


def test_invalid_subreddit():
    print("[TEST] Invalid subreddit scenario...")
    result = fetch_reddit_sentiment(subreddit="thissubdoesnotexist123", limit=5)
    pprint.pprint(result)
    assert isinstance(result, dict)
    print("[PASS] Invalid subreddit scenario")


if __name__ == "__main__":
    test_default()
    test_high_limit()
    test_no_credentials()
    test_invalid_subreddit()
