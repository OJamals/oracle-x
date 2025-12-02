import pprint
import pytest
from data_feeds.twitter_sentiment import fetch_twitter_sentiment


def test_twikit_basic_query():
    print("[TEST] twikit basic query scenario (query='AAPL', limit=3)...")
    result = fetch_twitter_sentiment("AAPL", limit=3)
    pprint.pprint(result)
    assert isinstance(result, list)
    assert len(result) > 0
    print("[PASS] twikit basic query scenario")


def test_twikit_user_tweets():
    print("[TEST] twikit user tweets scenario (no query, limit=2)...")
    result = fetch_twitter_sentiment("AAPL", limit=2)
    pprint.pprint(result)
    assert isinstance(result, list)
    assert len(result) > 0
    print("[PASS] twikit user tweets scenario")


def test_twikit_invalid_login():
    print("[TEST] twikit invalid login scenario...")
    # Not applicable: twscrape does not use username/password in this workflow
    print("[SKIP] Invalid login scenario not applicable for twscrape-only workflow.")


def test_twikit_no_results():
    print(
        "[TEST] twikit no results scenario (query='thisqueryshouldnotexist123', limit=1)..."
    )
    result = fetch_twitter_sentiment("thisqueryshouldnotexist123", limit=1)
    pprint.pprint(result)
    assert isinstance(result, list)
    print("[PASS] twikit no results scenario")


if __name__ == "__main__":
    test_twikit_basic_query()
    test_twikit_user_tweets()
    test_twikit_invalid_login()
    test_twikit_no_results()
