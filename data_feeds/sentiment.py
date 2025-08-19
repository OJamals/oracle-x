def fetch_sentiment_data(tickers=None) -> dict:
    # Load environment variables from .env for local testing
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("[WARN] python-dotenv not installed; .env loading skipped.")
    """
    Fetch sentiment data from Twitter and Reddit. (Google Trends removed due to persistent rate limiting.)
    Returns:
        dict: Sentiment snapshot by ticker/source.
    """
    from data_feeds.reddit_sentiment import fetch_reddit_sentiment
    from data_feeds.twitter_sentiment import fetch_twitter_sentiment
    if tickers is None:
        tickers = ["TSLA", "NVDA"]
    reddit_counts = fetch_reddit_sentiment()
    twitter_data = {ticker: fetch_twitter_sentiment(ticker) for ticker in tickers}
    return {
        ticker: {
            "reddit_mentions": reddit_counts.get(ticker, 0),
            "twitter_tweets": len(twitter_data.get(ticker, [])),
        }
        for ticker in tickers
    }
