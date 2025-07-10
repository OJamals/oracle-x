from data_feeds.twitter_feed import TwitterSentimentFeed

def fetch_twitter_sentiment(query, limit=100):
    """
    Fetch Twitter sentiment data for a query using twscrape backend only.
    Returns a list of tweet dicts with sentiment and tickers.
    """
    feed = TwitterSentimentFeed()
    return feed.fetch(query, limit=limit)

