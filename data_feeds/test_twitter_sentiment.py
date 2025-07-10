import unittest
from data_feeds import twitter_sentiment


class TestTwitterSentiment(unittest.TestCase):
    def test_fetch_twitter_sentiment_basic(self):
        tweets = twitter_sentiment.fetch_twitter_sentiment("AAPL", limit=10)
        self.assertIsInstance(tweets, list)
        self.assertLessEqual(len(tweets), 10)
        self.assertGreater(len(tweets), 0, "No tweets returned for basic test.")
        self._assert_tweet_structure(tweets[0])

    def _assert_tweet_structure(self, tweet):
        self.assertIsInstance(tweet, dict)
        for key in ("text", "sentiment", "tickers", "lang", "filtered_out_reason"):
            self.assertIn(key, tweet)

    def test_english_only(self):
        tweet = self._extracted_from_test_ticker_extraction_2(
            "No tweets returned for english_only test."
        )
        self.assertEqual(tweet["lang"], "en")
        self.assertIsNone(tweet["filtered_out_reason"])

    def test_ticker_extraction(self):
        tweet = self._extracted_from_test_ticker_extraction_2(
            "No tweets returned for ticker_extraction test."
        )
        self.assertTrue(tweet["tickers"])
        t = tweet["tickers"][0]
        self.assertTrue(t.isupper())
        self.assertTrue(2 <= len(t) <= 5)

    # TODO Rename this here and in `test_english_only` and `test_ticker_extraction`
    def _extracted_from_test_ticker_extraction_2(self, arg0):
        tweets = twitter_sentiment.fetch_twitter_sentiment("AAPL", limit=10)
        self.assertGreater(len(tweets), 0, arg0)
        return tweets[0]

    def test_sentiment_scoring(self):
        tweets = twitter_sentiment.fetch_twitter_sentiment("AAPL", limit=5)
        self.assertGreater(len(tweets), 0, "No tweets returned for sentiment_scoring test.")
        s = tweets[0]["sentiment"]
        self.assertIn("compound", s)
        self.assertIn("pos", s)
        self.assertIn("neu", s)
        self.assertIn("neg", s)
        self.assertTrue(-1.0 <= s["compound"] <= 1.0)

    def test_edge_cases(self):
        tweets = twitter_sentiment.fetch_twitter_sentiment("ZZZZZZZZZZ", limit=10)
        self.assertIsInstance(tweets, list)
        self.assertEqual(len(tweets), 0, "Expected no tweets for nonsense query.")

if __name__ == "__main__":
    unittest.main()
