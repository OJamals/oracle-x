
import asyncio
from twscrape import API


class TwitterSentimentFeed:
    """
    Interface for Twitter sentiment data using twscrape only.
    """
    def __init__(self, min_likes=1, min_retweets=0, sample_size=2000):
        self.min_likes = min_likes
        self.min_retweets = min_retweets
        self.sample_size = sample_size

    def fetch(self, query, limit=100):
        """
        Fetch sentiment data for a query using twscrape only.
        Args:
            query: Search term or ticker.
            limit: Number of tweets to fetch.
        Returns:
            List of filtered tweet dicts with sentiment and tickers.
        """
        return self._fetch_twscrape(query, limit=limit)

    def _fetch_twscrape(self, query, limit=100):
        import re
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from langdetect import detect, LangDetectException
        try:
            from textblob import TextBlob
            textblob_available = True
        except ImportError:
            textblob_available = False
        async def run():
            api = API()
            analyzer = SentimentIntensityAnalyzer()
            ticker_pattern = re.compile(r'(\$[A-Z]{2,5}|#[A-Z]{2,5}|\b[A-Z]{2,5}\b)')
            common_words = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN", "HAVE", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE"}
            try:
                from data_feeds.ticker_universe import fetch_ticker_universe
                valid_tickers = set(fetch_ticker_universe(sample_size=2000))
            except Exception:
                valid_tickers = {"AAPL", "TSLA", "MSFT", "GOOG", "AMZN", "NVDA", "AMD", "META", "NFLX", "SPY"}
            seen_texts = set()
            results = []
            count = 0
            async for tweet in api.search(query, limit=limit):
                if count >= limit:
                    break
                text = tweet.rawContent
                text_key = text.strip().lower()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                filtered_out_reason = None
                clean_text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
                # Loosen filters: allow retweets, non-English, short tweets, symbols, repeated chars
                # Comment out or remove strict filters for broader results
                # if text.strip().startswith("RT "):
                #     filtered_out_reason = "retweet"
                try:
                    lang = detect(clean_text)
                except LangDetectException:
                    lang = "unknown"
                # if lang != "en":
                #     filtered_out_reason = "non-english"
                # if not clean_text or len(clean_text.strip()) < 10:
                #     filtered_out_reason = "too-short"
                # if re.fullmatch(r"[\W_]+", clean_text):
                #     filtered_out_reason = "symbols-only"
                # if re.search(r"(.)\1{4,}", clean_text):
                #     filtered_out_reason = "repeated-chars"
                tickers = set()
                for match in ticker_pattern.findall(text):
                    t = match.replace("$", "").replace("#", "")
                    if t in valid_tickers and t not in common_words:
                        tickers.add(t)
                # Allow tweets without tickers for broader results
                # if not tickers:
                #     filtered_out_reason = "no-ticker"
                sentiment = analyzer.polarity_scores(text)
                if textblob_available and TextBlob is not None:
                    try:
                        tb = TextBlob(text)
                        sentiment["textblob_polarity"] = tb.sentiment.polarity
                        sentiment["textblob_subjectivity"] = tb.sentiment.subjectivity
                    except Exception:
                        sentiment["textblob_polarity"] = None
                        sentiment["textblob_subjectivity"] = None
                likes = getattr(tweet, 'likeCount', 0)
                retweets = getattr(tweet, 'retweetCount', 0)
                # Allow all engagement levels
                # if likes < 1 or retweets < 0:
                #     filtered_out_reason = "low-engagement"
                results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "tickers": list(tickers),
                    "lang": lang,
                    "likes": likes,
                    "retweets": retweets,
                    "filtered_out_reason": filtered_out_reason
                })
                count += 1
            filtered = [r for r in results if r["filtered_out_reason"] is None]
            print(f"[DEBUG] {len(filtered)} tweets passed all filters out of {len(results)}")
            return filtered
        return asyncio.run(run())
