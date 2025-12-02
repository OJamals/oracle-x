"""
Twitter Sentiment Feed using twscrape for tweet collection and analysis.
Provides TwitterSentimentFeed class for fetching and analyzing Twitter sentiment.
"""

import logging
import re
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TwitterSentimentFeed:
    """
    Twitter sentiment feed using twscrape for tweet collection.
    Features pre-compiled regex for tickers, cleaning, and sentiment analysis.
    """

    def __init__(self):
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5})')
        self.mention_pattern = re.compile(r'@[\w]+')
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.max_tweets = 20
        self._last_fetch = {}
        self._cache_duration = 300  # 5 minutes

    def _clean_text(self, text: str) -> str:
        """Clean tweet text by removing URLs, mentions, and normalizing."""
        if not text:
            return ""

        # Remove URLs
        text = self.url_pattern.sub('', text)
        # Remove mentions
        text = self.mention_pattern.sub('', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract ticker symbols from text using regex."""
        if not text:
            return []

        matches = self.ticker_pattern.findall(text.upper())
        return list(set(matches))  # Remove duplicates

    def _analyze_sentiment_basic(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis using VADER as fallback."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'pos': scores['pos'],
                'neu': scores['neu'],
                'neg': scores['neg']
            }
        except ImportError:
            logger.warning("VADER not available, using neutral sentiment")
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

    def fetch(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch tweets for a given query with sentiment analysis.

        Args:
            query: Search query (ticker symbol or search term)
            limit: Maximum number of tweets to fetch

        Returns:
            List of tweet dictionaries with sentiment analysis
        """
        # Check cache first
        cache_key = f"{query}_{limit}"
        current_time = time.time()

        if (cache_key in self._last_fetch and
            current_time - self._last_fetch[cache_key] < self._cache_duration):
            logger.debug(f"Using cached Twitter data for {query}")
            return self._last_fetch[cache_key]

        try:
            # Import twscrape
            from twscrape import API
            from twscrape.logger import set_log_level
            set_log_level("WARNING")  # Reduce twscrape logging

            # Initialize API (this will use existing accounts if configured)
            api = API()

            # Prepare search query
            search_query = query
            if not query.startswith('$') and not query.startswith('#'):
                # If it's a plain ticker, add $ prefix for better results
                search_query = f"${query}"

            # Add language filter and remove retweets for better quality
            full_query = f"{search_query} lang:en -filter:replies"
            logger.debug(f"Twitter search query: {full_query}")

            # Fetch tweets
            tweets_data = []
            limit = min(limit, self.max_tweets)  # Cap at max_tweets

            try:
                # Use asyncio to handle the async generator
                import asyncio

                async def collect_tweets():
                    tweets = api.search(full_query, limit=limit)
                    tweets_list = []
                    async for tweet in tweets:
                        tweets_list.append(tweet)
                        if len(tweets_list) >= limit:
                            break
                    return tweets_list

                tweets_list = asyncio.run(collect_tweets())

                for tweet in tweets_list:
                    try:
                        # Extract tweet data
                        tweet_text = tweet.rawContent if hasattr(tweet, 'rawContent') else str(tweet)
                        clean_text = self._clean_text(tweet_text)

                        if not clean_text:
                            continue

                        # Extract tickers
                        tickers = self._extract_tickers(tweet_text)

                        # Analyze sentiment
                        sentiment = self._analyze_sentiment_basic(clean_text)

                        # Create tweet dict
                        tweet_dict = {
                            'text': clean_text,
                            'raw_text': tweet_text,
                            'tickers': tickers,
                            'sentiment': sentiment,
                            'timestamp': datetime.now(),
                            'tweet_id': getattr(tweet, 'id', None),
                            'username': getattr(tweet, 'username', None),
                            'likes': getattr(tweet, 'likes', 0),
                            'retweets': getattr(tweet, 'retweets', 0),
                            'replies': getattr(tweet, 'replies', 0)
                        }

                        tweets_data.append(tweet_dict)

                    except Exception as e:
                        logger.debug(f"Error processing tweet: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Error fetching tweets for {query}: {e}")
                # Return empty list on error
                tweets_data = []

            # Cache results
            self._last_fetch[cache_key] = tweets_data
            self._last_fetch[f"{cache_key}_time"] = current_time

            logger.info(f"Fetched {len(tweets_data)} tweets for {query}")
            return tweets_data

        except ImportError:
            logger.warning("twscrape not available, returning empty tweet list")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Twitter fetch: {e}")
            return []

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the Twitter feed."""
        return {
            'service': 'twitter_feed',
            'status': 'operational' if self._check_twscrape_available() else 'twscrape_unavailable',
            'max_tweets': self.max_tweets,
            'cache_duration': self._cache_duration,
            'cached_queries': len([k for k in self._last_fetch.keys() if not k.endswith('_time')])
        }

    def _check_twscrape_available(self) -> bool:
        """Check if twscrape is available."""
        try:
            import twscrape
            return True
        except ImportError:
            return False