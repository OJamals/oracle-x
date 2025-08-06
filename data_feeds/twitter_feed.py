
import asyncio
import re
import logging
from typing import Dict, List, Optional, Any, Set
import asyncio
import re
import logging
from twscrape import API
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging to reduce verbosity
logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TextBlob = None
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available, using VADER sentiment only")

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    detect = None
    LangDetectException = Exception
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available, language detection disabled")


class TwitterSentimentFeed:
    """
    Efficient Twitter sentiment data interface using twscrape.
    
    Optimized for performance with:
    - Pre-compiled regex patterns
    - Efficient text processing
    - Deduplication
    - Configurable filters
    """
    
    # Pre-compiled patterns for efficiency
    TICKER_PATTERN = re.compile(r'(\$[A-Z]{2,5}|#[A-Z]{2,5}|\b[A-Z]{2,5}\b)')
    URL_PATTERN = re.compile(r'http\S+|www\S+')
    MENTION_PATTERN = re.compile(r'@\w+')
    HASHTAG_PATTERN = re.compile(r'#\w+')
    
    # Common words to exclude from ticker detection
    COMMON_WORDS = frozenset({
        "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN", 
        "HAVE", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", 
        "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", 
        "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "BUY",
        "GOT", "CAN", "MAY", "SET", "LOL", "OMG", "WTF", "CEO", "ETF", "USA"
    })
    
    def __init__(self, min_likes: int = 0, min_retweets: int = 0):
        """
        Initialize TwitterSentimentFeed.
        
        Args:
            min_likes: Minimum likes required (disabled by default for broader results)
            min_retweets: Minimum retweets required (disabled by default)
        """
        self.min_likes = min_likes
        self.min_retweets = min_retweets
        self.analyzer = SentimentIntensityAnalyzer()
        self._valid_tickers: Optional[Set[str]] = None
        
    def _get_valid_tickers(self) -> Set[str]:
        """Lazy load valid tickers with fallback."""
        if self._valid_tickers is None:
            try:
                from data_feeds.ticker_universe import fetch_ticker_universe
                valid_tickers = set(fetch_ticker_universe(sample_size=2000))
                self._valid_tickers = valid_tickers
                logger.info(f"Loaded {len(valid_tickers)} valid tickers")
            except Exception as e:
                logger.warning(f"Failed to load ticker universe: {e}, using fallback")
                self._valid_tickers = {
                    "AAPL", "TSLA", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "AMD", 
                    "META", "NFLX", "SPY", "QQQ", "IWM", "VTI", "SHOP", "ROKU",
                    "PLTR", "COIN", "HOOD", "SQ", "PYPL", "DIS", "BABA", "TSM"
                }
        return self._valid_tickers
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing URLs, mentions, and hashtags."""
        cleaned = self.URL_PATTERN.sub("", text)
        cleaned = self.MENTION_PATTERN.sub("", cleaned)
        cleaned = self.HASHTAG_PATTERN.sub("", cleaned)
        return cleaned.strip()
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract valid ticker symbols from text."""
        valid_tickers = self._get_valid_tickers()
        tickers = set()
        
        for match in self.TICKER_PATTERN.findall(text):
            ticker = match.replace("$", "").replace("#", "").upper()
            if ticker in valid_tickers and ticker not in self.COMMON_WORDS:
                tickers.add(ticker)
        
        return list(tickers)
    
    def _get_sentiment(self, text: str) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis."""
        sentiment = dict(self.analyzer.polarity_scores(text))
        
        if TEXTBLOB_AVAILABLE and TextBlob is not None:
            try:
                tb = TextBlob(text)
                sentiment["textblob_polarity"] = tb.sentiment.polarity
                sentiment["textblob_subjectivity"] = tb.sentiment.subjectivity
            except Exception:
                # Use a special marker value instead of None
                sentiment["textblob_polarity"] = 0.0
                sentiment["textblob_subjectivity"] = 0.0
        
        return sentiment
    
    def _get_language(self, text: str) -> str:
        """Detect text language with fallback."""
        if not LANGDETECT_AVAILABLE or detect is None:
            return "unknown"
        
        try:
            return detect(text)
        except Exception:
            return "unknown"
    
    def _should_include_tweet(self, tweet_data: Dict[str, Any]) -> bool:
        """Determine if tweet should be included based on filters."""
        # Minimal filtering for broader results
        likes = tweet_data.get("likes", 0)
        retweets = tweet_data.get("retweets", 0)
        
        # Only apply engagement filters if explicitly set
        if self.min_likes > 0 and likes < self.min_likes:
            return False
        if self.min_retweets > 0 and retweets < self.min_retweets:
            return False
        
        return True

    def fetch(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch sentiment data for a query using twscrape.
        
        Args:
            query: Search term or ticker symbol
            limit: Maximum number of tweets to fetch
            
        Returns:
            List of tweet dictionaries with sentiment and ticker data
        """
        return asyncio.run(self._fetch_async(query, limit))
    
    async def _fetch_async(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Async implementation of tweet fetching."""
        api = API()
        seen_texts = set()
        results = []
        processed_count = 0
        
        try:
            async for tweet in api.search(query, limit=limit):
                if processed_count >= limit:
                    break
                
                # Get tweet content
                text = getattr(tweet, 'rawContent', '') or getattr(tweet, 'content', '')
                if not text:
                    continue
                
                # Deduplicate by text content
                text_key = text.strip().lower()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                
                # Clean text for analysis
                clean_text = self._clean_text(text)
                if not clean_text or len(clean_text.strip()) < 5:
                    continue
                
                # Extract data
                tickers = self._extract_tickers(text)
                sentiment = self._get_sentiment(text)
                language = self._get_language(clean_text)
                
                # Get engagement metrics
                likes = getattr(tweet, 'likeCount', 0) or 0
                retweets = getattr(tweet, 'retweetCount', 0) or 0
                
                tweet_data = {
                    "text": text,
                    "clean_text": clean_text,
                    "sentiment": sentiment,
                    "tickers": tickers,
                    "language": language,
                    "likes": likes,
                    "retweets": retweets,
                    "created_at": getattr(tweet, 'date', None),
                    "user_followers": getattr(tweet.user, 'followersCount', 0) if hasattr(tweet, 'user') else 0
                }
                
                # Apply filters
                if self._should_include_tweet(tweet_data):
                    results.append(tweet_data)
                
                processed_count += 1
                
        except Exception as e:
            logger.error(f"Error fetching tweets for query '{query}': {e}")
            
        logger.info(f"Fetched {len(results)} tweets for query '{query}' (processed {processed_count})")
        return results
