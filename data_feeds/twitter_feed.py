
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
logger.setLevel(logging.WARNING)

# Suppress httpx and other verbose logging
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logging.captureWarnings(True)
class CardFilter(logging.Filter):
    def filter(self, record):
        return "Unknown card type" not in record.getMessage()

logging.getLogger('py.warnings').addFilter(CardFilter())

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

from data_feeds.advanced_sentiment import analyze_text_sentiment


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
    
    def _get_sentiment(self, text: str, tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis using advanced sentiment engine."""
        # Determine symbol for sentiment analysis - use first ticker if available, otherwise "unknown"
        symbol = "unknown"
        if tickers and len(tickers) > 0:
            symbol = tickers[0]
        
        # Use advanced sentiment analysis with source="twitter" for model selection
        sentiment_result = analyze_text_sentiment(text, symbol, source="twitter")
        
        # Create VADER-like sentiment dict for backward compatibility
        ensemble_score = sentiment_result.ensemble_score
        if ensemble_score >= 0:
            pos = ensemble_score
            neu = 1 - ensemble_score
            neg = 0.0
        else:
            pos = 0.0
            neu = 1 + ensemble_score  # ensemble_score is negative, so 1 + neg value
            neg = -ensemble_score
        
        sentiment_dict = {
            'compound': ensemble_score,
            'positive': pos,
            'neutral': neu,
            'negative': neg,
            'ensemble_score': ensemble_score,
            'confidence': sentiment_result.confidence
        }
        
        # Include TextBlob if available (maintain existing behavior)
        if TEXTBLOB_AVAILABLE and TextBlob is not None:
            try:
                tb = TextBlob(text)
                sentiment_dict["textblob_polarity"] = tb.sentiment.polarity
                sentiment_dict["textblob_subjectivity"] = tb.sentiment.subjectivity
            except Exception:
                sentiment_dict["textblob_polarity"] = 0.0
                sentiment_dict["textblob_subjectivity"] = 0.0
        
        return sentiment_dict
    
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
        
        import sys
        import os
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        try:
            # Reduce limit to improve performance
            reduced_limit = min(limit, 20)  # Cap at 20 tweets for faster processing
            async for tweet in api.search(query, limit=reduced_limit):
                if processed_count >= reduced_limit:
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
                sentiment = self._get_sentiment(text, tickers)
                language = self._get_language(clean_text)
                
                # Get engagement metrics
                likes = getattr(tweet, 'likeCount', 0) or 0
                retweets = getattr(tweet, 'retweetCount', 0) or 0
                
                # Truncate text to reduce memory usage and token consumption
                truncated_text = text[:200] + "..." if len(text) > 200 else text
                truncated_clean = clean_text[:150] + "..." if len(clean_text) > 150 else clean_text
                
                # Further truncate text to reduce token consumption and clean up URLs
                if len(truncated_text) > 100:
                    # Remove long URLs and replace with placeholder
                    cleaned_truncated = self.URL_PATTERN.sub('[URL]', truncated_text[:100] + "...")
                else:
                    cleaned_truncated = truncated_text
                
                tweet_data = {
                    "text": cleaned_truncated,
                    "clean_text": truncated_clean,
                    "sentiment": sentiment,
                    "tickers": tickers[:3],  # Limit tickers
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
        finally:
            sys.stderr.close()
            sys.stderr = original_stderr
            
        logger.info(f"Fetched {len(results)} tweets for query '{query}' (processed {processed_count})")
        return results
