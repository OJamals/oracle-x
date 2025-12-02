"""
Unified News Adapter Dispatcher for Oracle-X

Centralizes news source management with config-driven dispatch,
consolidated sentiment analysis, and SourceAdapterProtocol compliance.
"""

import logging
import requests
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from urllib.parse import urljoin

from data_feeds.adapter_protocol import SourceAdapterProtocol
from data_feeds.consolidated_data_feed import NewsItem
from data_feeds.data_types import SentimentData

# Import sentiment engine
try:
    from sentiment.sentiment_engine import get_sentiment_engine, analyze_symbol_sentiment, SentimentSummary
    ADVANCED_SENTIMENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced sentiment analysis not available: {e}")
    ADVANCED_SENTIMENT_AVAILABLE = False

logger = logging.getLogger(__name__)

class NewsAdapter(SourceAdapterProtocol):
    """
    Unified news adapter implementing SourceAdapterProtocol.

    Supports multiple news sources with config-driven dispatch:
    - RSS-based sources (MarketWatch, Reuters, etc.)
    - API-based sources (future expansion)
    """

    # Configurable news sources
    NEWS_SOURCES = {
        "marketwatch": {
            "name": "MarketWatch",
            "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
            "type": "rss"
        },
        "reuters": {
            "name": "Reuters",
            "url": "https://feeds.reuters.com/reuters/topNews",
            "type": "rss"
        },
        "cnn_business": {
            "name": "CNN Business",
            "url": "http://rss.cnn.com/rss/money_latest.rss",
            "type": "rss"
        },
        "financial_times": {
            "name": "Financial Times",
            "url": "https://www.ft.com/rss/home/uk",
            "type": "rss"
        },
        "fortune": {
            "name": "Fortune",
            "url": "https://fortune.com/feed/",
            "type": "rss"
        },
        "seeking_alpha": {
            "name": "Seeking Alpha",
            "url": "https://seekingalpha.com/feed.xml",
            "type": "rss"
        },
        "benzinga": {
            "name": "Benzinga",
            "url": "https://www.benzinga.com/feed/",
            "type": "rss"
        }
    }

    def __init__(self, cache=None, rate_limiter=None, performance_tracker=None):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.performance_tracker = performance_tracker

        # Initialize sentiment engine
        if ADVANCED_SENTIMENT_AVAILABLE:
            try:
                self.sentiment_engine = get_sentiment_engine()
                logger.info("Advanced sentiment engine initialized for news adapter")
            except Exception as e:
                logger.error(f"Failed to initialize sentiment engine: {e}")
                self.sentiment_engine = None
        else:
            self.sentiment_engine = None

        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def capabilities(self) -> Set[str]:
        """SourceAdapterProtocol capabilities"""
        return {"news", "sentiment"}

    def health(self) -> Dict[str, Any]:
        """SourceAdapterProtocol health check"""
        return {
            "source": "news_adapter",
            "status": "operational",
            "sources_count": len(self.NEWS_SOURCES),
            "advanced_sentiment_available": ADVANCED_SENTIMENT_AVAILABLE,
            "sentiment_engine_loaded": self.sentiment_engine is not None
        }

    def fetch_quote(self, symbol: str) -> Optional[Any]:
        """Not supported - news adapter doesn't provide quotes"""
        raise NotImplementedError("News adapter does not support quote fetching")

    def fetch_historical(self, symbol: str, period: str = "1y", interval: Optional[str] = None,
                        from_date: Optional[str] = None, to_date: Optional[str] = None) -> Optional[Any]:
        """Not supported - news adapter doesn't provide historical data"""
        raise NotImplementedError("News adapter does not support historical data fetching")

    def fetch_company_info(self, symbol: str) -> Optional[Any]:
        """Not supported - news adapter doesn't provide company info"""
        raise NotImplementedError("News adapter does not support company info fetching")

    def fetch_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        """SourceAdapterProtocol news fetch"""
        articles = self._fetch_news_articles(symbol, limit)
        return [self._convert_to_news_item(article) for article in articles]

    def fetch_sentiment(self, symbol: str, **kwargs) -> Any:
        """SourceAdapterProtocol sentiment fetch"""
        limit = kwargs.get("limit", 20)
        return self.get_sentiment(symbol, limit)

    def _fetch_news_articles(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch news articles from all configured sources"""
        all_articles = []

        for source_key, source_config in self.NEWS_SOURCES.items():
            try:
                articles = self._fetch_from_source(source_key, source_config, symbol, limit // len(self.NEWS_SOURCES) + 1)
                all_articles.extend(articles)
            except Exception as e:
                logger.warning(f"Failed to fetch from {source_key}: {e}")
                continue

        # Sort by relevance and recency, then limit
        all_articles.sort(key=lambda x: (self._calculate_relevance_score(x, symbol), x.get('published', '')), reverse=True)
        return all_articles[:limit]

    def _fetch_from_source(self, source_key: str, source_config: Dict, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch articles from a specific source"""
        if source_config["type"] == "rss":
            return self._fetch_from_rss(source_key, source_config["url"], symbol, limit)
        else:
            logger.warning(f"Unsupported source type: {source_config['type']}")
            return []

    def _fetch_from_rss(self, source_key: str, rss_url: str, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch articles from RSS feed with retry logic"""
        import feedparser
        import time

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching RSS from {source_key} (attempt {attempt + 1})")

                response = requests.get(rss_url, headers=self.headers, timeout=15, verify=True)
                response.raise_for_status()

                feed = feedparser.parse(response.content)
                articles = []

                for entry in feed.entries[:limit * 3]:  # Fetch more to filter
                    article = {
                        'title': entry.get('title', ''),
                        'description': entry.get('description', '') or entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': source_key
                    }

                    if self._is_relevant_to_symbol(article, symbol):
                        articles.append(article)

                    if len(articles) >= limit:
                        break

                logger.debug(f"Fetched {len(articles)} relevant articles from {source_key} for {symbol}")
                return articles

            except Exception as e:
                logger.warning(f"RSS fetch failed for {source_key} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                continue

        return []

    def _is_relevant_to_symbol(self, article: Dict[str, Any], symbol: str) -> bool:
        """Check if article is relevant to symbol"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()

        # Direct symbol match
        if f"${symbol.lower()}" in text or f" {symbol.lower()} " in text:
            return True

        # Company name mappings
        company_mappings = {
            'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'tim cook'],
            'TSLA': ['tesla', 'elon musk', 'electric vehicle', 'ev'],
            'MSFT': ['microsoft', 'windows', 'xbox', 'azure'],
            'GOOGL': ['google', 'alphabet', 'youtube', 'android'],
            'GOOG': ['google', 'alphabet', 'youtube', 'android'],
            'AMZN': ['amazon', 'aws', 'prime', 'bezos'],
            'META': ['facebook', 'meta', 'instagram', 'whatsapp', 'zuckerberg'],
            'NVDA': ['nvidia', 'graphics card', 'gpu', 'ai chip']
        }

        if symbol.upper() in company_mappings:
            if any(term in text for term in company_mappings[symbol.upper()]):
                return True

        # Financial keywords for broader relevance
        financial_keywords = [
            'stock', 'stocks', 'earnings', 'revenue', 'profit', 'market',
            'trading', 'investor', 'nasdaq', 'nyse', 'sp 500', 'financial'
        ]

        return any(keyword in text for keyword in financial_keywords)

    def _calculate_relevance_score(self, article: Dict[str, Any], symbol: str) -> float:
        """Calculate relevance score for article sorting"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        score = 0.0

        # Direct symbol mentions get highest score
        if f"${symbol.lower()}" in text:
            score += 1.0
        if f" {symbol.lower()} " in text:
            score += 0.8

        # Company name mentions
        company_mappings = {
            'AAPL': ['apple'], 'TSLA': ['tesla'], 'MSFT': ['microsoft'],
            'GOOGL': ['google', 'alphabet'], 'AMZN': ['amazon'], 'META': ['meta', 'facebook']
        }

        if symbol.upper() in company_mappings:
            if any(term in text for term in company_mappings[symbol.upper()]):
                score += 0.6

        # Financial keywords
        financial_keywords = ['earnings', 'revenue', 'profit', 'stock', 'market']
        keyword_count = sum(1 for keyword in financial_keywords if keyword in text)
        score += min(keyword_count * 0.1, 0.5)

        return score

    def _convert_to_news_item(self, article: Dict[str, Any]) -> NewsItem:
        """Convert article dict to NewsItem"""
        return NewsItem(
            title=article.get('title', ''),
            description=article.get('description', ''),
            url=article.get('link', ''),
            published_at=article.get('published', ''),
            source=article.get('source', 'unknown')
        )

    def get_sentiment(self, symbol: str, limit: int = 20) -> Optional[SentimentData]:
        """Get consolidated sentiment analysis from news sources"""
        try:
            articles = self._fetch_news_articles(symbol, limit)

            if not articles:
                logger.warning(f"No relevant articles found for {symbol}")
                return None

            # Extract text content
            texts = []
            article_metadata = []

            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                combined_text = f"{title}. {description}".strip()

                if combined_text:
                    texts.append(combined_text)
                    article_metadata.append({
                        'title': title,
                        'link': article.get('link', ''),
                        'published': article.get('published', ''),
                        'source': article.get('source', '')
                    })

            if not texts:
                return None

            # Use advanced sentiment analysis if available
            if self.sentiment_engine and ADVANCED_SENTIMENT_AVAILABLE:
                try:
                    sentiment_summary = analyze_symbol_sentiment(
                        symbol=symbol,
                        texts=texts,
                        sources=[meta['source'] for meta in article_metadata]
                    )

                    return SentimentData(
                        symbol=symbol,
                        sentiment_score=sentiment_summary.overall_sentiment,
                        confidence=sentiment_summary.confidence,
                        source="news_consolidated",
                        timestamp=datetime.now(),
                        sample_size=sentiment_summary.sample_size,
                        raw_data={
                            'articles': articles,
                            'article_metadata': article_metadata,
                            'bullish_mentions': sentiment_summary.bullish_mentions,
                            'bearish_mentions': sentiment_summary.bearish_mentions,
                            'trending_direction': sentiment_summary.trending_direction,
                            'quality_score': sentiment_summary.quality_score
                        }
                    )

                except Exception as e:
                    logger.error(f"Advanced sentiment analysis failed: {e}")

            # Fallback to basic analysis
            return self._basic_sentiment_analysis(symbol, articles, texts, article_metadata)

        except Exception as e:
            logger.error(f"News sentiment analysis failed for {symbol}: {e}")
            return None

    def get_news_texts(self, symbol: str, limit: int = 20) -> List[str]:
        """Get raw news texts for aggregation in advanced sentiment analysis"""
        try:
            articles = self._fetch_news_articles(symbol, limit)
            texts = []

            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                combined_text = f"{title}. {description}".strip()

                if combined_text:
                    texts.append(combined_text)

            return texts

        except Exception as e:
            logger.error(f"Failed to get news texts for {symbol}: {e}")
            return []

    def _basic_sentiment_analysis(self, symbol: str, articles: List[Dict], texts: List[str],
                                 article_metadata: List[Dict]) -> Optional[SentimentData]:
        """Basic keyword-based sentiment analysis"""
        try:
            positive_words = ['bullish', 'positive', 'gain', 'rise', 'growth', 'profit', 'strong', 'beat']
            negative_words = ['bearish', 'negative', 'loss', 'fall', 'decline', 'weak', 'miss', 'below']

            sentiment_scores = []

            for text in texts:
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)

                if positive_count + negative_count > 0:
                    score = (positive_count - negative_count) / (positive_count + negative_count)
                else:
                    score = 0.0

                sentiment_scores.append(score)

            if not sentiment_scores:
                return None

            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            confidence = min(0.8, max(0.2, sum(abs(s) for s in sentiment_scores) / len(sentiment_scores)))

            return SentimentData(
                symbol=symbol,
                sentiment_score=overall_sentiment,
                confidence=confidence,
                source="news_basic",
                timestamp=datetime.now(),
                sample_size=len(sentiment_scores),
                raw_data={
                    'articles': articles,
                    'article_metadata': article_metadata,
                    'individual_sentiments': sentiment_scores
                }
            )

        except Exception as e:
            logger.error(f"Basic sentiment analysis failed: {e}")
            return None