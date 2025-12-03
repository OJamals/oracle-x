"""
Enhanced sentiment pipeline using unified sentiment engine.
"""

from data_feeds.data_types import SentimentData
from sentiment.sentiment_engine import get_sentiment_engine


class EnhancedSentimentPipeline:
    def __init__(self):
        self.engine = get_sentiment_engine()

    def get_sentiment_analysis(
        self, symbol: str, include_reddit: bool = False
    ) -> SentimentData:
        """Enhanced sentiment analysis using multi-model ensemble."""
        # Dummy texts from multiple sources for demonstration
        texts = [
            f"Enhanced news headline about {symbol}.",
            f"Recent Twitter sentiment on {symbol} is positive.",
        ]
        if include_reddit:
            texts.append(f"Reddit discussion on {symbol} shows bullish trend.")
        return self.engine.news_sentiment(
            symbol, texts
        )  # Use news model as proxy for enhanced


def get_enhanced_sentiment_pipeline() -> EnhancedSentimentPipeline:
    """Get the enhanced sentiment pipeline."""
    return EnhancedSentimentPipeline()
