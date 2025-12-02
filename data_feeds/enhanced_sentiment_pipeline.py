"""
Stub for enhanced_sentiment_pipeline to fix test imports during refactor.
"""

from data_feeds.data_types import SentimentData

class EnhancedSentimentPipeline:
    def get_sentiment_analysis(self, symbol: str, include_reddit: bool = False) -> SentimentData:
        """Stub implementation for sentiment analysis."""
        return SentimentData(
            symbol=symbol,
            sentiment_score=0.5,
            confidence=0.7,
            source="enhanced_pipeline_stub",
            timestamp=None,
            sample_size=20,
        )

def get_enhanced_sentiment_pipeline() -> EnhancedSentimentPipeline:
    """Get the enhanced sentiment pipeline."""
    return EnhancedSentimentPipeline()