"""
Stub for twitter_sentiment to fix test imports during refactor.
"""

def fetch_twitter_sentiment(symbols: list[str], limit: int = 50) -> dict:
    """Stub implementation for Twitter sentiment data."""
    return {
        s: {
            "sentiment_score": 0.55,
            "confidence": 0.65,
            "sample_size": limit,
            "sample_texts": [f"Stub Twitter post about {s}."] * 3,
        }
        for s in symbols
    }