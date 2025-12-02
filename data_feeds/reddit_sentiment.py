"""
Stub for reddit_sentiment to fix test imports during refactor.
"""

def fetch_reddit_sentiment(symbols: list[str], limit: int = 50) -> dict:
    """Stub implementation for Reddit sentiment data."""
    return {
        s: {
            "sentiment_score": 0.5,
            "confidence": 0.6,
            "sample_size": limit,
            "sample_texts": [f"Stub Reddit post about {s}."] * 3,
        }
        for s in symbols
    }