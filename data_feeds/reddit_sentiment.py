"""
Reddit sentiment using unified sentiment engine.
"""

from typing import Dict, List

from sentiment.sentiment_engine import get_sentiment_engine


def fetch_reddit_sentiment(symbols: List[str], limit: int = 50) -> Dict[str, dict]:
    """Fetch Reddit sentiment using sentiment engine."""
    engine = get_sentiment_engine()
    results = {}
    dummy_texts = [f"Dummy Reddit post about stock."] * limit
    for s in symbols:
        sd = engine.reddit_sentiment(s, dummy_texts)
        results[s] = {
            "sentiment_score": sd.sentiment_score if sd else 0.5,
            "confidence": sd.confidence if sd else 0.6,
            "sample_size": limit,
            "sample_texts": dummy_texts[:3],
        }
    return results
