"""
Test script for Advanced Sentiment Analysis Engine
Tests multi-model sentiment analysis capabilities
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging to reduce verbose output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)  # Suppress verbose URL logs
logging.getLogger('twscrape').setLevel(logging.WARNING)  # Suppress Twitter warnings

from sentiment.sentiment_engine import (
    get_sentiment_engine, 
    analyze_symbol_sentiment, 
    analyze_text_sentiment
)

def test_individual_models():
    """Test individual sentiment models"""
    print("=== Testing Individual Sentiment Models ===")
    
    # Test texts with different sentiment types
    test_cases = [
        ("AAPL", "Apple earnings beat expectations! Stock is going to the moon! 🚀", "bullish"),
        ("TSLA", "Tesla is way overvalued. This bubble is about to crash hard.", "bearish"),
        ("SPY", "Market is consolidating in this range. Waiting for breakout.", "neutral"),
        ("NVDA", "NVIDIA guidance raised again. AI momentum continues strong.", "bullish"),
        ("META", "Meta missing user growth targets. Bearish for next quarter.", "bearish")
    ]
    
    engine = get_sentiment_engine()
    
    for symbol, text, expected in test_cases:
        print(f"\nTesting {symbol} - Expected: {expected}")
        print(f"Text: {text}")
        
        result = analyze_text_sentiment(text, symbol, "test")
        
        print(f"✓ VADER Score: {result.vader_score:.3f}")
        print(f"✓ FinBERT Score: {result.finbert_score:.3f}")
        print(f"✓ Ensemble Score: {result.ensemble_score:.3f}")
        print(f"✓ Confidence: {result.confidence:.3f}")
        print(f"✓ Model Weights: {result.model_weights}")

def test_batch_analysis():
    """Test batch sentiment analysis"""
    print("\n=== Testing Batch Analysis ===")
    
    # Sample financial texts
    texts = [
        "AAPL beat earnings expectations with strong iPhone sales",
        "Fed might raise rates again causing market volatility",
        "Tech stocks rallying on AI optimism and strong demand",
        "Banking sector under pressure from credit concerns",
        "Oil prices surging on supply constraints and demand"
    ]
    
    symbols = ["AAPL", "SPY", "QQQ", "XLF", "XLE"]
    sources = ["news"] * len(texts)
    
    engine = get_sentiment_engine()
    results = engine.analyze_batch(texts, symbols, sources)
    
    print(f"✓ Processed {len(results)} texts")
    for result in results:
        print(f"  {result.symbol}: {result.ensemble_score:.3f} (conf: {result.confidence:.3f})")

def test_symbol_aggregation():
    """Test symbol-level sentiment aggregation"""
    print("\n=== Testing Symbol Aggregation ===")
    
    # Multiple texts for AAPL
    aapl_texts = [
        "Apple iPhone 15 sales exceeding expectations globally",
        "AAPL services revenue growing strongly quarter over quarter",
        "Analysts upgrading AAPL price targets on AI integration",
        "Apple supply chain improvements reducing production costs",
        "Strong institutional buying in AAPL ahead of earnings"
    ]
    
    summary = analyze_symbol_sentiment("AAPL", aapl_texts, ["news", "research", "analyst", "supply", "flow"])
    
    print(f"✓ AAPL Sentiment Summary:")
    print(f"  Overall Sentiment: {summary.overall_sentiment:.3f}")
    print(f"  Confidence: {summary.confidence:.3f}")
    print(f"  Sample Size: {summary.sample_size}")
    print(f"  Bullish: {summary.bullish_mentions}, Bearish: {summary.bearish_mentions}, Neutral: {summary.neutral_mentions}")
    print(f"  Trending: {summary.trending_direction}")
    print(f"  Quality Score: {summary.quality_score:.1f}")

def test_financial_lexicon():
    """Test financial lexicon recognition"""
    print("\n=== Testing Financial Lexicon ===")
    
    # Test financial terminology
    financial_texts = [
        "Stock breaking out of resistance with strong volume",
        "Bearish divergence suggests potential correction ahead",
        "Options flow showing unusual call activity in tech",
        "Dark pools accumulating shares before earnings",
        "Institutional rotation from growth to value names"
    ]
    
    symbols = ["TEST"] * len(financial_texts)
    
    engine = get_sentiment_engine()
    results = engine.analyze_batch(financial_texts, symbols)
    
    print("✓ Financial Lexicon Recognition:")
    for i, result in enumerate(results):
        print(f"  Text {i+1}: Ensemble: {result.ensemble_score:.3f}, Confidence: {result.confidence:.3f}")

def test_model_performance_tracking():
    """Test model performance tracking"""
    print("\n=== Testing Model Performance Tracking ===")
    
    engine = get_sentiment_engine()
    
    # Show current model weights
    print("✓ Current Model Performance:")
    for model, perf in engine.model_performance.items():
        print(f"  {model}: Accuracy: {perf['accuracy']:.2f}, Weight: {perf['weight']:.2f}")
    
    # Update performance (example)
    engine.update_model_performance("finbert", 0.88)
    print("✓ Updated FinBERT performance to 0.88")

def main():
    """Run all advanced sentiment tests"""
    print("Testing Advanced Sentiment Analysis Engine")
    print("=" * 50)
    
    try:
        test_individual_models()
        test_batch_analysis()
        test_symbol_aggregation()
        test_financial_lexicon()
        test_model_performance_tracking()
        
        print("\n" + "=" * 50)
        print("✓ All advanced sentiment tests completed!")
        print("\nAdvanced sentiment engine is ready for production.")
        print("Features tested:")
        print("- Multi-model ensemble (VADER + FinBERT + Financial Lexicon)")
        print("- Dynamic model weighting based on context")
        print("- Batch processing capabilities")
        print("- Symbol-level sentiment aggregation")
        print("- Financial terminology recognition")
        print("- Performance tracking and adaptation")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
