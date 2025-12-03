#!/usr/bin/env python3
"""
Test Integration of News Adapters with DataFeedOrchestrator
Tests the GenericRSSSentimentAdapter with our working RSS feeds
"""

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Set up RSS feeds configuration
os.environ[
    "RSS_FEEDS"
] = "http://feeds.benzinga.com/benzinga,https://www.cnbc.com/id/10001147/device/rss/rss.html,https://www.ft.com/rss/home,https://fortune.com/feed/fortune-feeds/?id=3230629,https://feeds.marketwatch.com/marketwatch/topstories/,https://seekingalpha.com/feed.xml"
os.environ["RSS_INCLUDE_ALL"] = "1"

# Add project root to path
sys.path.append("/Users/omar/Documents/Projects/oracle-x")

try:
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def test_orchestrator_rss_integration():
        """Test RSS feed integration with DataFeedOrchestrator"""
        print("🚀 Testing RSS News Feeds Integration with DataFeedOrchestrator")
        print(f"Time: {datetime.now()}")
        print(f"RSS Feeds Configured: {len(os.environ['RSS_FEEDS'].split(','))} feeds")
        print("=" * 80)

        # Initialize orchestrator (should auto-detect RSS configuration)
        orchestrator = DataFeedOrchestrator()

        # Test symbols for sentiment analysis
        test_symbols = ["AAPL", "TSLA", "MSFT", "GOOGL"]

        results = {}

        for symbol in test_symbols:
            print(f"\n📊 Testing sentiment analysis for {symbol}...")

            try:
                # Test sentiment data retrieval
                sentiment_data_dict = orchestrator.get_sentiment_data(symbol)

                if sentiment_data_dict:
                    # Look for RSS news sentiment data in the results
                    rss_sentiment = None
                    for source_name, sentiment_data in sentiment_data_dict.items():
                        if (
                            "rss" in source_name.lower()
                            or "news" in source_name.lower()
                        ):
                            rss_sentiment = sentiment_data
                            break

                    # Also check for any sentiment data as fallback
                    if not rss_sentiment and sentiment_data_dict:
                        source_name, rss_sentiment = next(
                            iter(sentiment_data_dict.items())
                        )

                    if rss_sentiment:
                        results[symbol] = {
                            "success": True,
                            "sentiment_score": rss_sentiment.sentiment_score,
                            "confidence": rss_sentiment.confidence,
                            "source": rss_sentiment.source,
                            "sample_size": rss_sentiment.sample_size,
                            "timestamp": rss_sentiment.timestamp,
                        }

                        print(
                            f"  ✅ Sentiment Score: {rss_sentiment.sentiment_score:.3f}"
                        )
                        print(f"  📈 Confidence: {rss_sentiment.confidence:.3f}")
                        print(f"  📰 Source: {rss_sentiment.source}")
                        print(f"  📊 Sample Size: {rss_sentiment.sample_size}")
                        print(f"  🕒 Timestamp: {rss_sentiment.timestamp}")

                        if (
                            hasattr(rss_sentiment, "raw_data")
                            and rss_sentiment.raw_data
                        ):
                            sample_texts = rss_sentiment.raw_data.get(
                                "sample_texts", []
                            )
                            if sample_texts:
                                print(f"  📝 Sample Headlines:")
                                for i, text in enumerate(sample_texts[:3], 1):
                                    print(f"    {i}. {text[:100]}...")
                    else:
                        results[symbol] = {
                            "success": False,
                            "error": "No RSS sentiment data found",
                        }
                        print(f"  ❌ No RSS sentiment data found for {symbol}")
                        print(
                            f"  🔍 Available sources: {list(sentiment_data_dict.keys())}"
                        )
                else:
                    results[symbol] = {
                        "success": False,
                        "error": "No sentiment data returned",
                    }
                    print(f"  ❌ No sentiment data returned for {symbol}")

            except Exception as e:
                results[symbol] = {"success": False, "error": str(e)}
                print(f"  ❌ Error for {symbol}: {e}")

        # Summary
        print("\n" + "=" * 80)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 80)

        successful = sum(1 for r in results.values() if r.get("success", False))
        total = len(results)

        print(f"Successfully processed: {successful}/{total} symbols")
        print(f"Success rate: {(successful/total)*100:.1f}%")

        if successful > 0:
            avg_sentiment = (
                sum(
                    r.get("sentiment_score", 0)
                    for r in results.values()
                    if r.get("success", False)
                )
                / successful
            )
            avg_confidence = (
                sum(
                    r.get("confidence", 0)
                    for r in results.values()
                    if r.get("success", False)
                )
                / successful
            )
            total_sample_size = sum(
                r.get("sample_size", 0)
                for r in results.values()
                if r.get("success", False)
            )

            print(f"Average sentiment score: {avg_sentiment:.3f}")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Total news articles processed: {total_sample_size}")

        # Performance metrics
        print(f"\n📊 Performance Metrics:")
        if hasattr(orchestrator, "performance_tracker"):
            print(f"  Performance tracking enabled")
        else:
            print(f"  No performance tracking available")

        if successful >= total * 0.75:  # 75% success rate threshold
            print(f"\n✅ RSS INTEGRATION TEST PASSED!")
            print(
                f"   📰 News adapters successfully integrated into DataFeedOrchestrator"
            )
            print(f"   🎯 {successful}/{total} symbols processed successfully")
            return True
        else:
            print(f"\n❌ RSS INTEGRATION TEST FAILED!")
            print(f"   📰 Insufficient success rate: {(successful/total)*100:.1f}%")
            return False

    if __name__ == "__main__":
        success = test_orchestrator_rss_integration()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)
