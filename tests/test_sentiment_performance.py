#!/usr/bin/env python3
"""
Performance Test Script for Enhanced Sentiment Pipeline
Tests the optimization improvements to ensure processing time is reduced from 6+ seconds to <2 seconds
"""

import time
import sys
import os
from typing import List, Dict, Any
import statistics

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_feeds.enhanced_sentiment_pipeline import get_enhanced_sentiment_pipeline


def run_performance_test(
    symbols: List[str], num_iterations: int = 3, include_reddit: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive performance test for the sentiment pipeline

    Args:
        symbols: List of stock symbols to test
        num_iterations: Number of iterations to run for each symbol
        include_reddit: Whether to include Reddit sentiment

    Returns:
        Performance test results
    """
    print("🚀 Starting Enhanced Sentiment Pipeline Performance Test")
    print("=" * 60)

    pipeline = get_enhanced_sentiment_pipeline()

    # Show pipeline configuration
    health = pipeline.get_health_status()
    print(f"📊 Pipeline Configuration:")
    print(f"   • Sources: {health['sources_count']}")
    print(f"   • Max Workers: {health['max_workers']}")
    print(f"   • Timeout: {health['timeout_seconds']}s")
    print(f"   • Batch Size: {health['batch_size']}")
    print(f"   • Reddit Available: {health['reddit_available']}")
    print()

    all_results = []
    all_times = []

    # Test each symbol multiple times
    for symbol in symbols:
        print(f"📈 Testing {symbol}...")
        symbol_times = []

        for i in range(num_iterations):
            print(f"   Iteration {i+1}/{num_iterations}...", end=" ")

            start_time = time.time()
            try:
                result = pipeline.get_sentiment_analysis(
                    symbol, include_reddit=include_reddit
                )
                end_time = time.time()
                processing_time = end_time - start_time

                symbol_times.append(processing_time)
                all_times.append(processing_time)

                # Show result summary
                sources = result.get("sources_count", 0)
                sentiment = result.get("overall_sentiment", 0.0)
                confidence = result.get("confidence", 0.0)
                cached = result.get("cached", False)

                print(
                    f"✅ {processing_time:.2f}s "
                    f"({sources} sources, sentiment: {sentiment:.3f}, "
                    f"confidence: {confidence:.3f})"
                    f"{' [CACHED]' if cached else ''}"
                )

                all_results.append(
                    {
                        "symbol": symbol,
                        "iteration": i + 1,
                        "processing_time": processing_time,
                        "sources_count": sources,
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "cached": cached,
                    }
                )

            except Exception as e:
                end_time = time.time()
                processing_time = end_time - start_time
                symbol_times.append(processing_time)
                all_times.append(processing_time)

                print(f"❌ {processing_time:.2f}s (Error: {e})")

                all_results.append(
                    {
                        "symbol": symbol,
                        "iteration": i + 1,
                        "processing_time": processing_time,
                        "error": str(e),
                    }
                )

        # Show symbol summary
        if symbol_times:
            avg_time = statistics.mean(symbol_times)
            min_time = min(symbol_times)
            max_time = max(symbol_times)
            print(
                f"   📊 {symbol} Summary: Avg={avg_time:.2f}s, Min={min_time:.2f}s, Max={max_time:.2f}s"
            )
            print()

    # Overall performance analysis
    print("🎯 PERFORMANCE ANALYSIS")
    print("=" * 60)

    if all_times:
        avg_time = statistics.mean(all_times)
        median_time = statistics.median(all_times)
        min_time = min(all_times)
        max_time = max(all_times)

        print(f"📈 Overall Performance:")
        print(f"   • Average Time: {avg_time:.2f}s")
        print(f"   • Median Time: {median_time:.2f}s")
        print(f"   • Fastest: {min_time:.2f}s")
        print(f"   • Slowest: {max_time:.2f}s")
        print(f"   • Total Requests: {len(all_times)}")

        # Performance targets
        target_under_2s = sum(1 for t in all_times if t < 2.0)
        target_under_1s = sum(1 for t in all_times if t < 1.0)

        print(f"🎯 Target Achievement:")
        print(
            f"   • Under 2 seconds: {target_under_2s}/{len(all_times)} ({target_under_2s/len(all_times)*100:.1f}%)"
        )
        print(
            f"   • Under 1 second: {target_under_1s}/{len(all_times)} ({target_under_1s/len(all_times)*100:.1f}%)"
        )

        # Performance rating
        if avg_time < 1.0:
            rating = "🚀 EXCELLENT"
            color = "✅"
        elif avg_time < 2.0:
            rating = "👍 VERY GOOD"
            color = "✅"
        elif avg_time < 3.0:
            rating = "👌 GOOD"
            color = "✅"
        elif avg_time < 4.0:
            rating = "⚠️  NEEDS IMPROVEMENT"
            color = "⚠️"
        else:
            rating = "❌ POOR PERFORMANCE"
            color = "❌"

        print(f"   • Performance Rating: {color} {rating}")
        print(
            f"   • Improvement from 6+ seconds: {((6.0 - avg_time) / 6.0) * 100:.1f}% faster"
        )

        # Success criteria
        success = avg_time < 2.0
        print(f"   • Target Met (<2s): {'✅ YES' if success else '❌ NO'}")

    # Show pipeline health
    print()
    print("🏥 Pipeline Health Status:")
    health = pipeline.get_health_status()
    print(f"   • Status: {health['pipeline_status']}")
    print(f"   • Cache Size: {health['cache_size']}")
    print(f"   • Performance Stats: {health['performance_stats']}")

    return {
        "results": all_results,
        "performance_stats": {
            "average_time": statistics.mean(all_times) if all_times else 0,
            "median_time": statistics.median(all_times) if all_times else 0,
            "min_time": min(all_times) if all_times else 0,
            "max_time": max(all_times) if all_times else 0,
            "total_requests": len(all_times),
            "target_met": statistics.mean(all_times) < 2.0 if all_times else False,
        },
        "pipeline_health": health,
    }


def main():
    """Main function to run performance tests"""

    # Test symbols - mix of popular and less common stocks
    test_symbols = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "META", "AMZN"]

    print("Performance Test Configuration:")
    print(f"  • Test Symbols: {test_symbols}")
    print(f"  • Iterations per Symbol: 3")
    print(f"  • Include Reddit: True")
    print(f"  • Target Performance: <2 seconds")
    print()

    # Run the performance test
    results = run_performance_test(
        symbols=test_symbols, num_iterations=3, include_reddit=True
    )

    # Save results to file
    import json

    with open("sentiment_performance_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print("📄 Results saved to: sentiment_performance_results.json")

    # Final summary
    stats = results["performance_stats"]
    if stats["target_met"]:
        print(
            "🎉 SUCCESS: Performance target achieved! Processing time reduced to <2 seconds."
        )
    else:
        print(
            f"⚠️  WARNING: Target not fully achieved. Average time: {stats['average_time']:.2f}s"
        )

    return stats["target_met"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
