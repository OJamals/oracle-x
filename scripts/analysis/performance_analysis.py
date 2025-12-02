#!/usr/bin/env python3
"""
Performance Analysis for Oracle-X Data Feed System

Analyzes timing results from comprehensive testing to identify optimization opportunities.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any


def analyze_performance():
    """Analyze performance data from comprehensive testing"""

    # Load test results
    results_file = "test_results_comprehensive.json"
    if not os.path.exists(results_file):
        print(f"❌ Test results file not found: {results_file}")
        print(
            "Run the comprehensive tests first: python test_data_feeds_comprehensive.py"
        )
        return

    with open(results_file, "r") as f:
        test_results = json.load(f)

    print("=== Oracle-X Performance Analysis ===")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Categorize tests
    adapter_tests = []
    orchestrator_tests = []
    cli_tests = []

    for result in test_results:
        if result.get("timing_ms") is None:
            continue  # Skip skipped tests

        test_name = result["test"]
        timing = result["timing_ms"]

        if "orchestrator" in test_name:
            orchestrator_tests.append((test_name, timing))
        elif "cli" in test_name:
            cli_tests.append((test_name, timing))
        else:
            adapter_tests.append((test_name, timing))

    # Individual Adapter Performance
    print("## Individual Adapter Performance")
    print("| Adapter | Time (ms) | Performance |")
    print("|---------|-----------|-------------|")

    adapter_tests.sort(key=lambda x: x[1])  # Sort by timing

    for test_name, timing in adapter_tests:
        if timing < 500:
            performance = "🚀 Excellent"
        elif timing < 1000:
            performance = "✅ Good"
        elif timing < 2000:
            performance = "⚠️ Acceptable"
        else:
            performance = "🐌 Slow"

        print(f"| {test_name} | {timing:.0f} | {performance} |")

    print()

    # Orchestrator Performance
    print("## Orchestrator Performance")
    print("| Operation | Time (ms) | Performance |")
    print("|-----------|-----------|-------------|")

    orchestrator_tests.sort(key=lambda x: x[1])

    for test_name, timing in orchestrator_tests:
        operation = test_name.replace("orchestrator_", "")

        if operation == "init":
            if timing < 100:
                performance = "🚀 Excellent"
            else:
                performance = "⚠️ Check initialization"
        elif operation == "quote":
            if timing < 1000:
                performance = "🚀 Excellent"
            elif timing < 2000:
                performance = "✅ Good"
            else:
                performance = "⚠️ Review sources"
        elif operation == "sentiment":
            if timing < 5000:
                performance = "✅ Good"
            elif timing < 10000:
                performance = "⚠️ Acceptable"
            else:
                performance = "🐌 Needs optimization"
        elif operation == "market_data":
            if timing < 1000:
                performance = "🚀 Excellent"
            elif timing < 2000:
                performance = "✅ Good"
            else:
                performance = "⚠️ Review sources"
        else:
            performance = "📊 Analysis needed"

        print(f"| {operation} | {timing:.0f} | {performance} |")

    print()

    # CLI Performance
    print("## CLI Integration Performance")
    print("| CLI Command | Time (ms) | Performance |")
    print("|-------------|-----------|-------------|")

    cli_tests.sort(key=lambda x: x[1])

    for test_name, timing in cli_tests:
        command = test_name.replace("cli_validate_", "")

        if timing < 2000:
            performance = "🚀 Excellent"
        elif timing < 4000:
            performance = "✅ Good"
        else:
            performance = "⚠️ Review overhead"

        print(f"| {command} | {timing:.0f} | {performance} |")

    print()

    # Performance Recommendations
    print("## Performance Optimization Recommendations")
    print()

    # Find slowest adapters
    slow_adapters = [test for test in adapter_tests if test[1] > 2000]
    if slow_adapters:
        print("### 🐌 Slow Adapters (>2s)")
        for test_name, timing in slow_adapters:
            print(f"- **{test_name}**: {timing:.0f}ms")

            if "reddit" in test_name:
                print("  - Consider caching Reddit API responses")
                print("  - Implement concurrent processing for multiple subreddits")
                print("  - Add request batching")
            elif "twelvedata" in test_name:
                print("  - Check API response time vs rate limits")
                print("  - Consider upgrading to premium tier")
                print("  - Implement intelligent fallback ordering")
            elif "twitter" in test_name:
                print("  - Review Twitter API v2 performance")
                print("  - Consider reducing search scope")
                print("  - Implement streaming for real-time data")
        print()

    # Sentiment analysis optimization
    sentiment_tests = [test for test in orchestrator_tests if "sentiment" in test[0]]
    if sentiment_tests and sentiment_tests[0][1] > 5000:
        print("### 📊 Sentiment Analysis Optimization")
        print(f"- Current sentiment processing: {sentiment_tests[0][1]:.0f}ms")
        print("- **Recommended optimizations:**")
        print("  - Implement parallel sentiment processing")
        print("  - Add sentiment caching with appropriate TTL")
        print("  - Consider pre-processing popular tickers")
        print("  - Optimize VADER sentiment model performance")
        print()

    # Fast performers to highlight
    fast_adapters = [test for test in adapter_tests if test[1] < 500]
    if fast_adapters:
        print("### 🚀 High-Performance Adapters (<500ms)")
        for test_name, timing in fast_adapters:
            print(f"- **{test_name}**: {timing:.0f}ms - Excellent performance!")
        print()

    # Overall system health
    total_tests = len([r for r in test_results if r.get("status") == "PASS"])
    print("## System Health Summary")
    print(f"- ✅ **Tests Passing**: {total_tests}/22")
    print(f"- 🚀 **Fast Adapters**: {len(fast_adapters)} adapters under 500ms")
    print(f"- 🐌 **Slow Adapters**: {len(slow_adapters)} adapters over 2s")

    # Calculate average performance
    all_timings = [test[1] for test in adapter_tests + orchestrator_tests + cli_tests]
    avg_timing = sum(all_timings) / len(all_timings)
    print(f"- 📊 **Average Response Time**: {avg_timing:.0f}ms")

    if avg_timing < 1000:
        print("- 🎯 **Overall Performance**: Excellent system performance!")
    elif avg_timing < 2000:
        print("- 🎯 **Overall Performance**: Good system performance")
    else:
        print("- 🎯 **Overall Performance**: Consider optimization opportunities")


if __name__ == "__main__":
    analyze_performance()
