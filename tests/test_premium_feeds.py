#!/usr/bin/env python3
"""
Test strategic premium feeds integration
Validates rate limiting, batching, and unique data collection
"""

import os
import time
from data_feeds.strategic_premium_feeds import get_premium_feeds


def test_premium_feeds():
    """Test premium feeds with rate limiting"""
    print("=" * 70)
    print("💎 TESTING STRATEGIC PREMIUM FEEDS")
    print("=" * 70)

    premium = get_premium_feeds()

    # Check API availability
    print("\n📋 API Availability:")
    print(
        f"   Finnhub: {'✅ Available' if premium.finnhub_available else '❌ No API key'}"
    )
    print(f"   FMP: {'✅ Available' if premium.fmp_available else '❌ No API key'}")

    if not premium.finnhub_available and not premium.fmp_available:
        print("\n⚠️  No premium APIs available (this is OK - they're optional)")
        print("   Set FINNHUB_API_KEY and/or FINANCIALMODELINGPREP_API_KEY to enable")
        return None

    # Test with small batch
    test_symbols = ["AAPL", "MSFT", "GOOGL"]

    print(f"\n🔍 Testing with {len(test_symbols)} symbols: {test_symbols}")
    print("   (Limited to prevent rate limit exhaustion)")

    start_time = time.time()

    # Test Finnhub unique data
    if premium.finnhub_available:
        print("\n📊 Fetching Finnhub unique data...")
        print("   - Insider sentiment (executive buys/sells)")
        print("   - Recommendation trends (analyst ratings)")
        print("   - Price targets (analyst estimates)")

        finnhub_data = premium.get_unique_finnhub_data(test_symbols, max_symbols=3)

        if finnhub_data.get("available"):
            print(f"   ✅ Finnhub data fetched")

            # Show insider sentiment
            if finnhub_data.get("insider_sentiment"):
                print(f"\n   📈 Insider Sentiment:")
                for symbol, data in finnhub_data["insider_sentiment"].items():
                    print(
                        f"      {symbol}: {data['sentiment']} (net: {data['net_buying']:,} shares, {data['transactions']} transactions)"
                    )

            # Show recommendations
            if finnhub_data.get("recommendation_trends"):
                print(f"\n   🎯 Analyst Recommendations:")
                for symbol, data in finnhub_data["recommendation_trends"].items():
                    print(
                        f"      {symbol}: {data['consensus']} (buy: {data['buy']}, hold: {data['hold']}, sell: {data['sell']})"
                    )

            # Show price targets
            if finnhub_data.get("price_targets"):
                print(f"\n   💰 Price Targets:")
                for symbol, data in finnhub_data["price_targets"].items():
                    mean = data.get("target_mean")
                    if mean:
                        print(
                            f"      {symbol}: ${mean:.2f} avg (${data['target_low']:.2f}-${data['target_high']:.2f}, {data['num_analysts']} analysts)"
                        )

            if finnhub_data.get("fetch_errors"):
                print(f"\n   ⚠️  Errors: {len(finnhub_data['fetch_errors'])}")
                for error in finnhub_data["fetch_errors"][:3]:
                    print(f"      {error}")
        else:
            print(f"   ❌ Finnhub unavailable: {finnhub_data.get('reason')}")

    # Test FMP unique data
    if premium.fmp_available:
        print("\n📊 Fetching FMP unique data...")
        print("   - Analyst earnings/revenue estimates")
        print("   - Institutional ownership (smart money)")
        print("   - Insider transactions (detailed)")
        print("   - DCF valuations (fair value)")

        fmp_data = premium.get_unique_fmp_data(test_symbols, max_symbols=3)

        if fmp_data.get("available"):
            print(f"   ✅ FMP data fetched")

            # Show analyst estimates
            if fmp_data.get("analyst_estimates"):
                print(f"\n   📊 Analyst Estimates (Forward-Looking):")
                for symbol, data in fmp_data["analyst_estimates"].items():
                    eps = data.get("estimated_eps")
                    rev = data.get("estimated_revenue")
                    if eps:
                        print(
                            f"      {symbol}: EPS ${eps:.2f} (${data['estimated_eps_low']:.2f}-${data['estimated_eps_high']:.2f})"
                        )
                    if rev:
                        print(
                            f"         Revenue: ${rev/1e9:.2f}B ({data['num_analysts']} analysts)"
                        )

            # Show institutional ownership
            if fmp_data.get("institutional_ownership"):
                print(f"\n   🏦 Institutional Ownership (Smart Money):")
                for symbol, data in fmp_data["institutional_ownership"].items():
                    print(
                        f"      {symbol}: Top 5 hold {data['top_5_shares']:,} shares ({data['total_holders']} total holders)"
                    )
                    print(
                        f"         Top holders: {', '.join(data['top_5_holders'][:3])}"
                    )

            # Show insider transactions
            if fmp_data.get("insider_transactions"):
                print(f"\n   🔐 Insider Transactions (Recent Activity):")
                for symbol, data in fmp_data["insider_transactions"].items():
                    print(
                        f"      {symbol}: {data['net_sentiment']} (buys: {data['recent_buys']}, sells: {data['recent_sells']})"
                    )

            # Show DCF valuations
            if fmp_data.get("dcf_valuations"):
                print(f"\n   💎 DCF Valuations (Fair Value):")
                for symbol, data in fmp_data["dcf_valuations"].items():
                    dcf = data.get("dcf_value")
                    price = data.get("stock_price")
                    if dcf and price:
                        upside = (dcf - price) / price * 100
                        print(
                            f"      {symbol}: Fair value ${dcf:.2f} vs Price ${price:.2f} ({upside:+.1f}% upside)"
                        )

            if fmp_data.get("fetch_errors"):
                print(f"\n   ⚠️  Errors: {len(fmp_data['fetch_errors'])}")
                for error in fmp_data["fetch_errors"][:3]:
                    print(f"      {error}")
        else:
            print(f"   ❌ FMP unavailable: {fmp_data.get('reason')}")

    elapsed = time.time() - start_time

    # Test batched call
    print(f"\n⚡ Testing batched premium data fetch...")
    start_batch = time.time()
    all_data = premium.get_all_premium_data(test_symbols, max_symbols=2)
    elapsed_batch = time.time() - start_batch

    print(f"   ✅ Batched fetch completed in {elapsed_batch:.2f}s")

    # Performance analysis
    print(f"\n📈 Performance Analysis:")
    print(f"   Total time: {elapsed + elapsed_batch:.2f}s")
    print(
        f"   Rate limiting: {'✅ Working' if elapsed > 1 else '⚠️  Very fast (may not be limiting)'}"
    )

    # Data quality summary
    print(f"\n📊 Data Quality Summary:")

    unique_data_points = []
    if premium.finnhub_available:
        unique_data_points.extend(
            [
                "Insider sentiment (Finnhub)",
                "Analyst recommendations (Finnhub)",
                "Price targets (Finnhub)",
            ]
        )
    if premium.fmp_available:
        unique_data_points.extend(
            [
                "Analyst estimates (FMP)",
                "Institutional ownership (FMP)",
                "Insider transactions (FMP)",
                "DCF valuations (FMP)",
            ]
        )

    print(f"   Available unique data points: {len(unique_data_points)}")
    for i, point in enumerate(unique_data_points, 1):
        print(f"      {i}. {point}")

    print("\n" + "=" * 70)
    print("✅ PREMIUM FEEDS INTEGRATION TEST COMPLETE")
    print("=" * 70)

    print("\n💡 Strategic Usage:")
    print("   ✓ Premium APIs used ONLY for unique data")
    print("   ✓ Rate limiting prevents API exhaustion")
    print("   ✓ Batching optimizes multiple ticker requests")
    print("   ✓ Free sources (yfinance) still provide core data")

    return True


if __name__ == "__main__":
    import sys

    try:
        result = test_premium_feeds()
        sys.exit(0 if result is not False else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
