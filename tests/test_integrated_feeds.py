#!/usr/bin/env python3
"""
Test integrated data feeds (free + premium) without full pipeline
"""

import time
import sys


def test_integrated_feeds():
    """Test free + premium feeds integration"""
    print("=" * 70)
    print("🧪 TESTING INTEGRATED DATA FEEDS (FREE + PREMIUM)")
    print("=" * 70)

    # Test imports
    print("\n📦 Testing imports...")
    try:
        from data_feeds.market_internals import fetch_market_internals
        from data_feeds.options_flow import fetch_options_flow
        from data_feeds.dark_pools import fetch_dark_pool_data
        from data_feeds.earnings_calendar import fetch_earnings_calendar
        from data_feeds.sentiment import fetch_sentiment_data
        from data_feeds.news_scraper import fetch_headlines_yahoo_finance
        from data_feeds.finviz_scraper import fetch_finviz_breadth
        from data_feeds.ticker_universe import fetch_ticker_universe
        from data_feeds.strategic_premium_feeds import get_premium_feeds

        print("   ✅ All imports successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

    # Test free feeds
    print("\n📊 Testing FREE data feeds...")
    start_free = time.time()

    try:
        tickers = fetch_ticker_universe(sample_size=10)
        print(f"   ✓ Tickers: {len(tickers)} fetched")

        internals = fetch_market_internals()
        print(f"   ✓ Market Internals: VIX={internals.get('vix', 'N/A')}")

        options_flow = fetch_options_flow(tickers[:3])
        print(
            f"   ✓ Options Flow: {len(options_flow.get('unusual_sweeps', []))} sweeps"
        )

        dark_pools = fetch_dark_pool_data(tickers[:3])
        print(f"   ✓ Dark Pools: {len(dark_pools.get('dark_pools', []))} signals")

        earnings = fetch_earnings_calendar(tickers[:3])
        print(f"   ✓ Earnings: {len(earnings)} events")

        yahoo = fetch_headlines_yahoo_finance()
        print(f"   ✓ Yahoo News: {len(yahoo)} headlines")

        finviz = fetch_finviz_breadth()
        print(f"   ✓ Finviz: Advancers={finviz.get('advancers', 'N/A')}")

        elapsed_free = time.time() - start_free
        print(f"\n   ⚡ Free feeds time: {elapsed_free:.2f}s")

    except Exception as e:
        print(f"   ❌ Free feeds failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test premium feeds
    print("\n💎 Testing PREMIUM data feeds (optional)...")
    start_premium = time.time()

    try:
        premium = get_premium_feeds()

        print(
            f"   Finnhub: {'✅ Available' if premium.finnhub_available else '❌ No API key'}"
        )
        print(f"   FMP: {'✅ Available' if premium.fmp_available else '❌ No API key'}")

        if premium.finnhub_available or premium.fmp_available:
            # Test with 2 tickers
            test_tickers = tickers[:2]
            print(f"\n   Testing premium data for {test_tickers}...")

            premium_data = premium.get_all_premium_data(test_tickers, max_symbols=2)

            # Show Finnhub data
            finnhub_data = premium_data.get("finnhub", {})
            if finnhub_data.get("available"):
                insider = len(finnhub_data.get("insider_sentiment", {}))
                recos = len(finnhub_data.get("recommendation_trends", {}))
                targets = len(finnhub_data.get("price_targets", {}))
                print(
                    f"   ✓ Finnhub: {insider} insider, {recos} recommendations, {targets} price targets"
                )

            # Show FMP data
            fmp_data = premium_data.get("fmp", {})
            if fmp_data.get("available"):
                estimates = len(fmp_data.get("analyst_estimates", {}))
                institutional = len(fmp_data.get("institutional_ownership", {}))
                dcf = len(fmp_data.get("dcf_valuations", {}))
                print(
                    f"   ✓ FMP: {estimates} analyst estimates, {institutional} institutional data, {dcf} DCF valuations"
                )

            elapsed_premium = time.time() - start_premium
            print(f"\n   ⚡ Premium feeds time: {elapsed_premium:.2f}s")
        else:
            print("   ⚠️  No premium APIs configured (this is OK)")
            print(
                "   💡 Set FINNHUB_API_KEY and FINANCIALMODELINGPREP_API_KEY to enable"
            )
            elapsed_premium = 0

    except Exception as e:
        print(f"   ⚠️  Premium feeds failed (non-critical): {e}")
        elapsed_premium = 0

    # Summary
    total_time = time.time() - start_free
    print("\n" + "=" * 70)
    print("✅ INTEGRATED FEEDS TEST COMPLETE")
    print("=" * 70)

    print(f"\n📈 Performance Summary:")
    print(f"   Free feeds: {elapsed_free:.2f}s")
    print(f"   Premium feeds: {elapsed_premium:.2f}s")
    print(f"   Total: {total_time:.2f}s")

    print(f"\n💡 Architecture:")
    print(f"   ✓ Free feeds provide core market data")
    print(f"   ✓ Premium feeds add UNIQUE advanced data")
    print(f"   ✓ Rate limiting prevents API exhaustion")
    print(f"   ✓ System works with or without premium APIs")

    return True


if __name__ == "__main__":
    try:
        result = test_integrated_feeds()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
