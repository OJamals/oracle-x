import pandas as pd
from data_feeds.finviz_adapter import FinVizAdapter
from data_feeds.models import MarketBreadth, GroupPerformance


def test_finviz_adapter():
    """Test the refactored FinViz adapter"""
    print("Testing refactored FinViz adapter...")

    # Create adapter instance
    adapter = FinVizAdapter()

    # Test market breadth
    print("\n=== Testing Market Breadth ===")
    try:
        breadth = adapter.get_market_breadth()
        if breadth:
            print(f"✅ Market breadth fetched successfully")
            print(f"   Advancers: {breadth.advancers}")
            print(f"   Decliners: {breadth.decliners}")
            print(f"   New Highs: {breadth.new_highs}")
            print(f"   New Lows: {breadth.new_lows}")
            print(f"   Source: {breadth.source}")
        else:
            print("❌ Market breadth fetch failed")
    except Exception as e:
        print(f"❌ Market breadth test failed with error: {e}")

    # Test sector performance
    print("\n=== Testing Sector Performance ===")
    try:
        sectors = adapter.get_sector_performance()
        if sectors:
            print(f"✅ Sector performance fetched successfully")
            print(f"   Number of sectors: {len(sectors)}")
            if sectors:
                sector = sectors[0]
                print(f"   First sector: {sector.group_name}")
                print(f"   1D Performance: {sector.perf_1d}")
                print(f"   1W Performance: {sector.perf_1w}")
                print(f"   Source: {sector.source}")
        else:
            print("❌ Sector performance fetch returned empty list")
    except Exception as e:
        print(f"❌ Sector performance test failed with error: {e}")

    # Test news
    print("\n=== Testing News ===")
    try:
        news_data = adapter.get_news()
        if news_data:
            print(f"✅ News fetched successfully")
            print(f"   News keys: {list(news_data.keys())}")
            if "news" in news_data and not news_data["news"].empty:
                print(f"   News items: {len(news_data['news'])}")
                print(f"   News columns: {list(news_data['news'].columns)}")
            if "blogs" in news_data and not news_data["blogs"].empty:
                print(f"   Blog items: {len(news_data['blogs'])}")
        else:
            print("❌ News fetch returned empty data")
    except Exception as e:
        print(f"❌ News test failed with error: {e}")

    # Test insider trading
    print("\n=== Testing Insider Trading ===")
    try:
        insider_data = adapter.get_insider_trading()
        if not insider_data.empty:
            print(f"✅ Insider trading fetched successfully")
            print(f"   Insider trades: {len(insider_data)}")
            print(f"   Insider columns: {list(insider_data.columns)}")
        else:
            print("⚠️  Insider trading fetch returned empty DataFrame")
    except Exception as e:
        print(f"❌ Insider trading test failed with error: {e}")

    # Test earnings
    print("\n=== Testing Earnings ===")
    try:
        earnings_data = adapter.get_earnings()
        if earnings_data:
            print(f"✅ Earnings fetched successfully")
            print(f"   Earnings days: {list(earnings_data.keys())}")
            if earnings_data:
                first_day = list(earnings_data.keys())[0]
                print(f"   First day items: {len(earnings_data[first_day])}")
                print(f"   First day columns: {list(earnings_data[first_day].columns)}")
        else:
            print("⚠️  Earnings fetch returned empty data")
    except Exception as e:
        print(f"❌ Earnings test failed with error: {e}")

    # Test forex
    print("\n=== Testing Forex ===")
    try:
        forex_data = adapter.get_forex()
        if not forex_data.empty:
            print(f"✅ Forex fetched successfully")
            print(f"   Forex pairs: {len(forex_data)}")
            print(f"   Forex columns: {list(forex_data.columns)}")
        else:
            print("⚠️  Forex fetch returned empty DataFrame")
    except Exception as e:
        print(f"❌ Forex test failed with error: {e}")

    # Test crypto
    print("\n=== Testing Crypto ===")
    try:
        crypto_data = adapter.get_crypto()
        if not crypto_data.empty:
            print(f"✅ Crypto fetched successfully")
            print(f"   Crypto assets: {len(crypto_data)}")
            print(f"   Crypto columns: {list(crypto_data.columns)}")
        else:
            print("⚠️  Crypto fetch returned empty DataFrame")
    except Exception as e:
        print(f"❌ Crypto test failed with error: {e}")


def test_finviz_scraper_directly():
    """Test the scraper functions directly"""
    print("\n" + "=" * 50)
    print("Testing finviz_scraper functions directly...")
    print("=" * 50)

    from data_feeds.finviz_scraper import (
        fetch_finviz_breadth,
        fetch_finviz_sector_performance,
        fetch_finviz_news,
        fetch_finviz_insider_trading,
        fetch_finviz_earnings,
        fetch_finviz_forex,
        fetch_finviz_crypto,
    )

    # Test breadth
    print("\n=== Testing fetch_finviz_breadth ===")
    try:
        breadth = fetch_finviz_breadth()
        print(f"✅ Breadth data: {breadth}")
    except Exception as e:
        print(f"❌ Breadth fetch failed: {e}")

    # Test sector performance
    print("\n=== Testing fetch_finviz_sector_performance ===")
    try:
        sectors = fetch_finviz_sector_performance()
        print(f"✅ Sector performance items: {len(sectors)}")
        if sectors:
            print(f"   First sector: {sectors[0]['sector_name']}")
    except Exception as e:
        print(f"❌ Sector performance fetch failed: {e}")

    # Test news
    print("\n=== Testing fetch_finviz_news ===")
    try:
        news = fetch_finviz_news()
        print(f"✅ News keys: {list(news.keys())}")
    except Exception as e:
        print(f"❌ News fetch failed: {e}")

    # Test insider trading
    print("\n=== Testing fetch_finviz_insider_trading ===")
    try:
        insider = fetch_finviz_insider_trading()
        print(f"✅ Insider trades: {len(insider)}")
    except Exception as e:
        print(f"❌ Insider trading fetch failed: {e}")

    # Test earnings
    print("\n=== Testing fetch_finviz_earnings ===")
    try:
        earnings = fetch_finviz_earnings()
        print(f"✅ Earnings days: {list(earnings.keys())}")
    except Exception as e:
        print(f"❌ Earnings fetch failed: {e}")

    # Test forex
    print("\n=== Testing fetch_finviz_forex ===")
    try:
        forex = fetch_finviz_forex()
        print(f"✅ Forex pairs: {len(forex)}")
    except Exception as e:
        print(f"❌ Forex fetch failed: {e}")

    # Test crypto
    print("\n=== Testing fetch_finviz_crypto ===")
    try:
        crypto = fetch_finviz_crypto()
        print(f"✅ Crypto assets: {len(crypto)}")
    except Exception as e:
        print(f"❌ Crypto fetch failed: {e}")


if __name__ == "__main__":
    print("Running comprehensive FinViz refactored implementation tests...")
    test_finviz_adapter()
    test_finviz_scraper_directly()
    print("\n" + "=" * 50)
    print("Testing completed!")
    print("=" * 50)
