import pandas as pd
from finvizfinance.group.performance import Performance
from finvizfinance.group.overview import Overview
from finvizfinance.news import News
from finvizfinance.insider import Insider
from finvizfinance.earnings import Earnings
from finvizfinance.forex import Forex
from finvizfinance.crypto import Crypto

def test_sector_performance():
    """Test sector performance data"""
    print("=== Sector Performance ===")
    try:
        fgperformance = Performance()
        df = fgperformance.screener_view(group='Sector')
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("First 3 rows:")
        print(df.head(3))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_market_breadth():
    """Test market breadth data"""
    print("\n=== Market Breadth (Sector Overview) ===")
    try:
        fgoverview = Overview()
        df = fgoverview.screener_view(group='Sector')
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("First 3 rows:")
        print(df.head(3))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_news():
    """Test news data"""
    print("\n=== News ===")
    try:
        fnews = News()
        all_news = fnews.get_news()
        print("News keys:", list(all_news.keys()))
        if 'news' in all_news:
            news_df = all_news['news']
            print(f"News shape: {news_df.shape}")
            print(f"News columns: {list(news_df.columns)}")
            print("First 2 news items:")
            print(news_df.head(2))
        if 'blogs' in all_news:
            blogs_df = all_news['blogs']
            print(f"Blogs shape: {blogs_df.shape}")
            print(f"Blogs columns: {list(blogs_df.columns)}")
            print("First 2 blog items:")
            print(blogs_df.head(2))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_insider_trading():
    """Test insider trading data"""
    print("\n=== Insider Trading ===")
    try:
        finsider = Insider(option='top owner trade')
        insider_df = finsider.get_insider()
        print(f"Insider shape: {insider_df.shape}")
        print(f"Insider columns: {list(insider_df.columns)}")
        print("First 3 insider trades:")
        print(insider_df.head(3))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_earnings():
    """Test earnings data"""
    print("\n=== Earnings ===")
    try:
        fEarnings = Earnings()
        df_days = fEarnings.partition_days(mode='financial')
        print(f"Earnings days: {list(df_days.keys())}")
        if df_days:
            first_day = list(df_days.keys())[0]
            print(f"First day data shape: {df_days[first_day].shape}")
            print(f"First day columns: {list(df_days[first_day].columns)}")
            print("First 2 earnings:")
            print(df_days[first_day].head(2))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_forex():
    """Test forex data"""
    print("\n=== Forex ===")
    try:
        fforex = Forex()
        forex_df = fforex.performance()
        print(f"Forex shape: {forex_df.shape}")
        print(f"Forex columns: {list(forex_df.columns)}")
        print("Forex data:")
        print(forex_df.head(3))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_crypto():
    """Test crypto data"""
    print("\n=== Crypto ===")
    try:
        fcrypto = Crypto()
        crypto_df = fcrypto.performance()
        print(f"Crypto shape: {crypto_df.shape}")
        print(f"Crypto columns: {list(crypto_df.columns)}")
        print("Crypto data:")
        print(crypto_df.head(3))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing comprehensive finvizfinance functionality...")
    
    tests = [
        test_sector_performance,
        test_market_breadth,
        test_news,
        test_insider_trading,
        test_earnings,
        test_forex,
        test_crypto
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    print(f"\n=== SUMMARY ===")
    print(f"Passed: {passed}/{total}")
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")