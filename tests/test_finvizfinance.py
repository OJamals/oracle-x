import pandas as pd
from finvizfinance.group.performance import Performance
from finvizfinance.group.overview import Overview


def test_finvizfinance_sector_performance():
    """Test finvizfinance sector performance functionality"""
    print("Testing finvizfinance sector performance...")

    # Test sector performance
    try:
        fgperformance = Performance()
        df = fgperformance.screener_view(group="Sector")
        print("Sector Performance Data:")
        print(df.head())
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        return True
    except Exception as e:
        print(f"Error getting sector performance: {e}")
        return False


def test_finvizfinance_market_breadth():
    """Test if we can get market breadth data"""
    print("\nTesting market breadth data...")

    # Try to get market overview data
    try:
        fgoverview = Overview()
        df = fgoverview.screener_view(group="Sector")
        print("Market Overview Data:")
        print(df.head())
        print(f"Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"Error getting market overview: {e}")
        return False


if __name__ == "__main__":
    print("Testing finvizfinance library...")
    success1 = test_finvizfinance_sector_performance()
    success2 = test_finvizfinance_market_breadth()

    if success1 and success2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
