"""
Financial Calculation Integration Test
Test the new financial calculator with real Oracle-X data.
"""

import pytest
import time
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
from data_feeds.financial_calculator import FinancialCalculator

@pytest.mark.integration
@pytest.mark.network
@pytest.mark.api
def test_financial_integration():
    """Test financial calculator integration with orchestrator"""
    
    print("=== Financial Calculator Integration Test ===")
    
    # Initialize orchestrator
    orchestrator = DataFeedOrchestrator()
    
    # Test with a real quote
    symbol = "AAPL"
    
    print(f"\\n🔍 Testing financial calculations for {symbol}...")
    
    # Get quote data
    start_time = time.time()
    quote = orchestrator.get_quote(symbol)
    quote_time = time.time() - start_time
    
    if not quote:
        print("❌ Failed to get quote")
        return
    
    print(f"✅ Quote retrieved in {quote_time:.2f}s")
    print(f"   Price: ${quote.price}")
    print(f"   Volume: {quote.volume:,}" if quote.volume else "   Volume: N/A")
    print(f"   Market Cap: ${quote.market_cap:,}" if quote.market_cap else "   Market Cap: N/A")
    
    # Get market data for enhanced calculations
    start_time = time.time()
    market_data = orchestrator.get_market_data(symbol, period="1mo")
    market_time = time.time() - start_time
    
    if market_data is not None and not market_data.data.empty:
        print(f"✅ Market data retrieved in {market_time:.2f}s")
        print(f"   Rows: {len(market_data.data)}, Columns: {list(market_data.data.columns)}")
    else:
        print("⚠️ No market data available")
        market_data = None
    
    # Calculate comprehensive metrics
    start_time = time.time()
    metrics = FinancialCalculator.calculate_comprehensive_metrics(quote, market_data)
    calc_time = time.time() - start_time
    
    print(f"\\n📊 Financial Metrics (calculated in {calc_time:.3f}s):")
    print(f"   Symbol: {metrics.symbol}")
    print(f"   Current Price: ${metrics.price}")
    
    if metrics.price_change is not None:
        print(f"   Price Change: ${metrics.price_change} ({metrics.price_change_percent}%)")
    
    if metrics.volume and metrics.avg_volume:
        print(f"   Volume: {metrics.volume:,} (Avg: {metrics.avg_volume:,.0f})")
        print(f"   Volume Ratio: {metrics.volume_ratio:.2f}x")
    
    if metrics.volatility_1d:
        print(f"   Daily Volatility: {metrics.volatility_1d:.4f}")
    
    if metrics.volatility_30d:
        print(f"   30-Day Volatility: {metrics.volatility_30d:.2f}%")
    
    if metrics.sma_20:
        print(f"   20-Day SMA: ${metrics.sma_20:.2f}")
    
    if metrics.sma_50:
        print(f"   50-Day SMA: ${metrics.sma_50:.2f}")
    
    if metrics.rsi_14:
        print(f"   RSI (14): {metrics.rsi_14:.1f}")
    
    print(f"   Data Quality Score: {metrics.data_quality_score:.1f}/100")
    
    # Test portfolio-level calculations
    print("\n📈 Testing Portfolio Calculations...")
    quotes = []
    symbols = ["AAPL", "TSLA", "MSFT"]
    
    for sym in symbols:
        q = orchestrator.get_quote(sym)
        if q:
            quotes.append(q)
            print(f"   ✅ {sym}: ${q.price}")
        else:
            print(f"   ❌ {sym}: Failed to retrieve")
    
    if quotes:
        portfolio_metrics = FinancialCalculator.calculate_portfolio_metrics(quotes)
        print("\n📊 Portfolio Summary:")
        print(f"   Total Symbols: {portfolio_metrics.get('total_symbols', 0)}")
        print(f"   Combined Market Cap: ${portfolio_metrics.get('total_market_cap', 0):,}")
        print(f"   Combined Volume: {portfolio_metrics.get('total_volume', 0):,}")
        if portfolio_metrics.get('avg_pe_ratio'):
            print(f"   Average P/E Ratio: {portfolio_metrics['avg_pe_ratio']:.2f}")
        if portfolio_metrics.get('avg_price_change'):
            print(f"   Average Price Change: {portfolio_metrics['avg_price_change']:.2f}%")
    
    print("\n✅ Financial calculator integration test completed!")

if __name__ == "__main__":
    test_financial_integration()
