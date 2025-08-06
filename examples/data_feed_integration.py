#!/usr/bin/env python3
"""
Integration example showing how to use the Consolidated Data Feed
in real Oracle-X trading scenarios
"""

import sys
sys.path.append('/Users/omar/Documents/Projects/oracle-x')

from consolidated_data_feed import ConsolidatedDataFeed, get_quote, get_historical, get_company_info
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def demonstrate_basic_usage():
    """Basic usage examples"""
    print("🔹 Basic Data Feed Usage")
    print("-" * 40)
    
    # Get real-time quotes
    quote = get_quote("AAPL")
    if quote:
        print(f"AAPL: ${quote.price:.2f} ({quote.change:+.2f}, {quote.change_percent:+.2f}%)")
        volume_str = f"{quote.volume:,}" if quote.volume else "N/A"
        market_cap_str = f"${quote.market_cap/1e9:.1f}B" if quote.market_cap else "N/A"
        print(f"Volume: {volume_str} | Market Cap: {market_cap_str}")
        print(f"Data source: {quote.source}")
    
    print()

def demonstrate_portfolio_analysis():
    """Portfolio analysis using consolidated data"""
    print("🔹 Portfolio Analysis")
    print("-" * 40)
    
    # Define a sample portfolio
    portfolio = {
        "AAPL": 100,  # 100 shares
        "GOOGL": 50,  # 50 shares
        "MSFT": 75,   # 75 shares
        "TSLA": 25,   # 25 shares
        "NVDA": 40    # 40 shares
    }
    
    feed = ConsolidatedDataFeed()
    
    # Get current quotes for all positions
    quotes = feed.get_multiple_quotes(list(portfolio.keys()))
    
    total_value = 0
    total_change = 0
    
    print("Portfolio Holdings:")
    print(f"{'Symbol':<6} {'Shares':<8} {'Price':<10} {'Value':<12} {'Change':<10} {'Source':<15}")
    print("-" * 75)
    
    for symbol, shares in portfolio.items():
        if symbol in quotes:
            quote = quotes[symbol]
            value = float(quote.price) * shares
            change = float(quote.change) * shares
            
            total_value += value
            total_change += change
            
            print(f"{symbol:<6} {shares:<8} ${quote.price:<9.2f} ${value:<11.2f} "
                  f"{change:+<9.2f} {quote.source:<15}")
    
    print("-" * 75)
    print(f"Portfolio Value: ${total_value:,.2f}")
    print(f"Total Change: ${total_change:+.2f} ({(total_change/total_value)*100:+.2f}%)")
    print()

def demonstrate_technical_analysis():
    """Technical analysis using historical data"""
    print("🔹 Technical Analysis")
    print("-" * 40)
    
    symbol = "AAPL"
    
    # Get 6 months of historical data
    hist = get_historical(symbol, period="6mo")
    
    if hist is not None and not hist.empty:
        # Calculate technical indicators
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        hist['BB_Upper'], hist['BB_Lower'] = calculate_bollinger_bands(hist['Close'])
        
        # Current values
        current_price = hist['Close'].iloc[-1]
        sma_20 = hist['SMA_20'].iloc[-1]
        sma_50 = hist['SMA_50'].iloc[-1]
        rsi = hist['RSI'].iloc[-1]
        bb_upper = hist['BB_Upper'].iloc[-1]
        bb_lower = hist['BB_Lower'].iloc[-1]
        
        print(f"Technical Analysis for {symbol}:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"20-day SMA: ${sma_20:.2f}")
        print(f"50-day SMA: ${sma_50:.2f}")
        print(f"RSI: {rsi:.1f}")
        print(f"Bollinger Bands: ${bb_lower:.2f} - ${bb_upper:.2f}")
        
        # Generate signals
        signals = []
        if current_price > sma_20 > sma_50:
            signals.append("📈 Bullish trend (price > SMA20 > SMA50)")
        if rsi < 30:
            signals.append("🔵 Oversold (RSI < 30)")
        elif rsi > 70:
            signals.append("🔴 Overbought (RSI > 70)")
        if current_price < bb_lower:
            signals.append("🎯 Below lower Bollinger Band")
        
        if signals:
            print("\nSignals:")
            for signal in signals:
                print(f"  {signal}")
        else:
            print("\nNo strong signals detected")
    
    print()

def demonstrate_risk_monitoring():
    """Risk monitoring using real-time data"""
    print("🔹 Risk Monitoring")
    print("-" * 40)
    
    # Monitor high-volatility stocks
    watchlist = ["TSLA", "NVDA", "AMD", "MSTR", "GME"]
    
    feed = ConsolidatedDataFeed()
    alerts = []
    
    print("Volatility Monitor:")
    print(f"{'Symbol':<6} {'Change %':<10} {'Status':<15} {'Alert'}")
    print("-" * 50)
    
    for symbol in watchlist:
        quote = feed.get_quote(symbol)
        if quote:
            change_pct = float(quote.change_percent)
            
            if abs(change_pct) > 10:
                status = "🚨 HIGH VOLATILITY"
                alerts.append(f"{symbol}: {change_pct:+.1f}% move")
            elif abs(change_pct) > 5:
                status = "⚠️ MODERATE"
            else:
                status = "✅ NORMAL"
            
            alert_icon = "🔔" if abs(change_pct) > 5 else ""
            
            print(f"{symbol:<6} {change_pct:+<9.1f}% {status:<15} {alert_icon}")
    
    if alerts:
        print(f"\n🚨 Risk Alerts:")
        for alert in alerts:
            print(f"  • {alert}")
    
    print()

def demonstrate_news_sentiment():
    """News and sentiment analysis"""
    print("🔹 News & Sentiment Analysis")
    print("-" * 40)
    
    symbol = "AAPL"
    feed = ConsolidatedDataFeed()
    
    news = feed.get_news(symbol, limit=5)
    
    if news:
        print(f"Recent news for {symbol}:")
        for i, item in enumerate(news, 1):
            # Truncate title for display
            title = item.title[:60] + "..." if len(item.title) > 60 else item.title
            published = item.published.strftime('%Y-%m-%d %H:%M')
            print(f"{i}. {title}")
            print(f"   Published: {published} | Source: {item.source}")
            print()
    else:
        print(f"No recent news found for {symbol}")
    
    print()

def demonstrate_market_screening():
    """Market screening using search capabilities"""
    print("🔹 Market Screening")
    print("-" * 40)
    
    feed = ConsolidatedDataFeed()
    
    # Screen for technology stocks (using finance database)
    try:
        results = feed.search_securities(country="United States")
        
        if results and 'equities' in results:
            tech_stocks = results['equities']
            print(f"Found {len(tech_stocks)} securities in database")
            
            # Show a few examples
            if tech_stocks:
                sample_symbols = list(tech_stocks.keys())[:5]
                print(f"Sample symbols: {', '.join(sample_symbols)}")
        else:
            print("No screening results available")
    except Exception as e:
        print(f"Screening not available: {e}")
    
    print()

def calculate_rsi(prices, window=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def demonstrate_performance_metrics():
    """Show performance and caching statistics"""
    print("🔹 Performance Metrics")
    print("-" * 40)
    
    feed = ConsolidatedDataFeed()
    
    # Make some calls to populate cache
    start_time = datetime.now()
    
    # First call (cache miss)
    quote1 = feed.get_quote("AAPL")
    first_call_time = (datetime.now() - start_time).total_seconds()
    
    # Second call (cache hit)
    start_time = datetime.now()
    quote2 = feed.get_quote("AAPL")
    second_call_time = (datetime.now() - start_time).total_seconds()
    
    print(f"First call (cache miss): {first_call_time:.3f}s")
    print(f"Second call (cache hit): {second_call_time:.3f}s")
    print(f"Cache speedup: {first_call_time/second_call_time:.1f}x faster")
    
    # Cache statistics
    stats = feed.get_cache_stats()
    print(f"Cache items: {stats['total_cached_items']}")
    
    print()

def main():
    """Run all demonstration examples"""
    print("=" * 60)
    print("CONSOLIDATED DATA FEED - INTEGRATION EXAMPLES")
    print("=" * 60)
    print()
    
    try:
        demonstrate_basic_usage()
        demonstrate_portfolio_analysis()
        demonstrate_technical_analysis()
        demonstrate_risk_monitoring()
        demonstrate_news_sentiment()
        demonstrate_market_screening()
        demonstrate_performance_metrics()
        
        print("🎉 All examples completed successfully!")
        print("\n💡 The Consolidated Data Feed provides:")
        print("   • Unified access to multiple data sources")
        print("   • Automatic fallback and error handling")
        print("   • Intelligent caching for performance")
        print("   • Rate limiting to prevent quota exhaustion")
        print("   • Consistent data formats across sources")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
