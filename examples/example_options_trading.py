#!/usr/bin/env python3
"""
Oracle-X Options Trading Example Script

This script demonstrates how to use the Oracle Options Pipeline to:
1. Identify optimal options to purchase
2. Analyze specific tickers
3. Scan the market for opportunities
4. Monitor existing positions
5. Generate trade recommendations
"""

import json
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import the pipeline
from oracle_options_pipeline import (
    create_pipeline,
    OracleOptionsPipeline,
    PipelineConfig,
    RiskTolerance,
    OptionStrategy,
    OptionRecommendation
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def example_basic_analysis():
    """Example 1: Basic single ticker analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Single Ticker Analysis")
    print("="*60)
    
    # Create pipeline with default configuration
    pipeline = create_pipeline()
    
    # Analyze Apple options
    symbol = "AAPL"
    print(f"\n🔍 Analyzing {symbol} options...")
    
    recommendations = pipeline.analyze_ticker(symbol)
    
    if not recommendations:
        print(f"No opportunities found for {symbol}")
        return
    
    print(f"✅ Found {len(recommendations)} opportunities\n")
    
    # Display top 3 recommendations
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"--- Opportunity #{i} ---")
        print(f"Type: {rec.contract.option_type.value.upper()}")
        print(f"Strike: ${rec.contract.strike:.2f}")
        print(f"Expiry: {rec.contract.expiry.strftime('%Y-%m-%d')}")
        print(f"Opportunity Score: {rec.opportunity_score:.1f}/100")
        print(f"Entry Price: ${rec.entry_price:.2f}")
        print(f"Target: ${rec.target_price:.2f}")
        print(f"Expected Return: {rec.expected_return:.1%}")
        print(f"Probability of Profit: {rec.probability_of_profit:.1%}")
        print(f"Key Reasons: {', '.join(rec.key_reasons[:2])}")
        print()


def example_custom_configuration():
    """Example 2: Custom configuration for conservative investor"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Conservative Configuration")
    print("="*60)
    
    # Create conservative configuration
    config = {
        'risk_tolerance': 'conservative',
        'max_position_size': 0.02,  # 2% max position
        'min_opportunity_score': 80.0,  # Higher threshold
        'min_confidence': 0.7,  # Higher confidence required
        'min_days_to_expiry': 30,  # Avoid short-term options
        'max_days_to_expiry': 120,  # Longer time horizon
        'min_volume': 500,  # Higher liquidity requirement
        'min_open_interest': 1000
    }
    
    pipeline = create_pipeline(config)
    
    # Analyze multiple blue-chip stocks
    symbols = ["MSFT", "JNJ", "PG"]
    
    all_recommendations = []
    for symbol in symbols:
        print(f"\n🔍 Analyzing {symbol}...")
        recs = pipeline.analyze_ticker(symbol)
        all_recommendations.extend(recs)
        print(f"   Found {len(recs)} conservative opportunities")
    
    # Sort all recommendations by score
    all_recommendations.sort(key=lambda x: x.opportunity_score, reverse=True)
    
    if all_recommendations:
        best = all_recommendations[0]
        print(f"\n🏆 Best Conservative Opportunity:")
        print(f"   {best.symbol} {best.contract.option_type.value.upper()}")
        print(f"   Strike: ${best.contract.strike:.2f}")
        print(f"   Score: {best.opportunity_score:.1f}")
        print(f"   Position Size: {best.position_size:.1%} of portfolio")
        print(f"   Max Loss: ${best.max_loss:.2f}")


def example_market_scan():
    """Example 3: Market-wide opportunity scan"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Market-Wide Scan")
    print("="*60)
    
    # Create pipeline for aggressive trading
    config = {
        'risk_tolerance': 'aggressive',
        'max_position_size': 0.10,  # 10% positions
        'min_opportunity_score': 65.0,  # Lower threshold
        'min_days_to_expiry': 7,  # Include short-term
        'max_days_to_expiry': 45  # Focus on near-term
    }
    
    pipeline = create_pipeline(config)
    
    print("\n🌐 Scanning top liquid options for opportunities...")
    
    # Scan default universe (top 32 liquid symbols)
    result = pipeline.scan_market(max_symbols=15)
    
    print(f"\n📊 Scan Results:")
    print(f"   Symbols Analyzed: {result.symbols_analyzed}")
    print(f"   Opportunities Found: {result.opportunities_found}")
    print(f"   Execution Time: {result.execution_time:.2f} seconds")
    
    if result.recommendations:
        print(f"\n🏆 Top 5 Market Opportunities:")
        for i, rec in enumerate(result.recommendations[:5], 1):
            print(f"\n{i}. {rec.symbol} {rec.contract.option_type.value.upper()}")
            print(f"   Strike: ${rec.contract.strike:.2f} | Expiry: {rec.contract.expiry.strftime('%m/%d')}")
            print(f"   Score: {rec.opportunity_score:.1f} | Entry: ${rec.entry_price:.2f}")
            print(f"   Expected Return: {rec.expected_return:.1%}")


def example_generate_recommendations():
    """Example 4: Generate formatted trade recommendations"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Generate Trade Recommendations")
    print("="*60)
    
    pipeline = create_pipeline()
    
    # Generate recommendations for specific symbols
    symbols = ["NVDA", "AMD", "TSLA"]
    
    print(f"\n📋 Generating recommendations for: {', '.join(symbols)}")
    
    # Get recommendations in dictionary format
    recommendations = pipeline.generate_recommendations(symbols, output_format="dict")
    
    if recommendations:
        print(f"\n✅ Generated {len(recommendations)} trade recommendations\n")
        
        # Display first recommendation in detail
        rec = recommendations[0]
        print("📊 DETAILED TRADE RECOMMENDATION")
        print("-" * 40)
        print(f"Symbol: {rec['symbol']}")
        print(f"Strategy: {rec['strategy']}")
        print(f"Contract: {rec['contract']['type'].upper()} ${rec['contract']['strike']:.2f}")
        print(f"Expiry: {rec['contract']['expiry'][:10]}")
        print("\n💰 Trade Setup:")
        print(f"   Entry Price: ${rec['trade']['entry_price']:.2f}")
        print(f"   Target: ${rec['trade']['target_price']:.2f}")
        print(f"   Stop Loss: ${rec['trade']['stop_loss']:.2f}")
        print(f"   Position Size: {rec['trade']['position_size']:.1%}")
        print(f"   Max Contracts: {rec['trade']['max_contracts']}")
        print("\n📈 Risk/Reward:")
        print(f"   Expected Return: {rec['risk']['expected_return']:.1%}")
        print(f"   Win Probability: {rec['risk']['probability_of_profit']:.1%}")
        print(f"   Risk/Reward: {rec['risk']['risk_reward_ratio']:.1f}:1")
        print("\n✅ Key Reasons:")
        for reason in rec['analysis']['key_reasons']:
            print(f"   • {reason}")
        
        # Save to file
        with open('recommendations.json', 'w') as f:
            json.dump(recommendations[:3], f, indent=2)
        print("\n💾 Top 3 recommendations saved to 'recommendations.json'")


def example_position_monitoring():
    """Example 5: Monitor existing positions"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Position Monitoring")
    print("="*60)
    
    # Create sample positions (in practice, load from your broker/database)
    positions = [
        {
            'symbol': 'AAPL',
            'strike': 180.0,
            'expiry': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'type': 'call',
            'entry_price': 5.50,
            'quantity': 10
        },
        {
            'symbol': 'SPY',
            'strike': 440.0,
            'expiry': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
            'type': 'put',
            'entry_price': 3.25,
            'quantity': 5
        }
    ]
    
    pipeline = create_pipeline()
    
    print("\n📍 Monitoring existing positions...")
    updates = pipeline.monitor_positions(positions)
    
    if updates:
        print(f"\n📊 Position Updates:\n")
        for update in updates:
            pos = update['position']
            print(f"{pos['symbol']} {pos['type'].upper()} ${pos['strike']:.0f}")
            print(f"   Entry: ${pos['entry_price']:.2f}")
            print(f"   Current: ${update['current_price']:.2f}")
            print(f"   P&L: {update['pnl_percent']:.1f}%")
            print(f"   Action: {update['action'].upper()}")
            print(f"   Valuation Score: {update['valuation_score']:.1f}")
            print()


def example_advanced_filtering():
    """Example 6: Advanced filtering and analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Advanced Filtering")
    print("="*60)
    
    pipeline = create_pipeline()
    
    # Get top opportunities across multiple symbols
    print("\n🔍 Finding high-confidence opportunities...")
    
    symbols = ["QQQ", "IWM", "GLD", "TLT"]
    high_confidence_recs = []
    
    for symbol in symbols:
        recs = pipeline.analyze_ticker(symbol)
        # Filter for high confidence only
        filtered = [r for r in recs if r.ml_confidence > 0.75 and r.probability_of_profit > 0.6]
        high_confidence_recs.extend(filtered)
    
    if high_confidence_recs:
        # Sort by expected return
        high_confidence_recs.sort(key=lambda x: x.expected_return, reverse=True)
        
        print(f"\n✅ Found {len(high_confidence_recs)} high-confidence opportunities")
        print("\nTop 3 by Expected Return:")
        
        for i, rec in enumerate(high_confidence_recs[:3], 1):
            print(f"\n{i}. {rec.symbol} {rec.contract.option_type.value.upper()}")
            print(f"   Expected Return: {rec.expected_return:.1%}")
            print(f"   ML Confidence: {rec.ml_confidence:.1%}")
            print(f"   Win Probability: {rec.probability_of_profit:.1%}")
            print(f"   Risk/Reward: {rec.risk_reward_ratio:.2f}:1")


def example_performance_stats():
    """Example 7: Pipeline performance statistics"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Performance Statistics")
    print("="*60)
    
    pipeline = create_pipeline()
    
    # Run some analyses to populate cache
    print("\n📊 Running analyses to gather statistics...")
    
    test_symbols = ["AAPL", "GOOGL", "AMZN", "META", "NFLX"]
    for symbol in test_symbols:
        pipeline.analyze_ticker(symbol)
    
    # Get performance stats
    stats = pipeline.get_performance_stats()
    
    print("\n📈 Pipeline Performance Statistics:")
    print(f"   Cache Size: {stats['cache_size']} symbols")
    print(f"   Total Recommendations: {stats['total_recommendations']}")
    print(f"   Avg Opportunity Score: {stats['avg_opportunity_score']:.1f}")
    print(f"   Avg ML Confidence: {stats['avg_ml_confidence']:.1%}")
    
    if stats['top_symbols']:
        print("\n🏆 Top Symbols by Opportunities:")
        for symbol, count in stats['top_symbols']:
            print(f"   {symbol}: {count} opportunities")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("ORACLE-X OPTIONS TRADING PIPELINE EXAMPLES")
    print("="*60)
    print("\nThis script demonstrates various ways to use the pipeline")
    print("to identify optimal options trading opportunities.")
    
    # Run examples
    try:
        example_basic_analysis()
        example_custom_configuration()
        example_market_scan()
        example_generate_recommendations()
        example_position_monitoring()
        example_advanced_filtering()
        example_performance_stats()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nYou can now use these patterns in your own trading scripts!")
        print("\nKey Takeaways:")
        print("1. Use analyze_ticker() for single symbol analysis")
        print("2. Use scan_market() for finding opportunities across symbols")
        print("3. Customize configuration for your risk tolerance")
        print("4. Monitor existing positions for exit signals")
        print("5. Filter recommendations by confidence and probability")
        print("\n💡 Remember: Always validate recommendations with your own analysis!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()