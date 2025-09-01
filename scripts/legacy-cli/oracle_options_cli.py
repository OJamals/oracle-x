#!/usr/bin/env python3
"""
Oracle-X Options Pipeline CLI Interface

Command-line interface for easy interaction with the Oracle Options Pipeline.
Supports analyzing individual tickers, scanning markets, and monitoring positions.
"""

import argparse
import logging
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
from tabulate import tabulate

# Add CLI directory and project root to sys.path for imports
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir))
sys.path.insert(0, this_dir)
sys.path.insert(0, project_root)

from oracle_options_pipeline import (
    create_pipeline,
    OracleOptionsPipeline,
    PipelineConfig,
    RiskTolerance,
    OptionStrategy
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def print_recommendation(rec: Dict[str, Any], verbose: bool = False):
    """Pretty print a recommendation"""
    print("\n" + "="*60)
    print(f"📊 {rec['symbol']} - {rec['contract']['type'].upper()} Option")
    print(f"   Strike: ${rec['contract']['strike']:.2f} | Expiry: {rec['contract']['expiry'][:10]}")
    print("-"*60)
    
    # Scores
    scores = rec['scores']
    print(f"🎯 Opportunity Score: {scores['opportunity']:.1f}/100")
    print(f"   ML Confidence: {scores['ml_confidence']:.1%}")
    print(f"   Valuation Score: {scores['valuation']:.1%}")
    
    # Trade parameters
    trade = rec['trade']
    print(f"\n💰 Trade Parameters:")
    print(f"   Entry Price: ${trade['entry_price']:.2f}")
    print(f"   Target Price: ${trade['target_price']:.2f} ({(trade['target_price']/trade['entry_price']-1)*100:.1f}%)")
    print(f"   Stop Loss: ${trade['stop_loss']:.2f} ({(trade['stop_loss']/trade['entry_price']-1)*100:.1f}%)")
    print(f"   Position Size: {trade['position_size']:.1%} of portfolio")
    print(f"   Max Contracts: {trade['max_contracts']}")
    
    # Risk metrics
    risk = rec['risk']
    print(f"\n📈 Risk/Reward:")
    print(f"   Expected Return: {risk['expected_return']:.1%}")
    print(f"   Probability of Profit: {risk['probability_of_profit']:.1%}")
    print(f"   Risk/Reward Ratio: {risk['risk_reward_ratio']:.2f}:1")
    print(f"   Max Loss: ${risk['max_loss']:.2f}")
    print(f"   Breakeven: ${risk['breakeven_price']:.2f}")
    
    # Analysis
    if verbose:
        analysis = rec['analysis']
        print(f"\n✅ Key Reasons:")
        for reason in analysis['key_reasons']:
            print(f"   • {reason}")
        
        if analysis['risk_factors']:
            print(f"\n⚠️  Risk Factors:")
            for risk in analysis['risk_factors']:
                print(f"   • {risk}")
        
        if analysis['entry_signals']:
            print(f"\n🚦 Entry Signals:")
            for signal in analysis['entry_signals']:
                print(f"   • {signal}")
    
    print("="*60)


def cmd_analyze(args):
    """Analyze a single ticker"""
    print(f"\n🔍 Analyzing {args.symbol}...")
    # Enable detailed debug logs from pipeline if verbose
    if args.verbose:
        logging.getLogger('oracle_options_pipeline').setLevel(logging.DEBUG)
    
    # Create pipeline with config
    config = {
        'risk_tolerance': args.risk,
        'min_opportunity_score': args.min_score,
        'min_days_to_expiry': args.min_days,
        'max_days_to_expiry': args.max_days
    }
    
    pipeline = create_pipeline(config)
    
    # Analyze ticker
    recommendations = pipeline.analyze_ticker(args.symbol, args.expiry)
    
    if not recommendations:
        print(f"❌ No opportunities found for {args.symbol}")
        return
    
    print(f"\n✅ Found {len(recommendations)} opportunities for {args.symbol}")
    
    # Show top N recommendations
    for i, rec in enumerate(recommendations[:args.limit], 1):
        print(f"\n--- Recommendation #{i} ---")
        print_recommendation(rec.to_dict(), verbose=args.verbose)
    
    # Save to file if requested
    if args.output:
        output_data = {
            'symbol': args.symbol,
            'timestamp': datetime.now().isoformat(),
            'recommendations': [r.to_dict() for r in recommendations[:args.limit]]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


def cmd_scan(args):
    """Scan market for opportunities"""
    print(f"\n🌐 Scanning market for opportunities...")
    # Enable detailed debug logs from pipeline if verbose
    if args.verbose:
        logging.getLogger('oracle_options_pipeline').setLevel(logging.DEBUG)
    
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = args.symbols.split(',')
        print(f"   Scanning symbols: {', '.join(symbols)}")
    else:
        print(f"   Scanning top {args.max_symbols} liquid options")
    
    # Create pipeline
    config = {
        'risk_tolerance': args.risk,
        'min_opportunity_score': args.min_score,
        'min_days_to_expiry': args.min_days,
        'max_days_to_expiry': args.max_days
    }
    
    pipeline = create_pipeline(config)
    
    # Scan market
    result = pipeline.scan_market(symbols, max_symbols=args.max_symbols)
    
    print(f"\n📊 Scan Results:")
    print(f"   Symbols Analyzed: {result.symbols_analyzed}")
    print(f"   Opportunities Found: {result.opportunities_found}")
    print(f"   Execution Time: {result.execution_time:.2f}s")
    
    if result.errors:
        print(f"\n⚠️  Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            print(f"   • {error}")
    
    if not result.recommendations:
        print("\n❌ No opportunities found")
        return
    
    # Display top opportunities
    print(f"\n🏆 Top {min(args.top, len(result.recommendations))} Opportunities:")
    
    # Create summary table
    table_data = []
    for rec in result.recommendations[:args.top]:
        table_data.append([
            rec.symbol,
            rec.contract.option_type.value.upper(),
            f"${rec.contract.strike:.0f}",
            rec.contract.expiry.strftime('%m/%d'),
            f"{rec.opportunity_score:.1f}",
            f"${rec.entry_price:.2f}",
            f"{rec.expected_return:.1%}",
            f"{rec.probability_of_profit:.1%}"
        ])
    
    headers = ['Symbol', 'Type', 'Strike', 'Expiry', 'Score', 'Entry', 'Return', 'PoP']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Show detailed view if verbose
    if args.verbose:
        for i, rec in enumerate(result.recommendations[:3], 1):
            print(f"\n--- Top Opportunity #{i} ---")
            print_recommendation(rec.to_dict(), verbose=True)
    
    # Save results
    if args.output:
        output_data = {
            'scan_timestamp': result.timestamp.isoformat(),
            'symbols_analyzed': result.symbols_analyzed,
            'opportunities_found': result.opportunities_found,
            'execution_time': result.execution_time,
            'recommendations': [r.to_dict() for r in result.recommendations[:args.top]]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


def cmd_monitor(args):
    """Monitor existing positions"""
    print(f"\n📍 Monitoring positions from {args.positions_file}...")
    
    # Load positions
    try:
        with open(args.positions_file, 'r') as f:
            positions = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {args.positions_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in {args.positions_file}")
        return
    
    if not positions:
        print("❌ No positions to monitor")
        return
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Monitor positions
    updates = pipeline.monitor_positions(positions)
    
    if not updates:
        print("❌ Failed to monitor positions")
        return
    
    print(f"\n📊 Position Updates:")
    
    # Create monitoring table
    table_data = []
    for update in updates:
        pos = update['position']
        pnl_color = '🟢' if update['pnl_percent'] > 0 else '🔴'
        
        table_data.append([
            pos['symbol'],
            pos['type'].upper(),
            f"${pos['strike']:.0f}",
            pos['expiry'][:10],
            f"${pos['entry_price']:.2f}",
            f"${update['current_price']:.2f}",
            f"{pnl_color} {update['pnl_percent']:.1f}%",
            update['action'].upper()
        ])
    
    headers = ['Symbol', 'Type', 'Strike', 'Expiry', 'Entry', 'Current', 'P&L', 'Action']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Show recommendations
    for update in updates:
        if update['action'] != 'hold':
            pos = update['position']
            print(f"\n⚠️  {update['action'].upper()}: {pos['symbol']} {pos['type']} ${pos['strike']}")
            print(f"   Current P&L: {update['pnl_percent']:.1f}%")
            print(f"   Valuation Score: {update['valuation_score']:.1f}")
    
    # Save updates
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(updates, f, indent=2)
        print(f"\n💾 Updates saved to {args.output}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Oracle-X Options Prediction Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single ticker
  %(prog)s analyze AAPL --limit 5 --verbose
  
  # Scan market for opportunities
  %(prog)s scan --top 10 --min-score 75
  
  # Scan specific symbols
  %(prog)s scan --symbols AAPL,MSFT,GOOGL --top 5
  
  # Monitor existing positions
  %(prog)s monitor positions.json --output updates.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single ticker')
    analyze_parser.add_argument('symbol', type=str, help='Stock symbol to analyze')
    analyze_parser.add_argument('--expiry', type=str, help='Specific expiry date (YYYY-MM-DD)')
    analyze_parser.add_argument('--limit', type=int, default=5, help='Number of recommendations to show')
    analyze_parser.add_argument('--min-score', type=float, default=70, help='Minimum opportunity score')
    analyze_parser.add_argument('--risk', type=str, default='moderate', 
                               choices=['conservative', 'moderate', 'aggressive'],
                               help='Risk tolerance level')
    analyze_parser.add_argument('--min-days', type=int, default=7, help='Minimum days to expiry')
    analyze_parser.add_argument('--max-days', type=int, default=90, help='Maximum days to expiry')
    analyze_parser.add_argument('--verbose', action='store_true', help='Show detailed analysis')
    analyze_parser.add_argument('--output', type=str, help='Save results to JSON file')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan market for opportunities')
    scan_parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    scan_parser.add_argument('--max-symbols', type=int, default=20, help='Maximum symbols to scan')
    scan_parser.add_argument('--top', type=int, default=10, help='Number of top opportunities to show')
    scan_parser.add_argument('--min-score', type=float, default=70, help='Minimum opportunity score')
    scan_parser.add_argument('--min-days', type=int, default=3, help='Minimum days to expiry')
    scan_parser.add_argument('--max-days', type=int, default=120, help='Maximum days to expiry')
    scan_parser.add_argument('--risk', type=str, default='moderate',
                            choices=['conservative', 'moderate', 'aggressive'],
                            help='Risk tolerance level')
    scan_parser.add_argument('--verbose', action='store_true', help='Show detailed analysis')
    scan_parser.add_argument('--output', type=str, help='Save results to JSON file')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor existing positions')
    monitor_parser.add_argument('positions_file', type=str, help='JSON file with positions')
    monitor_parser.add_argument('--output', type=str, help='Save updates to JSON file')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'scan':
            cmd_scan(args)
        elif args.command == 'monitor':
            cmd_monitor(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()