"""
Test script for the comprehensive backtesting framework
Demonstrates walk-forward validation with multiple strategies
"""


import logging
import sys
from pathlib import Path
# Ensure project root is in sys.path for imports
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import our backtesting framework
from backtest_tracker.comprehensive_backtest import (
    BacktestEngine, BacktestConfig, BacktestMode,
    run_strategy_backtest, create_backtest_engine
)
from backtest_tracker.example_strategies import create_strategy, get_available_strategies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_backtest():
    """Test basic backtesting functionality"""
    logger.info("Testing basic backtesting functionality...")
    
    # Create configuration
    config = BacktestConfig(
        initial_capital=100000.0,
        commission_per_trade=1.0,
        slippage_pct=0.001,
        max_position_size=0.1,
        training_period_days=120,  # Shorter for testing
        testing_period_days=30,
        min_trades_required=5
    )
    
    # Test symbols (major stocks with good liquidity)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    # Test period (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    # Test momentum strategy
    strategy = create_strategy('momentum')
    logger.info(f"Testing strategy: {strategy.name}")
    
    try:
        # Run backtest with expanding window (simpler than walk-forward for testing)
        engine = create_backtest_engine(config)
        results = engine.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            mode=BacktestMode.EXPANDING_WINDOW
        )
        
        # Display results
        performance = results.get('performance')
        if performance:
            logger.info("=== BACKTEST RESULTS ===")
            logger.info(f"Total Return: {performance.total_return:.2%}")
            logger.info(f"Annualized Return: {performance.annualized_return:.2%}")
            logger.info(f"Volatility: {performance.volatility:.2%}")
            logger.info(f"Sharpe Ratio: {performance.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {performance.max_drawdown:.2%}")
            logger.info(f"Win Rate: {performance.win_rate:.2%}")
            logger.info(f"Total Trades: {performance.total_trades}")
            logger.info(f"Options Trades: {performance.options_trades}")
            
            if performance.total_trades > 0:
                logger.info(f"Average Win: ${performance.avg_win:.2f}")
                logger.info(f"Average Loss: ${performance.avg_loss:.2f}")
                logger.info(f"Profit Factor: {performance.profit_factor:.2f}")
        else:
            logger.warning("No performance metrics generated")
            
        # Show positions summary
        positions = results.get('positions', [])
        closed_positions = [p for p in positions if not p.is_open]
        logger.info(f"Positions opened: {len(positions)}")
        logger.info(f"Positions closed: {len(closed_positions)}")
        
        if closed_positions:
            total_pnl = sum(p.pnl for p in closed_positions)
            winning_positions = [p for p in closed_positions if p.pnl > 0]
            logger.info(f"Total P&L: ${total_pnl:.2f}")
            logger.info(f"Winning positions: {len(winning_positions)}")
            
            # Show a few example trades
            logger.info("=== SAMPLE TRADES ===")
            for i, pos in enumerate(closed_positions[:5]):  # Show first 5 trades
                logger.info(f"Trade {i+1}: {pos.symbol} {pos.position_type.value} "
                          f"Entry: ${pos.entry_price:.2f} Exit: ${pos.exit_price:.2f} "
                          f"P&L: ${pos.pnl:.2f} ({pos.return_pct:.1%}) "
                          f"Reason: {pos.exit_reason}")
        
        return True
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_comparison():
    """Test multiple strategies and compare performance"""
    logger.info("Testing strategy comparison...")
    
    config = BacktestConfig(
        initial_capital=100000.0,
        training_period_days=60,  # Shorter for testing
        testing_period_days=20
    )
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year
    
    results_comparison = {}
    
    for strategy_name in get_available_strategies():
        logger.info(f"Testing {strategy_name} strategy...")
        
        try:
            strategy = create_strategy(strategy_name)
            
            results = run_strategy_backtest(
                strategy=strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                config=config,
                mode=BacktestMode.EXPANDING_WINDOW
            )
            
            performance = results.get('performance')
            if performance:
                results_comparison[strategy_name] = {
                    'total_return': performance.total_return,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'max_drawdown': performance.max_drawdown,
                    'win_rate': performance.win_rate,
                    'total_trades': performance.total_trades
                }
            
        except Exception as e:
            logger.error(f"Failed to test {strategy_name}: {e}")
    
    # Display comparison
    if results_comparison:
        logger.info("\n=== STRATEGY COMPARISON ===")
        logger.info(f"{'Strategy':<15} {'Return':<8} {'Sharpe':<8} {'Drawdown':<10} {'Win Rate':<9} {'Trades':<7}")
        logger.info("-" * 65)
        
        for name, metrics in results_comparison.items():
            logger.info(f"{name:<15} {metrics['total_return']:<7.1%} "
                       f"{metrics['sharpe_ratio']:<7.2f} {metrics['max_drawdown']:<9.1%} "
                       f"{metrics['win_rate']:<8.1%} {metrics['total_trades']:<7}")
    
    return len(results_comparison) > 0

def test_data_manager():
    """Test the data manager functionality"""
    logger.info("Testing data manager...")
    
    from backtest_tracker.comprehensive_backtest import DataManager
    
    data_manager = DataManager()
    
    # Test data fetching
    symbols = ['AAPL', 'GOOGL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    try:
        data = data_manager.get_historical_data(symbols, start_date, end_date)
        
        logger.info(f"Fetched data for {len(data)} symbols")
        
        for symbol, df in data.items():
            logger.info(f"{symbol}: {len(df)} days, columns: {list(df.columns)}")
            
            # Check technical indicators
            required_indicators = ['SMA_20', 'RSI', 'MACD', 'BB_Upper', 'Volume_Ratio']
            missing = [ind for ind in required_indicators if ind not in df.columns]
            if missing:
                logger.warning(f"{symbol} missing indicators: {missing}")
            else:
                logger.info(f"{symbol} has all required technical indicators")
        
        # Test point-in-time data
        pit_date = end_date - timedelta(days=30)
        pit_data = data_manager.get_point_in_time_data('AAPL', pit_date, lookback_days=50)
        
        if pit_data is not None:
            logger.info(f"Point-in-time data: {len(pit_data)} days ending on {pit_date.date()}")
            logger.info(f"Latest date in data: {pit_data.index[-1].date()}")
            
            # Verify no lookahead bias
            if pit_data.index[-1].date() <= pit_date.date():
                logger.info("✓ No lookahead bias detected")
            else:
                logger.error("✗ Lookahead bias detected!")
        
        return True
        
    except Exception as e:
        logger.error(f"Data manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all backtesting tests"""
    logger.info("Starting comprehensive backtesting framework tests...")
    
    tests = [
        ("Data Manager", test_data_manager),
        ("Basic Backtest", test_basic_backtest),
        ("Strategy Comparison", test_strategy_comparison)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"{test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    overall_success = all(results.values())
    logger.info(f"\nOverall: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
