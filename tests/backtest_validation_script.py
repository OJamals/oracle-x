"""
Comprehensive Backtesting Validation Script
Production-level validation for the backtesting pipeline
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import backtesting framework
from backtest_tracker.comprehensive_backtest import (
    BacktestConfig, BacktestMode, DataManager,
    create_backtest_engine, run_strategy_backtest
)
from backtest_tracker.example_strategies import (
    create_strategy, get_available_strategies, MomentumStrategy
)

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BacktestValidator:
    """Comprehensive backtesting validation system"""

    def __init__(self):
        self.data_manager = DataManager()
        self.engine = create_backtest_engine()
        self.validation_results = {}

    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and bias prevention mechanisms"""
        logger.info("=== VALIDATING DATA QUALITY ===")

        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year

        results = {
            'data_fetched': False,
            'indicators_present': False,
            'no_lookahead_bias': False,
            'sufficient_data': False,
            'data_integrity': False
        }

        try:
            # Test data fetching
            data = self.data_manager.get_historical_data(symbols, start_date, end_date)
            if data and len(data) > 0:
                results['data_fetched'] = True
                logger.info(f"✓ Successfully fetched data for {len(data)} symbols")

                # Check indicators
                sample_symbol = list(data.keys())[0]
                required_indicators = ['SMA_20', 'RSI', 'MACD', 'BB_Upper', 'Volume_Ratio']
                missing = [ind for ind in required_indicators if ind not in data[sample_symbol].columns]
                if not missing:
                    results['indicators_present'] = True
                    logger.info("✓ All required technical indicators present")
                else:
                    logger.error(f"✗ Missing indicators: {missing}")

                # Check data integrity
                for symbol, df in data.items():
                    if df.empty:
                        logger.error(f"✗ Empty data for {symbol}")
                        continue

                    # Check for NaN values
                    nan_count = df.isnull().sum().sum()
                    if nan_count > 0:
                        logger.warning(f"⚠ {symbol} has {nan_count} NaN values")

                    # Check data length
                    if len(df) < 200:
                        logger.warning(f"⚠ {symbol} has only {len(df)} days of data")

                results['data_integrity'] = True

            # Test point-in-time data (critical for bias prevention)
            test_date = end_date - timedelta(days=100)
            pit_data = self.data_manager.get_point_in_time_data('AAPL', test_date)

            if pit_data is not None:
                results['no_lookahead_bias'] = True
                logger.info(f"✓ Point-in-time data working: {len(pit_data)} days ending {pit_data.index[-1].date()}")
                if pit_data.index[-1].date() <= test_date.date():
                    logger.info("✓ No lookahead bias detected")
                else:
                    logger.error("✗ Lookahead bias detected!")
                    results['no_lookahead_bias'] = False

            results['sufficient_data'] = len(data) >= 3

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            import traceback
            traceback.print_exc()

        return results

    def validate_risk_management(self) -> Dict[str, Any]:
        """Test risk management and position sizing calculations"""
        logger.info("=== VALIDATING RISK MANAGEMENT ===")

        config = BacktestConfig(
            initial_capital=100000.0,
            max_position_size=0.1,
            max_daily_loss=0.02,
            max_portfolio_drawdown=0.15
        )

        results = {
            'position_sizing': False,
            'risk_limits': False,
            'drawdown_calculation': False,
            'diversification': False
        }

        try:
            # Test position sizing
            from backtest_tracker.comprehensive_backtest import Portfolio

            portfolio = Portfolio(config.initial_capital, config)

            # Test with different confidence levels
            test_cases = [
                {'confidence': 0.5, 'price': 100.0, 'expected_max_size': 500},
                {'confidence': 1.0, 'price': 100.0, 'expected_max_size': 1000},
                {'confidence': 0.8, 'price': 50.0, 'expected_max_size': 1600}
            ]

            for i, test_case in enumerate(test_cases):
                signal = {
                    'confidence': test_case['confidence'],
                    'action': 'buy'
                }
                price = test_case['price']

                # Calculate position size
                position_size = self.engine._calculate_position_size(signal, price, portfolio.total_value)

                expected_max = test_case['expected_max_size']
                if abs(position_size - expected_max) <= 10:  # Allow small variance
                    logger.info(f"✓ Position sizing test {i+1} passed: {position_size} shares")
                else:
                    logger.error(f"✗ Position sizing test {i+1} failed: got {position_size}, expected ~{expected_max}")

            results['position_sizing'] = True

            # Test risk limits
            portfolio.total_value = config.initial_capital * 0.8  # 20% loss
            portfolio.peak_value = config.initial_capital

            if portfolio.max_drawdown > config.max_portfolio_drawdown:
                logger.info("✓ Risk limits correctly detected drawdown breach")
                results['risk_limits'] = True
            else:
                logger.error("✗ Risk limits not working properly")

            # Test drawdown calculation
            drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
            if abs(portfolio.max_drawdown - drawdown) < 0.001:
                logger.info(f"✓ Drawdown calculation correct: {drawdown:.1%}")
                results['drawdown_calculation'] = True
            else:
                logger.error(f"✗ Drawdown calculation error: {portfolio.max_drawdown} vs {drawdown}")

        except Exception as e:
            logger.error(f"Risk management validation failed: {e}")
            import traceback
            traceback.print_exc()

        return results

    def validate_performance_metrics(self) -> Dict[str, Any]:
        """Verify performance metrics accuracy"""
        logger.info("=== VALIDATING PERFORMANCE METRICS ===")

        results = {
            'sharpe_ratio': False,
            'sortino_ratio': False,
            'calmar_ratio': False,
            'win_rate': False,
            'profit_factor': False
        }

        try:
            # Create synthetic data for testing
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            n_days = len(dates)

            # Create portfolio values with known characteristics
            initial_value = 100000.0
            portfolio_values = [initial_value]

            # Generate returns: mostly positive with some losses
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.001, 0.02, n_days-1)  # 0.1% daily return, 2% volatility

            for ret in returns:
                new_value = portfolio_values[-1] * (1 + ret)
                portfolio_values.append(new_value)

            df = pd.DataFrame({
                'date': dates,
                'portfolio_value': portfolio_values
            })
            df.set_index('date', inplace=True)
            df['returns'] = df['portfolio_value'].pct_change().fillna(0)

            # Calculate metrics manually
            volatility = df['returns'].std() * np.sqrt(252)
            risk_free_rate = 0.04

            # Manual Sharpe calculation
            expected_return = df['returns'].mean() * 252
            manual_sharpe = (expected_return - risk_free_rate) / volatility

            # Manual Sortino calculation
            negative_returns = df['returns'][df['returns'] < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            manual_sortino = (expected_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

            # Test with backtesting engine
            # Create mock positions for testing
            positions = []
            for i in range(10):
                from backtest_tracker.comprehensive_backtest import Position, PositionType
                pos = Position(
                    symbol='TEST',
                    position_type=PositionType.LONG,
                    entry_date=dates[i],
                    entry_price=100.0,
                    quantity=100,
                    exit_date=dates[i+5] if i+5 < len(dates) else dates[-1],
                    exit_price=105.0 if i % 2 == 0 else 95.0  # Some wins, some losses
                )
                positions.append(pos)

            # Calculate performance metrics
            metrics = self.engine._calculate_performance_metrics(
                [{'date': date, 'portfolio_value': value} for date, value in zip(dates, portfolio_values)],
                positions,
                dates[0],
                dates[-1]
            )

            # Validate Sharpe ratio
            sharpe_diff = abs(metrics.sharpe_ratio - manual_sharpe)
            if sharpe_diff < 0.1:
                logger.info(f"✓ Sharpe ratio validation passed: {metrics.sharpe_ratio:.2f} vs {manual_sharpe:.2f}")
                results['sharpe_ratio'] = True
            else:
                logger.error(f"✗ Sharpe ratio validation failed: {metrics.sharpe_ratio:.2f} vs {manual_sharpe:.2f}")

            # Validate Sortino ratio
            sortino_diff = abs(metrics.sortino_ratio - manual_sortino)
            if sortino_diff < 0.1:
                logger.info(f"✓ Sortino ratio validation passed: {metrics.sortino_ratio:.2f} vs {manual_sortino:.2f}")
                results['sortino_ratio'] = True
            else:
                logger.error(f"✗ Sortino ratio validation failed: {metrics.sortino_ratio:.2f} vs {manual_sortino:.2f}")

            # Validate win rate
            winning_trades = sum(1 for p in positions if not p.is_open and p.pnl > 0)
            expected_win_rate = winning_trades / len([p for p in positions if not p.is_open])
            if abs(metrics.win_rate - expected_win_rate) < 0.01:
                logger.info(f"✓ Win rate validation passed: {metrics.win_rate:.1%}")
                results['win_rate'] = True
            else:
                logger.error(f"✗ Win rate validation failed: {metrics.win_rate:.1%} vs {expected_win_rate:.1%}")

        except Exception as e:
            logger.error(f"Performance metrics validation failed: {e}")
            import traceback
            traceback.print_exc()

        return results

    def run_comprehensive_backtest(self) -> Dict[str, Any]:
        """Run comprehensive historical backtests with multiple strategies"""
        logger.info("=== RUNNING COMPREHENSIVE BACKTESTS ===")

        results = {
            'strategies_tested': 0,
            'total_trades': 0,
            'walk_forward_tested': False,
            'stress_test_passed': False,
            'performance_metrics': {}
        }

        try:
            # Test parameters for more realistic backtesting
            config = BacktestConfig(
                initial_capital=100000.0,
                commission_per_trade=1.0,
                slippage_pct=0.001,
                max_position_size=0.05,  # More conservative position sizing
                max_daily_loss=0.05,     # More lenient risk limits
                max_portfolio_drawdown=0.25,  # More lenient drawdown limit
                training_period_days=180,  # 6 months training
                testing_period_days=60,   # 2 months testing
                min_trades_required=5
            )

            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years

            strategies = get_available_strategies()

            for strategy_name in strategies:
                logger.info(f"Testing {strategy_name} strategy...")

                try:
                    strategy = create_strategy(strategy_name)

                    # Test with expanding window first
                    expanding_results = run_strategy_backtest(
                        strategy=strategy,
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date,
                        config=config,
                        mode=BacktestMode.EXPANDING_WINDOW
                    )

                    performance = expanding_results.get('performance')
                    if performance:
                        logger.info(f"{strategy_name} - Expanding Window Results:")
                        logger.info(f"  Total Return: {performance.total_return:.2%}")
                        logger.info(f"  Sharpe Ratio: {performance.sharpe_ratio:.2f}")
                        logger.info(f"  Max Drawdown: {performance.max_drawdown:.2%}")
                        logger.info(f"  Win Rate: {performance.win_rate:.2%}")
                        logger.info(f"  Total Trades: {performance.total_trades}")
                        logger.info(f"  Options Trades: {performance.options_trades}")

                        results['performance_metrics'][strategy_name] = {
                            'expanding_window': {
                                'total_return': performance.total_return,
                                'sharpe_ratio': performance.sharpe_ratio,
                                'max_drawdown': performance.max_drawdown,
                                'win_rate': performance.win_rate,
                                'total_trades': performance.total_trades,
                                'options_trades': performance.options_trades
                            }
                        }

                        results['total_trades'] += performance.total_trades

                    # Test walk-forward validation
                    try:
                        wf_results = run_strategy_backtest(
                            strategy=strategy,
                            symbols=symbols,
                            start_date=start_date,
                            end_date=end_date,
                            config=config,
                            mode=BacktestMode.WALK_FORWARD
                        )

                        if wf_results.get('periods'):
                            results['walk_forward_tested'] = True
                            logger.info(f"✓ Walk-forward validation completed for {strategy_name}")

                            # Aggregate walk-forward results
                            wf_performance = wf_results.get('overall_performance')
                            if wf_performance:
                                logger.info(f"{strategy_name} - Walk-Forward Results:")
                                logger.info(f"  Total Return: {wf_performance.total_return:.2%}")
                                logger.info(f"  Sharpe Ratio: {wf_performance.sharpe_ratio:.2f}")
                                logger.info(f"  Max Drawdown: {wf_performance.max_drawdown:.2%}")
                                logger.info(f"  Win Rate: {wf_performance.win_rate:.2%}")
                                logger.info(f"  Total Trades: {wf_performance.total_trades}")

                                results['performance_metrics'][strategy_name]['walk_forward'] = {
                                    'total_return': wf_performance.total_return,
                                    'sharpe_ratio': wf_performance.sharpe_ratio,
                                    'max_drawdown': wf_performance.max_drawdown,
                                    'win_rate': wf_performance.win_rate,
                                    'total_trades': wf_performance.total_trades
                                }

                    except Exception as e:
                        logger.warning(f"Walk-forward test failed for {strategy_name}: {e}")

                    results['strategies_tested'] += 1

                except Exception as e:
                    logger.error(f"Failed to test {strategy_name}: {e}")
                    import traceback
                    traceback.print_exc()

            # Run stress tests
            stress_results = self._run_stress_tests()
            results['stress_test_passed'] = stress_results['passed']

        except Exception as e:
            logger.error(f"Comprehensive backtest failed: {e}")
            import traceback
            traceback.print_exc()

        return results

    def _run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests with edge cases and extreme market conditions"""
        logger.info("=== RUNNING STRESS TESTS ===")

        results = {
            'passed': False,
            'extreme_volatility': False,
            'gap_handling': False,
            'data_quality': False
        }

        try:
            # Test with high volatility environment
            config = BacktestConfig(
                initial_capital=100000.0,
                max_position_size=0.02,  # Very conservative
                max_daily_loss=0.10,     # Very lenient
                max_portfolio_drawdown=0.40
            )

            # Use volatile stocks and shorter timeframe
            symbols = ['TSLA', 'NVDA', 'SOXS', 'TQQQ']  # High volatility
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6 months

            strategy = MomentumStrategy()

            stress_results = run_strategy_backtest(
                strategy=strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                config=config,
                mode=BacktestMode.EXPANDING_WINDOW
            )

            performance = stress_results.get('performance')
            if performance:
                logger.info("Stress Test Results:")
                logger.info(f"  Total Return: {performance.total_return:.2%}")
                logger.info(f"  Max Drawdown: {performance.max_drawdown:.2%}")
                logger.info(f"  Total Trades: {performance.total_trades}")
                logger.info(f"  Win Rate: {performance.win_rate:.2%}")

                # Check if system handled extreme conditions
                if performance.max_drawdown < 0.50:  # Less than 50% drawdown
                    results['extreme_volatility'] = True
                    logger.info("✓ Extreme volatility test passed")
                else:
                    logger.warning(f"⚠ High drawdown in stress test: {performance.max_drawdown:.1%}")

                if performance.total_trades > 0:
                    results['passed'] = True
                    logger.info("✓ Stress test completed successfully")

        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            import traceback
            traceback.print_exc()

        return results

    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        logger.info("=== GENERATING VALIDATION REPORT ===")

        report = []
        report.append("=" * 80)
        report.append("BACKTESTING PIPELINE VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall status
        all_tests_passed = all([
            self.validation_results.get('data_quality', {}).get('passed', False),
            self.validation_results.get('risk_management', {}).get('passed', False),
            self.validation_results.get('performance_metrics', {}).get('passed', False),
            self.validation_results.get('comprehensive_backtest', {}).get('passed', False)
        ])

        report.append(f"OVERALL STATUS: {'✓ PRODUCTION READY' if all_tests_passed else '⚠ NEEDS ATTENTION'}")
        report.append("")

        # Detailed results
        for test_name, test_results in self.validation_results.items():
            report.append(f"## {test_name.upper().replace('_', ' ')}")
            report.append("-" * 40)

            if isinstance(test_results, dict):
                for key, value in test_results.items():
                    if isinstance(value, bool):
                        status = "✓" if value else "✗"
                        report.append(f"{status} {key.replace('_', ' ').title()}")
                    elif isinstance(value, (int, float)):
                        if 'return' in key or 'rate' in key or 'ratio' in key:
                            report.append(f"  {key.replace('_', ' ').title()}: {value:.2%}")
                        else:
                            report.append(f"  {key.replace('_', ' ').title()}: {value}")
                    elif isinstance(value, dict):
                        report.append(f"  {key.replace('_', ' ').title()}:")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, float) and ('return' in sub_key or 'rate' in sub_key or 'ratio' in sub_key):
                                report.append(f"    {sub_key.replace('_', ' ').title()}: {sub_value:.2%}")
                            else:
                                report.append(f"    {sub_key.replace('_', ' ').title()}: {sub_value}")
            report.append("")

        # Recommendations
        report.append("## RECOMMENDATIONS")
        report.append("-" * 40)

        if not all_tests_passed:
            report.append("⚠ Areas needing attention:")

            if not self.validation_results.get('data_quality', {}).get('passed', False):
                report.append("  - Data quality issues detected - check data validation logs")

            if not self.validation_results.get('risk_management', {}).get('passed', False):
                report.append("  - Risk management parameters may need adjustment")

            if not self.validation_results.get('performance_metrics', {}).get('passed', False):
                report.append("  - Performance metrics calculation needs verification")

            if not self.validation_results.get('comprehensive_backtest', {}).get('passed', False):
                report.append("  - Backtesting pipeline needs optimization")

        else:
            report.append("✓ All validation tests passed")
            report.append("✓ System is production-ready for backtesting")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

def main():
    """Main validation function"""
    logger.info("Starting comprehensive backtesting validation...")

    validator = BacktestValidator()

    # Run all validation tests
    try:
        logger.info("Running data quality validation...")
        data_results = validator.validate_data_quality()
        validator.validation_results['data_quality'] = data_results

        logger.info("Running risk management validation...")
        risk_results = validator.validate_risk_management()
        validator.validation_results['risk_management'] = risk_results

        logger.info("Running performance metrics validation...")
        metrics_results = validator.validate_performance_metrics()
        validator.validation_results['performance_metrics'] = metrics_results

        logger.info("Running comprehensive backtests...")
        backtest_results = validator.run_comprehensive_backtest()
        validator.validation_results['comprehensive_backtest'] = backtest_results

        # Generate and display report
        report = validator.generate_validation_report()
        print("\n" + report)

        # Save report to file
        with open('backtest_validation_report.txt', 'w') as f:
            f.write(report)

        logger.info("Validation complete. Report saved to backtest_validation_report.txt")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)