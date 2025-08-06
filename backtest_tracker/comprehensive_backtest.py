"""
Comprehensive Backtesting Framework
Advanced backtesting system with walk-forward validation, risk metrics, and strategy optimization
Designed for options swing trading with proper bias prevention and realistic execution modeling
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class PositionType(Enum):
    LONG = "long"
    SHORT = "short"
    CALL = "call"
    PUT = "put"

class BacktestMode(Enum):
    WALK_FORWARD = "walk_forward"
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"

@dataclass
class Position:
    """Trading position with complete tracking"""
    symbol: str
    position_type: PositionType
    entry_date: datetime
    entry_price: float
    quantity: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    commission: float = 0.0
    slippage: float = 0.0
    option_details: Optional[Dict] = None  # For options: strike, expiry, type
    
    @property
    def is_open(self) -> bool:
        return self.exit_date is None
    
    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        
        if self.position_type in [PositionType.LONG, PositionType.CALL]:
            gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SHORT or PUT
            gross_pnl = (self.entry_price - self.exit_price) * self.quantity
        
        return gross_pnl - self.commission - self.slippage
    
    @property
    def return_pct(self) -> float:
        if self.exit_price is None:
            return 0.0
        investment = abs(self.entry_price * self.quantity)
        return self.pnl / investment if investment > 0 else 0.0
    
    @property
    def holding_period(self) -> int:
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days

@dataclass
class BacktestConfig:
    """Backtesting configuration parameters"""
    initial_capital: float = 100000.0
    commission_per_trade: float = 1.0
    commission_pct: float = 0.0  # Percentage commission
    slippage_pct: float = 0.001  # 0.1% slippage
    max_position_size: float = 0.1  # 10% of portfolio per position
    risk_free_rate: float = 0.04  # 4% annual risk-free rate
    
    # Options-specific settings
    options_multiplier: int = 100  # Options contract multiplier
    max_options_allocation: float = 0.2  # 20% max allocation to options
    
    # Risk management
    max_daily_loss: float = 0.02  # 2% max daily loss
    max_portfolio_drawdown: float = 0.15  # 15% max drawdown before stopping
    
    # Walk-forward settings
    training_period_days: int = 252  # 1 year training
    testing_period_days: int = 63   # 1 quarter testing
    min_trades_required: int = 10   # Minimum trades for valid test

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_period: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Options-specific metrics
    options_trades: int = 0
    options_win_rate: float = 0.0
    options_avg_return: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0  # Value at Risk 95%
    expected_shortfall: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_date: datetime) -> List[Dict]:
        """
        Generate trading signals based on data up to current_date
        Must be implemented by subclasses
        
        Args:
            data: Dictionary mapping symbol names to their historical DataFrames
            current_date: Current date for signal generation (no lookahead bias)
        
        Returns:
            List of signal dictionaries with format:
            {
                'symbol': str,
                'action': 'buy'|'sell'|'buy_call'|'buy_put',
                'quantity': int,
                'stop_loss': float,
                'take_profit': float,
                'confidence': float,
                'reason': str
            }
        """
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def update_parameters(self, new_params: Dict):
        """Update strategy parameters"""
        self.parameters.update(new_params)

class DataManager:
    """Manages historical data with bias prevention"""
    
    def __init__(self, cache_dir: str = "backtest_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.data_cache = {}
        
    def get_historical_data(self, symbols: List[str], start_date: datetime, 
                          end_date: datetime, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get historical data with proper caching and bias prevention
        """
        data = {}
        
        for symbol in symbols:
            cache_key = f"{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                # Load from cache
                try:
                    with open(cache_file, 'rb') as f:
                        symbol_data = pickle.load(f)
                    logger.debug(f"Loaded {symbol} data from cache")
                except Exception as e:
                    logger.warning(f"Failed to load cache for {symbol}: {e}")
                    symbol_data = self._fetch_data(symbol, start_date, end_date, interval)
            else:
                # Fetch new data
                symbol_data = self._fetch_data(symbol, start_date, end_date, interval)
                
                # Cache the data
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(symbol_data, f)
                except Exception as e:
                    logger.warning(f"Failed to cache data for {symbol}: {e}")
            
            if symbol_data is not None and not symbol_data.empty:
                data[symbol] = symbol_data
        
        return data
    
    def _fetch_data(self, symbol: str, start_date: datetime, 
                   end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol} from {start_date} to {end_date}")
                return None
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to prevent recalculation"""
        # Simple moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Exponential moving averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        return data
    
    def get_point_in_time_data(self, symbol: str, as_of_date: datetime, 
                              lookback_days: int = 252) -> Optional[pd.DataFrame]:
        """
        Get data as it would have been available at a specific point in time
        Critical for preventing lookahead bias
        """
        start_date = as_of_date - timedelta(days=lookback_days + 30)  # Extra buffer
        end_date = as_of_date
        
        data = self.get_historical_data([symbol], start_date, end_date)
        if symbol not in data:
            return None
        
        symbol_data = data[symbol]
        
        # Ensure we only have data up to as_of_date
        # Convert as_of_date to timezone-aware if data index is timezone-aware
        if symbol_data.index.tz is not None:
            if as_of_date.tzinfo is None:
                # Make as_of_date timezone-aware using the same timezone as the data
                as_of_date = as_of_date.replace(tzinfo=symbol_data.index.tz)
        
        symbol_data = symbol_data[symbol_data.index <= as_of_date]
        
        return symbol_data.tail(lookback_days) if len(symbol_data) > 0 else None

class BacktestEngine:
    """Main backtesting engine with walk-forward validation"""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.data_manager = DataManager()
        self.results_cache = {}
        
    def run_backtest(self, strategy: TradingStrategy, symbols: List[str], 
                    start_date: datetime, end_date: datetime, 
                    mode: BacktestMode = BacktestMode.WALK_FORWARD) -> Dict[str, Any]:
        """
        Run comprehensive backtest with specified mode
        """
        logger.info(f"Starting backtest: {strategy.name} on {len(symbols)} symbols")
        logger.info(f"Period: {start_date} to {end_date}, Mode: {mode.value}")
        
        if mode == BacktestMode.WALK_FORWARD:
            return self._run_walk_forward_backtest(strategy, symbols, start_date, end_date)
        elif mode == BacktestMode.EXPANDING_WINDOW:
            return self._run_expanding_window_backtest(strategy, symbols, start_date, end_date)
        else:  # ROLLING_WINDOW
            return self._run_rolling_window_backtest(strategy, symbols, start_date, end_date)
    
    def _run_walk_forward_backtest(self, strategy: TradingStrategy, symbols: List[str], 
                                  start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Walk-forward backtesting - the gold standard for strategy validation
        """
        results = {
            'strategy_name': strategy.name,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'mode': 'walk_forward',
            'periods': [],
            'overall_performance': None,
            'parameter_stability': {},
            'out_of_sample_performance': {}
        }
        
        current_date = start_date
        period_number = 1
        
        while current_date + timedelta(days=self.config.training_period_days + self.config.testing_period_days) <= end_date:
            # Define training and testing periods
            training_start = current_date
            training_end = current_date + timedelta(days=self.config.training_period_days)
            testing_start = training_end + timedelta(days=1)
            testing_end = testing_start + timedelta(days=self.config.testing_period_days)
            
            logger.info(f"Walk-forward period {period_number}: Train {training_start.date()} to {training_end.date()}, Test {testing_start.date()} to {testing_end.date()}")
            
            # Get training data
            training_data = self.data_manager.get_historical_data(
                symbols, training_start, training_end
            )
            
            if not training_data:
                logger.warning(f"No training data available for period {period_number}")
                current_date = testing_end
                period_number += 1
                continue
            
            # Optimize strategy parameters on training data (placeholder for now)
            # TODO: Implement parameter optimization
            
            # Run backtest on testing period
            testing_results = self._run_single_period_backtest(
                strategy, symbols, testing_start, testing_end, training_data
            )
            
            testing_results['period_number'] = period_number
            testing_results['training_period'] = (training_start, training_end)
            testing_results['testing_period'] = (testing_start, testing_end)
            
            results['periods'].append(testing_results)
            
            # Move to next period
            current_date = testing_end
            period_number += 1
        
        # Aggregate results across all periods
        results['overall_performance'] = self._aggregate_walk_forward_results(results['periods'])
        
        return results
    
    def _run_single_period_backtest(self, strategy: TradingStrategy, symbols: List[str],
                                   start_date: datetime, end_date: datetime,
                                   training_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Run backtest for a single period with realistic execution modeling
        """
        portfolio = Portfolio(self.config.initial_capital, self.config)
        all_positions = []
        daily_portfolio_values = []
        trade_log = []
        
        # Get testing data
        testing_data = self.data_manager.get_historical_data(symbols, start_date, end_date)
        
        if not testing_data:
            return self._empty_backtest_result(start_date, end_date)
        
        # Create unified date range
        all_dates = set()
        for symbol_data in testing_data.values():
            all_dates.update(symbol_data.index)
        
        trading_dates = sorted(all_dates)
        
        for current_date in trading_dates:
            # Update portfolio with current prices
            current_prices = {}
            for symbol, data in testing_data.items():
                if current_date in data.index:
                    current_prices[symbol] = data.loc[current_date, 'Close']
            
            portfolio.update_portfolio_value(current_date, current_prices)
            daily_portfolio_values.append({
                'date': current_date,
                'portfolio_value': portfolio.total_value,
                'cash': portfolio.cash,
                'positions_value': portfolio.positions_value
            })
            
            # Check for risk management triggers
            if self._check_risk_limits(portfolio):
                logger.warning(f"Risk limits breached on {current_date}, stopping backtest")
                break
            
            # Generate signals (only using data available up to current_date)
            point_in_time_data = {}
            for symbol in symbols:
                pit_data = self.data_manager.get_point_in_time_data(symbol, current_date)
                if pit_data is not None:
                    point_in_time_data[symbol] = pit_data
            
            if point_in_time_data:
                signals = strategy.generate_signals(point_in_time_data, current_date)
                
                # Execute signals
                for signal in signals:
                    position = self._execute_signal(signal, current_date, current_prices, portfolio)
                    if position:
                        all_positions.append(position)
                        trade_log.append({
                            'date': current_date,
                            'action': 'open',
                            'signal': signal,
                            'position': position
                        })
            
            # Check exit conditions for open positions
            exits = portfolio.check_exit_conditions(current_date, current_prices)
            for exit_position in exits:
                trade_log.append({
                    'date': current_date,
                    'action': 'close',
                    'position': exit_position,
                    'reason': exit_position.exit_reason
                })
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(
            daily_portfolio_values, all_positions, start_date, end_date
        )
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'performance': performance,
            'positions': all_positions,
            'daily_values': daily_portfolio_values,
            'trade_log': trade_log,
            'final_portfolio_value': portfolio.total_value
        }
    
    def _execute_signal(self, signal: Dict, current_date: datetime, 
                       current_prices: Dict[str, float], portfolio) -> Optional[Position]:
        """Execute a trading signal with realistic modeling"""
        symbol = signal['symbol']
        action = signal['action']
        
        if symbol not in current_prices:
            return None
        
        current_price = current_prices[symbol]
        
        # Calculate position size based on risk management
        position_size = self._calculate_position_size(
            signal, current_price, portfolio.total_value
        )
        
        if position_size <= 0:
            return None
        
        # Calculate costs
        commission = max(self.config.commission_per_trade, 
                        current_price * position_size * self.config.commission_pct)
        slippage = current_price * position_size * self.config.slippage_pct
        
        # Create position
        if action == 'buy':
            position_type = PositionType.LONG
        elif action == 'sell':
            position_type = PositionType.SHORT
        elif action == 'buy_call':
            position_type = PositionType.CALL
        elif action == 'buy_put':
            position_type = PositionType.PUT
        else:
            return None
        
        # Check if we have enough capital
        required_capital = current_price * position_size + commission + slippage
        if required_capital > portfolio.cash:
            # Reduce position size to fit available capital
            position_size = int((portfolio.cash - commission - slippage) / current_price)
            if position_size <= 0:
                return None
        
        position = Position(
            symbol=symbol,
            position_type=position_type,
            entry_date=current_date,
            entry_price=current_price,
            quantity=position_size,
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit'),
            commission=commission,
            slippage=slippage
        )
        
        # Update portfolio
        portfolio.add_position(position)
        
        return position
    
    def _calculate_position_size(self, signal: Dict, price: float, portfolio_value: float) -> int:
        """Calculate appropriate position size based on risk management"""
        confidence = signal.get('confidence', 0.5)
        
        # Base position size as percentage of portfolio
        base_allocation = self.config.max_position_size * confidence
        
        # For options, apply additional constraints
        if signal['action'] in ['buy_call', 'buy_put']:
            base_allocation = min(base_allocation, self.config.max_options_allocation)
        
        position_value = portfolio_value * base_allocation
        position_size = int(position_value / price)
        
        return max(0, position_size)
    
    def _check_risk_limits(self, portfolio) -> bool:
        """Check if risk limits have been breached"""
        # Check maximum drawdown
        if portfolio.max_drawdown > self.config.max_portfolio_drawdown:
            return True
        
        # Check daily loss limit
        daily_loss = (portfolio.initial_capital - portfolio.total_value) / portfolio.initial_capital
        if daily_loss > self.config.max_daily_loss:
            return True
        
        return False
    
    def _calculate_performance_metrics(self, daily_values: List[Dict], 
                                     positions: List[Position], 
                                     start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not daily_values or not positions:
            return PerformanceMetrics()
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(daily_values)
        df.set_index('date', inplace=True)
        
        # Calculate returns
        df['returns'] = df['portfolio_value'].pct_change().fillna(0)
        
        # Basic metrics
        total_return = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1
        trading_days = len(df)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        volatility = df['returns'].std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Downside deviation for Sortino ratio
        negative_returns = df['returns'][df['returns'] < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        df['cumulative'] = (1 + df['returns']).cumprod()
        df['running_max'] = df['cumulative'].expanding().max()
        df['drawdown'] = (df['cumulative'] / df['running_max']) - 1
        max_drawdown = df['drawdown'].min()
        
        # Trade analysis
        closed_positions = [p for p in positions if not p.is_open]
        winning_trades = [p for p in closed_positions if p.pnl > 0]
        losing_trades = [p for p in closed_positions if p.pnl < 0]
        
        win_rate = len(winning_trades) / len(closed_positions) if closed_positions else 0
        avg_win = np.mean([p.pnl for p in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([p.pnl for p in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(p.pnl for p in winning_trades) / sum(p.pnl for p in losing_trades)) if losing_trades else float('inf')
        
        # Options-specific metrics
        options_positions = [p for p in closed_positions if p.position_type in [PositionType.CALL, PositionType.PUT]]
        options_winning = [p for p in options_positions if p.pnl > 0]
        options_win_rate = len(options_winning) / len(options_positions) if options_positions else 0
        options_avg_return = np.mean([p.return_pct for p in options_positions]) if options_positions else 0
        
        # VaR and Expected Shortfall
        var_95 = np.percentile(df['returns'], 5) if len(df) > 0 else 0
        expected_shortfall = df['returns'][df['returns'] <= var_95].mean() if len(df) > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            avg_holding_period=float(np.mean([p.holding_period for p in closed_positions]) if closed_positions else 0),
            total_trades=len(closed_positions),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            options_trades=len(options_positions),
            options_win_rate=options_win_rate,
            options_avg_return=float(options_avg_return),
            var_95=float(var_95),
            expected_shortfall=float(expected_shortfall)
        )
    
    def _aggregate_walk_forward_results(self, periods: List[Dict]) -> PerformanceMetrics:
        """Aggregate performance across walk-forward periods"""
        if not periods:
            return PerformanceMetrics()
        
        # Combine all periods for overall statistics
        all_positions = []
        all_daily_values = []
        
        for period in periods:
            all_positions.extend(period['positions'])
            all_daily_values.extend(period['daily_values'])
        
        if not all_daily_values:
            return PerformanceMetrics()
        
        # Sort by date
        all_daily_values.sort(key=lambda x: x['date'])
        
        start_date = all_daily_values[0]['date']
        end_date = all_daily_values[-1]['date']
        
        return self._calculate_performance_metrics(all_daily_values, all_positions, start_date, end_date)
    
    def _empty_backtest_result(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Return empty backtest result structure"""
        return {
            'start_date': start_date,
            'end_date': end_date,
            'performance': PerformanceMetrics(),
            'positions': [],
            'daily_values': [],
            'trade_log': [],
            'final_portfolio_value': self.config.initial_capital
        }
    
    def _run_expanding_window_backtest(self, strategy: TradingStrategy, symbols: List[str], 
                                     start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Expanding window backtest - train on all previous data"""
        # TODO: Implement expanding window logic
        return self._run_single_period_backtest(strategy, symbols, start_date, end_date)
    
    def _run_rolling_window_backtest(self, strategy: TradingStrategy, symbols: List[str], 
                                   start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Rolling window backtest - fixed training window size"""
        # TODO: Implement rolling window logic
        return self._run_single_period_backtest(strategy, symbols, start_date, end_date)

class Portfolio:
    """Portfolio management with position tracking and risk monitoring"""
    
    def __init__(self, initial_capital: float, config: BacktestConfig):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: List[Position] = []
        self.config = config
        self.total_value = initial_capital
        self.positions_value = 0.0
        self.max_drawdown = 0.0
        self.peak_value = initial_capital
    
    def add_position(self, position: Position):
        """Add a new position to the portfolio"""
        # Deduct cash for the position
        cost = position.entry_price * position.quantity + position.commission + position.slippage
        self.cash -= cost
        self.positions.append(position)
    
    def update_portfolio_value(self, current_date: datetime, current_prices: Dict[str, float]):
        """Update portfolio value based on current prices"""
        positions_value = 0.0
        
        for position in self.positions:
            if position.is_open and position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                if position.position_type in [PositionType.LONG, PositionType.CALL]:
                    position_value = current_price * position.quantity
                else:  # SHORT or PUT
                    position_value = position.entry_price * position.quantity - (current_price - position.entry_price) * position.quantity
                positions_value += position_value
        
        self.positions_value = positions_value
        self.total_value = self.cash + positions_value
        
        # Update drawdown tracking
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
        
        current_drawdown = (self.peak_value - self.total_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def check_exit_conditions(self, current_date: datetime, current_prices: Dict[str, float]) -> List[Position]:
        """Check and execute exit conditions for open positions"""
        exits = []
        
        for position in self.positions:
            if not position.is_open or position.symbol not in current_prices:
                continue
            
            current_price = current_prices[position.symbol]
            should_exit = False
            exit_reason = None
            
            # Check stop loss
            if position.stop_loss:
                if ((position.position_type in [PositionType.LONG, PositionType.CALL] and current_price <= position.stop_loss) or
                    (position.position_type in [PositionType.SHORT, PositionType.PUT] and current_price >= position.stop_loss)):
                    should_exit = True
                    exit_reason = "stop_loss"
            
            # Check take profit
            if position.take_profit and not should_exit:
                if ((position.position_type in [PositionType.LONG, PositionType.CALL] and current_price >= position.take_profit) or
                    (position.position_type in [PositionType.SHORT, PositionType.PUT] and current_price <= position.take_profit)):
                    should_exit = True
                    exit_reason = "take_profit"
            
            # Check for options expiration (placeholder - would need actual options data)
            if position.position_type in [PositionType.CALL, PositionType.PUT] and not should_exit:
                # For now, assume options expire after 30 days
                if (current_date - position.entry_date).days >= 30:
                    should_exit = True
                    exit_reason = "expiration"
            
            if should_exit and exit_reason:
                self._close_position(position, current_date, current_price, exit_reason)
                exits.append(position)
        
        return exits
    
    def _close_position(self, position: Position, exit_date: datetime, 
                       exit_price: float, exit_reason: str):
        """Close a position and update cash"""
        position.exit_date = exit_date
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        
        # Add proceeds to cash
        proceeds = exit_price * position.quantity - position.commission - position.slippage
        self.cash += proceeds

# ============================================================================
# Public Interface Functions
# ============================================================================

def create_backtest_engine(config: Optional[BacktestConfig] = None) -> BacktestEngine:
    """Create a backtesting engine with specified configuration"""
    return BacktestEngine(config)

def run_strategy_backtest(strategy: TradingStrategy, symbols: List[str], 
                         start_date: datetime, end_date: datetime,
                         config: Optional[BacktestConfig] = None, 
                         mode: BacktestMode = BacktestMode.WALK_FORWARD) -> Dict[str, Any]:
    """
    Run a complete backtest for a trading strategy
    
    Args:
        strategy: Trading strategy to test
        symbols: List of symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        config: Backtesting configuration
        mode: Backtesting mode (walk-forward, expanding, rolling)
    
    Returns:
        Complete backtest results with performance metrics
    """
    engine = create_backtest_engine(config)
    return engine.run_backtest(strategy, symbols, start_date, end_date, mode)
