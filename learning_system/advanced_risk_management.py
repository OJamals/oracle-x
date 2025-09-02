"""
Advanced Risk Management System for Oracle-X
Dynamic Position Sizing, Portfolio Optimization, and Risk Analytics
"""

import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Try to import optimization libraries
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import norm, t
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.covariance import LedoitWolf, EmpiricalCovariance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class RiskConfig:
    """Configuration for risk management system"""
    max_portfolio_risk: float = 0.02  # 2% daily VaR
    max_position_size: float = 0.1    # 10% max position
    max_correlation: float = 0.7      # Max correlation between positions
    confidence_level: float = 0.95    # VaR confidence level
    lookback_period: int = 252        # Days for risk calculations
    rebalance_threshold: float = 0.05 # 5% deviation triggers rebalance
    use_kelly_criterion: bool = True
    use_black_litterman: bool = True
    dynamic_hedging: bool = True

@dataclass
class PositionRisk:
    """Risk metrics for individual positions"""
    symbol: str
    position_size: float
    market_value: float
    daily_var: float
    expected_return: float
    volatility: float
    beta: float
    correlation_with_portfolio: float
    sharpe_ratio: float
    max_drawdown: float
    risk_contribution: float

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_value: float
    daily_var: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    diversification_ratio: float
    concentration_risk: float
    tail_risk: float

class RiskCalculator:
    """Advanced risk calculation engine"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.price_history = {}
        self.return_history = {}
        self.covariance_matrix = None
        
    def update_price_history(self, symbol: str, prices: pd.Series):
        """Update price history for a symbol"""
        self.price_history[symbol] = prices
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        self.return_history[symbol] = returns
        
        # Update covariance matrix if we have multiple symbols
        if len(self.return_history) > 1:
            self._update_covariance_matrix()
    
    def _update_covariance_matrix(self):
        """Update the covariance matrix of returns"""
        if not self.return_history:
            return
        
        # Align all return series
        returns_df = pd.DataFrame(self.return_history)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 30:  # Need sufficient data
            return
        
        if SKLEARN_AVAILABLE:
            # Use Ledoit-Wolf shrinkage estimator for better covariance estimation
            lw = LedoitWolf()
            self.covariance_matrix = pd.DataFrame(
                lw.fit(returns_df).covariance_,
                index=returns_df.columns,
                columns=returns_df.columns
            )
        else:
            # Simple empirical covariance
            self.covariance_matrix = returns_df.cov()
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = None) -> float:
        """Calculate Value at Risk"""
        if confidence_level is None:
            confidence_level = self.config.confidence_level
        
        if len(returns) < 30:
            return 0.0
        
        # Parametric VaR (assumes normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        
        if SCIPY_AVAILABLE:
            var_parametric = norm.ppf(1 - confidence_level, mean_return, std_return)
        else:
            # Simple approximation
            z_score = 1.96 if confidence_level == 0.95 else 2.33  # 95% or 99%
            var_parametric = mean_return - z_score * std_return
        
        # Historical VaR
        var_historical = returns.quantile(1 - confidence_level)
        
        # Use the more conservative estimate
        return min(var_parametric, var_historical)
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = None) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if confidence_level is None:
            confidence_level = self.config.confidence_level
        
        var = self.calculate_var(returns, confidence_level)
        
        # Expected shortfall is the mean of returns below VaR
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) > 0:
            return tail_returns.mean()
        else:
            return var
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market"""
        if len(asset_returns) < 30 or len(market_returns) < 30:
            return 1.0
        
        # Align series
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 30:
            return 1.0
        
        covariance = aligned_data['asset'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        if market_variance > 0:
            return covariance / market_variance
        else:
            return 1.0
    
    def calculate_position_risk(self, symbol: str, position_size: float, 
                              market_value: float, market_returns: pd.Series = None) -> PositionRisk:
        """Calculate comprehensive risk metrics for a position"""
        if symbol not in self.return_history:
            # Return default risk metrics
            return PositionRisk(
                symbol=symbol,
                position_size=position_size,
                market_value=market_value,
                daily_var=0.02 * market_value,  # Default 2% VaR
                expected_return=0.0,
                volatility=0.2,  # Default 20% volatility
                beta=1.0,
                correlation_with_portfolio=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                risk_contribution=0.0
            )
        
        returns = self.return_history[symbol]
        
        # Basic metrics
        expected_return = returns.mean() * 252  # Annualized
        volatility = returns.std() * np.sqrt(252)  # Annualized
        daily_var = abs(self.calculate_var(returns)) * market_value
        
        # Beta calculation
        beta = 1.0
        if market_returns is not None:
            beta = self.calculate_beta(returns, market_returns)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Risk contribution (simplified)
        risk_contribution = (position_size * volatility) ** 2
        
        return PositionRisk(
            symbol=symbol,
            position_size=position_size,
            market_value=market_value,
            daily_var=daily_var,
            expected_return=expected_return,
            volatility=volatility,
            beta=beta,
            correlation_with_portfolio=0.0,  # Will be calculated at portfolio level
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            risk_contribution=risk_contribution
        )

class PositionSizer:
    """Dynamic position sizing using various methods"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.risk_calculator = RiskCalculator(config)
    
    def kelly_criterion_size(self, expected_return: float, volatility: float, 
                           win_rate: float = 0.55) -> float:
        """Calculate position size using Kelly Criterion"""
        if not self.config.use_kelly_criterion or volatility <= 0:
            return 0.05  # Default 5% position
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        
        # Simplified Kelly for continuous returns
        # f = μ / σ²
        kelly_fraction = expected_return / (volatility ** 2)
        
        # Apply safety factor (typically 0.25 to 0.5 of full Kelly)
        safe_kelly = kelly_fraction * 0.25
        
        # Constrain to maximum position size
        return min(abs(safe_kelly), self.config.max_position_size)
    
    def risk_parity_size(self, volatility: float, target_risk: float) -> float:
        """Calculate position size for risk parity"""
        if volatility <= 0:
            return 0.0
        
        # Position size = target risk / volatility
        position_size = target_risk / volatility
        
        return min(position_size, self.config.max_position_size)
    
    def volatility_adjusted_size(self, base_size: float, current_vol: float, 
                               target_vol: float = 0.15) -> float:
        """Adjust position size based on volatility"""
        if current_vol <= 0:
            return base_size
        
        # Scale inversely with volatility
        vol_adjustment = target_vol / current_vol
        adjusted_size = base_size * vol_adjustment
        
        return min(adjusted_size, self.config.max_position_size)
    
    def confidence_adjusted_size(self, base_size: float, confidence: float) -> float:
        """Adjust position size based on prediction confidence"""
        # Scale position size with confidence
        confidence_multiplier = max(0.1, min(2.0, confidence))
        adjusted_size = base_size * confidence_multiplier
        
        return min(adjusted_size, self.config.max_position_size)

class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.risk_calculator = RiskCalculator(config)
    
    def mean_variance_optimization(self, expected_returns: pd.Series, 
                                 covariance_matrix: pd.DataFrame,
                                 target_return: float = None) -> pd.Series:
        """Optimize portfolio using mean-variance optimization"""
        if not SCIPY_AVAILABLE:
            # Equal weight fallback
            n_assets = len(expected_returns)
            return pd.Series([1.0/n_assets] * n_assets, index=expected_returns.index)
        
        n_assets = len(expected_returns)
        
        # Objective function: minimize portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(covariance_matrix.values, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        if target_return is not None:
            # Add return constraint
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.dot(x, expected_returns.values) - target_return
            })
        
        # Bounds: no short selling, max position size
        bounds = [(0, self.config.max_position_size) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        x0 = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            # Fallback to equal weights
            return pd.Series([1.0/n_assets] * n_assets, index=expected_returns.index)
    
    def risk_parity_optimization(self, covariance_matrix: pd.DataFrame) -> pd.Series:
        """Optimize for risk parity (equal risk contribution)"""
        if not SCIPY_AVAILABLE:
            n_assets = len(covariance_matrix)
            return pd.Series([1.0/n_assets] * n_assets, index=covariance_matrix.index)
        
        n_assets = len(covariance_matrix)
        
        def risk_parity_objective(weights):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix.values, weights)))
            marginal_contrib = np.dot(covariance_matrix.values, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Minimize sum of squared deviations from equal risk contribution
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.01, self.config.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return pd.Series(result.x, index=covariance_matrix.index)
        else:
            return pd.Series([1.0/n_assets] * n_assets, index=covariance_matrix.index)
    
    def black_litterman_optimization(self, market_caps: pd.Series, 
                                   expected_returns: pd.Series,
                                   covariance_matrix: pd.DataFrame,
                                   views: Dict[str, float] = None,
                                   confidence: float = 0.5) -> pd.Series:
        """Black-Litterman portfolio optimization"""
        if not SCIPY_AVAILABLE or not self.config.use_black_litterman:
            return self.mean_variance_optimization(expected_returns, covariance_matrix)
        
        # Market equilibrium returns (reverse optimization)
        risk_aversion = 3.0  # Typical value
        market_weights = market_caps / market_caps.sum()
        
        # Equilibrium returns: λ * Σ * w_market
        equilibrium_returns = risk_aversion * np.dot(covariance_matrix.values, market_weights.values)
        equilibrium_returns = pd.Series(equilibrium_returns, index=market_caps.index)
        
        if views is None or len(views) == 0:
            # No views, use equilibrium
            return self.mean_variance_optimization(equilibrium_returns, covariance_matrix)
        
        # Incorporate views
        # This is a simplified Black-Litterman implementation
        # In practice, you'd want a more sophisticated view incorporation
        
        adjusted_returns = equilibrium_returns.copy()
        for symbol, view_return in views.items():
            if symbol in adjusted_returns.index:
                # Blend equilibrium and view based on confidence
                adjusted_returns[symbol] = (
                    (1 - confidence) * adjusted_returns[symbol] + 
                    confidence * view_return
                )
        
        return self.mean_variance_optimization(adjusted_returns, covariance_matrix)

class AdvancedRiskManager:
    """Main advanced risk management system"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        
        # Component systems
        self.risk_calculator = RiskCalculator(config)
        self.position_sizer = PositionSizer(config)
        self.portfolio_optimizer = PortfolioOptimizer(config)
        
        # State tracking
        self.current_positions = {}
        self.risk_history = deque(maxlen=1000)
        self.rebalance_history = []
        
        logger.info("Advanced risk management system initialized")
    
    def update_market_data(self, price_data: Dict[str, pd.Series]):
        """Update market data for all symbols"""
        for symbol, prices in price_data.items():
            self.risk_calculator.update_price_history(symbol, prices)
    
    def calculate_optimal_position_size(self, symbol: str, prediction: Dict[str, Any],
                                      current_portfolio_value: float) -> float:
        """Calculate optimal position size for a new trade"""
        expected_return = prediction.get('expected_return', 0.0)
        confidence = prediction.get('confidence', 0.5)
        
        # Get historical volatility
        if symbol in self.risk_calculator.return_history:
            returns = self.risk_calculator.return_history[symbol]
            volatility = returns.std() * np.sqrt(252)  # Annualized
        else:
            volatility = 0.2  # Default 20% volatility
        
        # Calculate base position size using Kelly Criterion
        kelly_size = self.position_sizer.kelly_criterion_size(expected_return, volatility)
        
        # Adjust for confidence
        confidence_adjusted = self.position_sizer.confidence_adjusted_size(kelly_size, confidence)
        
        # Adjust for current volatility
        target_vol = 0.15  # 15% target volatility
        vol_adjusted = self.position_sizer.volatility_adjusted_size(
            confidence_adjusted, volatility, target_vol
        )
        
        # Convert to dollar amount
        position_value = vol_adjusted * current_portfolio_value
        
        logger.info(f"Optimal position size for {symbol}: {vol_adjusted:.1%} "
                   f"(${position_value:,.0f})")
        
        return vol_adjusted
    
    def assess_portfolio_risk(self, positions: Dict[str, Dict[str, float]],
                            market_returns: pd.Series = None) -> PortfolioRisk:
        """Assess comprehensive portfolio risk"""
        if not positions:
            return PortfolioRisk(
                total_value=0, daily_var=0, expected_return=0, volatility=0,
                sharpe_ratio=0, sortino_ratio=0, max_drawdown=0, calmar_ratio=0,
                diversification_ratio=0, concentration_risk=0, tail_risk=0
            )
        
        total_value = sum(pos['market_value'] for pos in positions.values())
        
        # Calculate individual position risks
        position_risks = {}
        portfolio_returns = []
        
        for symbol, position_data in positions.items():
            position_risk = self.risk_calculator.calculate_position_risk(
                symbol, 
                position_data['position_size'],
                position_data['market_value'],
                market_returns
            )
            position_risks[symbol] = position_risk
            
            # Collect returns for portfolio calculations
            if symbol in self.risk_calculator.return_history:
                weight = position_data['market_value'] / total_value
                returns = self.risk_calculator.return_history[symbol] * weight
                portfolio_returns.append(returns)
        
        # Portfolio-level calculations
        if portfolio_returns:
            portfolio_return_series = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            
            expected_return = portfolio_return_series.mean() * 252
            volatility = portfolio_return_series.std() * np.sqrt(252)
            daily_var = abs(self.risk_calculator.calculate_var(portfolio_return_series)) * total_value
            
            # Risk-adjusted metrics
            risk_free_rate = 0.02
            sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = portfolio_return_series[portfolio_return_series < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = (expected_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
            
            # Max drawdown
            cumulative = (1 + portfolio_return_series).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # Calmar ratio
            calmar_ratio = expected_return / max_drawdown if max_drawdown > 0 else 0
            
        else:
            expected_return = volatility = daily_var = 0
            sharpe_ratio = sortino_ratio = max_drawdown = calmar_ratio = 0
        
        # Concentration risk (Herfindahl index)
        weights = [pos['market_value'] / total_value for pos in positions.values()]
        concentration_risk = sum(w**2 for w in weights)
        
        # Diversification ratio (simplified)
        individual_vols = [risk.volatility for risk in position_risks.values()]
        avg_individual_vol = np.mean(individual_vols) if individual_vols else 0
        diversification_ratio = avg_individual_vol / volatility if volatility > 0 else 1
        
        # Tail risk (Expected Shortfall)
        if portfolio_returns:
            tail_risk = abs(self.risk_calculator.calculate_expected_shortfall(portfolio_return_series)) * total_value
        else:
            tail_risk = 0
        
        return PortfolioRisk(
            total_value=total_value,
            daily_var=daily_var,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            diversification_ratio=diversification_ratio,
            concentration_risk=concentration_risk,
            tail_risk=tail_risk
        )
    
    def check_risk_limits(self, portfolio_risk: PortfolioRisk) -> Dict[str, Any]:
        """Check if portfolio exceeds risk limits"""
        violations = []
        
        # VaR limit
        var_limit = self.config.max_portfolio_risk * portfolio_risk.total_value
        if portfolio_risk.daily_var > var_limit:
            violations.append({
                'type': 'var_limit',
                'current': portfolio_risk.daily_var,
                'limit': var_limit,
                'severity': 'high'
            })
        
        # Concentration limit
        if portfolio_risk.concentration_risk > 0.5:  # 50% concentration threshold
            violations.append({
                'type': 'concentration_risk',
                'current': portfolio_risk.concentration_risk,
                'limit': 0.5,
                'severity': 'medium'
            })
        
        # Drawdown limit
        if portfolio_risk.max_drawdown > 0.15:  # 15% max drawdown
            violations.append({
                'type': 'max_drawdown',
                'current': portfolio_risk.max_drawdown,
                'limit': 0.15,
                'severity': 'high'
            })
        
        return {
            'violations': violations,
            'risk_score': len(violations),
            'within_limits': len(violations) == 0
        }
    
    def generate_rebalancing_recommendations(self, current_positions: Dict[str, Dict],
                                          target_allocations: Dict[str, float] = None) -> Dict[str, Any]:
        """Generate portfolio rebalancing recommendations"""
        if not current_positions:
            return {'recommendations': [], 'rebalance_needed': False}
        
        total_value = sum(pos['market_value'] for pos in current_positions.values())
        current_weights = {
            symbol: pos['market_value'] / total_value 
            for symbol, pos in current_positions.items()
        }
        
        recommendations = []
        
        if target_allocations:
            # Compare current vs target allocations
            for symbol in set(list(current_weights.keys()) + list(target_allocations.keys())):
                current_weight = current_weights.get(symbol, 0)
                target_weight = target_allocations.get(symbol, 0)
                deviation = abs(current_weight - target_weight)
                
                if deviation > self.config.rebalance_threshold:
                    action = 'increase' if target_weight > current_weight else 'decrease'
                    amount = abs(target_weight - current_weight) * total_value
                    
                    recommendations.append({
                        'symbol': symbol,
                        'action': action,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'deviation': deviation,
                        'amount': amount
                    })
        
        rebalance_needed = len(recommendations) > 0
        
        return {
            'recommendations': recommendations,
            'rebalance_needed': rebalance_needed,
            'total_deviation': sum(rec['deviation'] for rec in recommendations)
        }
    
    def optimize_portfolio_allocation(self, symbols: List[str], 
                                    expected_returns: Dict[str, float] = None,
                                    optimization_method: str = 'mean_variance') -> Dict[str, float]:
        """Optimize portfolio allocation"""
        if len(symbols) < 2:
            return {symbols[0]: 1.0} if symbols else {}
        
        # Prepare data
        if expected_returns is None:
            expected_returns = {}
            for symbol in symbols:
                if symbol in self.risk_calculator.return_history:
                    returns = self.risk_calculator.return_history[symbol]
                    expected_returns[symbol] = returns.mean() * 252  # Annualized
                else:
                    expected_returns[symbol] = 0.08  # Default 8% expected return
        
        expected_returns_series = pd.Series(expected_returns)
        
        # Get covariance matrix
        if self.risk_calculator.covariance_matrix is not None:
            available_symbols = [s for s in symbols if s in self.risk_calculator.covariance_matrix.index]
            if len(available_symbols) >= 2:
                cov_matrix = self.risk_calculator.covariance_matrix.loc[available_symbols, available_symbols]
                expected_returns_series = expected_returns_series[available_symbols]
            else:
                # Create identity covariance matrix
                cov_matrix = pd.DataFrame(
                    np.eye(len(symbols)) * 0.04,  # 20% volatility
                    index=symbols, columns=symbols
                )
        else:
            # Create identity covariance matrix
            cov_matrix = pd.DataFrame(
                np.eye(len(symbols)) * 0.04,  # 20% volatility
                index=symbols, columns=symbols
            )
        
        # Optimize based on method
        if optimization_method == 'risk_parity':
            optimal_weights = self.portfolio_optimizer.risk_parity_optimization(cov_matrix)
        elif optimization_method == 'black_litterman':
            # Create dummy market caps (equal for simplicity)
            market_caps = pd.Series([1.0] * len(symbols), index=symbols)
            optimal_weights = self.portfolio_optimizer.black_litterman_optimization(
                market_caps, expected_returns_series, cov_matrix
            )
        else:  # mean_variance
            optimal_weights = self.portfolio_optimizer.mean_variance_optimization(
                expected_returns_series, cov_matrix
            )
        
        return optimal_weights.to_dict()
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data"""
        dashboard = {
            'timestamp': datetime.now(),
            'config': self.config.__dict__,
            'risk_history_length': len(self.risk_history),
            'rebalance_history_length': len(self.rebalance_history),
            'available_symbols': list(self.risk_calculator.return_history.keys()),
            'covariance_matrix_available': self.risk_calculator.covariance_matrix is not None
        }
        
        return dashboard
