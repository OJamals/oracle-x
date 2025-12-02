"""
Options Valuation Engine for Oracle-X Options Prediction Pipeline

This module provides multi-model option pricing, fair value calculation,
mispricing detection, and volatility analysis for identifying undervalued options.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from decimal import Decimal

# Import existing options math functionality
from data_feeds.sources.options_math import (
    bs_price,
    bs_greeks,
    implied_vol,
    bs_d1,
    bs_d2,
)

# Configure logging
logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type enumeration"""

    CALL = "call"
    PUT = "put"


class OptionStyle(Enum):
    """Option exercise style"""

    EUROPEAN = "european"
    AMERICAN = "american"


class PricingModel(Enum):
    """Available pricing models"""

    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"
    MONTE_CARLO = "monte_carlo"


@dataclass
class OptionContract:
    """Option contract data structure"""

    symbol: str
    strike: float
    expiry: datetime
    option_type: OptionType
    style: OptionStyle = OptionStyle.AMERICAN
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None
    underlying_price: Optional[float] = None

    @property
    def market_price(self) -> Optional[float]:
        """Get the best available market price"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def time_to_expiry(self) -> float:
        """Calculate time to expiry in years"""
        days_to_expiry = (self.expiry - datetime.now()).days
        return max(0, days_to_expiry / 365.0)


@dataclass
class ValuationResult:
    """Comprehensive valuation result"""

    contract: OptionContract
    theoretical_value: float
    market_price: float
    mispricing: float  # theoretical - market
    mispricing_ratio: float  # (theoretical - market) / market
    confidence_score: float
    pricing_breakdown: Dict[str, float]
    greeks: Dict[str, float]
    implied_volatility: Optional[float] = None
    historical_volatility: Optional[float] = None
    iv_rank: Optional[float] = None  # IV percentile over past year
    iv_skew: Optional[float] = None

    @property
    def is_undervalued(self) -> bool:
        """Check if option is undervalued"""
        return self.mispricing > 0 and self.mispricing_ratio > 0.05

    @property
    def opportunity_score(self) -> float:
        """Calculate opportunity score (0-100)"""
        score = 0.0

        # Mispricing component (40%)
        if self.is_undervalued:
            mispricing_score = min(40, self.mispricing_ratio * 100)
            score += mispricing_score

        # Confidence component (30%)
        score += self.confidence_score * 0.3

        # Liquidity component (20%)
        if self.contract.volume and self.contract.open_interest:
            liquidity_score = min(
                20,
                (
                    min(self.contract.volume / 100, 1.0) * 10
                    + min(self.contract.open_interest / 1000, 1.0) * 10
                ),
            )
            score += liquidity_score

        # Greeks favorability (10%)
        if self.greeks:
            # Favorable delta for the position
            delta_score = abs(self.greeks.get("delta", 0)) * 10
            score += min(10, delta_score)

        return min(100, score)


@dataclass
class IVSurfacePoint:
    """Point on the implied volatility surface"""

    strike: float
    expiry: datetime
    implied_vol: float
    moneyness: float  # strike / spot
    time_to_expiry: float


@dataclass
class OpportunityAnalysis:
    """Complete opportunity analysis"""

    valuation: ValuationResult
    expected_return: float
    probability_of_profit: float
    risk_reward_ratio: float
    max_loss: float
    max_gain: float
    breakeven_price: float
    suggested_position_size: float
    confidence_level: str  # 'high', 'medium', 'low'
    key_risks: List[str]
    entry_signals: List[str]


class BinomialModel:
    """Binomial tree model for option pricing"""

    def __init__(self, steps: int = 100):
        self.steps = steps

    def price(
        self,
        S: float,
        K: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        option_type: OptionType,
        style: OptionStyle,
    ) -> float:
        """
        Calculate option price using binomial tree

        Args:
            S: Current stock price
            K: Strike price
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility
            T: Time to expiry
            option_type: Call or Put
            style: European or American
        """
        if T <= 0:
            return max(0, S - K) if option_type == OptionType.CALL else max(0, K - S)

        dt = T / self.steps
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialize stock price tree
        stock_tree = np.zeros((self.steps + 1, self.steps + 1))
        for i in range(self.steps + 1):
            for j in range(i + 1):
                stock_tree[j, i] = S * (u ** (i - j)) * (d**j)

        # Initialize option value tree
        option_tree = np.zeros((self.steps + 1, self.steps + 1))

        # Calculate terminal payoffs
        for j in range(self.steps + 1):
            if option_type == OptionType.CALL:
                option_tree[j, self.steps] = max(0, stock_tree[j, self.steps] - K)
            else:
                option_tree[j, self.steps] = max(0, K - stock_tree[j, self.steps])

        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                # Discounted expected value
                option_tree[j, i] = np.exp(-r * dt) * (
                    p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                )

                # For American options, check early exercise
                if style == OptionStyle.AMERICAN:
                    if option_type == OptionType.CALL:
                        intrinsic = stock_tree[j, i] - K
                    else:
                        intrinsic = K - stock_tree[j, i]
                    option_tree[j, i] = max(option_tree[j, i], intrinsic)

        return option_tree[0, 0]


class MonteCarloModel:
    """Monte Carlo simulation for option pricing"""

    def __init__(self, simulations: int = 10000, seed: Optional[int] = None):
        self.simulations = simulations
        if seed:
            np.random.seed(seed)

    def price(
        self,
        S: float,
        K: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        option_type: OptionType,
    ) -> Tuple[float, float]:
        """
        Calculate option price using Monte Carlo simulation

        Returns:
            Tuple of (price, standard_error)
        """
        if T <= 0:
            price = max(0, S - K) if option_type == OptionType.CALL else max(0, K - S)
            return price, 0.0

        # Generate random paths
        Z = np.random.standard_normal(self.simulations)

        # Calculate terminal stock prices
        ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

        # Calculate payoffs
        if option_type == OptionType.CALL:
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.simulations)

        return option_price, standard_error


class OptionsValuationEngine:
    """
    Main options valuation engine for multi-model pricing and opportunity detection
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        cache_ttl: int = 300,
        confidence_threshold: float = 0.7,
        max_workers: int = 4,
    ):
        """
        Initialize the valuation engine

        Args:
            risk_free_rate: Default risk-free rate
            cache_ttl: Cache time-to-live in seconds
            confidence_threshold: Minimum confidence for recommendations
            max_workers: Maximum parallel workers for batch processing
        """
        self.risk_free_rate = risk_free_rate
        self.cache_ttl = cache_ttl
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers

        # Initialize pricing models
        self.models = {
            PricingModel.BLACK_SCHOLES: self._price_black_scholes,
            PricingModel.BINOMIAL: BinomialModel(steps=100),
            PricingModel.MONTE_CARLO: MonteCarloModel(simulations=10000),
        }

        # Model weights for consensus pricing
        self.model_weights = {
            PricingModel.BLACK_SCHOLES: 0.4,
            PricingModel.BINOMIAL: 0.3,
            PricingModel.MONTE_CARLO: 0.3,
        }

        # Cache for expensive calculations
        self._cache = {}
        self._cache_timestamps = {}

        # Historical data storage for IV analysis
        self._iv_history = {}

        logger.info("OptionsValuationEngine initialized")

    def _price_black_scholes(
        self,
        S: float,
        K: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        option_type: OptionType,
        style: Optional[OptionStyle] = None,
    ) -> float:
        """Wrapper for Black-Scholes pricing using existing implementation"""
        put_call = "call" if option_type == OptionType.CALL else "put"
        return bs_price(S, K, r, q, sigma, T, put_call)

    def _get_cache_key(self, contract: OptionContract, method: str) -> str:
        """Generate cache key for a calculation"""
        return f"{contract.symbol}_{contract.strike}_{contract.expiry}_{contract.option_type.value}_{method}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self._cache:
            timestamp = self._cache_timestamps.get(key, 0)
            if time.time() - timestamp < self.cache_ttl:
                return self._cache[key]
            else:
                # Clean up expired cache
                del self._cache[key]
                del self._cache_timestamps[key]
        return None

    def _set_cache(self, key: str, value: Any):
        """Store value in cache"""
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()

    def calculate_fair_value(
        self,
        contract: OptionContract,
        underlying_price: float,
        volatility: Optional[float] = None,
        dividend_yield: float = 0.0,
        use_cache: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate fair value using multiple pricing models

        Args:
            contract: Option contract to value
            underlying_price: Current underlying price
            volatility: Volatility to use (if None, uses implied volatility)
            dividend_yield: Dividend yield
            use_cache: Whether to use cached calculations

        Returns:
            Tuple of (consensus_price, model_prices_dict)
        """
        cache_key = self._get_cache_key(contract, "fair_value")

        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                return cached

        # Use implied volatility if available, otherwise estimate
        if volatility is None:
            if contract.implied_volatility:
                volatility = contract.implied_volatility
            else:
                # Estimate from market price if available
                if contract.market_price:
                    volatility = self._estimate_implied_volatility(
                        contract, underlying_price, dividend_yield
                    )
                else:
                    volatility = 0.25  # Default fallback

        model_prices = {}
        T = contract.time_to_expiry

        # Black-Scholes pricing
        model_prices[PricingModel.BLACK_SCHOLES] = self._price_black_scholes(
            underlying_price,
            contract.strike,
            self.risk_free_rate,
            dividend_yield,
            volatility,
            T,
            contract.option_type,
        )

        # Binomial model pricing
        binomial_model = self.models[PricingModel.BINOMIAL]
        model_prices[PricingModel.BINOMIAL] = binomial_model.price(
            underlying_price,
            contract.strike,
            self.risk_free_rate,
            dividend_yield,
            volatility,
            T,
            contract.option_type,
            contract.style,
        )

        # Monte Carlo pricing
        mc_model = self.models[PricingModel.MONTE_CARLO]
        mc_price, mc_error = mc_model.price(
            underlying_price,
            contract.strike,
            self.risk_free_rate,
            dividend_yield,
            volatility,
            T,
            contract.option_type,
        )
        model_prices[PricingModel.MONTE_CARLO] = mc_price

        # Calculate weighted consensus price
        consensus_price = sum(
            price * self.model_weights[model] for model, price in model_prices.items()
        )

        result = (consensus_price, {m.value: p for m, p in model_prices.items()})

        if use_cache:
            self._set_cache(cache_key, result)

        return result

    def _estimate_implied_volatility(
        self, contract: OptionContract, underlying_price: float, dividend_yield: float
    ) -> float:
        """Estimate implied volatility from market price"""
        if not contract.market_price:
            return 0.25  # Default

        put_call = "call" if contract.option_type == OptionType.CALL else "put"

        try:
            iv = implied_vol(
                contract.market_price,
                underlying_price,
                contract.strike,
                self.risk_free_rate,
                dividend_yield,
                contract.time_to_expiry,
                put_call,
            )
            return iv if iv else 0.25
        except Exception as e:
            logger.warning(f"Failed to calculate IV: {e}")
            return 0.25

    def detect_mispricing(
        self,
        contract: OptionContract,
        underlying_price: float,
        market_data: Optional[pd.DataFrame] = None,
    ) -> ValuationResult:
        """
        Detect mispricing by comparing market price to theoretical value

        Args:
            contract: Option contract to analyze
            underlying_price: Current underlying price
            market_data: Historical market data for volatility calculation

        Returns:
            ValuationResult with mispricing analysis
        """
        # Calculate historical volatility if market data provided
        hist_vol = None
        if market_data is not None and not market_data.empty:
            hist_vol = self._calculate_historical_volatility(market_data)

        # Get market price
        market_price = contract.market_price
        if not market_price:
            raise ValueError("Market price not available for mispricing detection")

        # Calculate fair value
        fair_value, model_prices = self.calculate_fair_value(
            contract, underlying_price, volatility=hist_vol
        )

        # Calculate mispricing
        mispricing = fair_value - market_price
        mispricing_ratio = mispricing / market_price if market_price != 0 else 0

        # Calculate Greeks
        put_call = "call" if contract.option_type == OptionType.CALL else "put"
        volatility = hist_vol or contract.implied_volatility or 0.25

        greeks = bs_greeks(
            underlying_price,
            contract.strike,
            self.risk_free_rate,
            0.0,  # dividend yield
            volatility,
            contract.time_to_expiry,
            put_call,
        )

        # Calculate confidence score
        confidence = self._calculate_confidence_score(
            contract, model_prices, market_price
        )

        # Get IV metrics
        iv = self._estimate_implied_volatility(contract, underlying_price, 0.0)
        iv_rank = self._calculate_iv_rank(contract.symbol, iv)
        iv_skew = self._calculate_iv_skew(contract, underlying_price)

        return ValuationResult(
            contract=contract,
            theoretical_value=fair_value,
            market_price=market_price,
            mispricing=mispricing,
            mispricing_ratio=mispricing_ratio,
            confidence_score=confidence,
            pricing_breakdown=model_prices,
            greeks=greeks,
            implied_volatility=iv,
            historical_volatility=hist_vol,
            iv_rank=iv_rank,
            iv_skew=iv_skew,
        )

    def _calculate_historical_volatility(
        self, market_data: pd.DataFrame, periods: int = 20
    ) -> float:
        """Calculate historical volatility from market data"""
        if "Close" not in market_data.columns or len(market_data) < periods:
            return 0.25

        # Calculate log returns
        returns = np.log(market_data["Close"] / market_data["Close"].shift(1))

        # Calculate annualized volatility
        volatility = returns.std() * np.sqrt(252)

        return float(volatility) if not np.isnan(volatility) else 0.25

    def _calculate_confidence_score(
        self,
        contract: OptionContract,
        model_prices: Dict[str, float],
        market_price: float,
    ) -> float:
        """Calculate confidence score for valuation"""
        confidence = 100.0

        # Check model agreement
        prices = list(model_prices.values())
        if prices:
            std_dev = np.std(prices)
            mean_price = np.mean(prices)
            if mean_price > 0:
                coefficient_of_variation = std_dev / mean_price
                # Penalize high disagreement between models
                confidence -= min(30, float(coefficient_of_variation) * 100)

        # Check liquidity
        if not contract.volume or contract.volume < 10:
            confidence -= 20
        if not contract.open_interest or contract.open_interest < 100:
            confidence -= 15

        # Check bid-ask spread
        if contract.bid and contract.ask:
            spread = contract.ask - contract.bid
            if contract.ask > 0:
                spread_ratio = spread / contract.ask
                confidence -= min(20, spread_ratio * 100)

        # Time decay penalty for very short-term options
        if contract.time_to_expiry < 0.02:  # Less than ~7 days
            confidence -= 25

        return max(0, min(100, confidence))

    def _calculate_iv_rank(self, symbol: str, current_iv: float) -> float:
        """Calculate IV rank (percentile over past year)"""
        # This would typically query historical IV data
        # For now, return a placeholder
        return 50.0

    def _calculate_iv_skew(
        self, contract: OptionContract, underlying_price: float
    ) -> float:
        """Calculate IV skew relative to ATM options"""
        # Calculate moneyness
        moneyness = contract.strike / underlying_price

        # This would typically compare to other strikes
        # For now, return a simple skew estimate
        if contract.option_type == OptionType.PUT:
            # Puts typically have higher IV when OTM
            if moneyness < 0.95:
                return 0.1 * (1 - moneyness)
        else:
            # Calls typically have lower IV when OTM
            if moneyness > 1.05:
                return -0.1 * (moneyness - 1)

        return 0.0

    def analyze_iv_surface(
        self, options_chain: List[OptionContract], underlying_price: float
    ) -> List[IVSurfacePoint]:
        """
        Analyze implied volatility surface across strikes and expiries

        Args:
            options_chain: List of option contracts
            underlying_price: Current underlying price

        Returns:
            List of IV surface points
        """
        surface_points = []

        for contract in options_chain:
            if not contract.market_price:
                continue

            # Calculate implied volatility
            iv = self._estimate_implied_volatility(contract, underlying_price, 0.0)

            # Create surface point
            point = IVSurfacePoint(
                strike=contract.strike,
                expiry=contract.expiry,
                implied_vol=iv,
                moneyness=contract.strike / underlying_price,
                time_to_expiry=contract.time_to_expiry,
            )

            surface_points.append(point)

        return surface_points

    def calculate_expected_returns(
        self,
        valuation: ValuationResult,
        target_price: float,
        probability_model: str = "normal",
    ) -> Tuple[float, float]:
        """
        Calculate expected returns and probability of profit

        Args:
            valuation: Valuation result
            target_price: Target price for underlying
            probability_model: Probability distribution model

        Returns:
            Tuple of (expected_return, probability_of_profit)
        """
        contract = valuation.contract
        current_price = contract.underlying_price or target_price

        # Calculate expected option value at target
        if contract.option_type == OptionType.CALL:
            expected_value = max(0, target_price - contract.strike)
        else:
            expected_value = max(0, contract.strike - target_price)

        # Calculate return
        if valuation.market_price > 0:
            expected_return = (
                expected_value - valuation.market_price
            ) / valuation.market_price
        else:
            expected_return = 0.0

        # Calculate probability of profit
        volatility = valuation.implied_volatility or 0.25
        time_to_expiry = contract.time_to_expiry

        if probability_model == "normal":
            # Use Black-Scholes probability
            from scipy.stats import norm

            # Calculate d2 for probability of being ITM
            d1 = bs_d1(
                current_price,
                contract.strike,
                self.risk_free_rate,
                0.0,
                volatility,
                time_to_expiry,
            )
            d2 = bs_d2(
                current_price,
                contract.strike,
                self.risk_free_rate,
                0.0,
                volatility,
                time_to_expiry,
            )

            if contract.option_type == OptionType.CALL:
                prob_itm = norm.cdf(d2)
            else:
                prob_itm = norm.cdf(-d2)

            # Adjust for breakeven
            breakeven = (
                contract.strike + valuation.market_price
                if contract.option_type == OptionType.CALL
                else contract.strike - valuation.market_price
            )

            d2_breakeven = bs_d2(
                current_price,
                breakeven,
                self.risk_free_rate,
                0.0,
                volatility,
                time_to_expiry,
            )

            if contract.option_type == OptionType.CALL:
                probability_of_profit = norm.cdf(d2_breakeven)
            else:
                probability_of_profit = norm.cdf(-d2_breakeven)
        else:
            # Simple probability estimate
            probability_of_profit = 0.5

        return expected_return, float(probability_of_profit)

    def scan_opportunities(
        self,
        options_chain: List[OptionContract],
        underlying_price: float,
        market_data: Optional[pd.DataFrame] = None,
        min_opportunity_score: float = 70.0,
    ) -> List[OpportunityAnalysis]:
        """
        Scan multiple options for opportunities using parallel processing

        Args:
            options_chain: List of option contracts to analyze
            underlying_price: Current underlying price
            market_data: Historical market data
            min_opportunity_score: Minimum score to consider

        Returns:
            List of opportunity analyses sorted by score
        """
        opportunities = []

        # Process options in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit valuation tasks
            future_to_contract = {
                executor.submit(
                    self._analyze_single_opportunity,
                    contract,
                    underlying_price,
                    market_data,
                ): contract
                for contract in options_chain
                if contract.market_price is not None
            }

            # Collect results
            for future in as_completed(future_to_contract):
                contract = future_to_contract[future]
                try:
                    analysis = future.result()
                    if (
                        analysis
                        and analysis.valuation.opportunity_score
                        >= min_opportunity_score
                    ):
                        opportunities.append(analysis)
                except Exception as e:
                    logger.error(f"Failed to analyze {contract.symbol}: {e}")

        # Sort by opportunity score
        opportunities.sort(key=lambda x: x.valuation.opportunity_score, reverse=True)

        return opportunities

    def _analyze_single_opportunity(
        self,
        contract: OptionContract,
        underlying_price: float,
        market_data: Optional[pd.DataFrame],
    ) -> Optional[OpportunityAnalysis]:
        """Analyze a single option for opportunity"""
        try:
            # Get valuation
            valuation = self.detect_mispricing(contract, underlying_price, market_data)

            # Skip if not undervalued
            if not valuation.is_undervalued:
                return None

            # Calculate expected returns
            target_price = underlying_price * 1.05  # 5% move assumption
            expected_return, prob_profit = self.calculate_expected_returns(
                valuation, target_price
            )

            # Calculate risk metrics
            max_loss = valuation.market_price

            if contract.option_type == OptionType.CALL:
                max_gain = float("inf")  # Unlimited upside
                breakeven = contract.strike + valuation.market_price
            else:
                max_gain = contract.strike - valuation.market_price
                breakeven = contract.strike - valuation.market_price

            risk_reward = max_gain / max_loss if max_loss > 0 else float("inf")

            # Determine confidence level
            if valuation.confidence_score >= 80:
                confidence_level = "high"
            elif valuation.confidence_score >= 60:
                confidence_level = "medium"
            else:
                confidence_level = "low"

            # Identify key risks
            risks = []
            if contract.time_to_expiry < 0.08:  # Less than 30 days
                risks.append("High time decay risk")
            if valuation.implied_volatility and valuation.implied_volatility > 0.5:
                risks.append("High volatility risk")
            if contract.volume and contract.volume < 100:
                risks.append("Low liquidity")

            # Entry signals
            signals = []
            if valuation.mispricing_ratio > 0.1:
                signals.append(f"Undervalued by {valuation.mispricing_ratio:.1%}")
            if valuation.iv_rank and valuation.iv_rank < 30:
                signals.append("Low IV rank - good entry")
            if prob_profit > 0.6:
                signals.append(f"High probability of profit: {prob_profit:.1%}")

            # Suggest position size (simplified Kelly criterion)
            kelly_fraction = (
                prob_profit * risk_reward - (1 - prob_profit)
            ) / risk_reward
            suggested_size = min(
                0.1, max(0.01, kelly_fraction * 0.25)
            )  # Conservative sizing

            return OpportunityAnalysis(
                valuation=valuation,
                expected_return=expected_return,
                probability_of_profit=prob_profit,
                risk_reward_ratio=risk_reward,
                max_loss=max_loss,
                max_gain=max_gain,
                breakeven_price=breakeven,
                suggested_position_size=suggested_size,
                confidence_level=confidence_level,
                key_risks=risks,
                entry_signals=signals,
            )

        except Exception as e:
            logger.error(f"Failed to analyze opportunity: {e}")
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        active_entries = sum(
            1
            for key in self._cache
            if current_time - self._cache_timestamps.get(key, 0) < self.cache_ttl
        )

        return {
            "total_entries": len(self._cache),
            "active_entries": active_entries,
            "expired_entries": len(self._cache) - active_entries,
            "cache_ttl": self.cache_ttl,
        }

    def clear_cache(self):
        """Clear all cached calculations"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")


# Example usage and integration functions
def create_valuation_engine(
    config: Optional[Dict[str, Any]] = None,
) -> OptionsValuationEngine:
    """
    Factory function to create configured valuation engine

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured OptionsValuationEngine instance
    """
    if config is None:
        config = {}

    return OptionsValuationEngine(
        risk_free_rate=config.get("risk_free_rate", 0.05),
        cache_ttl=config.get("cache_ttl", 300),
        confidence_threshold=config.get("confidence_threshold", 0.7),
        max_workers=config.get("max_workers", 4),
    )


def analyze_options_chain(
    engine: OptionsValuationEngine,
    symbol: str,
    options_data: List[Dict[str, Any]],
    underlying_price: float,
) -> List[OpportunityAnalysis]:
    """
    Analyze an entire options chain for opportunities

    Args:
        engine: Valuation engine instance
        symbol: Underlying symbol
        options_data: Raw options chain data
        underlying_price: Current underlying price

    Returns:
        List of opportunity analyses
    """
    # Convert raw data to OptionContract objects
    contracts = []
    for opt in options_data:
        try:
            contract = OptionContract(
                symbol=symbol,
                strike=float(opt["strike"]),
                expiry=pd.to_datetime(opt["expiry"]),
                option_type=(
                    OptionType.CALL if opt["type"].lower() == "call" else OptionType.PUT
                ),
                bid=opt.get("bid"),
                ask=opt.get("ask"),
                last=opt.get("last"),
                volume=opt.get("volume"),
                open_interest=opt.get("openInterest"),
                implied_volatility=opt.get("impliedVolatility"),
                underlying_price=underlying_price,
            )
            contracts.append(contract)
        except Exception as e:
            logger.warning(f"Failed to parse option data: {e}")
            continue

    # Scan for opportunities
    return engine.scan_opportunities(contracts, underlying_price)
