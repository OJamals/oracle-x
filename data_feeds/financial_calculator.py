"""
Enhanced Financial Calculator for Oracle-X
Optimized financial calculation utilities for Quote objects and market data.
"""

from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Import Quote from the correct location
try:
    from data_feeds.models import Quote
except ImportError:
    # Fallback for development/testing
    from typing import NamedTuple

    class Quote(NamedTuple):
        symbol: str
        price: Decimal
        change: Optional[Decimal] = None
        change_percent: Optional[Decimal] = None
        volume: Optional[int] = None
        market_cap: Optional[int] = None
        pe_ratio: Optional[Decimal] = None


@dataclass
class FinancialMetrics:
    """Comprehensive financial metrics container"""

    symbol: str
    price: Decimal

    # Price metrics
    price_change: Optional[Decimal] = None
    price_change_percent: Optional[Decimal] = None

    # Volume metrics
    volume: Optional[int] = None
    avg_volume: Optional[float] = None
    volume_ratio: Optional[float] = None

    # Valuation metrics
    market_cap: Optional[int] = None
    pe_ratio: Optional[Decimal] = None
    pb_ratio: Optional[Decimal] = None

    # Volatility metrics
    beta: Optional[Decimal] = None
    volatility_1d: Optional[float] = None
    volatility_30d: Optional[float] = None

    # Technical indicators
    rsi_14: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None

    # Quality metrics
    data_quality_score: float = 0.0
    calculation_timestamp: Optional[pd.Timestamp] = None


class FinancialCalculator:
    """High-performance financial calculations for Oracle-X data feeds"""

    @staticmethod
    def calculate_comprehensive_metrics(
        quote: Quote, market_data: Optional[pd.DataFrame] = None
    ) -> FinancialMetrics:
        """Calculate comprehensive financial metrics from Quote and optional market data"""

        metrics = FinancialMetrics(
            symbol=quote.symbol,
            price=quote.price,
            price_change=quote.change,
            price_change_percent=quote.change_percent,
            volume=quote.volume,
            market_cap=quote.market_cap,
            pe_ratio=quote.pe_ratio,
            pb_ratio=getattr(quote, "pb_ratio", None),
            beta=getattr(quote, "beta", None),
            calculation_timestamp=pd.Timestamp.now(),
        )

        # Enhanced calculations if market data is available
        if market_data is not None and not market_data.data.empty:
            try:
                # Volume analysis
                if "Volume" in market_data.data.columns:
                    avg_volume = market_data.data["Volume"].mean()
                    if avg_volume > 0 and quote.volume:
                        metrics.avg_volume = float(avg_volume)
                        metrics.volume_ratio = float(quote.volume / avg_volume)

                # Volatility calculations
                if "Close" in market_data.data.columns:
                    close_prices = market_data.data["Close"].dropna()
                    if len(close_prices) >= 2:
                        # Daily volatility
                        daily_returns = close_prices.pct_change().dropna()
                        if len(daily_returns) > 0:
                            metrics.volatility_1d = float(daily_returns.std())

                        # 30-day volatility (if we have enough data)
                        if len(daily_returns) >= 20:
                            metrics.volatility_30d = float(
                                daily_returns.tail(20).std() * np.sqrt(252)
                            )

                    # Technical indicators
                    if len(close_prices) >= 20:
                        metrics.sma_20 = float(close_prices.tail(20).mean())

                        # RSI calculation
                        if len(close_prices) >= 15:
                            metrics.rsi_14 = FinancialCalculator._calculate_rsi(
                                close_prices, 14
                            )

                    if len(close_prices) >= 50:
                        metrics.sma_50 = float(close_prices.tail(50).mean())

            except Exception as e:
                # Don't fail the entire calculation for optional metrics
                pass

        # Calculate data quality score
        metrics.data_quality_score = FinancialCalculator._calculate_data_quality(
            metrics
        )

        return metrics

    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < window + 1:
                return None

            delta = prices.diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except Exception:
            return None

    @staticmethod
    def _calculate_data_quality(metrics: FinancialMetrics) -> float:
        """Calculate comprehensive data quality score"""
        score = 0.0
        total_weight = 0.0

        # Core price data (weight: 40%)
        if metrics.price and metrics.price > 0:
            score += 40
        total_weight += 40

        # Volume data (weight: 20%)
        if metrics.volume and metrics.volume > 0:
            score += 20
        total_weight += 20

        # Change data (weight: 15%)
        if (
            metrics.price_change is not None
            and metrics.price_change_percent is not None
        ):
            score += 15
        total_weight += 15

        # Valuation metrics (weight: 15%)
        valuation_count = sum(
            [
                1
                for x in [metrics.market_cap, metrics.pe_ratio, metrics.pb_ratio]
                if x is not None
            ]
        )
        score += (valuation_count / 3) * 15
        total_weight += 15

        # Technical indicators (weight: 10%)
        technical_count = sum(
            [
                1
                for x in [metrics.rsi_14, metrics.sma_20, metrics.volatility_1d]
                if x is not None
            ]
        )
        if technical_count > 0:
            score += (technical_count / 3) * 10
        total_weight += 10

        return round(score / total_weight * 100, 1) if total_weight > 0 else 0.0

    @staticmethod
    def calculate_portfolio_metrics(quotes: List[Quote]) -> Dict[str, Any]:
        """Calculate portfolio-level metrics from multiple quotes"""
        if not quotes:
            return {}

        total_market_cap = 0
        total_volume = 0
        pe_ratios = []
        price_changes = []

        for quote in quotes:
            if quote.market_cap:
                total_market_cap += quote.market_cap

            if quote.volume:
                total_volume += quote.volume

            if quote.pe_ratio:
                pe_ratios.append(float(quote.pe_ratio))

            if quote.change_percent:
                price_changes.append(float(quote.change_percent))

        metrics = {
            "total_symbols": len(quotes),
            "total_market_cap": total_market_cap,
            "total_volume": total_volume,
            "avg_pe_ratio": np.mean(pe_ratios) if pe_ratios else None,
            "avg_price_change": np.mean(price_changes) if price_changes else None,
            "price_change_volatility": (
                np.std(price_changes) if len(price_changes) > 1 else None
            ),
        }

        return metrics

    @staticmethod
    def calculate_correlations(
        market_data_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Calculate correlation matrix for multiple symbols"""
        if not market_data_dict:
            return pd.DataFrame()

        try:
            # Extract close prices for each symbol
            price_data = {}
            for symbol, data in market_data_dict.items():
                if "Close" in data.columns and not data.empty:
                    price_data[symbol] = data["Close"]

            if not price_data:
                return pd.DataFrame()

            # Create aligned DataFrame
            prices_df = pd.DataFrame(price_data)

            # Calculate correlation matrix
            return prices_df.corr()

        except Exception:
            return pd.DataFrame()

    @staticmethod
    def optimize_quote_for_calculations(quote: Quote) -> Dict[str, Any]:
        """Extract optimized data from Quote object for financial calculations"""
        optimized = {
            "symbol": quote.symbol,
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "market_cap": getattr(quote, "market_cap", None),
            "pe_ratio": getattr(quote, "pe_ratio", None),
        }

        # Ensure numeric fields are properly typed
        if quote.price:
            optimized["price"] = Decimal(str(quote.price)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        if quote.change:
            optimized["change"] = Decimal(str(quote.change)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        if quote.change_percent:
            optimized["change_percent"] = Decimal(str(quote.change_percent)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Ensure volume is integer
        if quote.volume:
            optimized["volume"] = int(quote.volume)

        return optimized


# Quick utility functions for common calculations
def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def calculate_simple_moving_average(
    prices: List[float], window: int
) -> Optional[float]:
    """Calculate simple moving average"""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def calculate_market_cap_billions(market_cap: int) -> float:
    """Convert market cap to billions for readability"""
    return round(market_cap / 1_000_000_000, 2)


def format_currency(amount: Union[Decimal, float], currency: str = "USD") -> str:
    """Format currency amounts for display"""
    if isinstance(amount, Decimal):
        amount = float(amount)

    if amount >= 1_000_000_000:
        return f"${amount/1_000_000_000:.2f}B {currency}"
    elif amount >= 1_000_000:
        return f"${amount/1_000_000:.2f}M {currency}"
    elif amount >= 1_000:
        return f"${amount/1_000:.2f}K {currency}"
    else:
        return f"${amount:.2f} {currency}"
