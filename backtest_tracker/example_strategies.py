"""
Example Trading Strategies for Backtesting
Demonstrates implementation of TradingStrategy base class with realistic trading logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from .comprehensive_backtest import TradingStrategy

logger = logging.getLogger(__name__)

class MomentumStrategy(TradingStrategy):
    """
    Simple momentum strategy for swing trading
    Buys stocks showing momentum with RSI confirmation
    """
    
    def __init__(self, name: str = "Momentum Strategy"):
        super().__init__(name)
        self.parameters = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'momentum_period': 20,
            'volume_threshold': 1.5,  # Volume must be 1.5x average
            'min_price': 10.0,  # Minimum stock price
            'max_price': 500.0,  # Maximum stock price
            'stop_loss_pct': 0.08,  # 8% stop loss
            'take_profit_pct': 0.15,  # 15% take profit
            'min_confidence': 0.6
        }
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_date: datetime) -> List[Dict]:
        """Generate momentum-based trading signals"""
        signals = []
        
        for symbol, df in data.items():
            if df.empty or len(df) < self.parameters['momentum_period']:
                continue
            
            try:
                # Get latest data point
                latest = df.iloc[-1]
                
                # Basic filters
                current_price = latest['Close']
                if (current_price < self.parameters['min_price'] or 
                    current_price > self.parameters['max_price']):
                    continue
                
                # Check if we have required indicators
                required_cols = ['RSI', 'SMA_20', 'Volume_Ratio', 'MACD', 'BB_Upper', 'BB_Lower']
                if not all(col in df.columns for col in required_cols):
                    continue
                
                # Momentum signal
                signal = self._check_momentum_signal(df, symbol, current_date)
                if signal:
                    signals.append(signal)
                    
                # Options signal (for high volatility situations)
                options_signal = self._check_options_signal(df, symbol, current_date)
                if options_signal:
                    signals.append(options_signal)
                    
            except Exception as e:
                logger.warning(f"Error generating signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _check_momentum_signal(self, df: pd.DataFrame, symbol: str, current_date: datetime) -> Optional[Dict]:
        """Check for momentum trading opportunity"""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        current_price = latest['Close']
        rsi = latest['RSI']
        volume_ratio = latest['Volume_Ratio']
        sma_20 = latest['SMA_20']
        macd = latest['MACD']
        macd_signal = latest['MACD_Signal']
        
        # Skip if missing data
        if pd.isna(rsi) or pd.isna(volume_ratio) or pd.isna(sma_20):
            return None
        
        # Long signal conditions
        momentum_bullish = (
            current_price > sma_20 and  # Above 20-day SMA
            rsi > self.parameters['rsi_oversold'] and rsi < self.parameters['rsi_overbought'] and  # RSI in range
            volume_ratio > self.parameters['volume_threshold'] and  # High volume
            macd > macd_signal and  # MACD bullish
            current_price > prev['Close']  # Price momentum
        )
        
        if momentum_bullish:
            # Calculate confidence based on signal strength
            confidence = self._calculate_momentum_confidence(df, latest)
            
            if confidence >= self.parameters['min_confidence']:
                return {
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': 100,  # Will be adjusted by position sizing
                    'stop_loss': current_price * (1 - self.parameters['stop_loss_pct']),
                    'take_profit': current_price * (1 + self.parameters['take_profit_pct']),
                    'confidence': confidence,
                    'reason': f'Momentum signal: RSI={rsi:.1f}, Vol={volume_ratio:.1f}x, Price>{sma_20:.2f}'
                }
        
        return None
    
    def _check_options_signal(self, df: pd.DataFrame, symbol: str, current_date: datetime) -> Optional[Dict]:
        """Check for options trading opportunity (high volatility situations)"""
        latest = df.iloc[-1]
        
        current_price = latest['Close']
        rsi = latest['RSI']
        bb_upper = latest['BB_Upper']
        bb_lower = latest['BB_Lower']
        volume_ratio = latest['Volume_Ratio']
        
        # Skip if missing data
        if pd.isna(rsi) or pd.isna(bb_upper) or pd.isna(bb_lower):
            return None
        
        # High volatility call option signal
        high_vol_bullish = (
            current_price > bb_upper and  # Breaking out of Bollinger Bands
            rsi < 80 and  # Not extremely overbought
            volume_ratio > 2.0  # Very high volume
        )
        
        # High volatility put option signal
        high_vol_bearish = (
            current_price < bb_lower and  # Breaking down from Bollinger Bands
            rsi > 20 and  # Not extremely oversold
            volume_ratio > 2.0  # Very high volume
        )
        
        if high_vol_bullish:
            confidence = min(0.8, volume_ratio / 3.0)  # Cap at 0.8
            return {
                'symbol': symbol,
                'action': 'buy_call',
                'quantity': 5,  # Options contracts
                'stop_loss': current_price * 0.9,  # 10% stop for options
                'take_profit': current_price * 1.2,  # 20% target for options
                'confidence': confidence,
                'reason': f'High vol call: Price>{bb_upper:.2f}, Vol={volume_ratio:.1f}x'
            }
        
        if high_vol_bearish:
            confidence = min(0.8, volume_ratio / 3.0)
            return {
                'symbol': symbol,
                'action': 'buy_put',
                'quantity': 5,
                'stop_loss': current_price * 1.1,  # 10% stop for puts
                'take_profit': current_price * 0.8,  # 20% target for puts
                'confidence': confidence,
                'reason': f'High vol put: Price<{bb_lower:.2f}, Vol={volume_ratio:.1f}x'
            }
        
        return None
    
    def _calculate_momentum_confidence(self, df: pd.DataFrame, latest: pd.Series) -> float:
        """Calculate confidence score for momentum signal"""
        # Base confidence from RSI position
        rsi = latest['RSI']
        rsi_confidence = 1.0 - abs(rsi - 50) / 50.0  # Higher confidence near RSI 50
        
        # Volume confirmation
        volume_ratio = latest['Volume_Ratio']
        volume_confidence = min(1.0, volume_ratio / 3.0)  # Scale up to 3x volume
        
        # Price momentum confirmation
        price_change = (latest['Close'] / df.iloc[-5]['Close'] - 1) if len(df) >= 5 else 0
        momentum_confidence = min(1.0, abs(price_change) * 10)  # Scale 10% move to 1.0
        
        # Average the confidence scores
        confidence = (rsi_confidence + volume_confidence + momentum_confidence) / 3.0
        
        return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0

class MeanReversionStrategy(TradingStrategy):
    """
    Mean reversion strategy for oversold/overbought conditions
    """
    
    def __init__(self, name: str = "Mean Reversion Strategy"):
        super().__init__(name)
        self.parameters = {
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'bb_deviation': 0.02,  # 2% outside Bollinger Bands
            'volume_threshold': 1.2,
            'stop_loss_pct': 0.06,
            'take_profit_pct': 0.10,
            'min_confidence': 0.5
        }
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_date: datetime) -> List[Dict]:
        """Generate mean reversion signals"""
        signals = []
        
        for symbol, df in data.items():
            if df.empty or len(df) < 20:
                continue
            
            try:
                signal = self._check_mean_reversion_signal(df, symbol, current_date)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Error in mean reversion signal for {symbol}: {e}")
        
        return signals
    
    def _check_mean_reversion_signal(self, df: pd.DataFrame, symbol: str, current_date: datetime) -> Optional[Dict]:
        """Check for mean reversion opportunity"""
        latest = df.iloc[-1]
        
        current_price = latest['Close']
        rsi = latest['RSI']
        bb_upper = latest['BB_Upper']
        bb_lower = latest['BB_Lower']
        volume_ratio = latest['Volume_Ratio']
        
        if pd.isna(rsi) or pd.isna(bb_upper) or pd.isna(bb_lower):
            return None
        
        # Oversold condition (buy signal)
        oversold = (
            rsi <= self.parameters['rsi_oversold'] and
            current_price <= bb_lower * (1 + self.parameters['bb_deviation']) and
            volume_ratio >= self.parameters['volume_threshold']
        )
        
        # Overbought condition (sell signal)
        overbought = (
            rsi >= self.parameters['rsi_overbought'] and
            current_price >= bb_upper * (1 - self.parameters['bb_deviation']) and
            volume_ratio >= self.parameters['volume_threshold']
        )
        
        if oversold:
            confidence = (self.parameters['rsi_oversold'] - rsi) / self.parameters['rsi_oversold'] + 0.3
            confidence = min(1.0, max(0.1, confidence))
            
            return {
                'symbol': symbol,
                'action': 'buy',
                'quantity': 100,
                'stop_loss': current_price * (1 - self.parameters['stop_loss_pct']),
                'take_profit': current_price * (1 + self.parameters['take_profit_pct']),
                'confidence': confidence,
                'reason': f'Mean reversion buy: RSI={rsi:.1f}, Price={current_price:.2f} vs BB_Low={bb_lower:.2f}'
            }
        
        if overbought:
            confidence = (rsi - self.parameters['rsi_overbought']) / (100 - self.parameters['rsi_overbought']) + 0.3
            confidence = min(1.0, max(0.1, confidence))
            
            return {
                'symbol': symbol,
                'action': 'sell',
                'quantity': 100,
                'stop_loss': current_price * (1 + self.parameters['stop_loss_pct']),
                'take_profit': current_price * (1 - self.parameters['take_profit_pct']),
                'confidence': confidence,
                'reason': f'Mean reversion sell: RSI={rsi:.1f}, Price={current_price:.2f} vs BB_High={bb_upper:.2f}'
            }
        
        return None

class BreakoutStrategy(TradingStrategy):
    """
    Breakout strategy targeting price breakouts with volume confirmation
    """
    
    def __init__(self, name: str = "Breakout Strategy"):
        super().__init__(name)
        self.parameters = {
            'breakout_period': 20,  # Look for 20-day highs/lows
            'volume_multiplier': 2.0,  # Volume must be 2x average
            'atr_multiplier': 1.5,  # ATR-based stops
            'min_price_change': 0.03,  # Minimum 3% breakout
            'stop_loss_atr': 2.0,  # 2x ATR stop loss
            'take_profit_ratio': 2.0,  # 2:1 reward/risk ratio
            'min_confidence': 0.6
        }
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_date: datetime) -> List[Dict]:
        """Generate breakout signals"""
        signals = []
        
        for symbol, df in data.items():
            if df.empty or len(df) < self.parameters['breakout_period']:
                continue
            
            try:
                signal = self._check_breakout_signal(df, symbol, current_date)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Error in breakout signal for {symbol}: {e}")
        
        return signals
    
    def _check_breakout_signal(self, df: pd.DataFrame, symbol: str, current_date: datetime) -> Optional[Dict]:
        """Check for breakout opportunity"""
        latest = df.iloc[-1]
        lookback_period = self.parameters['breakout_period']
        
        current_price = latest['Close']
        volume_ratio = latest['Volume_Ratio']
        
        # Calculate ATR for position sizing
        df_copy = df.copy()
        df_copy['tr'] = np.maximum(
            df_copy['High'] - df_copy['Low'],
            np.maximum(
                abs(df_copy['High'] - df_copy['Close'].shift(1)),
                abs(df_copy['Low'] - df_copy['Close'].shift(1))
            )
        )
        atr = df_copy['tr'].rolling(window=14).mean().iloc[-1]
        
        if pd.isna(atr) or pd.isna(volume_ratio):
            return None
        
        # Recent price data for breakout detection
        recent_data = df.tail(lookback_period)
        recent_high = recent_data['High'].max()
        recent_low = recent_data['Low'].min()
        
        # Upside breakout
        upside_breakout = (
            current_price > recent_high and
            (current_price / recent_data['Close'].iloc[0] - 1) >= self.parameters['min_price_change'] and
            volume_ratio >= self.parameters['volume_multiplier']
        )
        
        # Downside breakout (for short positions)
        downside_breakout = (
            current_price < recent_low and
            (recent_data['Close'].iloc[0] / current_price - 1) >= self.parameters['min_price_change'] and
            volume_ratio >= self.parameters['volume_multiplier']
        )
        
        if upside_breakout:
            stop_loss = current_price - (atr * self.parameters['stop_loss_atr'])
            take_profit = current_price + (current_price - stop_loss) * self.parameters['take_profit_ratio']
            
            # Confidence based on volume and price momentum
            price_momentum = (current_price / recent_data['Close'].iloc[0] - 1) * 10
            volume_strength = min(1.0, volume_ratio / 4.0)
            confidence = (price_momentum + volume_strength) / 2.0
            confidence = min(1.0, max(0.1, confidence))
            
            if confidence >= self.parameters['min_confidence']:
                return {
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': 100,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'reason': f'Upside breakout: Price={current_price:.2f} > High={recent_high:.2f}, Vol={volume_ratio:.1f}x'
                }
        
        if downside_breakout:
            stop_loss = current_price + (atr * self.parameters['stop_loss_atr'])
            take_profit = current_price - (stop_loss - current_price) * self.parameters['take_profit_ratio']
            
            price_momentum = (recent_data['Close'].iloc[0] / current_price - 1) * 10
            volume_strength = min(1.0, volume_ratio / 4.0)
            confidence = (price_momentum + volume_strength) / 2.0
            confidence = min(1.0, max(0.1, confidence))
            
            if confidence >= self.parameters['min_confidence']:
                return {
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': 100,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'reason': f'Downside breakout: Price={current_price:.2f} < Low={recent_low:.2f}, Vol={volume_ratio:.1f}x'
                }
        
        return None

# Factory function to create strategies
def create_strategy(strategy_name: str) -> TradingStrategy:
    """Create a trading strategy by name"""
    strategies = {
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'breakout': BreakoutStrategy
    }
    
    if strategy_name.lower() not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name.lower()]()

def get_available_strategies() -> List[str]:
    """Get list of available strategy names"""
    return ['momentum', 'mean_reversion', 'breakout']
