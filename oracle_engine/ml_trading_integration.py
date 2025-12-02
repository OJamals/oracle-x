"""
ML Trading Integration Module
Integrates the ML prediction engine with the trading system
Provides unified interface for ML-powered trading decisions
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Import components
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
from sentiment.sentiment_engine import AdvancedSentimentEngine
from oracle_engine.ensemble_ml_engine import (
    EnsemblePredictionEngine, PredictionType, PredictionResult, create_prediction_engine
)
from backtest_tracker.comprehensive_backtest import BacktestEngine

logger = logging.getLogger(__name__)

@dataclass
class MLTradingSignal:
    """Trading signal enhanced with ML predictions"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    
    # ML predictions
    price_direction_prediction: Optional[PredictionResult] = None
    price_target_prediction: Optional[PredictionResult] = None
    volatility_prediction: Optional[PredictionResult] = None
    
    # Signal strength breakdown
    technical_strength: float = 0.0
    sentiment_strength: float = 0.0
    ml_strength: float = 0.0
    
    # Risk metrics
    risk_score: float = 0.5  # 0.0 (low) to 1.0 (high)
    position_size_factor: float = 1.0  # Multiplier for position sizing
    
    # Context
    market_regime: str = "normal"
    data_quality: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    features_used: List[str] = field(default_factory=list)
    model_weights: Dict[str, float] = field(default_factory=dict)
    prediction_horizon_days: int = 5

@dataclass
class MLTradingConfig:
    """Configuration for ML-enhanced trading"""
    # Signal generation
    min_confidence_threshold: float = 0.6
    min_data_quality_threshold: float = 0.7
    
    # Prediction horizons
    direction_horizon_days: int = 5
    target_horizon_days: int = 10
    volatility_horizon_days: int = 5
    
    # Risk management
    max_risk_score: float = 0.8
    min_position_size_factor: float = 0.1
    max_position_size_factor: float = 2.0
    
    # Model management
    retrain_frequency_days: int = 30
    min_training_samples: int = 500
    
    # Signal combination weights
    technical_weight: float = 0.3
    sentiment_weight: float = 0.2
    ml_weight: float = 0.5

class MLTradingOrchestrator:
    """
    Main orchestrator for ML-enhanced trading
    Combines technical analysis, sentiment, and ML predictions
    """
    
    def __init__(self, 
                 data_orchestrator: DataFeedOrchestrator,
                 sentiment_engine: AdvancedSentimentEngine,
                 config: MLTradingConfig = None):
        
        self.data_orchestrator = data_orchestrator
        self.sentiment_engine = sentiment_engine
        self.config = config or MLTradingConfig()
        
        # Initialize ML prediction engine
        self.ml_engine = create_prediction_engine(
            data_orchestrator, sentiment_engine
        )
        
        # Initialize backtesting for validation
        self.backtest_engine = BacktestEngine()
        
        # State
        self.last_training_time = None
        self.signal_history = []
        self.performance_metrics = {}
        
        logger.info("ML Trading Orchestrator initialized")
    
    def generate_trading_signal(self, symbol: str) -> Optional[MLTradingSignal]:
        """
        Generate comprehensive trading signal with ML predictions
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            ML-enhanced trading signal or None if insufficient data
        """
        try:
            logger.info(f"Generating ML trading signal for {symbol}")
            
            # Get ML predictions
            direction_pred = self.ml_engine.predict(
                symbol, PredictionType.PRICE_DIRECTION, 
                self.config.direction_horizon_days
            )
            
            target_pred = self.ml_engine.predict(
                symbol, PredictionType.PRICE_TARGET,
                self.config.target_horizon_days
            )
            
            volatility_pred = self.ml_engine.predict(
                symbol, PredictionType.VOLATILITY,
                self.config.volatility_horizon_days
            )
            
            # Get current market data for technical analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            try:
                data = self.data_orchestrator.get_historical_data(
                    symbol, start_date, end_date
                )
            except AttributeError:
                # Fallback for different API
                data = self._get_market_data_fallback(symbol, start_date, end_date)
            
            if data.empty:
                logger.warning(f"No market data available for {symbol}")
                return None
            
            # Get sentiment analysis
            try:
                sentiment = self.sentiment_engine.get_sentiment_analysis(symbol)
            except AttributeError:
                # Fallback for different API
                sentiment = self._get_sentiment_fallback(symbol)
            
            # Analyze data quality
            data_quality = self._assess_data_quality(data, direction_pred, sentiment)
            
            if data_quality < self.config.min_data_quality_threshold:
                logger.warning(f"Data quality too low for {symbol}: {data_quality}")
                return None
            
            # Calculate technical indicators
            technical_signal = self._calculate_technical_signal(data)
            
            # Calculate sentiment signal
            sentiment_signal = self._calculate_sentiment_signal(sentiment)
            
            # Calculate ML signal
            ml_signal = self._calculate_ml_signal(direction_pred, target_pred)
            
            # Combine signals
            combined_signal = self._combine_signals(
                technical_signal, sentiment_signal, ml_signal
            )
            
            # Calculate risk metrics
            risk_score = self._calculate_risk_score(
                data, direction_pred, volatility_pred
            )
            
            # Determine position sizing
            position_size_factor = self._calculate_position_size(
                combined_signal['confidence'], risk_score, volatility_pred
            )
            
            # Create final trading signal
            signal = MLTradingSignal(
                symbol=symbol,
                signal_type=combined_signal['type'],
                confidence=combined_signal['confidence'],
                price_direction_prediction=direction_pred,
                price_target_prediction=target_pred,
                volatility_prediction=volatility_pred,
                technical_strength=technical_signal['strength'],
                sentiment_strength=sentiment_signal['strength'],
                ml_strength=ml_signal['strength'],
                risk_score=risk_score,
                position_size_factor=position_size_factor,
                market_regime=direction_pred.market_regime if direction_pred else "normal",
                data_quality=data_quality,
                features_used=self._get_features_used(direction_pred, target_pred),
                model_weights=self._get_model_weights(direction_pred, target_pred),
                prediction_horizon_days=self.config.direction_horizon_days
            )
            
            # Store signal history
            self.signal_history.append(signal)
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(days=30)
            self.signal_history = [
                s for s in self.signal_history if s.timestamp > cutoff_time
            ]
            
            logger.info(f"Generated signal for {symbol}: {signal.signal_type} "
                       f"(confidence: {signal.confidence:.3f}, risk: {signal.risk_score:.3f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate trading signal for {symbol}: {e}")
            return None
    
    def _get_market_data_fallback(self, symbol: str, start_date: datetime, 
                                 end_date: datetime) -> pd.DataFrame:
        """Fallback method to get market data"""
        # This would be implemented based on the actual data orchestrator API
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def _get_sentiment_fallback(self, symbol: str) -> Dict:
        """Fallback method to get sentiment data"""
        return {
            'overall_sentiment': 0.0,
            'confidence': 0.5,
            'momentum': 0.0,
            'article_count': 0
        }
    
    def _assess_data_quality(self, data: pd.DataFrame, 
                           direction_pred: Optional[PredictionResult],
                           sentiment: Dict) -> float:
        """Assess overall data quality for signal generation"""
        quality_factors = []
        
        # Market data quality
        if not data.empty:
            data_completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            data_recency = min(1.0, len(data) / 50.0)  # Prefer 50+ days of data
            quality_factors.extend([data_completeness, data_recency])
        
        # ML prediction quality
        if direction_pred:
            quality_factors.append(direction_pred.data_quality_score)
        
        # Sentiment quality
        if sentiment:
            sentiment_quality = sentiment.get('confidence', 0.5)
            quality_factors.append(sentiment_quality)
        
        # Overall quality as average
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _calculate_technical_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical analysis signal"""
        if len(data) < 20:
            return {'type': 'HOLD', 'strength': 0.0}
        
        current_price = data['Close'].iloc[-1]
        
        # Simple moving average signals
        sma_20 = data['Close'].rolling(20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
        
        # RSI signal
        rsi = data.get('RSI', pd.Series([50])).iloc[-1]
        
        # Trend strength
        trend_strength = (current_price / sma_20 - 1) * 100  # Percentage above/below SMA
        
        # Signal logic
        signals = []
        
        if current_price > sma_20 and sma_20 > sma_50:
            signals.append(1)  # Bullish
        elif current_price < sma_20 and sma_20 < sma_50:
            signals.append(-1)  # Bearish
        else:
            signals.append(0)  # Neutral
        
        if rsi < 30:
            signals.append(1)  # Oversold
        elif rsi > 70:
            signals.append(-1)  # Overbought
        else:
            signals.append(0)
        
        # Combine signals
        avg_signal = np.mean(signals)
        
        if avg_signal > 0.3:
            signal_type = 'BUY'
        elif avg_signal < -0.3:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        
        strength = min(1.0, abs(avg_signal))
        
        return {
            'type': signal_type,
            'strength': strength,
            'trend_strength': trend_strength,
            'rsi': rsi
        }
    
    def _calculate_sentiment_signal(self, sentiment: Dict) -> Dict[str, Any]:
        """Calculate sentiment-based signal"""
        if not sentiment:
            return {'type': 'HOLD', 'strength': 0.0}
        
        sentiment_score = sentiment.get('overall_sentiment', 0.0)
        confidence = sentiment.get('confidence', 0.5)
        
        # Sentiment thresholds
        if sentiment_score > 0.2 and confidence > 0.6:
            signal_type = 'BUY'
        elif sentiment_score < -0.2 and confidence > 0.6:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        
        strength = abs(sentiment_score) * confidence
        
        return {
            'type': signal_type,
            'strength': strength,
            'sentiment_score': sentiment_score,
            'confidence': confidence
        }
    
    def _calculate_ml_signal(self, direction_pred: Optional[PredictionResult],
                           target_pred: Optional[PredictionResult]) -> Dict[str, Any]:
        """Calculate ML-based signal"""
        if not direction_pred:
            return {'type': 'HOLD', 'strength': 0.0}
        
        # Direction prediction signal
        direction_confidence = direction_pred.confidence
        direction_prediction = direction_pred.prediction
        
        # Price target signal
        target_signal = 0.0
        if target_pred and target_pred.confidence > 0.5:
            target_signal = target_pred.prediction  # Expected return
        
        # Combine ML signals
        if direction_prediction > 0.7 and direction_confidence > 0.6:
            signal_type = 'BUY'
        elif direction_prediction < 0.3 and direction_confidence > 0.6:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        
        # Strength based on confidence and prediction magnitude
        strength = direction_confidence * abs(direction_prediction - 0.5) * 2
        
        return {
            'type': signal_type,
            'strength': strength,
            'direction_prediction': direction_prediction,
            'direction_confidence': direction_confidence,
            'target_signal': target_signal
        }
    
    def _combine_signals(self, technical: Dict, sentiment: Dict, ml: Dict) -> Dict[str, Any]:
        """Combine technical, sentiment, and ML signals"""
        # Convert signal types to scores
        def signal_to_score(signal_type: str) -> float:
            return {'BUY': 1.0, 'SELL': -1.0, 'HOLD': 0.0}.get(signal_type, 0.0)
        
        # Weighted combination
        technical_score = signal_to_score(technical['type']) * technical['strength'] * self.config.technical_weight
        sentiment_score = signal_to_score(sentiment['type']) * sentiment['strength'] * self.config.sentiment_weight
        ml_score = signal_to_score(ml['type']) * ml['strength'] * self.config.ml_weight
        
        combined_score = technical_score + sentiment_score + ml_score
        
        # Determine final signal
        if combined_score > 0.3:
            final_type = 'BUY'
        elif combined_score < -0.3:
            final_type = 'SELL'
        else:
            final_type = 'HOLD'
        
        # Combined confidence
        confidence = min(1.0, abs(combined_score))
        
        return {
            'type': final_type,
            'confidence': confidence,
            'combined_score': combined_score
        }
    
    def _calculate_risk_score(self, data: pd.DataFrame,
                            direction_pred: Optional[PredictionResult],
                            volatility_pred: Optional[PredictionResult]) -> float:
        """Calculate risk score for the position"""
        risk_factors = []
        
        # Historical volatility risk
        if len(data) >= 20:
            returns = data['Close'].pct_change().dropna()
            historical_vol = returns.rolling(20).std().iloc[-1]
            risk_factors.append(min(1.0, historical_vol * 50))  # Scale volatility
        
        # ML prediction uncertainty risk
        if direction_pred:
            uncertainty_risk = direction_pred.uncertainty
            risk_factors.append(uncertainty_risk)
        
        # Volatility prediction risk
        if volatility_pred and volatility_pred.confidence > 0.5:
            vol_risk = volatility_pred.prediction * 2  # Scale predicted volatility
            risk_factors.append(min(1.0, vol_risk))
        
        # Market regime risk
        if direction_pred:
            regime_risk = {
                'normal': 0.3,
                'volatile': 0.8,
                'trending': 0.4,
                'sideways': 0.5
            }.get(direction_pred.market_regime, 0.5)
            risk_factors.append(regime_risk)
        
        # Average risk score
        return np.mean(risk_factors) if risk_factors else 0.5
    
    def _calculate_position_size(self, confidence: float, risk_score: float,
                               volatility_pred: Optional[PredictionResult]) -> float:
        """Calculate position size factor based on confidence and risk"""
        # Base size from confidence
        base_size = confidence
        
        # Risk adjustment
        risk_adjustment = 1.0 - risk_score
        
        # Volatility adjustment
        vol_adjustment = 1.0
        if volatility_pred and volatility_pred.confidence > 0.5:
            # Reduce size in high volatility
            vol_adjustment = max(0.5, 1.0 - volatility_pred.prediction)
        
        # Combined position size
        position_size = base_size * risk_adjustment * vol_adjustment
        
        # Apply constraints
        position_size = max(self.config.min_position_size_factor, position_size)
        position_size = min(self.config.max_position_size_factor, position_size)
        
        return position_size
    
    def _get_features_used(self, *predictions) -> List[str]:
        """Get list of features used in predictions"""
        features = set()
        for pred in predictions:
            if pred and pred.feature_importance:
                features.update(pred.feature_importance.keys())
        return list(features)
    
    def _get_model_weights(self, *predictions) -> Dict[str, float]:
        """Get model weights from predictions"""
        weights = {}
        for pred in predictions:
            if pred and pred.model_contributions:
                weights.update(pred.model_contributions)
        return weights
    
    def train_models(self, symbols: List[str], lookback_days: int = 252) -> Dict[str, Any]:
        """Train ML models with recent data"""
        logger.info(f"Training ML models for {len(symbols)} symbols")
        
        result = self.ml_engine.train_models(symbols, lookback_days)
        
        if result:
            self.last_training_time = datetime.now()
            logger.info("ML model training completed")
        
        return result
    
    def should_retrain_models(self) -> bool:
        """Check if models should be retrained"""
        if not self.last_training_time:
            return True
        
        days_since_training = (datetime.now() - self.last_training_time).days
        return days_since_training >= self.config.retrain_frequency_days
    
    def get_signal_performance(self) -> Dict[str, Any]:
        """Get performance metrics for generated signals"""
        if not self.signal_history:
            return {}
        
        recent_signals = [s for s in self.signal_history 
                         if s.timestamp > datetime.now() - timedelta(days=7)]
        
        if not recent_signals:
            return {}
        
        # Calculate basic statistics
        total_signals = len(recent_signals)
        buy_signals = len([s for s in recent_signals if s.signal_type == 'BUY'])
        sell_signals = len([s for s in recent_signals if s.signal_type == 'SELL'])
        hold_signals = len([s for s in recent_signals if s.signal_type == 'HOLD'])
        
        avg_confidence = np.mean([s.confidence for s in recent_signals])
        avg_risk = np.mean([s.risk_score for s in recent_signals])
        avg_position_size = np.mean([s.position_size_factor for s in recent_signals])
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'average_confidence': avg_confidence,
            'average_risk_score': avg_risk,
            'average_position_size_factor': avg_position_size,
            'last_signal_time': recent_signals[-1].timestamp.isoformat()
        }

# ============================================================================
# Public Interface Functions
# ============================================================================

def create_ml_trading_orchestrator(
    data_orchestrator: DataFeedOrchestrator,
    sentiment_engine: AdvancedSentimentEngine,
    config: MLTradingConfig = None
) -> MLTradingOrchestrator:
    """Create and initialize ML trading orchestrator"""
    return MLTradingOrchestrator(data_orchestrator, sentiment_engine, config)

def get_default_ml_config() -> MLTradingConfig:
    """Get default ML trading configuration"""
    return MLTradingConfig()
