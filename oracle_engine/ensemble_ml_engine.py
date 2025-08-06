"""
Ensemble ML Prediction Engine - Main orchestrator for the ML system
Combines multiple models for robust predictions with uncertainty quantification
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our components
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
from data_feeds.advanced_sentiment import AdvancedSentimentEngine

try:
    # Import from ml_prediction_engine without conflicts
    import oracle_engine.ml_prediction_engine as ml_engine
    ML_ENGINE_AVAILABLE = True
except ImportError as e:
    ML_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Define our own types to avoid conflicts
from enum import Enum

class PredictionType(Enum):
    PRICE_DIRECTION = "price_direction"
    PRICE_TARGET = "price_target"
    VOLATILITY = "volatility"
    OPTION_VALUE = "option_value"

class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

@dataclass
class PredictionResult:
    symbol: str
    prediction_type: PredictionType
    prediction: float
    confidence: float
    uncertainty: float
    feature_importance: Dict[str, float]
    model_contributions: Dict[str, float]
    timestamp: datetime
    horizon_days: int
    data_quality_score: float = 0.0
    market_regime: str = "normal"
    prediction_context: Dict[str, Any] = field(default_factory=dict)

# Define base classes and functions
class BaseMLModel:
    def __init__(self):
        self.is_trained = False
        self.performance = None
    
    def train(self, X, y):
        return {"status": "success"}
    
    def predict(self, X):
        return np.array([0.0]), np.array([0.0])
    
    def update(self, X, y):
        return True
    
    def get_feature_importance(self):
        return {}

class FeatureEngineer:
    def engineer_features(self, data, sentiment_data=None, target_horizon_days=5):
        """
        Engineer features including technical indicators, sentiment features, and target variables
        """
        if not data:
            return pd.DataFrame()
        
        # Combine data from all symbols
        all_features = []
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            symbol_df = df.copy()
            symbol_df['symbol'] = symbol
            
            # Handle both 'close' and 'Close' column names
            close_col = 'Close' if 'Close' in symbol_df.columns else 'close'
            high_col = 'High' if 'High' in symbol_df.columns else 'high'
            low_col = 'Low' if 'Low' in symbol_df.columns else 'low'
            volume_col = 'Volume' if 'Volume' in symbol_df.columns else 'volume'
            
            if close_col not in symbol_df.columns:
                continue
                
            # Technical indicators
            symbol_df['returns'] = symbol_df[close_col].pct_change()
            symbol_df['volatility'] = symbol_df['returns'].rolling(20).std()
            symbol_df['sma_20'] = symbol_df[close_col].rolling(20).mean()
            symbol_df['sma_50'] = symbol_df[close_col].rolling(50).mean()
            symbol_df['ema_12'] = symbol_df[close_col].ewm(span=12).mean()
            symbol_df['ema_26'] = symbol_df[close_col].ewm(span=26).mean()
            
            # RSI calculation (simplified)
            delta = symbol_df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            symbol_df['rsi'] = 100 - (100 / (1 + rs))
            
            # Price ratios
            symbol_df['price_to_sma20'] = symbol_df[close_col] / symbol_df['sma_20']
            symbol_df['price_to_sma50'] = symbol_df[close_col] / symbol_df['sma_50']
            
            # Volume features if available
            if volume_col in symbol_df.columns:
                symbol_df['volume_sma'] = symbol_df[volume_col].rolling(20).mean()
                symbol_df['volume_ratio'] = symbol_df[volume_col] / symbol_df['volume_sma']
            else:
                symbol_df['volume_ratio'] = 1.0
            
            # High/Low features if available
            if high_col in symbol_df.columns and low_col in symbol_df.columns:
                symbol_df['high_low_ratio'] = symbol_df[high_col] / symbol_df[low_col]
                symbol_df['close_position'] = (symbol_df[close_col] - symbol_df[low_col]) / (symbol_df[high_col] - symbol_df[low_col])
            else:
                symbol_df['high_low_ratio'] = 1.0
                symbol_df['close_position'] = 0.5
            
            # Sentiment features
            if sentiment_data and symbol in sentiment_data:
                sentiment = sentiment_data[symbol]
                if sentiment:
                    symbol_df['sentiment_score'] = sentiment.overall_sentiment
                    symbol_df['sentiment_confidence'] = sentiment.confidence
                    symbol_df['sentiment_quality'] = sentiment.quality_score
                    symbol_df['bullish_ratio'] = sentiment.bullish_mentions / max(sentiment.sample_size, 1)
                    symbol_df['bearish_ratio'] = sentiment.bearish_mentions / max(sentiment.sample_size, 1)
                else:
                    symbol_df['sentiment_score'] = 0.0
                    symbol_df['sentiment_confidence'] = 0.0
                    symbol_df['sentiment_quality'] = 0.0
                    symbol_df['bullish_ratio'] = 0.0
                    symbol_df['bearish_ratio'] = 0.0
            else:
                symbol_df['sentiment_score'] = 0.0
                symbol_df['sentiment_confidence'] = 0.0
                symbol_df['sentiment_quality'] = 0.0
                symbol_df['bullish_ratio'] = 0.0
                symbol_df['bearish_ratio'] = 0.0
            
            # Create target variables for different horizons
            horizons = [1, 5, 10, 20]
            
            for horizon in horizons:
                if len(symbol_df) > horizon:
                    # Future returns for regression
                    future_prices = symbol_df[close_col].shift(-horizon)
                    current_prices = symbol_df[close_col]
                    returns = (future_prices / current_prices - 1)
                    symbol_df[f'target_return_{horizon}d'] = returns
                    
                    # Price direction for classification (1 = up, 0 = down)
                    symbol_df[f'target_direction_{horizon}d'] = (returns > 0).astype(int)
                else:
                    symbol_df[f'target_return_{horizon}d'] = np.nan
                    symbol_df[f'target_direction_{horizon}d'] = np.nan
            
            # Add timestamp if not present
            if 'timestamp' not in symbol_df.columns:
                symbol_df['timestamp'] = pd.to_datetime(symbol_df.index)
            
            all_features.append(symbol_df)
        
        if not all_features:
            return pd.DataFrame()
        
        # Combine all symbols
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Fill NaN values (but keep target NaNs for proper filtering)
        feature_cols = [col for col in combined_df.columns 
                       if not col.startswith('target_') and col not in ['symbol', 'timestamp']]
        
        for col in feature_cols:
            if combined_df[col].dtype in ['float64', 'int64']:
                combined_df[col] = combined_df[col].fillna(0)
        
        return combined_df

def create_ml_model(model_type, prediction_type, **kwargs):
    """Create a model instance"""
    if ML_ENGINE_AVAILABLE:
        try:
            # Import ml_engine here to avoid unbound variable issues
            import oracle_engine.ml_prediction_engine as ml_engine
            # Convert our types to ml_engine types
            ml_model_type = getattr(ml_engine.ModelType, model_type.name)
            ml_prediction_type = getattr(ml_engine.PredictionType, prediction_type.name)
            return ml_engine.create_ml_model(ml_model_type, ml_prediction_type, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to create ML model: {e}")
    
    # Fallback to our basic model
    return BaseMLModel()

def get_available_models():
    """Get list of available models"""
    if ML_ENGINE_AVAILABLE:
        try:
            import oracle_engine.ml_prediction_engine as ml_engine
            return ml_engine.get_available_models()
        except Exception as e:
            logger.warning(f"Failed to get available models: {e}")
    
    return [ModelType.RANDOM_FOREST, ModelType.XGBOOST]

logger = logging.getLogger(__name__)

class EnsemblePredictionEngine:
    """
    Main ML prediction engine that orchestrates multiple models
    Provides unified interface for training, prediction, and model management
    """
    
    def __init__(self, data_orchestrator: DataFeedOrchestrator, 
                 sentiment_engine: AdvancedSentimentEngine):
        self.data_orchestrator = data_orchestrator
        self.sentiment_engine = sentiment_engine
        
        # Model management
        self.models: Dict[str, Any] = {}  # Use Any to avoid type conflicts
        self.model_weights: Dict[str, float] = {}
        self.ensemble_performance = {}
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer() if ML_ENGINE_AVAILABLE else None
        
        # Configuration
        self.prediction_horizons = [1, 5, 10, 20]  # Days
        self.model_configs = self._get_default_model_configs()
        
        # State management
        self._lock = threading.Lock()
        self.last_training_time = None
        self.prediction_cache = {}
        self.performance_history = []
        
        # Initialize models if available
        if ML_ENGINE_AVAILABLE:
            self._initialize_models()
        else:
            logger.warning("ML engine not fully available. Using fallback prediction methods.")
    
    def _get_default_model_configs(self) -> Dict[str, Dict]:
        """Get default configuration for each model type"""
        return {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            },
            'neural_network': {
                'hidden_size': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'learning_rate': 0.001
            }
        }
    
    def _initialize_models(self):
        """Initialize all available ML models"""
        if not ML_ENGINE_AVAILABLE:
            return
        
        available_models = get_available_models()
        logger.info(f"Available ML models: {[m.value for m in available_models]}")
        
        for prediction_type in [PredictionType.PRICE_DIRECTION, PredictionType.PRICE_TARGET]:
            for model_type in available_models:
                model_key = None  # Initialize to avoid unbound variable
                try:
                    model_key = f"{model_type.value}_{prediction_type.value}"
                    config = self.model_configs.get(model_type.value, {})
                    
                    model = create_ml_model(model_type, prediction_type, **config)
                    self.models[model_key] = model
                    self.model_weights[model_key] = 1.0  # Equal weight initially
                    
                    logger.info(f"Initialized model: {model_key}")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize {model_key or 'unknown'}: {e}")
    
    def train_models(self, symbols: List[str], 
                    lookback_days: int = 252,
                    update_existing: bool = True) -> Dict[str, Any]:
        """
        Train all models on historical data
        
        Args:
            symbols: List of symbols to train on
            lookback_days: How many days of history to use
            update_existing: Whether to update existing models or retrain
            
        Returns:
            Training results and performance metrics
        """
        logger.info(f"Training models on {len(symbols)} symbols with {lookback_days} days of data")
        
        if not ML_ENGINE_AVAILABLE:
            return self._fallback_training(symbols, lookback_days)
        
        training_results = {}
        
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            historical_data = {}
            sentiment_data = {}
            
            for symbol in symbols:
                try:
                    # Get price data using the correct method name
                    market_data = self.data_orchestrator.get_market_data(
                        symbol, period="1y", interval="1d"
                    )
                    if market_data and not market_data.data.empty:
                        historical_data[symbol] = market_data.data
                    
                    # Get sentiment data with actual texts
                    raw_sentiment_data = self.data_orchestrator.get_sentiment_data(symbol)
                    
                    # Extract texts from sentiment data for analysis
                    all_texts = []
                    all_sources = []
                    for source_name, sentiment_data_obj in raw_sentiment_data.items():
                        if (hasattr(sentiment_data_obj, 'raw_data') and 
                            sentiment_data_obj.raw_data and 
                            'sample_texts' in sentiment_data_obj.raw_data):
                            texts = sentiment_data_obj.raw_data['sample_texts']
                            if texts:
                                all_texts.extend(texts)
                                all_sources.extend([source_name] * len(texts))
                    
                    # Get advanced sentiment analysis using actual texts
                    if all_texts:
                        sentiment = self.sentiment_engine.get_symbol_sentiment_summary(symbol, all_texts, all_sources)
                        if sentiment:
                            sentiment_data[symbol] = sentiment
                    
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue
            
            if not historical_data:
                logger.error("No historical data available for training")
                return {}
            
            # Engineer features
            if self.feature_engineer:
                features_df = self.feature_engineer.engineer_features(
                    historical_data, sentiment_data
                )
            else:
                # Create basic features if no feature engineer available
                features_df = pd.DataFrame()
                for symbol, data in historical_data.items():
                    df = data.copy()
                    df['symbol'] = symbol
                    close_col = 'Close' if 'Close' in df.columns else 'close'
                    df['returns'] = df[close_col].pct_change()
                    df['volatility'] = df['returns'].rolling(20).std()
                    features_df = pd.concat([features_df, df], ignore_index=True)
            
            if features_df.empty:
                logger.error("No features could be engineered")
                return {}
            
            logger.info(f"Engineered {len(features_df)} feature samples")
            
            # Train models for each prediction type and horizon
            for prediction_type in [PredictionType.PRICE_DIRECTION, PredictionType.PRICE_TARGET]:
                for horizon in self.prediction_horizons:
                    target_col = f"target_{'direction' if prediction_type == PredictionType.PRICE_DIRECTION else 'return'}_{horizon}d"
                    
                    if target_col not in features_df.columns:
                        continue
                    
                    # Prepare training data
                    feature_cols = [col for col in features_df.columns 
                                  if not col.startswith('target_') and 
                                     col not in ['symbol', 'timestamp']]
                    
                    X = features_df[feature_cols].copy()
                    y = features_df[target_col].copy()
                    
                    # Remove NaN values
                    mask = ~(X.isna().any(axis=1) | y.isna())
                    X = X[mask]
                    y = y[mask]
                    
                    if len(X) < 50:  # Need minimum samples
                        logger.warning(f"Insufficient samples for {prediction_type.value}_{horizon}d: {len(X)}")
                        continue
                    
                    # Train all models for this prediction type
                    horizon_results = self._train_models_for_target(
                        X, y, prediction_type, horizon, update_existing
                    )
                    
                    training_results[f"{prediction_type.value}_{horizon}d"] = horizon_results
            
            # Update ensemble weights based on validation performance
            self._update_ensemble_weights()
            
            self.last_training_time = datetime.now()
            
            logger.info("Model training completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {}
    
    def _train_models_for_target(self, X: pd.DataFrame, y: pd.Series,
                                prediction_type: PredictionType, horizon: int,
                                update_existing: bool) -> Dict[str, Any]:
        """Train all models for a specific target variable"""
        results = {}
        
        for model_key, model in self.models.items():
            if prediction_type.value not in model_key:
                continue
            
            try:
                logger.info(f"Training {model_key} for {horizon}d horizon")
                
                if update_existing and model.is_trained:
                    # Try incremental update first
                    if model.update(X, y):
                        logger.info(f"Successfully updated {model_key}")
                        continue
                
                # Full training
                result = model.train(X, y)
                results[model_key] = result
                
                logger.info(f"Trained {model_key}: {result.get('validation_metrics', {})}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_key}: {e}")
                results[model_key] = {'error': str(e)}
        
        return results
    
    def _update_ensemble_weights(self):
        """Update model weights based on validation performance"""
        # Simple performance-based weighting
        for model_key, model in self.models.items():
            if model.is_trained:
                performance = model.performance
                
                if performance:
                    # Weight based on accuracy for classification, inverse MSE for regression
                    if hasattr(performance, 'prediction_type') and performance.prediction_type == PredictionType.PRICE_DIRECTION:
                        weight = max(0.1, getattr(performance, 'accuracy', 0.5))
                    else:
                        mse = getattr(performance, 'mse', 1.0)
                        weight = max(0.1, 1.0 / (1.0 + mse))
                else:
                    weight = 0.5  # Default weight if no performance data
                
                self.model_weights[model_key] = weight
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for key in self.model_weights:
                self.model_weights[key] /= total_weight
        
        logger.info(f"Updated model weights: {self.model_weights}")
    
    def predict(self, symbol: str, prediction_type: PredictionType,
                horizon_days: int = 5) -> Optional[PredictionResult]:
        """
        Make ensemble prediction for a symbol
        
        Args:
            symbol: Symbol to predict
            prediction_type: Type of prediction
            horizon_days: Prediction horizon in days
            
        Returns:
            Prediction result with confidence and uncertainty
        """
        cache_key = f"{symbol}_{prediction_type.value}_{horizon_days}"
        
        # Check cache (valid for 1 hour)
        if cache_key in self.prediction_cache:
            cached_result, cache_time = self.prediction_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < 3600:
                return cached_result
        
        try:
            if ML_ENGINE_AVAILABLE:
                result = self._ml_predict(symbol, prediction_type, horizon_days)
            else:
                result = self._fallback_predict(symbol, prediction_type, horizon_days)
            
            # Cache result
            if result:
                self.prediction_cache[cache_key] = (result, datetime.now())
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None
    
    def _ml_predict(self, symbol: str, prediction_type: PredictionType,
                   horizon_days: int) -> Optional[PredictionResult]:
        """Make ML-based prediction"""
        # Get current data for features
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # Need history for indicators
        
        try:
            # Get data using the correct method
            market_data = self.data_orchestrator.get_market_data(
                symbol, period="60d", interval="1d"
            )
            
            if not market_data or market_data.data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            data = market_data.data
            
            # Get sentiment data from data orchestrator (includes actual tweets/posts)
            raw_sentiment_data = self.data_orchestrator.get_sentiment_data(symbol)
            
            # Extract texts from sentiment data for analysis
            all_texts = []
            all_sources = []
            for source_name, sentiment_data_obj in raw_sentiment_data.items():
                if (hasattr(sentiment_data_obj, 'raw_data') and 
                    sentiment_data_obj.raw_data and 
                    'sample_texts' in sentiment_data_obj.raw_data):
                    texts = sentiment_data_obj.raw_data['sample_texts']
                    if texts:
                        all_texts.extend(texts)
                        all_sources.extend([source_name] * len(texts))
            
            # Get advanced sentiment analysis using actual texts
            sentiment = None
            if all_texts:
                sentiment = self.sentiment_engine.get_symbol_sentiment_summary(symbol, all_texts, all_sources)
            
            sentiment_data = {symbol: sentiment} if sentiment else {}
            
            # Engineer features for latest data point
            if self.feature_engineer:
                features_df = self.feature_engineer.engineer_features(
                    {symbol: data}, sentiment_data, horizon_days
                )
            else:
                # Create basic features if no feature engineer
                features_df = data.copy()
                # Handle both 'close' and 'Close' column names
                close_col = 'Close' if 'Close' in features_df.columns else 'close'
                features_df['returns'] = features_df[close_col].pct_change()
            
            if features_df.empty:
                logger.warning(f"No features available for {symbol}")
                return None
            
            # Get latest feature row
            latest_features = features_df.iloc[-1:].copy()
            
            # Remove target and metadata columns
            feature_cols = [col for col in latest_features.columns 
                          if not col.startswith('target_') and 
                             col not in ['symbol', 'timestamp']]
            
            X = latest_features[feature_cols]
            
            # Get predictions from relevant models
            predictions = []
            uncertainties = []
            model_contributions = {}
            feature_importance = {}
            
            for model_key, model in self.models.items():
                if (prediction_type.value in model_key and 
                    model.is_trained and 
                    model_key in self.model_weights):
                    
                    try:
                        pred, uncertainty = model.predict(X)
                        weight = self.model_weights[model_key]
                        
                        predictions.append(pred[0] * weight)
                        uncertainties.append(uncertainty[0] * weight)
                        model_contributions[model_key] = pred[0]
                        
                        # Accumulate feature importance
                        model_importance = model.get_feature_importance()
                        for feature, importance in model_importance.items():
                            if feature in feature_importance:
                                feature_importance[feature] += importance * weight
                            else:
                                feature_importance[feature] = importance * weight
                    
                    except Exception as e:
                        logger.warning(f"Prediction failed for {model_key}: {e}")
            
            if not predictions:
                logger.warning(f"No model predictions available for {symbol}")
                return None
            
            # Ensemble prediction
            ensemble_prediction = sum(predictions)
            ensemble_uncertainty = np.sqrt(sum([u**2 for u in uncertainties]))
            
            # Calculate confidence (inverse of normalized uncertainty)
            confidence = max(0.1, 1.0 / (1.0 + ensemble_uncertainty))
            
            # Data quality score (simple heuristic)
            data_quality = min(1.0, len(data) / 50.0)  # Better with more data
            
            # Market regime detection
            market_regime = self._detect_market_regime(data)
            
            return PredictionResult(
                symbol=symbol,
                prediction_type=prediction_type,
                prediction=ensemble_prediction,
                confidence=confidence,
                uncertainty=ensemble_uncertainty,
                feature_importance=feature_importance,
                model_contributions=model_contributions,
                timestamp=datetime.now(),
                horizon_days=horizon_days,
                data_quality_score=data_quality,
                market_regime=market_regime,
                prediction_context={
                    'models_used': len(predictions),
                    'data_points': len(data),
                    'sentiment_available': bool(sentiment_data)
                }
            )
            
        except Exception as e:
            logger.error(f"ML prediction failed for {symbol}: {e}")
            return None
    
    def _fallback_predict(self, symbol: str, prediction_type: PredictionType,
                         horizon_days: int) -> Optional[PredictionResult]:
        """Fallback prediction when ML models not available"""
        try:
            # Get recent data using the correct method
            market_data = self.data_orchestrator.get_market_data(
                symbol, period="30d", interval="1d"
            )
            
            if not market_data or market_data.data.empty:
                return None
            
            data = market_data.data
            
            # Simple technical analysis prediction
            close_col = 'Close' if 'Close' in data.columns else 'close'
            current_price = data[close_col].iloc[-1]
            sma_20 = data[close_col].rolling(20).mean().iloc[-1]
            
            # Direction prediction based on trend
            if prediction_type == PredictionType.PRICE_DIRECTION:
                prediction = 1.0 if current_price > sma_20 else 0.0
                confidence = 0.6  # Moderate confidence
            else:
                # Price target as simple trend extrapolation
                trend = (current_price / sma_20 - 1) * 0.5  # Dampened trend
                prediction = trend
                confidence = 0.5  # Lower confidence for price targets
            
            return PredictionResult(
                symbol=symbol,
                prediction_type=prediction_type,
                prediction=prediction,
                confidence=confidence,
                uncertainty=0.3,  # Default uncertainty
                feature_importance={'sma_trend': 1.0},
                model_contributions={'technical_fallback': prediction},
                timestamp=datetime.now(),
                horizon_days=horizon_days,
                data_quality_score=0.5,
                market_regime='normal',
                prediction_context={'fallback_method': True}
            )
            
        except Exception as e:
            logger.error(f"Fallback prediction failed for {symbol}: {e}")
            return None
    
    def _fallback_training(self, symbols: List[str], lookback_days: int) -> Dict[str, Any]:
        """Fallback training when ML not available"""
        return {
            'method': 'fallback',
            'symbols': len(symbols),
            'lookback_days': lookback_days,
            'message': 'ML models not available, using technical analysis fallback'
        }
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime"""
        if len(data) < 20:
            return 'normal'
        
        # Calculate recent volatility and trend
        close_col = 'Close' if 'Close' in data.columns else 'close'
        returns = data[close_col].pct_change().dropna()
        recent_vol = returns.tail(20).std()
        long_vol = returns.std()
        
        recent_trend = (data[close_col].iloc[-1] / data[close_col].iloc[-20] - 1)
        
        # Classify regime
        if recent_vol > long_vol * 1.5:
            return 'volatile'
        elif abs(recent_trend) > 0.1:
            return 'trending'
        elif recent_vol < long_vol * 0.7:
            return 'sideways'
        else:
            return 'normal'
    
    def get_model_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for all models"""
        performance = {}
        
        for model_key, model in self.models.items():
            if model.is_trained:
                perf = model.performance
                if perf:
                    performance[model_key] = {
                        'accuracy': getattr(perf, 'accuracy', 0.0),
                        'mse': getattr(perf, 'mse', 0.0),
                        'total_predictions': getattr(perf, 'total_predictions', 0),
                        'correct_predictions': getattr(perf, 'correct_predictions', 0),
                        'last_updated': getattr(perf, 'last_updated', datetime.now()).isoformat(),
                        'weight': self.model_weights.get(model_key, 0.0)
                    }
                else:
                    performance[model_key] = {
                        'accuracy': 0.0,
                        'mse': 0.0,
                        'total_predictions': 0,
                        'correct_predictions': 0,
                        'last_updated': datetime.now().isoformat(),
                        'weight': self.model_weights.get(model_key, 0.0)
                    }
        
        return performance
    
    def save_models(self, filepath: str) -> bool:
        """Save trained models to file"""
        try:
            model_data = {
                'model_weights': self.model_weights,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'prediction_horizons': self.prediction_horizons,
                'model_configs': self.model_configs
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Models saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """Load trained models from file"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.model_weights = model_data.get('model_weights', {})
            self.prediction_horizons = model_data.get('prediction_horizons', [1, 5, 10, 20])
            self.model_configs = model_data.get('model_configs', self._get_default_model_configs())
            
            if model_data.get('last_training_time'):
                self.last_training_time = datetime.fromisoformat(model_data['last_training_time'])
            
            logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

# ============================================================================
# Public Interface Functions
# ============================================================================

def create_prediction_engine(data_orchestrator: DataFeedOrchestrator,
                           sentiment_engine: AdvancedSentimentEngine) -> EnsemblePredictionEngine:
    """Create and initialize the ensemble prediction engine"""
    return EnsemblePredictionEngine(data_orchestrator, sentiment_engine)
