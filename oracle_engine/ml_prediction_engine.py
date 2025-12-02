"""
ML-Driven Prediction Engine with Online Learning
Advanced machine learning system for stock price and options prediction
Implements ensemble models with continuous learning and uncertainty quantification
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

import pickle
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from abc import ABC, abstractmethod

# ML libraries
try:
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        mean_squared_error,
        accuracy_score,
        classification_report,
    )

    SKLEARN_AVAILABLE = True
except ImportError:

    class DummyClass:
        def __init__(*a, **k):
            raise ImportError(
                "scikit-learn not available. Please install with: pip install scikit-learn"
            )

        def __call__(*a, **k):
            raise ImportError(
                "scikit-learn not available. Please install with: pip install scikit-learn"
            )

    sklearn = DummyClass
    RandomForestRegressor = DummyClass
    RandomForestClassifier = DummyClass
    train_test_split = lambda *a, **k: (_ for _ in ()).throw(
        ImportError(
            "scikit-learn not available. Please install with: pip install scikit-learn"
        )
    )
    cross_val_score = DummyClass
    StandardScaler = DummyClass
    LabelEncoder = DummyClass
    mean_squared_error = DummyClass
    accuracy_score = DummyClass
    classification_report = DummyClass
    SKLEARN_AVAILABLE = False
    logging.warning(
        "scikit-learn not available. Please install with: pip install scikit-learn"
    )

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:

    class DummyXGB:
        def __init__(*a, **k):
            raise ImportError(
                "XGBoost not available. Please install with: pip install xgboost"
            )

        def __call__(*a, **k):
            raise ImportError(
                "XGBoost not available. Please install with: pip install xgboost"
            )

    xgb = DummyXGB()
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Please install with: pip install xgboost")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:

    class DummyTorch:
        def __init__(*a, **k):
            raise ImportError(
                "PyTorch not available. Please install with: pip install torch"
            )

        def __call__(*a, **k):
            raise ImportError(
                "PyTorch not available. Please install with: pip install torch"
            )

    torch = DummyTorch()
    nn = DummyTorch()
    optim = DummyTorch()
    DataLoader = DummyTorch()
    TensorDataset = DummyTorch()
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Please install with: pip install torch")

# Import our components
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
from sentiment.sentiment_engine import AdvancedSentimentEngine

logger = logging.getLogger(__name__)


class PredictionType(Enum):
    PRICE_DIRECTION = "price_direction"  # Up/Down classification
    PRICE_TARGET = "price_target"  # Specific price regression
    VOLATILITY = "volatility"  # Volatility prediction
    OPTION_VALUE = "option_value"  # Options pricing


class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


@dataclass
class PredictionResult:
    """Comprehensive prediction result with uncertainty quantification"""

    symbol: str
    prediction_type: PredictionType
    prediction: float
    confidence: float  # 0.0 to 1.0
    uncertainty: float  # Standard deviation or confidence interval width
    feature_importance: Dict[str, float]
    model_contributions: Dict[str, float]  # Individual model predictions
    timestamp: datetime
    horizon_days: int  # Prediction horizon

    # Additional metadata
    data_quality_score: float = 0.0
    market_regime: str = "normal"  # normal, volatile, trending, sideways
    prediction_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Track model performance metrics"""

    model_name: str
    prediction_type: PredictionType
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    mse: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_predictions: int = 0
    correct_predictions: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    # Drift detection metrics
    performance_history: List[float] = field(default_factory=list)
    drift_score: float = 0.0
    needs_retraining: bool = False


class FeatureEngineer:
    """Advanced feature engineering for ML models"""

    def __init__(self):
        self.feature_cache = {}
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE and StandardScaler else None
        self.label_encoders = {}

    def engineer_features(
        self,
        data: Dict[str, pd.DataFrame],
        sentiment_data: Optional[Dict[str, Dict]] = None,
        target_horizon_days: int = 5,
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML models

        Args:
            data: Historical price data by symbol
            sentiment_data: Sentiment analysis results by symbol
            target_horizon_days: Prediction horizon for target variables

        Returns:
            DataFrame with engineered features and targets
        """
        features_list = []

        for symbol, df in data.items():
            if df.empty or len(df) < 50:  # Need sufficient history
                continue

            try:
                symbol_features = self._create_symbol_features(
                    df, symbol, sentiment_data, target_horizon_days
                )
                if not symbol_features.empty:
                    features_list.append(symbol_features)

            except Exception as e:
                logger.warning(f"Failed to engineer features for {symbol}: {e}")
                continue

        if not features_list:
            return pd.DataFrame()

        # Combine all symbol features
        combined_features = pd.concat(features_list, ignore_index=True)

        # Handle missing values
        combined_features = self._handle_missing_values(combined_features)

        # Scale numerical features
        if self.scaler and SKLEARN_AVAILABLE and StandardScaler:
            numerical_cols = combined_features.select_dtypes(
                include=[np.number]
            ).columns
            combined_features[numerical_cols] = self.scaler.fit_transform(
                combined_features[numerical_cols]
            )

        return combined_features

    def _create_symbol_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        sentiment_data: Optional[Dict] = None,
        target_horizon_days: int = 5,
    ) -> pd.DataFrame:
        """Create features for a single symbol"""
        df = df.copy()

        # Basic price features
        df["return_1d"] = df["Close"].pct_change()
        df["return_5d"] = df["Close"].pct_change(5)
        df["return_20d"] = df["Close"].pct_change(20)

        # Volatility features
        df["volatility_5d"] = df["return_1d"].rolling(5).std()
        df["volatility_20d"] = df["return_1d"].rolling(20).std()
        df["volatility_ratio"] = df["volatility_5d"] / df["volatility_20d"]

        # Technical indicator features (assuming they exist from DataManager)
        technical_features = self._add_technical_features(df)

        # Volume features
        df["volume_ma_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["volume_trend"] = (
            df["Volume"].rolling(5).mean() / df["Volume"].rolling(20).mean()
        )

        # Price momentum features
        df["momentum_5d"] = df["Close"] / df["Close"].shift(5) - 1
        df["momentum_20d"] = df["Close"] / df["Close"].shift(20) - 1
        df["momentum_acceleration"] = df["momentum_5d"] - df["momentum_20d"]

        # Support/Resistance features
        df["distance_to_high_20d"] = (df["Close"] - df["High"].rolling(20).max()) / df[
            "Close"
        ]
        df["distance_to_low_20d"] = (df["Close"] - df["Low"].rolling(20).min()) / df[
            "Close"
        ]

        # Market microstructure features
        df["spread"] = (df["High"] - df["Low"]) / df["Close"]
        df["body_size"] = abs(df["Close"] - df["Open"]) / df["Close"]
        df["upper_shadow"] = (df["High"] - np.maximum(df["Open"], df["Close"])) / df[
            "Close"
        ]
        df["lower_shadow"] = (np.minimum(df["Open"], df["Close"]) - df["Low"]) / df[
            "Close"
        ]

        # Sentiment features
        if sentiment_data and symbol in sentiment_data:
            sentiment_info = sentiment_data[symbol]
            df["sentiment_score"] = sentiment_info.get("overall_sentiment", 0.0)
            df["sentiment_confidence"] = sentiment_info.get("confidence", 0.5)
            df["sentiment_momentum"] = sentiment_info.get("momentum", 0.0)
            df["news_volume"] = sentiment_info.get("article_count", 0)
        else:
            df["sentiment_score"] = 0.0
            df["sentiment_confidence"] = 0.5
            df["sentiment_momentum"] = 0.0
            df["news_volume"] = 0

        # Market regime features
        df["market_regime"] = self._classify_market_regime(df)

        # Target variables (future returns)
        df[f"target_return_{target_horizon_days}d"] = (
            df["Close"].shift(-target_horizon_days) / df["Close"] - 1
        )
        df[f"target_direction_{target_horizon_days}d"] = (
            df[f"target_return_{target_horizon_days}d"] > 0
        ).astype(int)
        df[f"target_volatility_{target_horizon_days}d"] = (
            df["return_1d"]
            .shift(-target_horizon_days)
            .rolling(target_horizon_days)
            .std()
        )

        # Add symbol and timestamp
        df["symbol"] = symbol
        df["timestamp"] = df.index

        # Remove rows with insufficient data
        df = df.dropna()

        return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        # RSI features
        if "RSI" in df.columns:
            df["rsi_oversold"] = (df["RSI"] < 30).astype(int)
            df["rsi_overbought"] = (df["RSI"] > 70).astype(int)
            df["rsi_divergence"] = df["RSI"].diff()

        # MACD features
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            df["macd_signal"] = (df["MACD"] > df["MACD_Signal"]).astype(int)
            df["macd_momentum"] = df["MACD"] - df["MACD_Signal"]
            df["macd_divergence"] = df["macd_momentum"].diff()

        # Bollinger Band features
        if all(col in df.columns for col in ["BB_Upper", "BB_Lower", "BB_Middle"]):
            df["bb_position"] = (df["Close"] - df["BB_Lower"]) / (
                df["BB_Upper"] - df["BB_Lower"]
            )
            df["bb_squeeze"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
            df["bb_breakout_up"] = (df["Close"] > df["BB_Upper"]).astype(int)
            df["bb_breakout_down"] = (df["Close"] < df["BB_Lower"]).astype(int)

        # Moving Average features
        if "SMA_20" in df.columns and "SMA_50" in df.columns:
            df["sma_cross"] = (df["SMA_20"] > df["SMA_50"]).astype(int)
            df["price_vs_sma20"] = df["Close"] / df["SMA_20"] - 1
            df["price_vs_sma50"] = df["Close"] / df["SMA_50"] - 1

        return df

    def _classify_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify market regime for each time period"""
        regimes = []

        for i in range(len(df)):
            if i < 20:  # Need sufficient history
                regimes.append("normal")
                continue

            # Calculate recent volatility and trend
            recent_data = df.iloc[i - 20 : i]
            volatility = recent_data["return_1d"].std()
            trend = recent_data["Close"].iloc[-1] / recent_data["Close"].iloc[0] - 1

            # Classify regime
            if volatility > recent_data["return_1d"].rolling(60).std().iloc[-1] * 1.5:
                regime = "volatile"
            elif abs(trend) > 0.1:  # 10% move in 20 days
                regime = "trending"
            elif volatility < recent_data["return_1d"].rolling(60).std().iloc[-1] * 0.7:
                regime = "sideways"
            else:
                regime = "normal"

            regimes.append(regime)

        return pd.Series(regimes, index=df.index)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature dataset"""
        # Forward fill first, then backward fill
        df = df.ffill().bfill()

        # For remaining NaN values, fill with median for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(
                    df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
                )

        return df


class BaseMLModel(ABC):
    """Abstract base class for ML models"""

    def __init__(self, model_type: ModelType, prediction_type: PredictionType):
        self.model_type = model_type
        self.prediction_type = prediction_type
        self.model = None
        self.is_trained = False
        self.feature_importance = {}
        self.performance = ModelPerformance(
            model_name=f"{model_type.value}_{prediction_type.value}",
            prediction_type=prediction_type,
        )

    @abstractmethod
    def train(
        self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty"""
        pass

    @abstractmethod
    def update(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Update model with new data (online learning)"""
        pass

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()

    def validate_performance(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Validate model performance"""
        if not self.is_trained:
            return {}

        predictions, uncertainties = self.predict(X_test)

        if self.prediction_type == PredictionType.PRICE_DIRECTION:
            # Classification metrics
            if accuracy_score:
                accuracy = float(accuracy_score(y_test, predictions > 0.5))
                self.performance.accuracy = accuracy
                self.performance.correct_predictions = int(accuracy * len(y_test))
        else:
            # Regression metrics
            if mean_squared_error:
                mse = float(mean_squared_error(y_test, predictions))
                self.performance.mse = mse

        self.performance.total_predictions = len(y_test)
        self.performance.last_updated = datetime.now()

        return {
            "accuracy": self.performance.accuracy,
            "mse": self.performance.mse,
            "total_predictions": self.performance.total_predictions,
        }


class RandomForestPredictor(BaseMLModel):
    """Random Forest implementation with feature importance"""

    def __init__(self, prediction_type: PredictionType, **kwargs):
        super().__init__(ModelType.RANDOM_FOREST, prediction_type)

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for RandomForestPredictor")

        # Configure model based on prediction type
        if prediction_type == PredictionType.PRICE_DIRECTION:
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                random_state=42,
                n_jobs=-1,
            )

    def train(
        self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train Random Forest model"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Get feature importance
        self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))

        # Validate
        validation_metrics = self.validate_performance(X_val, y_val)

        return {
            "training_samples": len(X_train),
            "validation_metrics": validation_metrics,
            "feature_importance": self.feature_importance,
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimation"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get predictions from all trees
        if self.prediction_type == PredictionType.PRICE_DIRECTION:
            # For classification, get probability predictions
            predictions = self.model.predict_proba(X)[
                :, 1
            ]  # Probability of positive class

            # Uncertainty as entropy of the prediction
            uncertainties = -(
                predictions * np.log(predictions + 1e-8)
                + (1 - predictions) * np.log(1 - predictions + 1e-8)
            )
        else:
            # For regression, get predictions from individual trees
            tree_predictions = np.array(
                [tree.predict(X) for tree in self.model.estimators_]
            )
            predictions = np.mean(tree_predictions, axis=0)
            uncertainties = np.std(tree_predictions, axis=0)

        return predictions, uncertainties

    def update(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Update model with new data"""
        if not self.is_trained:
            return False

        # Random Forest doesn't support incremental learning
        # We would need to retrain with combined data
        # For now, indicate that retraining is needed
        self.performance.needs_retraining = True
        return False


class XGBoostPredictor(BaseMLModel):
    """XGBoost implementation with gradient boosting"""

    def __init__(self, prediction_type: PredictionType, **kwargs):
        super().__init__(ModelType.XGBOOST, prediction_type)

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required for XGBoostPredictor")
        # Defensive: if xgb is not available, this will error out
        if prediction_type == PredictionType.PRICE_DIRECTION:
            if not hasattr(xgb, "XGBClassifier"):
                raise ImportError(
                    "XGBoost not available. Please install with: pip install xgboost"
                )
            self.model = xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
                n_jobs=-1,
            )
        else:
            if not hasattr(xgb, "XGBRegressor"):
                raise ImportError(
                    "XGBoost not available. Please install with: pip install xgboost"
                )
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
                n_jobs=-1,
            )

    def train(
        self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train XGBoost model"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Only pass early_stopping_rounds if eval_set is provided and model supports it
        fit_kwargs = {"X": X_train, "y": y_train, "verbose": False}
        # XGBRegressor/XGBClassifier support early_stopping_rounds with eval_set
        if hasattr(self.model, "fit") and hasattr(self.model, "evals_result_"):
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["early_stopping_rounds"] = 10

        self.model.fit(**fit_kwargs)
        self.is_trained = True

        # Get feature importance
        self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))

        # Validate
        validation_metrics = self.validate_performance(X_val, y_val)

        return {
            "training_samples": len(X_train),
            "validation_metrics": validation_metrics,
            "feature_importance": self.feature_importance,
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimation"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        if self.prediction_type == PredictionType.PRICE_DIRECTION:
            predictions = self.model.predict_proba(X)[:, 1]
            # Uncertainty as distance from 0.5 (less confident near 0.5)
            uncertainties = 1 - 2 * np.abs(predictions - 0.5)
        else:
            predictions = self.model.predict(X)
            # For XGBoost regression, uncertainty is harder to estimate
            # Use a simple heuristic based on prediction magnitude
            uncertainties = (
                np.abs(predictions) * 0.1
            )  # 10% of prediction as uncertainty

        return predictions, uncertainties

    def update(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Update model with new data"""
        if not self.is_trained:
            return False

        # XGBoost supports incremental learning via xgb_model parameter
        try:
            self.model.fit(X, y, xgb_model=self.model.get_booster())
            return True
        except Exception as e:
            logger.warning(f"XGBoost incremental update failed: {e}")
            self.performance.needs_retraining = True
            return False


# Neural Network implementation placeholder
class NeuralNetworkPredictor(BaseMLModel):
    """Enhanced Neural Network implementation with PyTorch"""

    def __init__(self, prediction_type: PredictionType, **kwargs):
        super().__init__(ModelType.NEURAL_NETWORK, prediction_type)

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for NeuralNetworkPredictor")

        # Enhanced architecture parameters
        self.hidden_size = kwargs.get("hidden_size", 128)
        self.num_layers = kwargs.get("num_layers", 3)
        self.dropout = kwargs.get("dropout", 0.3)  # Increased for better regularization
        self.learning_rate = kwargs.get("learning_rate", 0.001)

        # Enhanced training configuration
        self.use_batch_norm = kwargs.get("use_batch_norm", True)
        self.use_early_stopping = kwargs.get("use_early_stopping", True)
        self.use_lr_scheduling = kwargs.get("use_lr_scheduling", True)
        self.patience = kwargs.get("patience", 15)
        self.min_lr = kwargs.get("min_lr", 1e-6)

        # Will be initialized in train()
        self.input_size = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

    def _create_model(self, input_size: int):
        """Create enhanced neural network model with batch normalization"""
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, self.hidden_size))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))

        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

        # Output layer
        if self.prediction_type == PredictionType.PRICE_DIRECTION:
            layers.append(nn.Linear(self.hidden_size, 1))
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Linear(self.hidden_size, 1))

        return nn.Sequential(*layers)

    def train(
        self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train neural network with early stopping and timeout"""
        import time

        # Initialize model
        self.input_size = X.shape[1]
        self.model = self._create_model(self.input_size)

        # Setup enhanced training
        if self.prediction_type == PredictionType.PRICE_DIRECTION:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()

        # Use AdamW optimizer for better performance
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,  # L2 regularization
        )

        # Add learning rate scheduler
        if self.use_lr_scheduling:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=self.min_lr,
                verbose=True,
            )

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1)

        # Training configuration
        max_epochs = 50  # Reduced from 100
        patience = 10  # Early stopping patience
        min_improvement = 1e-4  # Minimum improvement threshold
        timeout_seconds = 60  # Maximum training time

        # Training loop with early stopping and timeout
        start_time = time.time()
        best_val_loss = float("inf")
        patience_counter = 0

        self.model.train()

        for epoch in range(max_epochs):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                logger.warning(
                    f"Neural network training timeout after {timeout_seconds}s at epoch {epoch}"
                )
                break

            # Training step
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            train_loss = self.criterion(outputs, y_train_tensor)
            train_loss.backward()
            self.optimizer.step()

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
            self.model.train()

            # Early stopping check
            if val_loss < best_val_loss - min_improvement:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Update learning rate scheduler
            if self.use_lr_scheduling and self.scheduler:
                self.scheduler.step(val_loss)

            # Log progress with learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            if epoch % 10 == 0 or epoch < 5:
                logger.info(
                    f"NN Epoch {epoch}/{max_epochs}: train_loss={train_loss.item():.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}"
                )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                # Restore best model
                self.model.load_state_dict(best_model_state)
                break

            # Convergence check
            if train_loss.item() < 1e-6:
                logger.info(f"Converged at epoch {epoch} (loss < 1e-6)")
                break

        training_time = time.time() - start_time
        logger.info(
            f"Neural network training completed in {training_time:.2f}s ({epoch+1} epochs)"
        )

        self.is_trained = True

        # Simple feature importance (gradient-based)
        self.feature_importance = self._calculate_feature_importance(X_train_tensor)

        # Validate
        validation_metrics = self.validate_performance(X_val, y_val)

        return {
            "training_samples": len(X_train),
            "validation_metrics": validation_metrics,
            "feature_importance": self.feature_importance,
            "training_time": training_time,
            "epochs_completed": epoch + 1,
            "early_stopped": patience_counter >= patience,
            "best_val_loss": best_val_loss,
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimation"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            predictions = self.model(X_tensor).numpy().flatten()

            # Simple uncertainty estimation using dropout at inference
            uncertainties = self._estimate_uncertainty(X_tensor)

        return predictions, uncertainties

    def _estimate_uncertainty(
        self, X_tensor: torch.Tensor, n_samples: int = 10
    ) -> np.ndarray:
        """Estimate uncertainty using Monte Carlo dropout"""
        self.model.train()  # Enable dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X_tensor).numpy().flatten()
                predictions.append(pred)

        predictions = np.array(predictions)
        uncertainties = np.std(predictions, axis=0)

        self.model.eval()  # Return to eval mode
        return uncertainties

    def _calculate_feature_importance(self, X_tensor: torch.Tensor) -> Dict[str, float]:
        """Calculate feature importance using gradients"""
        # Simplified feature importance calculation
        self.model.eval()
        X_tensor.requires_grad_(True)

        output = self.model(X_tensor).sum()
        output.backward()

        importance = torch.abs(X_tensor.grad).mean(dim=0).detach().numpy()

        # Normalize
        importance = importance / importance.sum()

        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def update(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Update model with new data"""
        if not self.is_trained:
            return False

        # Neural networks can support online learning
        try:
            X_tensor = torch.FloatTensor(X.values)
            y_tensor = torch.FloatTensor(y.values).unsqueeze(1)

            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

            return True
        except Exception as e:
            logger.warning(f"Neural network incremental update failed: {e}")
            return False


# ============================================================================
# Public Interface Functions
# ============================================================================


def create_ml_model(
    model_type: ModelType, prediction_type: PredictionType, **kwargs
) -> BaseMLModel:
    """Factory function to create ML models"""
    if model_type == ModelType.RANDOM_FOREST:
        return RandomForestPredictor(prediction_type, **kwargs)
    elif model_type == ModelType.XGBOOST:
        return XGBoostPredictor(prediction_type, **kwargs)
    elif model_type == ModelType.NEURAL_NETWORK:
        return NeuralNetworkPredictor(prediction_type, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_available_models() -> List[ModelType]:
    """Get list of available model types based on installed libraries"""
    available = []

    if SKLEARN_AVAILABLE:
        available.append(ModelType.RANDOM_FOREST)

    if XGBOOST_AVAILABLE:
        available.append(ModelType.XGBOOST)

    if PYTORCH_AVAILABLE:
        available.append(ModelType.NEURAL_NETWORK)

    return available
