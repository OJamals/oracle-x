"""
Enhanced ML Learning System - Phase 1: Critical Training Fixes
Addresses memory issues, training completion, and error handling
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import gc
import threading
from dataclasses import dataclass
import json
from pathlib import Path

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Enhanced training result with detailed metrics"""

    model_key: str
    success: bool
    training_time: float
    samples_trained: int
    performance_metrics: Dict[str, float]
    error_message: Optional[str] = None
    memory_usage: Optional[Dict[str, float]] = None


class MemoryManager:
    """Manages memory usage during ML training to prevent crashes"""

    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = max_memory_mb
        self.memory_checkpoints = []

    def check_memory(self, operation: str) -> Dict[str, float]:
        """Check current memory usage"""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            checkpoint = {
                "operation": operation,
                "memory_mb": memory_mb,
                "timestamp": datetime.now().isoformat(),
            }

            self.memory_checkpoints.append(checkpoint)

            if memory_mb > self.max_memory_mb:
                logger.warning(
                    f"High memory usage: {memory_mb:.1f}MB > {self.max_memory_mb}MB during {operation}"
                )
                self.cleanup_memory()

            return checkpoint

        except ImportError:
            return {
                "operation": operation,
                "memory_mb": 0,
                "error": "psutil not available",
            }

    def cleanup_memory(self):
        """Force garbage collection and cleanup"""
        gc.collect()

        # Try to cleanup torch cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class SafeSentimentProcessor:
    """Memory-safe sentiment processing with batch handling"""

    def __init__(self, batch_size: int = 50, max_texts: int = 1000):
        self.batch_size = batch_size
        self.max_texts = max_texts
        self.memory_manager = MemoryManager()

    def process_sentiment_safely(
        self, sentiment_engine, symbol: str, raw_sentiment_data: Dict
    ) -> Optional[Dict]:
        """Process sentiment data in memory-safe batches"""
        try:
            self.memory_manager.check_memory("sentiment_start")

            all_texts = []
            all_sources = []

            # Extract texts with limits
            for source_name, sentiment_data_obj in raw_sentiment_data.items():
                if (
                    hasattr(sentiment_data_obj, "raw_data")
                    and sentiment_data_obj.raw_data
                    and "sample_texts" in sentiment_data_obj.raw_data
                ):

                    texts = sentiment_data_obj.raw_data["sample_texts"]
                    if texts:
                        # Limit texts per source
                        limited_texts = texts[: min(len(texts), self.batch_size)]
                        all_texts.extend(limited_texts)
                        all_sources.extend([source_name] * len(limited_texts))

                        if len(all_texts) >= self.max_texts:
                            break

            if not all_texts:
                return None

            # Limit total texts
            all_texts = all_texts[: self.max_texts]
            all_sources = all_sources[: self.max_texts]

            self.memory_manager.check_memory("sentiment_texts_extracted")

            # Process in batches
            sentiment_results = []
            for i in range(0, len(all_texts), self.batch_size):
                batch_texts = all_texts[i : i + self.batch_size]
                batch_sources = all_sources[i : i + self.batch_size]

                try:
                    # Use the correct sentiment analysis function
                    from sentiment.sentiment_engine import analyze_symbol_sentiment

                    batch_sentiment = analyze_symbol_sentiment(
                        symbol, batch_texts, batch_sources
                    )

                    if batch_sentiment:
                        sentiment_results.append(
                            {
                                "overall_sentiment": batch_sentiment.overall_sentiment,
                                "confidence": batch_sentiment.confidence,
                                "quality_score": batch_sentiment.quality_score,
                                "sample_size": len(batch_texts),
                            }
                        )

                    # Cleanup after each batch
                    self.memory_manager.cleanup_memory()

                except Exception as e:
                    logger.warning(f"Batch sentiment processing failed: {e}")
                    continue

            self.memory_manager.check_memory("sentiment_processing_complete")

            # Aggregate results
            if sentiment_results:
                total_samples = sum(r["sample_size"] for r in sentiment_results)
                weighted_sentiment = (
                    sum(
                        r["overall_sentiment"] * r["sample_size"]
                        for r in sentiment_results
                    )
                    / total_samples
                )
                avg_confidence = sum(r["confidence"] for r in sentiment_results) / len(
                    sentiment_results
                )
                avg_quality = sum(r["quality_score"] for r in sentiment_results) / len(
                    sentiment_results
                )

                return {
                    "overall_sentiment": weighted_sentiment,
                    "confidence": avg_confidence,
                    "quality_score": avg_quality,
                    "sample_size": total_samples,
                    "batch_count": len(sentiment_results),
                }

            return None

        except Exception as e:
            logger.error(f"Safe sentiment processing failed: {e}")
            self.memory_manager.cleanup_memory()
            return None


class RobustTrainingPipeline:
    """Robust training pipeline with fallbacks and error handling"""

    def __init__(self, engine):
        self.engine = engine
        self.memory_manager = MemoryManager()
        self.sentiment_processor = SafeSentimentProcessor()
        self.training_results = []

    def train_with_fallbacks(
        self, symbols: List[str], lookback_days: int = 252
    ) -> Dict[str, TrainingResult]:
        """Train models with multiple fallback strategies"""
        logger.info(f"Starting robust training for {len(symbols)} symbols")
        results = {}

        for symbol in symbols:
            symbol_result = self._train_symbol_with_fallbacks(symbol, lookback_days)
            results[symbol] = symbol_result

        return results

    def _train_symbol_with_fallbacks(
        self, symbol: str, lookback_days: int
    ) -> TrainingResult:
        """Train models for a single symbol with fallback strategies"""
        start_time = datetime.now()

        try:
            # Strategy 1: Full training with sentiment
            result = self._attempt_full_training(symbol, lookback_days)
            if result.success:
                return result

            logger.warning(
                f"Full training failed for {symbol}, trying without sentiment"
            )

            # Strategy 2: Training without sentiment
            result = self._attempt_training_without_sentiment(symbol, lookback_days)
            if result.success:
                return result

            logger.warning(
                f"Training without sentiment failed for {symbol}, trying minimal training"
            )

            # Strategy 3: Minimal training with basic features only
            result = self._attempt_minimal_training(symbol, lookback_days)
            if result.success:
                return result

            # All strategies failed
            training_time = (datetime.now() - start_time).total_seconds()
            return TrainingResult(
                model_key=f"{symbol}_training",
                success=False,
                training_time=training_time,
                samples_trained=0,
                performance_metrics={},
                error_message="All training strategies failed",
            )

        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Training pipeline failed for {symbol}: {e}")
            return TrainingResult(
                model_key=f"{symbol}_training",
                success=False,
                training_time=training_time,
                samples_trained=0,
                performance_metrics={},
                error_message=str(e),
            )

    def _attempt_full_training(self, symbol: str, lookback_days: int) -> TrainingResult:
        """Attempt full training with sentiment and all features"""
        start_time = datetime.now()

        try:
            self.memory_manager.check_memory("full_training_start")

            # Get market data
            market_data = self.engine.data_orchestrator.get_market_data(
                symbol, period="1y", interval="1d"
            )

            if not market_data or market_data.data.empty:
                raise ValueError(f"No market data available for {symbol}")

            self.memory_manager.check_memory("market_data_loaded")

            # Get sentiment data safely
            raw_sentiment_data = self.engine.data_orchestrator.get_sentiment_data(
                symbol
            )
            sentiment_data = self.sentiment_processor.process_sentiment_safely(
                self.engine.sentiment_engine, symbol, raw_sentiment_data
            )

            self.memory_manager.check_memory("sentiment_processed")

            # Prepare training data
            training_data = self._prepare_training_data(
                market_data.data, sentiment_data, symbol
            )

            if training_data.empty:
                raise ValueError(f"No valid training data for {symbol}")

            self.memory_manager.check_memory("training_data_prepared")

            # Train models
            training_success = self._train_models_on_data(training_data, symbol)

            training_time = (datetime.now() - start_time).total_seconds()

            return TrainingResult(
                model_key=f"{symbol}_full",
                success=training_success,
                training_time=training_time,
                samples_trained=len(training_data),
                performance_metrics=self._calculate_performance_metrics(symbol),
                memory_usage=(
                    dict(self.memory_manager.memory_checkpoints[-1])
                    if self.memory_manager.memory_checkpoints
                    else {}
                ),
            )

        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Full training failed for {symbol}: {e}")
            self.memory_manager.cleanup_memory()
            return TrainingResult(
                model_key=f"{symbol}_full",
                success=False,
                training_time=training_time,
                samples_trained=0,
                performance_metrics={},
                error_message=str(e),
            )

    def _attempt_training_without_sentiment(
        self, symbol: str, lookback_days: int
    ) -> TrainingResult:
        """Attempt training with market data only"""
        start_time = datetime.now()

        try:
            self.memory_manager.check_memory("no_sentiment_training_start")

            # Get market data
            market_data = self.engine.data_orchestrator.get_market_data(
                symbol, period="1y", interval="1d"
            )

            if not market_data or market_data.data.empty:
                raise ValueError(f"No market data available for {symbol}")

            # Prepare training data without sentiment
            training_data = self._prepare_training_data(market_data.data, None, symbol)

            if training_data.empty:
                raise ValueError(f"No valid training data for {symbol}")

            # Train models
            training_success = self._train_models_on_data(training_data, symbol)

            training_time = (datetime.now() - start_time).total_seconds()

            return TrainingResult(
                model_key=f"{symbol}_no_sentiment",
                success=training_success,
                training_time=training_time,
                samples_trained=len(training_data),
                performance_metrics=self._calculate_performance_metrics(symbol),
            )

        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Training without sentiment failed for {symbol}: {e}")
            return TrainingResult(
                model_key=f"{symbol}_no_sentiment",
                success=False,
                training_time=training_time,
                samples_trained=0,
                performance_metrics={},
                error_message=str(e),
            )

    def _attempt_minimal_training(
        self, symbol: str, lookback_days: int
    ) -> TrainingResult:
        """Attempt minimal training with basic features only"""
        start_time = datetime.now()

        try:
            # Get minimal market data
            market_data = self.engine.data_orchestrator.get_market_data(
                symbol, period="60d", interval="1d"  # Reduced data
            )

            if not market_data or market_data.data.empty:
                raise ValueError(f"No market data available for {symbol}")

            # Create minimal features
            df = market_data.data.copy()
            df["returns"] = df["Close"].pct_change()
            df["sma_20"] = df["Close"].rolling(20).mean()
            df["target_direction"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

            # Drop rows with NaN
            df = df.dropna()

            if len(df) < 20:
                raise ValueError(f"Insufficient data for {symbol}")

            # Simple training with basic features
            features = ["returns", "sma_20"]
            X = df[features].fillna(0)
            y = df["target_direction"]

            # Train at least one model
            success = self._train_single_model(X, y, symbol)

            training_time = (datetime.now() - start_time).total_seconds()

            return TrainingResult(
                model_key=f"{symbol}_minimal",
                success=success,
                training_time=training_time,
                samples_trained=len(df),
                performance_metrics={
                    "basic_accuracy": 0.5 if success else 0.0
                },  # Placeholder
            )

        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Minimal training failed for {symbol}: {e}")
            return TrainingResult(
                model_key=f"{symbol}_minimal",
                success=False,
                training_time=training_time,
                samples_trained=0,
                performance_metrics={},
                error_message=str(e),
            )

    def _prepare_training_data(
        self, market_data: pd.DataFrame, sentiment_data: Optional[Dict], symbol: str
    ) -> pd.DataFrame:
        """Prepare training data with features and targets"""
        try:
            df = market_data.copy()

            # Basic technical indicators
            df["returns"] = df["Close"].pct_change()
            df["volatility"] = df["returns"].rolling(20).std()
            df["sma_20"] = df["Close"].rolling(20).mean()
            df["sma_50"] = df["Close"].rolling(50).mean()
            df["rsi"] = self._calculate_rsi(df["Close"])

            # Price ratios
            df["price_to_sma20"] = df["Close"] / df["sma_20"]
            df["price_to_sma50"] = df["Close"] / df["sma_50"]

            # Volume features if available
            if "Volume" in df.columns:
                df["volume_sma"] = df["Volume"].rolling(20).mean()
                df["volume_ratio"] = df["Volume"] / df["volume_sma"]
            else:
                df["volume_ratio"] = 1.0

            # Sentiment features if available
            if sentiment_data:
                df["sentiment_score"] = sentiment_data["overall_sentiment"]
                df["sentiment_confidence"] = sentiment_data["confidence"]
                df["sentiment_quality"] = sentiment_data["quality_score"]
            else:
                df["sentiment_score"] = 0.0
                df["sentiment_confidence"] = 0.0
                df["sentiment_quality"] = 0.0

            # Target variables
            df["target_direction"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
            df["target_return"] = df["Close"].pct_change().shift(-1)

            # Drop rows with NaN
            df = df.dropna()

            return df

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _train_models_on_data(self, training_data: pd.DataFrame, symbol: str) -> bool:
        """Train actual ML models on prepared data"""
        try:
            # Define features and targets
            feature_cols = [
                "returns",
                "volatility",
                "sma_20",
                "sma_50",
                "rsi",
                "price_to_sma20",
                "price_to_sma50",
                "volume_ratio",
                "sentiment_score",
                "sentiment_confidence",
                "sentiment_quality",
            ]

            # Ensure all feature columns exist
            feature_cols = [col for col in feature_cols if col in training_data.columns]

            X = training_data[feature_cols].fillna(0)
            y_direction = training_data["target_direction"]
            y_return = training_data["target_return"].fillna(0)

            if len(X) < 10:
                raise ValueError("Insufficient training data")

            # Train each model type
            trained_count = 0

            # Train direction models
            for model_key, model in self.engine.models.items():
                if "direction" in model_key and symbol.upper() in model_key.upper():
                    try:
                        if hasattr(model, "train"):
                            model.train(X, y_direction)
                            trained_count += 1
                            logger.info(f"Successfully trained {model_key}")
                    except Exception as e:
                        logger.warning(f"Failed to train {model_key}: {e}")

            # Train target models
            for model_key, model in self.engine.models.items():
                if "target" in model_key and symbol.upper() in model_key.upper():
                    try:
                        if hasattr(model, "train"):
                            model.train(X, y_return)
                            trained_count += 1
                            logger.info(f"Successfully trained {model_key}")
                    except Exception as e:
                        logger.warning(f"Failed to train {model_key}: {e}")

            return trained_count > 0

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def _train_single_model(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> bool:
        """Train a single model as fallback"""
        try:
            # Find any available model for this symbol
            for model_key, model in self.engine.models.items():
                if symbol.upper() in model_key.upper():
                    try:
                        if hasattr(model, "train"):
                            model.train(X, y)
                            logger.info(
                                f"Successfully trained single model {model_key}"
                            )
                            return True
                    except Exception as e:
                        logger.warning(f"Failed to train single model {model_key}: {e}")
                        continue
            return False
        except Exception as e:
            logger.error(f"Single model training failed: {e}")
            return False

    def _calculate_performance_metrics(self, symbol: str) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        metrics = {}

        # Count trained models
        trained_models = 0
        for model_key, model in self.engine.models.items():
            if symbol.upper() in model_key.upper():
                if hasattr(model, "is_trained") and model.is_trained:
                    trained_models += 1

        metrics["trained_models"] = trained_models
        metrics["total_models"] = len(
            [k for k in self.engine.models.keys() if symbol.upper() in k.upper()]
        )
        metrics["training_completion_rate"] = trained_models / max(
            metrics["total_models"], 1
        )

        return metrics


# Enhanced ensemble engine with robust training
class EnhancedEnsembleEngine:
    """Enhanced ensemble engine with robust training pipeline"""

    def __init__(self, base_engine):
        self.base_engine = base_engine
        self.robust_trainer = RobustTrainingPipeline(base_engine)
        self.training_history = []

    def train_models_robustly(
        self, symbols: List[str], lookback_days: int = 252
    ) -> Dict[str, Any]:
        """Train models with robust error handling and fallbacks"""
        logger.info(f"Starting enhanced training for {len(symbols)} symbols")

        start_time = datetime.now()
        results = self.robust_trainer.train_with_fallbacks(symbols, lookback_days)
        training_time = (datetime.now() - start_time).total_seconds()

        # Aggregate results
        successful_symbols = [s for s, r in results.items() if r.success]
        failed_symbols = [s for s, r in results.items() if not r.success]
        total_samples = sum(r.samples_trained for r in results.values())

        summary = {
            "training_time": training_time,
            "total_symbols": len(symbols),
            "successful_symbols": len(successful_symbols),
            "failed_symbols": len(failed_symbols),
            "success_rate": len(successful_symbols) / len(symbols),
            "total_samples_trained": total_samples,
            "detailed_results": results,
        }

        # Store in history
        self.training_history.append(
            {"timestamp": datetime.now().isoformat(), "summary": summary}
        )

        logger.info(
            f"Enhanced training completed: {len(successful_symbols)}/{len(symbols)} symbols successful"
        )

        return summary


def create_enhanced_training_wrapper(base_engine):
    """Create enhanced training wrapper for existing engine"""
    return EnhancedEnsembleEngine(base_engine)
