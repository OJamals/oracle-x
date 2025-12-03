"""
Adaptive Feature Engineering System for Oracle-X
Automated Feature Selection, Generation, and Optimization
"""

import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Try to import feature engineering libraries
try:
    from sklearn.feature_selection import (
        SelectKBest,
        SelectPercentile,
        RFE,
        RFECV,
        mutual_info_regression,
        mutual_info_classif,
        f_regression,
        f_classif,
        chi2,
    )
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.decomposition import PCA, FastICA
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LassoCV, ElasticNetCV

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for adaptive feature engineering"""

    max_features: int = 100
    min_feature_importance: float = 0.01
    correlation_threshold: float = 0.95
    mutual_info_threshold: float = 0.1
    polynomial_degree: int = 2
    pca_variance_threshold: float = 0.95
    auto_generate_features: bool = True
    use_technical_indicators: bool = True
    use_statistical_features: bool = True
    use_interaction_features: bool = True
    feature_selection_method: str = (
        "mutual_info"  # 'mutual_info', 'rfe', 'lasso', 'forest'
    )


class TechnicalIndicatorGenerator:
    """Generate technical indicators for financial data"""

    def __init__(self):
        self.indicators = {}

    def generate_price_indicators(
        self, df: pd.DataFrame, price_col: str = "close"
    ) -> pd.DataFrame:
        """Generate price-based technical indicators"""
        result_df = df.copy()

        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found")
            return result_df

        prices = df[price_col].values

        if TALIB_AVAILABLE and len(prices) > 30:
            try:
                # Moving averages
                result_df["sma_5"] = talib.SMA(prices, timeperiod=5)
                result_df["sma_10"] = talib.SMA(prices, timeperiod=10)
                result_df["sma_20"] = talib.SMA(prices, timeperiod=20)
                result_df["ema_12"] = talib.EMA(prices, timeperiod=12)
                result_df["ema_26"] = talib.EMA(prices, timeperiod=26)

                # Momentum indicators
                result_df["rsi"] = talib.RSI(prices, timeperiod=14)
                (
                    result_df["macd"],
                    result_df["macd_signal"],
                    result_df["macd_hist"],
                ) = talib.MACD(prices)
                result_df["stoch_k"], result_df["stoch_d"] = talib.STOCH(
                    df.get("high", prices), df.get("low", prices), prices
                )

                # Volatility indicators
                if "high" in df.columns and "low" in df.columns:
                    (
                        result_df["bb_upper"],
                        result_df["bb_middle"],
                        result_df["bb_lower"],
                    ) = talib.BBANDS(prices)
                    result_df["atr"] = talib.ATR(df["high"], df["low"], prices)

                # Volume indicators (if volume available)
                if "volume" in df.columns:
                    volume = df["volume"].values
                    result_df["ad"] = talib.AD(df["high"], df["low"], prices, volume)
                    result_df["obv"] = talib.OBV(prices, volume)

            except Exception as e:
                logger.warning(f"Error generating TA-Lib indicators: {e}")

        # Manual calculations as fallback
        if not TALIB_AVAILABLE or len(prices) <= 30:
            result_df = self._manual_indicators(result_df, price_col)

        return result_df

    def _manual_indicators(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Manual calculation of technical indicators"""
        # Simple moving averages
        for period in [5, 10, 20]:
            df[f"sma_{period}"] = df[price_col].rolling(window=period).mean()

        # Exponential moving averages
        for period in [12, 26]:
            df[f"ema_{period}"] = df[price_col].ewm(span=period).mean()

        # RSI calculation
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        sma_20 = df[price_col].rolling(window=20).mean()
        std_20 = df[price_col].rolling(window=20).std()
        df["bb_upper"] = sma_20 + (std_20 * 2)
        df["bb_lower"] = sma_20 - (std_20 * 2)
        df["bb_middle"] = sma_20

        return df


class StatisticalFeatureGenerator:
    """Generate statistical features from time series data"""

    def __init__(self):
        self.lookback_periods = [5, 10, 20, 50]

    def generate_statistical_features(
        self, df: pd.DataFrame, target_cols: List[str] = None
    ) -> pd.DataFrame:
        """Generate statistical features"""
        result_df = df.copy()

        if target_cols is None:
            target_cols = [
                col for col in df.columns if df[col].dtype in ["float64", "int64"]
            ]

        for col in target_cols:
            if col not in df.columns:
                continue

            series = df[col]

            # Rolling statistics
            for period in self.lookback_periods:
                if len(series) > period:
                    result_df[f"{col}_mean_{period}"] = series.rolling(period).mean()
                    result_df[f"{col}_std_{period}"] = series.rolling(period).std()
                    result_df[f"{col}_min_{period}"] = series.rolling(period).min()
                    result_df[f"{col}_max_{period}"] = series.rolling(period).max()
                    result_df[f"{col}_skew_{period}"] = series.rolling(period).skew()
                    result_df[f"{col}_kurt_{period}"] = series.rolling(period).kurt()

            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                result_df[f"{col}_lag_{lag}"] = series.shift(lag)

            # Difference features
            result_df[f"{col}_diff_1"] = series.diff(1)
            result_df[f"{col}_diff_2"] = series.diff(2)
            result_df[f"{col}_pct_change"] = series.pct_change()

            # Z-score (standardized values)
            for period in [20, 50]:
                if len(series) > period:
                    rolling_mean = series.rolling(period).mean()
                    rolling_std = series.rolling(period).std()
                    result_df[f"{col}_zscore_{period}"] = (
                        series - rolling_mean
                    ) / rolling_std

        return result_df


class InteractionFeatureGenerator:
    """Generate interaction and polynomial features"""

    def __init__(self, max_degree: int = 2):
        self.max_degree = max_degree
        self.polynomial_features = None

    def generate_interaction_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        max_interactions: int = 50,
    ) -> pd.DataFrame:
        """Generate interaction features between columns"""
        result_df = df.copy()

        if feature_cols is None:
            feature_cols = [
                col for col in df.columns if df[col].dtype in ["float64", "int64"]
            ]

        # Limit to prevent explosion of features
        feature_cols = feature_cols[:10]

        interaction_count = 0

        # Pairwise interactions
        for i, col1 in enumerate(feature_cols):
            for j, col2 in enumerate(feature_cols[i + 1 :], i + 1):
                if interaction_count >= max_interactions:
                    break

                if col1 in df.columns and col2 in df.columns:
                    # Multiplication
                    result_df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

                    # Division (with safety check)
                    safe_col2 = df[col2].replace(0, np.nan)
                    result_df[f"{col1}_div_{col2}"] = df[col1] / safe_col2

                    # Ratio
                    result_df[f"{col1}_ratio_{col2}"] = df[col1] / (
                        df[col1] + df[col2] + 1e-8
                    )

                    interaction_count += 3

            if interaction_count >= max_interactions:
                break

        return result_df

    def generate_polynomial_features(
        self, df: pd.DataFrame, feature_cols: List[str] = None
    ) -> pd.DataFrame:
        """Generate polynomial features"""
        if not SKLEARN_AVAILABLE:
            return df

        result_df = df.copy()

        if feature_cols is None:
            feature_cols = [
                col for col in df.columns if df[col].dtype in ["float64", "int64"]
            ]

        # Limit features to prevent memory issues
        feature_cols = feature_cols[:5]

        if len(feature_cols) > 0:
            try:
                self.polynomial_features = PolynomialFeatures(
                    degree=self.max_degree, interaction_only=True, include_bias=False
                )

                poly_data = self.polynomial_features.fit_transform(
                    df[feature_cols].fillna(0)
                )
                poly_feature_names = self.polynomial_features.get_feature_names_out(
                    feature_cols
                )

                # Add polynomial features
                for i, name in enumerate(poly_feature_names):
                    if name not in feature_cols:  # Skip original features
                        result_df[f"poly_{name}"] = poly_data[:, i]

            except Exception as e:
                logger.warning(f"Error generating polynomial features: {e}")

        return result_df


class AdaptiveFeatureSelector:
    """Adaptive feature selection with multiple methods"""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.selected_features = []
        self.feature_scores = {}
        self.selection_history = []

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, method: str = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using specified method"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Sklearn not available, returning all features")
            return X, list(X.columns)

        method = method or self.config.feature_selection_method

        # Remove features with too many NaN values
        X_clean = self._clean_features(X)

        if method == "mutual_info":
            return self._mutual_info_selection(X_clean, y)
        elif method == "rfe":
            return self._rfe_selection(X_clean, y)
        elif method == "lasso":
            return self._lasso_selection(X_clean, y)
        elif method == "forest":
            return self._forest_selection(X_clean, y)
        elif method == "correlation":
            return self._correlation_selection(X_clean, y)
        else:
            logger.warning(f"Unknown selection method: {method}")
            return X_clean, list(X_clean.columns)

    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean features by removing high NaN columns and correlations"""
        # Remove columns with >50% NaN values
        nan_threshold = 0.5
        valid_cols = []

        for col in X.columns:
            nan_ratio = X[col].isna().sum() / len(X)
            if nan_ratio <= nan_threshold:
                valid_cols.append(col)

        X_clean = X[valid_cols].copy()

        # Fill remaining NaN values
        X_clean = X_clean.fillna(method="ffill").fillna(0)

        # Remove highly correlated features
        if len(X_clean.columns) > 1:
            corr_matrix = X_clean.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = [
                column
                for column in upper_tri.columns
                if any(upper_tri[column] > self.config.correlation_threshold)
            ]

            X_clean = X_clean.drop(columns=to_drop)

        return X_clean

    def _mutual_info_selection(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Feature selection using mutual information"""
        try:
            # Determine if classification or regression
            is_classification = len(y.unique()) < 10

            if is_classification:
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)

            # Create feature scores dictionary
            feature_scores = dict(zip(X.columns, mi_scores))
            self.feature_scores.update(feature_scores)

            # Select features above threshold
            selected_features = [
                feature
                for feature, score in feature_scores.items()
                if score >= self.config.mutual_info_threshold
            ]

            # Limit to max_features
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(
                    feature_scores.items(), key=lambda x: x[1], reverse=True
                )
                selected_features = [
                    f[0] for f in sorted_features[: self.config.max_features]
                ]

            self.selected_features = selected_features

            logger.info(
                f"Mutual info selection: {len(selected_features)} features selected"
            )
            return X[selected_features], selected_features

        except Exception as e:
            logger.error(f"Error in mutual info selection: {e}")
            return X, list(X.columns)

    def _rfe_selection(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Recursive feature elimination"""
        try:
            # Determine if classification or regression
            is_classification = len(y.unique()) < 10

            if is_classification:
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)

            # Use RFECV for automatic feature number selection
            selector = RFECV(
                estimator=estimator,
                step=1,
                cv=3,
                scoring="accuracy" if is_classification else "neg_mean_squared_error",
                n_jobs=-1,
            )

            selector.fit(X, y)

            selected_features = X.columns[selector.support_].tolist()
            self.selected_features = selected_features

            logger.info(f"RFE selection: {len(selected_features)} features selected")
            return X[selected_features], selected_features

        except Exception as e:
            logger.error(f"Error in RFE selection: {e}")
            return X, list(X.columns)

    def _lasso_selection(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Feature selection using Lasso regularization"""
        try:
            # Use LassoCV for automatic alpha selection
            lasso = LassoCV(cv=3, random_state=42, max_iter=1000)
            lasso.fit(X, y)

            # Select features with non-zero coefficients
            selected_mask = np.abs(lasso.coef_) > self.config.min_feature_importance
            selected_features = X.columns[selected_mask].tolist()

            # Store feature importance scores
            feature_scores = dict(zip(X.columns, np.abs(lasso.coef_)))
            self.feature_scores.update(feature_scores)

            self.selected_features = selected_features

            logger.info(f"Lasso selection: {len(selected_features)} features selected")
            return X[selected_features], selected_features

        except Exception as e:
            logger.error(f"Error in Lasso selection: {e}")
            return X, list(X.columns)

    def _forest_selection(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Feature selection using Random Forest importance"""
        try:
            # Determine if classification or regression
            is_classification = len(y.unique()) < 10

            if is_classification:
                forest = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                forest = RandomForestRegressor(n_estimators=100, random_state=42)

            forest.fit(X, y)

            # Get feature importances
            importances = forest.feature_importances_
            feature_scores = dict(zip(X.columns, importances))
            self.feature_scores.update(feature_scores)

            # Select features above importance threshold
            selected_features = [
                feature
                for feature, importance in feature_scores.items()
                if importance >= self.config.min_feature_importance
            ]

            # Limit to max_features
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(
                    feature_scores.items(), key=lambda x: x[1], reverse=True
                )
                selected_features = [
                    f[0] for f in sorted_features[: self.config.max_features]
                ]

            self.selected_features = selected_features

            logger.info(f"Forest selection: {len(selected_features)} features selected")
            return X[selected_features], selected_features

        except Exception as e:
            logger.error(f"Error in forest selection: {e}")
            return X, list(X.columns)

    def _correlation_selection(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Feature selection based on correlation with target"""
        try:
            # Calculate correlation with target
            correlations = X.corrwith(y).abs()

            # Select features above correlation threshold
            selected_features = correlations[
                correlations >= self.config.mutual_info_threshold
            ].index.tolist()

            # Store correlation scores
            feature_scores = correlations.to_dict()
            self.feature_scores.update(feature_scores)

            # Limit to max_features
            if len(selected_features) > self.config.max_features:
                sorted_features = correlations.sort_values(ascending=False)
                selected_features = sorted_features.head(
                    self.config.max_features
                ).index.tolist()

            self.selected_features = selected_features

            logger.info(
                f"Correlation selection: {len(selected_features)} features selected"
            )
            return X[selected_features], selected_features

        except Exception as e:
            logger.error(f"Error in correlation selection: {e}")
            return X, list(X.columns)


class AdaptiveFeatureEngineering:
    """Main adaptive feature engineering system"""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

        # Component systems
        self.technical_generator = TechnicalIndicatorGenerator()
        self.statistical_generator = StatisticalFeatureGenerator()
        self.interaction_generator = InteractionFeatureGenerator(
            self.config.polynomial_degree
        )
        self.feature_selector = AdaptiveFeatureSelector(self.config)

        # Feature tracking
        self.feature_pipeline = []
        self.feature_history = []
        self.performance_by_features = {}

        logger.info("Adaptive feature engineering system initialized")

    def engineer_features(
        self, df: pd.DataFrame, target_col: str = None
    ) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        result_df = df.copy()

        logger.info(
            f"Starting feature engineering on {len(df)} samples, {len(df.columns)} features"
        )

        # Generate technical indicators
        if self.config.use_technical_indicators:
            logger.info("Generating technical indicators...")
            result_df = self.technical_generator.generate_price_indicators(result_df)

        # Generate statistical features
        if self.config.use_statistical_features:
            logger.info("Generating statistical features...")
            result_df = self.statistical_generator.generate_statistical_features(
                result_df
            )

        # Generate interaction features
        if self.config.use_interaction_features:
            logger.info("Generating interaction features...")
            result_df = self.interaction_generator.generate_interaction_features(
                result_df
            )

        # Remove infinite and NaN values
        result_df = result_df.replace([np.inf, -np.inf], np.nan)

        logger.info(
            f"Feature engineering completed: {len(result_df.columns)} features generated"
        )

        return result_df

    def select_optimal_features(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select optimal features using configured method"""
        return self.feature_selector.select_features(X, y)

    def fit_transform(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Complete fit and transform pipeline"""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        # Separate features and target
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])

        # Engineer features
        X_engineered = self.engineer_features(X)

        # Select optimal features
        X_selected, selected_features = self.select_optimal_features(X_engineered, y)

        # Store pipeline information
        pipeline_info = {
            "timestamp": datetime.now(),
            "original_features": len(X.columns),
            "engineered_features": len(X_engineered.columns),
            "selected_features": len(selected_features),
            "feature_names": selected_features,
            "selection_method": self.config.feature_selection_method,
        }

        self.feature_pipeline.append(pipeline_info)

        logger.info(
            f"Feature pipeline completed: {len(X.columns)} -> "
            f"{len(X_engineered.columns)} -> {len(selected_features)} features"
        )

        return X_selected, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        if not self.feature_selector.selected_features:
            logger.warning("No fitted pipeline found, returning original data")
            return df

        # Engineer features
        X_engineered = self.engineer_features(df)

        # Select only the previously selected features
        available_features = [
            f
            for f in self.feature_selector.selected_features
            if f in X_engineered.columns
        ]

        if len(available_features) < len(self.feature_selector.selected_features):
            logger.warning(
                f"Only {len(available_features)} of "
                f"{len(self.feature_selector.selected_features)} features available"
            )

        return X_engineered[available_features]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_selector.feature_scores.copy()

    def update_feature_performance(
        self, performance_metric: float, feature_set: List[str] = None
    ):
        """Update performance tracking for feature sets"""
        if feature_set is None:
            feature_set = self.feature_selector.selected_features

        feature_key = str(sorted(feature_set))

        if feature_key not in self.performance_by_features:
            self.performance_by_features[feature_key] = []

        self.performance_by_features[feature_key].append(
            {
                "performance": performance_metric,
                "timestamp": datetime.now(),
                "num_features": len(feature_set),
            }
        )

        logger.info(
            f"Updated feature performance: {performance_metric:.4f} "
            f"for {len(feature_set)} features"
        )

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering pipeline"""
        summary = {
            "config": self.config.__dict__,
            "pipeline_history": self.feature_pipeline,
            "current_features": self.feature_selector.selected_features,
            "feature_scores": self.feature_selector.feature_scores,
            "performance_history": self.performance_by_features,
        }

        return summary

    def save_pipeline(self, filepath: str):
        """Save feature engineering pipeline"""
        pipeline_data = {
            "config": self.config.__dict__,
            "selected_features": self.feature_selector.selected_features,
            "feature_scores": self.feature_selector.feature_scores,
            "pipeline_history": self.feature_pipeline,
            "performance_history": self.performance_by_features,
        }

        with open(filepath, "w") as f:
            json.dump(pipeline_data, f, indent=2, default=str)

        logger.info(f"Feature engineering pipeline saved to {filepath}")

    def load_pipeline(self, filepath: str):
        """Load feature engineering pipeline"""
        try:
            with open(filepath, "r") as f:
                pipeline_data = json.load(f)

            # Restore configuration
            self.config = FeatureConfig(**pipeline_data["config"])

            # Restore feature selector state
            self.feature_selector.selected_features = pipeline_data["selected_features"]
            self.feature_selector.feature_scores = pipeline_data["feature_scores"]

            # Restore history
            self.feature_pipeline = pipeline_data["pipeline_history"]
            self.performance_by_features = pipeline_data["performance_history"]

            logger.info(f"Feature engineering pipeline loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
