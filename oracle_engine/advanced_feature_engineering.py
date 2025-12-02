"""
Enhanced Feature Engineering and Selection System
Part of Phase 2 Optimization for Oracle-X ML System
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
    RFECV,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance results"""

    feature_name: str
    importance_score: float
    ranking: int
    method: str


@dataclass
class FeatureEngineeringResult:
    """Results from feature engineering process"""

    features_created: int
    features_selected: int
    feature_importance: List[FeatureImportance]
    engineering_time: float
    selection_time: float


class AdvancedFeatureEngineer:
    """
    Enhanced feature engineering with automated selection
    Part of Phase 2 ML optimization
    """

    def __init__(self):
        self.feature_importance_history = {}
        self.scaler = None
        self.selected_features = None
        self.feature_selector = None

        # Configuration
        self.max_features = 50  # Limit to prevent overfitting
        self.selection_methods = ["mutual_info", "f_score", "rfe"]
        self.scaling_method = "robust"  # robust, standard, minmax

    def create_technical_features(
        self, df: pd.DataFrame, price_col: str = "close"
    ) -> pd.DataFrame:
        """Create comprehensive technical indicators"""
        if df.empty or price_col not in df.columns:
            return df

        result_df = df.copy()

        try:
            # Price-based features
            result_df["returns_1d"] = result_df[price_col].pct_change()
            result_df["returns_5d"] = result_df[price_col].pct_change(5)
            result_df["returns_10d"] = result_df[price_col].pct_change(10)

            # Volatility features
            result_df["volatility_5d"] = result_df["returns_1d"].rolling(5).std()
            result_df["volatility_10d"] = result_df["returns_1d"].rolling(10).std()
            result_df["volatility_20d"] = result_df["returns_1d"].rolling(20).std()

            # Moving averages
            for period in [5, 10, 20, 50]:
                result_df[f"sma_{period}"] = result_df[price_col].rolling(period).mean()
                result_df[f"ema_{period}"] = (
                    result_df[price_col].ewm(span=period).mean()
                )
                result_df[f"price_to_sma_{period}"] = (
                    result_df[price_col] / result_df[f"sma_{period}"]
                )

            # Advanced technical indicators
            result_df = self._add_rsi(result_df, price_col)
            result_df = self._add_macd(result_df, price_col)
            result_df = self._add_bollinger_bands(result_df, price_col)
            result_df = self._add_momentum_indicators(result_df, price_col)

            # Market structure features
            result_df = self._add_market_structure_features(result_df, price_col)

            logger.info(
                f"Created {len(result_df.columns) - len(df.columns)} technical features"
            )

        except Exception as e:
            logger.warning(f"Error creating technical features: {e}")

        return result_df

    def _add_rsi(
        self, df: pd.DataFrame, price_col: str, periods: List[int] = [14, 21]
    ) -> pd.DataFrame:
        """Add RSI indicators"""
        for period in periods:
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        return df

    def _add_macd(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add MACD indicators"""
        ema_12 = df[price_col].ewm(span=12).mean()
        ema_26 = df[price_col].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        return df

    def _add_bollinger_bands(
        self, df: pd.DataFrame, price_col: str, period: int = 20
    ) -> pd.DataFrame:
        """Add Bollinger Bands indicators"""
        sma = df[price_col].rolling(period).mean()
        std = df[price_col].rolling(period).std()
        df[f"bb_upper_{period}"] = sma + (2 * std)
        df[f"bb_lower_{period}"] = sma - (2 * std)
        df[f"bb_position_{period}"] = (df[price_col] - df[f"bb_lower_{period}"]) / (
            df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]
        )
        df[f"bb_width_{period}"] = (
            df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]
        ) / sma
        return df

    def _add_momentum_indicators(
        self, df: pd.DataFrame, price_col: str
    ) -> pd.DataFrame:
        """Add momentum-based indicators"""
        # Rate of Change
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = (
                (df[price_col] - df[price_col].shift(period))
                / df[price_col].shift(period)
            ) * 100

        # Williams %R
        for period in [14, 21]:
            if "high" in df.columns and "low" in df.columns:
                highest_high = df["high"].rolling(period).max()
                lowest_low = df["low"].rolling(period).min()
                df[f"williams_r_{period}"] = (
                    -100 * (highest_high - df[price_col]) / (highest_high - lowest_low)
                )

        return df

    def _add_market_structure_features(
        self, df: pd.DataFrame, price_col: str
    ) -> pd.DataFrame:
        """Add market structure features"""
        # Support and resistance levels (simplified)
        df["local_max"] = df[price_col].rolling(5, center=True).max() == df[price_col]
        df["local_min"] = df[price_col].rolling(5, center=True).min() == df[price_col]

        # Trend strength
        df["trend_strength_10"] = (
            df[price_col]
            .rolling(10)
            .apply(
                lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) == 10 else 0
            )
        )

        return df

    def create_sentiment_features(
        self, df: pd.DataFrame, sentiment_data: Dict
    ) -> pd.DataFrame:
        """Create sentiment-based features"""
        result_df = df.copy()

        if not sentiment_data:
            # Add default sentiment features
            result_df["sentiment_score"] = 0.0
            result_df["sentiment_confidence"] = 0.0
            result_df["sentiment_quality"] = 0.0
            return result_df

        # Add raw sentiment features
        result_df["sentiment_score"] = sentiment_data.get("overall_sentiment", 0.0)
        result_df["sentiment_confidence"] = sentiment_data.get("confidence", 0.0)
        result_df["sentiment_quality"] = sentiment_data.get("quality_score", 0.0)

        # Add derived sentiment features
        result_df["sentiment_momentum"] = result_df["sentiment_score"].diff()
        result_df["sentiment_volatility"] = (
            result_df["sentiment_score"].rolling(5).std()
        )
        result_df["sentiment_trend"] = (
            result_df["sentiment_score"]
            .rolling(10)
            .apply(
                lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) == 10 else 0
            )
        )

        return result_df

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, prediction_type: str = "regression"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Intelligent feature selection using multiple methods
        """
        import time

        start_time = time.time()

        if X.empty or len(X.columns) <= self.max_features:
            return X, list(X.columns)

        # Clean data
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)

        if len(X_clean.columns) == 0:
            logger.warning("No numeric features found for selection")
            return X, []

        # Scale features for selection
        scaler = RobustScaler() if self.scaling_method == "robust" else StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        X_scaled_df = pd.DataFrame(
            X_scaled, columns=X_clean.columns, index=X_clean.index
        )

        selected_features = set()
        importance_scores = {}

        try:
            # Method 1: Statistical tests
            k_best = min(self.max_features // 2, len(X_clean.columns))
            if prediction_type == "classification":
                selector = SelectKBest(score_func=f_classif, k=k_best)
            else:
                selector = SelectKBest(score_func=f_regression, k=k_best)

            X_selected = selector.fit_transform(X_scaled, y)
            statistical_features = X_clean.columns[selector.get_support()].tolist()
            selected_features.update(statistical_features)

            # Store importance scores
            for i, feature in enumerate(X_clean.columns):
                if selector.get_support()[i]:
                    importance_scores[feature] = selector.scores_[i]

            logger.info(f"Statistical selection: {len(statistical_features)} features")

        except Exception as e:
            logger.warning(f"Statistical feature selection failed: {e}")

        try:
            # Method 2: Mutual information
            k_mutual = min(self.max_features // 2, len(X_clean.columns))
            if prediction_type == "classification":
                mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_scaled, y, random_state=42)

            # Select top features by mutual information
            mi_indices = np.argsort(mi_scores)[-k_mutual:]
            mi_features = X_clean.columns[mi_indices].tolist()
            selected_features.update(mi_features)

            # Store MI scores
            for i, feature in enumerate(X_clean.columns):
                if i in mi_indices:
                    importance_scores[feature] = mi_scores[i]

            logger.info(f"Mutual information selection: {len(mi_features)} features")

        except Exception as e:
            logger.warning(f"Mutual information selection failed: {e}")

        try:
            # Method 3: Recursive feature elimination (limited for performance)
            if len(X_clean.columns) <= 100:  # Only for smaller feature sets
                if prediction_type == "classification":
                    estimator = RandomForestClassifier(
                        n_estimators=10, random_state=42, n_jobs=1
                    )
                else:
                    estimator = RandomForestRegressor(
                        n_estimators=10, random_state=42, n_jobs=1
                    )

                n_features = min(self.max_features // 3, len(X_clean.columns))
                rfe = RFE(estimator, n_features_to_select=n_features, step=5)
                rfe.fit(X_scaled, y)

                rfe_features = X_clean.columns[rfe.support_].tolist()
                selected_features.update(rfe_features)

                # Store RFE rankings as importance
                for i, feature in enumerate(X_clean.columns):
                    if rfe.support_[i]:
                        importance_scores[feature] = 1.0 / rfe.ranking_[i]

                logger.info(f"RFE selection: {len(rfe_features)} features")

        except Exception as e:
            logger.warning(f"RFE selection failed: {e}")

        # Combine and limit features
        final_features = list(selected_features)[: self.max_features]

        if not final_features:
            # Fallback: use correlation-based selection
            correlations = X_clean.corrwith(y).abs().sort_values(ascending=False)
            final_features = correlations.head(self.max_features).index.tolist()
            logger.info(
                f"Fallback correlation selection: {len(final_features)} features"
            )

        # Create importance results
        self.feature_importance_history = [
            FeatureImportance(
                feature_name=feature,
                importance_score=importance_scores.get(feature, 0.0),
                ranking=i + 1,
                method="combined",
            )
            for i, feature in enumerate(final_features)
        ]

        selection_time = time.time() - start_time
        logger.info(
            f"Feature selection completed in {selection_time:.2f}s: {len(final_features)}/{len(X.columns)} features"
        )

        # Return selected features
        X_selected = X[final_features] if final_features else X
        return X_selected, final_features

    def create_targets(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        horizons: List[int] = [1, 5, 10],
    ) -> pd.DataFrame:
        """Create prediction targets for different horizons"""
        result_df = df.copy()

        for horizon in horizons:
            # Price direction (classification)
            future_price = df[price_col].shift(-horizon)
            result_df[f"price_direction_{horizon}d"] = (
                future_price > df[price_col]
            ).astype(int)

            # Price target (regression)
            result_df[f"price_target_{horizon}d"] = (
                future_price / df[price_col] - 1
            ) * 100

        return result_df

    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of feature importance analysis"""
        if not self.feature_importance_history:
            return {}

        # Sort by importance score
        sorted_features = sorted(
            self.feature_importance_history,
            key=lambda x: x.importance_score,
            reverse=True,
        )

        return {
            "top_features": [f.feature_name for f in sorted_features[:10]],
            "importance_scores": {
                f.feature_name: f.importance_score for f in sorted_features
            },
            "total_features": len(sorted_features),
            "selection_method": "combined",
        }
