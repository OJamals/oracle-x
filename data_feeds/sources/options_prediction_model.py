"""
Options Prediction Model for Oracle-X Options Prediction Pipeline

This module implements a sophisticated ML-based prediction model for options price movement,
combining multiple signal sources, advanced feature engineering, and ensemble learning
to identify high-probability options trading opportunities.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Using alternative models.")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import existing components
from data_feeds.sources.options_valuation_engine import (
    OptionsValuationEngine,
    OptionContract,
    OptionType,
)
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine

# Configure logging
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals"""

    TECHNICAL = "technical"
    OPTIONS_FLOW = "options_flow"
    SENTIMENT = "sentiment"
    MARKET_STRUCTURE = "market_structure"
    FUNDAMENTAL = "fundamental"
    VALUATION = "valuation"


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TechnicalSignals:
    """Technical analysis signals"""

    rsi: float
    macd_signal: float
    moving_avg_signal: float
    bollinger_signal: float
    volume_signal: float
    support_resistance_signal: float
    momentum_score: float
    trend_strength: float
    volatility_regime: str  # 'low', 'normal', 'high'


@dataclass
class OptionsFlowSignals:
    """Options flow and positioning signals"""

    unusual_activity_score: float
    put_call_ratio: float
    put_call_ratio_change: float
    open_interest_change: float
    volume_oi_ratio: float
    smart_money_indicator: float
    large_block_ratio: float
    sweep_score: float


@dataclass
class SentimentSignals:
    """Market sentiment signals"""

    overall_sentiment: float
    sentiment_momentum: float
    news_sentiment: float
    social_sentiment: float
    analyst_rating_change: float
    retail_interest: float
    institutional_sentiment: float


@dataclass
class MarketStructureSignals:
    """Market microstructure signals"""

    gex_level: float  # Gamma exposure
    dex_level: float  # Delta exposure
    max_pain_distance: float
    dark_pool_ratio: float
    market_breadth: float
    vix_regime: str  # 'low', 'normal', 'elevated', 'crisis'
    correlation_score: float


@dataclass
class AggregatedSignals:
    """Aggregated signals from all sources"""

    technical: TechnicalSignals
    options_flow: OptionsFlowSignals
    sentiment: SentimentSignals
    market_structure: MarketStructureSignals
    valuation_score: float
    timestamp: datetime
    quality_score: float


@dataclass
class PredictionResult:
    """Options price movement prediction result"""

    contract: OptionContract
    price_increase_probability: float
    expected_return: float
    confidence: PredictionConfidence
    opportunity_score: float
    signals: AggregatedSignals
    feature_importance: Dict[str, float]
    risk_metrics: Dict[str, float]
    recommendation: str  # 'strong_buy', 'buy', 'hold', 'avoid'
    key_drivers: List[str]
    timestamp: datetime


@dataclass
class ModelPerformance:
    """Model performance metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return: float
    total_predictions: int
    timestamp: datetime


class FeatureEngineering:
    """Advanced feature engineering for options prediction"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names = []
        self.feature_stats = {}

    def engineer_features(
        self,
        market_data: pd.DataFrame,
        options_data: pd.DataFrame,
        sentiment_data: Optional[Dict] = None,
        market_internals: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Engineer comprehensive feature set for options prediction

        Returns:
            DataFrame with 50+ engineered features
        """
        features = pd.DataFrame()

        # Technical features
        tech_features = self._create_technical_features(market_data)
        features = pd.concat([features, tech_features], axis=1)

        # Options-specific features
        options_features = self._create_options_features(options_data)
        features = pd.concat([features, options_features], axis=1)

        # Sentiment features
        if sentiment_data:
            sent_features = self._create_sentiment_features(sentiment_data)
            features = pd.concat([features, sent_features], axis=1)

        # Market structure features
        if market_internals:
            struct_features = self._create_market_structure_features(market_internals)
            features = pd.concat([features, struct_features], axis=1)

        # Cross-feature interactions
        interaction_features = self._create_interaction_features(features)
        features = pd.concat([features, interaction_features], axis=1)

        # Time-based features
        time_features = self._create_time_features(options_data)
        features = pd.concat([features, time_features], axis=1)

        self.feature_names = features.columns.tolist()

        # Handle missing values using modern pandas syntax
        features = features.ffill().fillna(0)

        return features

    def _create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        features = pd.DataFrame(index=data.index)

        # Price-based features
        features["returns_1d"] = data["Close"].pct_change(1)
        features["returns_5d"] = data["Close"].pct_change(5)
        features["returns_20d"] = data["Close"].pct_change(20)

        # Moving averages
        features["sma_20"] = data["Close"].rolling(20).mean()
        features["sma_50"] = data["Close"].rolling(50).mean()
        features["ema_12"] = data["Close"].ewm(span=12).mean()
        features["ema_26"] = data["Close"].ewm(span=26).mean()

        # Price relative to MAs
        features["price_to_sma20"] = data["Close"] / features["sma_20"]
        features["price_to_sma50"] = data["Close"] / features["sma_50"]

        # MACD
        features["macd"] = features["ema_12"] - features["ema_26"]
        features["macd_signal"] = features["macd"].ewm(span=9).mean()
        features["macd_histogram"] = features["macd"] - features["macd_signal"]

        # RSI
        delta = data["Close"].diff().astype(float)
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta).where(delta < 0, 0.0).rolling(window=14).mean()
        rs = gain / loss
        features["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma = data["Close"].rolling(bb_period).mean()
        std = data["Close"].rolling(bb_period).std()
        features["bb_upper"] = sma + (bb_std * std)
        features["bb_lower"] = sma - (bb_std * std)
        features["bb_width"] = features["bb_upper"] - features["bb_lower"]
        features["bb_position"] = (data["Close"] - features["bb_lower"]) / features[
            "bb_width"
        ]

        # Volume features
        features["volume_sma"] = data["Volume"].rolling(20).mean()
        features["volume_ratio"] = data["Volume"] / features["volume_sma"]
        features["dollar_volume"] = data["Close"] * data["Volume"]

        # Volatility features
        features["volatility_20d"] = data["Close"].pct_change().rolling(
            20
        ).std() * np.sqrt(252)
        features["volatility_60d"] = data["Close"].pct_change().rolling(
            60
        ).std() * np.sqrt(252)
        features["volatility_ratio"] = (
            features["volatility_20d"] / features["volatility_60d"]
        )

        # ATR (Average True Range)
        high_low = data["High"] - data["Low"]
        high_close = (data["High"] - data["Close"].shift()).abs()
        low_close = (data["Low"] - data["Close"].shift()).abs()
        true_range = pd.DataFrame(
            {"hl": high_low, "hc": high_close, "lc": low_close}
        ).max(axis=1)
        features["atr"] = true_range.rolling(14).mean()
        features["atr_ratio"] = features["atr"] / data["Close"]

        # Support/Resistance
        features["distance_from_high"] = (
            data["High"].rolling(20).max() - data["Close"]
        ) / data["Close"]
        features["distance_from_low"] = (
            data["Close"] - data["Low"].rolling(20).min()
        ) / data["Close"]

        return features

    def _create_options_features(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Create options-specific features"""
        features = pd.DataFrame(
            index=options_data.index if not options_data.empty else [0]
        )

        if options_data.empty:
            return features

        # Implied volatility features
        features["iv_current"] = options_data.get("implied_volatility", 0)
        features["iv_rank"] = options_data.get("iv_rank", 50)
        features["iv_percentile"] = options_data.get("iv_percentile", 50)

        # Greeks features
        features["delta"] = options_data.get("delta", 0)
        features["gamma"] = options_data.get("gamma", 0)
        features["theta"] = options_data.get("theta", 0)
        features["vega"] = options_data.get("vega", 0)
        features["rho"] = options_data.get("rho", 0)

        # Greeks momentum
        if "delta" in options_data:
            features["delta_change"] = options_data["delta"].diff()
            features["gamma_change"] = options_data["gamma"].diff()

        # Moneyness features
        if "strike" in options_data and "underlying_price" in options_data:
            features["moneyness"] = (
                options_data["strike"] / options_data["underlying_price"]
            )
            features["distance_from_strike"] = (
                options_data["underlying_price"] - options_data["strike"]
            ) / options_data["strike"]

        # Time decay features
        if "days_to_expiry" in options_data:
            features["days_to_expiry"] = options_data["days_to_expiry"]
            features["time_decay_factor"] = 1 / (1 + features["days_to_expiry"])
            features["theta_decay_rate"] = (
                features["theta"] * features["time_decay_factor"]
            )

        # Volume and OI features
        features["volume"] = options_data.get("volume", 0)
        features["open_interest"] = options_data.get("open_interest", 0)
        features["volume_oi_ratio"] = features["volume"] / (
            features["open_interest"] + 1
        )

        # Put/Call ratios
        if "put_volume" in options_data and "call_volume" in options_data:
            features["put_call_volume_ratio"] = options_data["put_volume"] / (
                options_data["call_volume"] + 1
            )
        if "put_oi" in options_data and "call_oi" in options_data:
            features["put_call_oi_ratio"] = options_data["put_oi"] / (
                options_data["call_oi"] + 1
            )

        # Spread features
        if "bid" in options_data and "ask" in options_data:
            features["bid_ask_spread"] = options_data["ask"] - options_data["bid"]
            features["spread_ratio"] = features["bid_ask_spread"] / (
                (options_data["ask"] + options_data["bid"]) / 2 + 0.01
            )

        return features

    def _create_sentiment_features(self, sentiment_data: Dict) -> pd.DataFrame:
        """Create sentiment-based features"""
        features = pd.DataFrame(index=[0])

        # Overall sentiment scores
        features["sentiment_overall"] = sentiment_data.get("overall_sentiment", 0)
        features["sentiment_confidence"] = sentiment_data.get("confidence", 0.5)

        # Source-specific sentiment
        features["reddit_sentiment"] = sentiment_data.get("reddit_sentiment", 0)
        features["twitter_sentiment"] = sentiment_data.get("twitter_sentiment", 0)
        features["news_sentiment"] = sentiment_data.get("news_sentiment", 0)

        # Sentiment momentum
        features["sentiment_1d_change"] = sentiment_data.get("sentiment_1d_change", 0)
        features["sentiment_7d_change"] = sentiment_data.get("sentiment_7d_change", 0)

        # Sentiment dispersion
        features["sentiment_std"] = sentiment_data.get("sentiment_std", 0)
        features["bullish_ratio"] = sentiment_data.get("bullish_mentions", 0) / max(
            sentiment_data.get("total_mentions", 1), 1
        )
        features["bearish_ratio"] = sentiment_data.get("bearish_mentions", 0) / max(
            sentiment_data.get("total_mentions", 1), 1
        )

        # Analyst sentiment
        features["analyst_rating"] = sentiment_data.get(
            "analyst_rating", 3
        )  # 1-5 scale
        features["analyst_consensus"] = sentiment_data.get("analyst_consensus", 0)
        features["rating_changes"] = sentiment_data.get("rating_changes_30d", 0)

        return features

    def _create_market_structure_features(self, market_data: Dict) -> pd.DataFrame:
        """Create market microstructure features"""
        features = pd.DataFrame(index=[0])

        # Gamma exposure (GEX)
        features["total_gex"] = market_data.get("total_gex", 0)
        features["call_gex"] = market_data.get("call_gex", 0)
        features["put_gex"] = market_data.get("put_gex", 0)
        features["gex_tilt"] = features["call_gex"] / (features["put_gex"] + 1)

        # Max pain
        features["max_pain_level"] = market_data.get("max_pain", 0)
        features["distance_from_max_pain"] = market_data.get(
            "distance_from_max_pain", 0
        )

        # Dark pool activity
        features["dark_pool_volume"] = market_data.get("dark_pool_volume", 0)
        features["dark_pool_ratio"] = market_data.get("dark_pool_ratio", 0)
        features["dark_pool_sentiment"] = market_data.get("dark_pool_sentiment", 0)

        # Market breadth
        features["advance_decline_ratio"] = market_data.get("advance_decline_ratio", 1)
        features["new_highs_lows_ratio"] = market_data.get("new_highs_lows_ratio", 1)
        features["percent_above_ma"] = market_data.get("percent_above_200ma", 50)

        # VIX and volatility regime
        features["vix_level"] = market_data.get("vix", 20)
        features["vix_percentile"] = market_data.get("vix_percentile", 50)
        features["term_structure"] = market_data.get("vix9d_vix_ratio", 1)

        # Correlation and dispersion
        features["correlation_spy"] = market_data.get("correlation_spy", 0.5)
        features["sector_dispersion"] = market_data.get("sector_dispersion", 0)

        return features

    def _create_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different signals"""
        interaction_features = pd.DataFrame(index=features.index)

        # Technical × Sentiment interactions
        if "rsi" in features and "sentiment_overall" in features:
            interaction_features["rsi_sentiment"] = (
                features["rsi"] * features["sentiment_overall"]
            )

        # Volatility × Greeks interactions
        if "volatility_20d" in features and "vega" in features:
            interaction_features["vol_vega_exposure"] = (
                features["volatility_20d"] * features["vega"]
            )

        if "volatility_ratio" in features and "gamma" in features:
            interaction_features["vol_gamma_risk"] = (
                features["volatility_ratio"] * features["gamma"]
            )

        # Flow × Structure interactions
        if "volume_oi_ratio" in features and "total_gex" in features:
            interaction_features["flow_structure_signal"] = features[
                "volume_oi_ratio"
            ] * np.sign(features["total_gex"])

        # Momentum × Sentiment
        if "returns_5d" in features and "sentiment_1d_change" in features:
            interaction_features["momentum_sentiment_align"] = (
                features["returns_5d"] * features["sentiment_1d_change"]
            )

        # Time decay × Volatility
        if "time_decay_factor" in features and "iv_rank" in features:
            interaction_features["decay_vol_premium"] = (
                features["time_decay_factor"] * features["iv_rank"]
            )

        return interaction_features

    def _create_time_features(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        features = pd.DataFrame(
            index=options_data.index if not options_data.empty else [0]
        )

        # Day of week effects
        if "date" in options_data:
            dates = pd.to_datetime(options_data["date"])
            features["day_of_week"] = dates.dt.dayofweek
            features["is_monday"] = (features["day_of_week"] == 0).astype(int)
            features["is_friday"] = (features["day_of_week"] == 4).astype(int)

            # Month effects
            features["month"] = dates.dt.month
            features["is_month_end"] = (dates.dt.day > 25).astype(int)
            features["is_quarter_end"] = dates.dt.quarter.diff().fillna(0).astype(int)

        # Options expiry effects
        if "days_to_expiry" in options_data:
            features["is_expiry_week"] = (options_data["days_to_expiry"] <= 5).astype(
                int
            )
            features["is_expiry_month"] = (options_data["days_to_expiry"] <= 30).astype(
                int
            )

        return features


class SignalAggregator:
    """Aggregates and weights signals from multiple sources"""

    def __init__(self, orchestrator: DataFeedOrchestrator):
        self.orchestrator = orchestrator
        self.signal_weights = {
            SignalType.TECHNICAL: 0.25,
            SignalType.OPTIONS_FLOW: 0.20,
            SignalType.SENTIMENT: 0.15,
            SignalType.MARKET_STRUCTURE: 0.15,
            SignalType.FUNDAMENTAL: 0.10,
            SignalType.VALUATION: 0.15,
        }

    def aggregate_signals(
        self, symbol: str, contract: OptionContract, lookback_days: int = 30
    ) -> AggregatedSignals:
        """
        Aggregate all signal types for a given option contract
        """
        # Get market data
        market_data = self._get_market_data(symbol, lookback_days)

        # Get individual signal components
        technical = self._get_technical_signals(market_data)
        options_flow = self._get_options_flow_signals(symbol, contract)
        sentiment = self._get_sentiment_signals(symbol)
        market_structure = self._get_market_structure_signals(symbol)
        valuation_score = self._get_valuation_signal(contract)

        # Calculate quality score
        quality_score = self._calculate_signal_quality(
            technical, options_flow, sentiment, market_structure
        )

        return AggregatedSignals(
            technical=technical,
            options_flow=options_flow,
            sentiment=sentiment,
            market_structure=market_structure,
            valuation_score=valuation_score,
            timestamp=datetime.now(),
            quality_score=quality_score,
        )

    def _get_market_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Fetch market data from orchestrator"""
        try:
            market_data = self.orchestrator.get_market_data(
                symbol, period=f"{lookback_days}d", interval="1d"
            )
            if market_data and not market_data.data.empty:
                return market_data.data
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")

        # Return empty DataFrame if fetch fails
        return pd.DataFrame()

    def _get_technical_signals(self, market_data: pd.DataFrame) -> TechnicalSignals:
        """Calculate technical analysis signals"""
        if market_data.empty:
            return TechnicalSignals(
                rsi=50,
                macd_signal=0,
                moving_avg_signal=0,
                bollinger_signal=0,
                volume_signal=0,
                support_resistance_signal=0,
                momentum_score=0,
                trend_strength=0,
                volatility_regime="normal",
            )

        # Calculate RSI
        delta = market_data["Close"].diff().astype(float)
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta).where(delta < 0, 0.0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # MACD
        ema_12 = market_data["Close"].ewm(span=12).mean()
        ema_26 = market_data["Close"].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean().iloc[-1]

        # Moving average signals
        sma_20 = market_data["Close"].rolling(20).mean().iloc[-1]
        sma_50 = (
            market_data["Close"].rolling(50).mean().iloc[-1]
            if len(market_data) >= 50
            else sma_20
        )
        current_price = market_data["Close"].iloc[-1]

        ma_signal = 0
        if current_price > sma_20:
            ma_signal += 0.5
        if current_price > sma_50:
            ma_signal += 0.5

        # Bollinger Bands
        bb_sma = market_data["Close"].rolling(20).mean()
        bb_std = market_data["Close"].rolling(20).std()
        bb_upper = bb_sma + (2 * bb_std)
        bb_lower = bb_sma - (2 * bb_std)
        bb_position = (current_price - bb_lower.iloc[-1]) / (
            bb_upper.iloc[-1] - bb_lower.iloc[-1]
        )

        # Volume signal
        volume_sma = market_data["Volume"].rolling(20).mean().iloc[-1]
        current_volume = market_data["Volume"].iloc[-1]
        volume_signal = min(2.0, current_volume / volume_sma) - 1

        # Support/Resistance
        high_20 = market_data["High"].rolling(20).max().iloc[-1]
        low_20 = market_data["Low"].rolling(20).min().iloc[-1]
        sr_signal = (current_price - low_20) / (high_20 - low_20)

        # Momentum
        returns_5d = market_data["Close"].pct_change(5).iloc[-1]
        returns_20d = market_data["Close"].pct_change(20).iloc[-1]
        momentum = (returns_5d + returns_20d) / 2

        # Trend strength
        adx = self._calculate_adx(market_data)

        # Volatility regime
        volatility = market_data["Close"].pct_change().rolling(20).std().iloc[
            -1
        ] * np.sqrt(252)
        vol_regime = "normal"
        if volatility < 0.15:
            vol_regime = "low"
        elif volatility > 0.30:
            vol_regime = "high"

        return TechnicalSignals(
            rsi=rsi,
            macd_signal=macd_signal,
            moving_avg_signal=ma_signal,
            bollinger_signal=bb_position,
            volume_signal=volume_signal,
            support_resistance_signal=sr_signal,
            momentum_score=momentum,
            trend_strength=adx,
            volatility_regime=vol_regime,
        )

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)"""
        try:
            high = data["High"]
            low = data["Low"]
            close = data["Close"]

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()

            # Calculate directional movements
            up_move = high - high.shift()
            down_move = low.shift() - low

            pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

            pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
            neg_di = 100 * (neg_dm.rolling(period).mean() / atr)

            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
            adx = dx.rolling(period).mean().iloc[-1]

            return adx if not np.isnan(adx) else 0
        except:
            return 0

    def _get_options_flow_signals(
        self, symbol: str, contract: OptionContract
    ) -> OptionsFlowSignals:
        """Get options flow and positioning signals"""
        try:
            # Get options analytics from orchestrator
            analytics = self.orchestrator.get_options_analytics(
                symbol, include=["chain", "gex", "max_pain"]
            )

            if not analytics:
                return self._default_options_flow_signals()

            chain = analytics.get("chain", [])

            # Calculate flow metrics
            total_volume = sum(opt.get("volume", 0) for opt in chain)
            total_oi = sum(opt.get("open_interest", 0) for opt in chain)

            put_volume = sum(
                opt.get("volume", 0) for opt in chain if opt.get("put_call") == "put"
            )
            call_volume = sum(
                opt.get("volume", 0) for opt in chain if opt.get("put_call") == "call"
            )

            put_oi = sum(
                opt.get("open_interest", 0)
                for opt in chain
                if opt.get("put_call") == "put"
            )
            call_oi = sum(
                opt.get("open_interest", 0)
                for opt in chain
                if opt.get("put_call") == "call"
            )

            # Put/Call ratios
            pc_ratio = put_volume / max(call_volume, 1)
            pc_oi_ratio = put_oi / max(call_oi, 1)

            # Unusual activity detection
            avg_volume = total_volume / max(len(chain), 1)
            unusual_activity = 0
            large_blocks = 0

            for opt in chain:
                vol = opt.get("volume", 0)
                oi = opt.get("open_interest", 0)

                if vol > avg_volume * 3:
                    unusual_activity += 1

                if vol > 1000:  # Large block threshold
                    large_blocks += 1

            unusual_score = min(1.0, unusual_activity / max(len(chain), 1))
            large_block_ratio = large_blocks / max(len(chain), 1)

            # Smart money indicator (large trades relative to OI)
            smart_money = 0
            for opt in chain:
                vol = opt.get("volume", 0)
                oi = opt.get("open_interest", 0)
                if oi > 0 and vol / oi > 0.5:  # High volume relative to OI
                    smart_money += 1

            smart_money_indicator = min(1.0, smart_money / max(len(chain), 1))

            # Calculate sweep score (simplified)
            sweep_score = 0
            for opt in chain:
                vol = opt.get("volume", 0)
                if vol > 500:  # Large volume threshold
                    sweep_score += 1

            sweep_score = min(1.0, sweep_score / max(len(chain), 1))

            return OptionsFlowSignals(
                unusual_activity_score=unusual_score,
                put_call_ratio=pc_ratio,
                put_call_ratio_change=0,  # Would need historical data
                open_interest_change=0,  # Would need historical data
                volume_oi_ratio=total_volume / max(total_oi, 1),
                smart_money_indicator=smart_money_indicator,
                large_block_ratio=large_block_ratio,
                sweep_score=sweep_score,
            )

        except Exception as e:
            logger.error(f"Failed to get options flow signals: {e}")
            return self._default_options_flow_signals()

    def _default_options_flow_signals(self) -> OptionsFlowSignals:
        """Return default options flow signals when data is unavailable"""
        return OptionsFlowSignals(
            unusual_activity_score=0.0,
            put_call_ratio=1.0,
            put_call_ratio_change=0.0,
            open_interest_change=0.0,
            volume_oi_ratio=0.0,
            smart_money_indicator=0.0,
            large_block_ratio=0.0,
            sweep_score=0.0,
        )

    def _get_sentiment_signals(self, symbol: str) -> SentimentSignals:
        """Get sentiment signals for the symbol"""
        try:
            # Get sentiment data from orchestrator (returns Dict[str, SentimentData])
            sentiment_data_dict = self.orchestrator.get_sentiment_data(symbol)

            if not sentiment_data_dict:
                return self._default_sentiment_signals()

            # Aggregate sentiment from all sources
            total_sentiment = 0.0
            total_confidence = 0.0
            count = 0

            sentiment_momentum = 0.0
            news_list: List[float] = []
            social_list: List[float] = []
            analyst_rating_change = 0.0
            retail_interest = 0.0
            institutional_sentiment = 0.0

            for source, sentiment_data in sentiment_data_dict.items():
                # Defensive handling: tests or mocks may inject non-numeric or Mock objects.
                # Coerce sentiment_score and confidence to floats; skip entries that cannot be
                # converted to numeric values.
                try:
                    score = getattr(sentiment_data, "sentiment_score", None)
                    conf = getattr(sentiment_data, "confidence", 0.0)
                except Exception:
                    # sentiment_data doesn't have expected attributes; skip
                    continue

                try:
                    s_val = float(score)
                except Exception:
                    # Non-numeric score (e.g., Mock) — skip this source
                    continue

                try:
                    c_val = float(conf)
                except Exception:
                    c_val = 0.0

                total_sentiment += s_val
                total_confidence += c_val
                count += 1

                # Extract from raw_data if available
                raw_data = getattr(sentiment_data, "raw_data", {}) or {}

                # Map different sources to specific sentiment types
                if "news" in source.lower() or "yahoo" in source.lower():
                    news_list.append(s_val)
                elif "reddit" in source.lower() or "twitter" in source.lower():
                    social_list.append(s_val)
                    retail_interest += (
                        c_val  # Use confidence as proxy for retail interest
                    )

                # Extract additional metrics from raw_data defensively
                try:
                    sentiment_momentum += float(raw_data.get("sentiment_momentum", 0.0))
                except Exception:
                    sentiment_momentum += 0.0
                try:
                    analyst_rating_change += float(
                        raw_data.get("analyst_rating_change", 0.0)
                    )
                except Exception:
                    analyst_rating_change += 0.0
                try:
                    institutional_sentiment += float(
                        raw_data.get("institutional_sentiment", 0.0)
                    )
                except Exception:
                    institutional_sentiment += 0.0

            # Calculate averages
            if count > 0:
                overall_sentiment = total_sentiment / count
                sentiment_confidence = total_confidence / count
                news_sentiment = max(news_list) if len(news_list) > 0 else 0.0
                social_sentiment = max(social_list) if len(social_list) > 0 else 0.0
                retail_interest = retail_interest / count
                sentiment_momentum = sentiment_momentum / count
                analyst_rating_change = analyst_rating_change / count
                institutional_sentiment = institutional_sentiment / count
            else:
                return self._default_sentiment_signals()

            return SentimentSignals(
                overall_sentiment=overall_sentiment,
                sentiment_momentum=sentiment_momentum,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                analyst_rating_change=analyst_rating_change,
                retail_interest=retail_interest,
                institutional_sentiment=institutional_sentiment,
            )

        except Exception as e:
            logger.error(f"Failed to get sentiment signals: {e}")
            return self._default_sentiment_signals()

    def _default_sentiment_signals(self) -> SentimentSignals:
        """Return default sentiment signals when data is unavailable"""
        return SentimentSignals(
            overall_sentiment=0.0,
            sentiment_momentum=0.0,
            news_sentiment=0.0,
            social_sentiment=0.0,
            analyst_rating_change=0.0,
            retail_interest=0.0,
            institutional_sentiment=0.0,
        )

    def _get_market_structure_signals(self, symbol: str) -> MarketStructureSignals:
        """Get market structure signals"""
        try:
            # Get market data since get_market_internals doesn't exist in orchestrator
            market_data = self.orchestrator.get_market_data(
                symbol, period="1d", interval="1h"
            )

            if not market_data or market_data.data.empty:
                return self._default_market_structure_signals()

            df = market_data.data
            recent_data = df.tail(24)  # Last 24 hours

            # Calculate mock market structure indicators from price data
            volatility = (
                recent_data["Close"].pct_change().std() * 100
                if len(recent_data) > 1
                else 20.0
            )
            volume_ratio = (
                recent_data["Volume"].mean() / df["Volume"].mean()
                if len(df) > len(recent_data)
                else 1.0
            )

            # Mock structure metrics
            gex_level = min(
                max(volatility * -100, -10000), 10000
            )  # Scale volatility to GEX range
            dex_level = gex_level * 0.5  # DEX typically smaller than GEX
            max_pain_distance = (
                abs(recent_data["Close"].iloc[-1] - recent_data["Close"].mean())
                if len(recent_data) > 0
                else 0
            )
            dark_pool_ratio = 0.3  # Default assumption
            market_breadth = 0.5  # Neutral

            # VIX regime based on volatility
            vix_regime = "normal"
            if volatility < 15:
                vix_regime = "low"
            elif volatility > 25:
                vix_regime = "elevated"
            elif volatility > 35:
                vix_regime = "crisis"

            correlation_score = 0.5  # Default neutral correlation

            return MarketStructureSignals(
                gex_level=gex_level,
                dex_level=dex_level,
                max_pain_distance=max_pain_distance,
                dark_pool_ratio=dark_pool_ratio,
                market_breadth=market_breadth,
                vix_regime=vix_regime,
                correlation_score=correlation_score,
            )

        except Exception as e:
            logger.error(f"Failed to get market structure signals: {e}")
            return self._default_market_structure_signals()

    def _default_market_structure_signals(self) -> MarketStructureSignals:
        """Return default market structure signals when data is unavailable"""
        return MarketStructureSignals(
            gex_level=0.0,
            dex_level=0.0,
            max_pain_distance=0.0,
            dark_pool_ratio=0.0,
            market_breadth=0.0,
            vix_regime="normal",
            correlation_score=0.5,
        )

    def _get_valuation_signal(self, contract: OptionContract) -> float:
        """Get valuation signal from options valuation engine"""
        try:
            # Use the valuation engine from orchestrator if available
            valuation_engine = OptionsValuationEngine()

            # Get current market data for the underlying
            market_data = self._get_market_data(contract.symbol, 30)

            if market_data.empty:
                return 0.0

            current_price = market_data["Close"].iloc[-1]

            # Get valuation result
            valuation = valuation_engine.detect_mispricing(
                contract, current_price, market_data
            )

            if valuation:
                return valuation.mispricing_ratio
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Failed to get valuation signal: {e}")
            return 0.0

    def _calculate_signal_quality(
        self,
        technical: TechnicalSignals,
        options_flow: OptionsFlowSignals,
        sentiment: SentimentSignals,
        market_structure: MarketStructureSignals,
    ) -> float:
        """Calculate overall signal quality score"""
        quality_scores = []

        # Technical signal quality (based on data completeness)
        tech_score = 0.8 if technical.rsi != 50 else 0.5  # Default RSI is 50
        quality_scores.append(tech_score)

        # Options flow quality (based on volume and activity)
        flow_score = min(
            1.0, options_flow.unusual_activity_score + options_flow.volume_oi_ratio
        )
        quality_scores.append(flow_score)

        # Sentiment quality (based on sentiment availability)
        sent_score = 0.8 if sentiment.overall_sentiment != 0 else 0.3
        quality_scores.append(sent_score)

        # Market structure quality
        struct_score = 0.7 if market_structure.gex_level != 0 else 0.4
        quality_scores.append(struct_score)

        # Average quality
        return sum(quality_scores) / len(quality_scores)


class OptionsPredictionModel:
    """
    ML-based options prediction model that combines multiple signals
    """

    def __init__(
        self,
        orchestrator: DataFeedOrchestrator,
        valuation_engine: Optional[OptionsValuationEngine] = None,
        ensemble_engine: Optional["EnsemblePredictionEngine"] = None,
    ):
        """
        Initialize the options prediction model

        Args:
            ensemble_engine: ML ensemble engine for predictions
            orchestrator: Data feed orchestrator
            valuation_engine: optional valuation engine (for pricing/valuation)
        """
        # Ensemble engine may be injected (optional in tests)
        self.ensemble_engine: Optional["EnsemblePredictionEngine"] = ensemble_engine
        self.orchestrator = orchestrator
        self.signal_aggregator = SignalAggregator(orchestrator)
        self.feature_engineering = FeatureEngineering()

        # Model components
        self.models = {}
        self.is_trained = False

        # Performance tracking
        self.performance_history = []

        # Optional valuation engine (tests may inject a mock)
        self.valuation_engine: Optional[OptionsValuationEngine] = valuation_engine

        logger.info("Options prediction model initialized")

    def predict(
        self, symbol: str, contract: OptionContract, lookback_days: int = 30
    ) -> PredictionResult:
        """
        Generate prediction for an option contract

        Args:
            symbol: Stock symbol
            contract: Option contract to predict
            lookback_days: Days of historical data to use

        Returns:
            PredictionResult with comprehensive prediction
        """
        try:
            # Step 1: Aggregate all signals
            aggregated_signals = self.signal_aggregator.aggregate_signals(
                symbol, contract, lookback_days
            )

            # Step 2: Engineer features
            market_data = self.signal_aggregator._get_market_data(symbol, lookback_days)
            options_data = self._get_options_data(contract)

            features = self.feature_engineering.engineer_features(
                market_data=market_data,
                options_data=options_data,
                sentiment_data=self._signals_to_dict(aggregated_signals.sentiment),
                market_internals=self._signals_to_dict(
                    aggregated_signals.market_structure
                ),
            )

            # Step 3: Make prediction using ensemble
            if self.ensemble_engine and not features.empty:
                try:
                    from oracle_engine.ensemble_ml_engine import PredictionType

                    prediction_result = self.ensemble_engine.predict(
                        symbol=symbol,
                        prediction_type=PredictionType.PRICE_DIRECTION,
                        horizon_days=5,
                    )

                    if prediction_result:
                        probability = prediction_result.prediction
                        confidence_score = prediction_result.confidence
                    else:
                        probability = 0.5
                        confidence_score = 0.5

                except Exception as e:
                    logger.warning(f"Ensemble prediction failed: {e}")
                    probability = 0.5
                    confidence_score = 0.5
            else:
                # Fallback scoring
                probability, confidence_score = self._fallback_prediction(
                    aggregated_signals
                )

            # Step 4: Calculate expected return
            expected_return = self._calculate_expected_return(
                contract, aggregated_signals, probability
            )

            # Step 5: Determine confidence level
            confidence = self._determine_confidence(
                confidence_score, aggregated_signals.quality_score
            )

            # Step 6: Calculate opportunity score
            opportunity_score = self._calculate_opportunity_score(
                probability, expected_return, confidence_score
            )

            # Step 7: Generate feature importance
            feature_importance = self._calculate_feature_importance(features)

            # Step 8: Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                contract, probability, expected_return
            )

            # Step 9: Generate recommendation
            recommendation = self._generate_recommendation(
                opportunity_score, confidence, expected_return
            )

            # Step 10: Generate key drivers
            key_drivers = self._generate_key_drivers(
                aggregated_signals, feature_importance
            )

            return PredictionResult(
                contract=contract,
                price_increase_probability=probability,
                expected_return=expected_return,
                confidence=confidence,
                opportunity_score=opportunity_score,
                signals=aggregated_signals,
                feature_importance=feature_importance,
                risk_metrics=risk_metrics,
                recommendation=recommendation,
                key_drivers=key_drivers,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            # Return default prediction
            return self._default_prediction_result(contract)

    def _get_options_data(self, contract: OptionContract) -> pd.DataFrame:
        """Convert option contract to DataFrame for feature engineering"""
        data = {
            "implied_volatility": getattr(contract, "implied_volatility", 0.3),
            "delta": getattr(contract, "delta", 0.5),
            "gamma": getattr(contract, "gamma", 0.1),
            "theta": getattr(contract, "theta", -0.05),
            "vega": getattr(contract, "vega", 0.1),
            "rho": getattr(contract, "rho", 0.01),
            "strike": contract.strike,
            "underlying_price": contract.underlying_price,
            "days_to_expiry": contract.time_to_expiry * 365,
            "volume": contract.volume or 0,
            "open_interest": contract.open_interest or 0,
            "bid": float(contract.bid) if contract.bid else 0,
            "ask": float(contract.ask) if contract.ask else 0,
            "date": datetime.now(),
        }

        return pd.DataFrame([data])

    def _signals_to_dict(self, signals) -> Dict[str, Any]:
        """Convert signals dataclass to dictionary"""
        return asdict(signals)

    def _fallback_prediction(self, signals: AggregatedSignals) -> Tuple[float, float]:
        """Fallback prediction when ML model is unavailable"""
        # Simple rule-based prediction
        score = 0.5

        # Technical signals contribution
        if signals.technical.rsi < 30:  # Oversold
            score += 0.1
        elif signals.technical.rsi > 70:  # Overbought
            score -= 0.1

        if signals.technical.macd_signal > 0:
            score += 0.05

        # Sentiment contribution
        sentiment_score = (
            signals.sentiment.overall_sentiment
            if signals.sentiment.overall_sentiment is not None
            else 0.0
        )
        score += sentiment_score * 0.1

        # Options flow contribution
        if signals.options_flow.put_call_ratio < 0.8:  # Bullish flow
            score += 0.05
        elif signals.options_flow.put_call_ratio > 1.2:  # Bearish flow
            score -= 0.05

        # Valuation contribution
        valuation_score = (
            signals.valuation_score if signals.valuation_score is not None else 0.0
        )
        score += valuation_score * 0.1

        # Clamp between 0 and 1
        probability = max(0.0, min(1.0, score))
        confidence = signals.quality_score

        return probability, confidence

    def _calculate_expected_return(
        self, contract: OptionContract, signals: AggregatedSignals, probability: float
    ) -> float:
        """Calculate expected return for the option"""
        # Base expected move from volatility
        volatility = getattr(contract, "implied_volatility", 0.3)
        time_factor = np.sqrt(contract.time_to_expiry)
        expected_move = volatility * time_factor

        # Adjust based on probability and signals
        direction_factor = (probability - 0.5) * 2  # -1 to 1

        # Technical momentum adjustment
        momentum_factor = signals.technical.momentum_score
        if momentum_factor is None:
            momentum_factor = 0.0

        # Combine factors
        expected_return = expected_move * direction_factor * (1 + momentum_factor)

        return expected_return

    # Compatibility wrapper expected by older tests / callers
    def _determine_confidence(
        self,
        prediction_prob: Optional[float] = None,
        signal_quality: Optional[float] = None,
        feature_confidence: float = 0.0,
    ):
        """Compatibility shim used in tests. Returns PredictionConfidence."""
        # Normalize inputs
        p = float(prediction_prob) if prediction_prob is not None else 0.5
        sq = (
            float(signal_quality) / 100.0
            if signal_quality is not None and signal_quality > 1
            else (float(signal_quality) if signal_quality is not None else 0.5)
        )
        # feature_confidence is currently unused in aggregate but accepted
        combined = (p + sq) / 2
        if combined > 0.8:
            return PredictionConfidence.HIGH
        elif combined > 0.6:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW

    def _calculate_opportunity_score(
        self, probability: float, expected_return: float, confidence: float
    ) -> float:
        """Calculate overall opportunity score (0-100)"""
        # Guard against None values
        probability = probability if probability is not None else 0.5
        expected_return = expected_return if expected_return is not None else 0.0
        confidence = confidence if confidence is not None else 0.5

        # Edge over random (50%)
        edge = abs(probability - 0.5) * 2  # 0 to 1

        # Expected return magnitude
        return_magnitude = min(1.0, abs(expected_return) * 2)  # Cap at 100%

        # Confidence factor
        confidence_factor = confidence

        # Composite score
        score = (edge * 0.4 + return_magnitude * 0.4 + confidence_factor * 0.2) * 100

        return min(100, score)

    # Backwards-compatible public API wrappers expected by tests
    def calculate_opportunity_score(
        self, valuation_result, signals: AggregatedSignals, prediction_prob: float
    ) -> float:
        """Compatibility wrapper around internal opportunity score calculation."""
        # valuation_result may provide mispricing_ratio used as modifier
        try:
            val = getattr(valuation_result, "mispricing_ratio", 0.0)
        except Exception:
            val = 0.0

        momentum_score = (
            signals.technical.momentum_score
            if hasattr(signals, "technical")
            and signals.technical.momentum_score is not None
            else 0.0
        )
        base = self._calculate_opportunity_score(
            prediction_prob,
            momentum_score,
            signals.quality_score if hasattr(signals, "quality_score") else 0.5,
        )
        # incorporate valuation as small boost
        return base + (val * 10)

    def predict_price_movement(
        self, symbol: str, contract: OptionContract
    ) -> PredictionResult:
        """Compatibility alias for predict."""
        return self.predict(symbol, contract)

    def rank_opportunities(
        self, predictions: List[PredictionResult]
    ) -> List[PredictionResult]:
        """Rank prediction results by opportunity score descending."""
        return sorted(predictions, key=lambda x: x.opportunity_score, reverse=True)

    def get_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance from internal models or feature_engineering as fallback."""
        # If models have feature_importances_ attribute, aggregate them
        import numpy as _np

        if self.models:
            # Average importances across models
            importances = None
            for m in self.models.values():
                vals = getattr(m, "feature_importances_", None)
                if vals is None:
                    continue
                vals = _np.array(vals)
                if importances is None:
                    importances = vals
                else:
                    importances = importances + vals

            if importances is not None:
                importances = importances / len(self.models)
                names = getattr(self, "feature_engineer", None)
                if names is None:
                    names = getattr(self, "feature_engineering", None)
                feature_names = getattr(names, "feature_names", [])
                if len(feature_names) == len(importances):
                    total = float(_np.sum(importances)) or 1.0
                    return {
                        n: float(v) / total for n, v in zip(feature_names, importances)
                    }

        # Fallback: equal importance for known feature names
        names = getattr(self, "feature_engineer", None)
        if names is None:
            names = getattr(self, "feature_engineering", None)
        feature_names = getattr(names, "feature_names", [])
        n = len(feature_names)
        if n == 0:
            return {}
        imp = 1.0 / n
        return {name: imp for name in feature_names}

    def evaluate_model_performance(self) -> ModelPerformance:
        """Simple performance evaluator over stored prediction history."""
        preds = getattr(self, "predictions_history", [])
        actuals = getattr(self, "actuals_history", [])
        if not preds or not actuals or len(preds) != len(actuals):
            return ModelPerformance(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_roc=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                avg_return=0.0,
                total_predictions=0,
                timestamp=datetime.now(),
            )

        import numpy as _np

        tp = sum(
            1
            for p, a in zip(preds, actuals)
            if (p >= 0.5 and a == 1) or (p < 0.5 and a == 0)
        )
        accuracy = tp / len(preds)
        # Placeholder metrics
        return ModelPerformance(
            accuracy=accuracy,
            precision=accuracy,
            recall=accuracy,
            f1_score=accuracy,
            auc_roc=accuracy,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=accuracy,
            avg_return=float(_np.mean(preds)) if preds else 0.0,
            total_predictions=len(preds),
            timestamp=datetime.now(),
        )

    def _calculate_feature_importance(self, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance (simplified)"""
        if features.empty:
            return {}

        # For now, return equal importance for all features
        # In a real implementation, this would come from trained models
        feature_names = features.columns.tolist()
        n_features = len(feature_names)

        if n_features == 0:
            return {}

        importance = 1.0 / n_features
        return {name: importance for name in feature_names}

    def _calculate_risk_metrics(
        self, contract: OptionContract, probability: float, expected_return: float
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        # Guard against None values
        probability = probability if probability is not None else 0.5
        expected_return = expected_return if expected_return is not None else 0.0

        # Maximum loss (premium paid)
        premium = float(contract.last or contract.ask or 0)
        max_loss = premium

        # Potential max gain (for calls: unlimited, for puts: strike - premium)
        if contract.option_type == OptionType.CALL:
            max_gain = float("inf")  # Unlimited for calls
        else:
            max_gain = contract.strike - premium

        # Expected value
        prob_profit = probability if expected_return > 0 else 1 - probability
        expected_value = (
            prob_profit * abs(expected_return) * premium - (1 - prob_profit) * premium
        )

        # Sharpe-like ratio
        sharpe_ratio = expected_value / max(premium, 0.01)

        # Time decay risk
        theta = getattr(contract, "theta", -0.05)
        time_decay_risk = abs(theta) * 30  # 30-day time decay

        return {
            "max_loss": max_loss,
            "max_gain": max_gain if max_gain != float("inf") else 999999,
            "expected_value": expected_value,
            "sharpe_ratio": sharpe_ratio,
            "time_decay_risk": time_decay_risk,
            "probability_of_profit": prob_profit,
        }

    def _generate_recommendation(
        self,
        opportunity_score: float,
        confidence: PredictionConfidence,
        expected_return: float,
    ) -> str:
        """Generate trading recommendation"""
        if opportunity_score >= 80 and confidence == PredictionConfidence.HIGH:
            return "strong_buy"
        elif opportunity_score >= 65 and confidence in [
            PredictionConfidence.HIGH,
            PredictionConfidence.MEDIUM,
        ]:
            return "buy"
        elif opportunity_score >= 50:
            return "hold"
        else:
            return "avoid"

    def _generate_key_drivers(
        self, signals: AggregatedSignals, feature_importance: Dict[str, float]
    ) -> List[str]:
        """Generate key drivers for the prediction"""
        drivers = []

        # Technical drivers
        if signals.technical.rsi < 30:
            drivers.append("Oversold RSI condition")
        elif signals.technical.rsi > 70:
            drivers.append("Overbought RSI condition")

        if signals.technical.macd_signal > 0.01:
            drivers.append("Positive MACD signal")

        # Sentiment drivers
        if signals.sentiment.overall_sentiment > 0.3:
            drivers.append("Positive sentiment momentum")
        elif signals.sentiment.overall_sentiment < -0.3:
            drivers.append("Negative sentiment momentum")

        # Options flow drivers
        if signals.options_flow.unusual_activity_score > 0.5:
            drivers.append("Unusual options activity detected")

        if signals.options_flow.smart_money_indicator > 0.6:
            drivers.append("Smart money positioning")

        # Valuation drivers
        if signals.valuation_score > 0.2:
            drivers.append("Options mispricing opportunity")

        return drivers[:5]  # Limit to top 5 drivers

    def _default_prediction_result(self, contract: OptionContract) -> PredictionResult:
        """Return default prediction when analysis fails"""
        default_signals = AggregatedSignals(
            technical=TechnicalSignals(
                rsi=50,
                macd_signal=0,
                moving_avg_signal=0,
                bollinger_signal=0,
                volume_signal=0,
                support_resistance_signal=0,
                momentum_score=0,
                trend_strength=0,
                volatility_regime="normal",
            ),
            options_flow=OptionsFlowSignals(
                unusual_activity_score=0,
                put_call_ratio=1,
                put_call_ratio_change=0,
                open_interest_change=0,
                volume_oi_ratio=0,
                smart_money_indicator=0,
                large_block_ratio=0,
                sweep_score=0,
            ),
            sentiment=SentimentSignals(
                overall_sentiment=0,
                sentiment_momentum=0,
                news_sentiment=0,
                social_sentiment=0,
                analyst_rating_change=0,
                retail_interest=0,
                institutional_sentiment=0,
            ),
            market_structure=MarketStructureSignals(
                gex_level=0,
                dex_level=0,
                max_pain_distance=0,
                dark_pool_ratio=0,
                market_breadth=0,
                vix_regime="normal",
                correlation_score=0.5,
            ),
            valuation_score=0,
            timestamp=datetime.now(),
            quality_score=0.3,
        )

        return PredictionResult(
            contract=contract,
            price_increase_probability=0.5,
            expected_return=0.0,
            confidence=PredictionConfidence.LOW,
            opportunity_score=30.0,
            signals=default_signals,
            feature_importance={},
            risk_metrics={
                "max_loss": 0,
                "max_gain": 0,
                "expected_value": 0,
                "sharpe_ratio": 0,
            },
            recommendation="avoid",
            key_drivers=["Insufficient data for analysis"],
            timestamp=datetime.now(),
        )
