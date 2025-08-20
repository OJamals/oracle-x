"""
Unit tests for Options Prediction Model

Tests the sophisticated ML-based prediction model for options price movement,
including feature engineering, signal aggregation, and ensemble prediction.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd

from data_feeds.options_prediction_model import (
    FeatureEngineering,
    SignalAggregator,
    OptionsPredictionModel,
    TechnicalSignals,
    OptionsFlowSignals,
    SentimentSignals,
    MarketStructureSignals,
    AggregatedSignals,
    PredictionResult,
    PredictionConfidence,
    SignalType,
    ModelPerformance
)
from data_feeds.options_valuation_engine import (
    OptionContract,
    OptionType,
    ValuationResult
)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_engineer = FeatureEngineering()
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        self.market_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 115, 100),
            'Low': np.random.uniform(95, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        })
        self.market_data.set_index('Date', inplace=True)
        
        # Create sample options data
        self.options_data = pd.DataFrame({
            'strike': [100, 105, 110],
            'underlying_price': [105, 105, 105],
            'implied_volatility': [0.2, 0.25, 0.3],
            'delta': [0.6, 0.5, 0.4],
            'gamma': [0.02, 0.03, 0.02],
            'theta': [-0.05, -0.06, -0.07],
            'vega': [0.1, 0.12, 0.11],
            'volume': [1000, 1500, 800],
            'open_interest': [5000, 7000, 3000],
            'bid': [5.0, 3.0, 1.5],
            'ask': [5.2, 3.2, 1.7],
            'days_to_expiry': [30, 30, 30]
        })
    
    def test_engineer_features_creates_all_feature_types(self):
        """Test that all feature types are created"""
        features = self.feature_engineer.engineer_features(
            self.market_data,
            self.options_data
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 50)  # Should have 50+ features
        
        # Check for key feature categories
        feature_names = features.columns.tolist()
        
        # Technical features
        self.assertIn('rsi', feature_names)
        self.assertIn('macd', feature_names)
        self.assertIn('bb_position', feature_names)
        
        # Options features
        self.assertIn('iv_current', feature_names)
        self.assertIn('delta', feature_names)
        self.assertIn('moneyness', feature_names)
        
        # Volume features
        self.assertIn('volume_ratio', feature_names)
        self.assertIn('volume_oi_ratio', feature_names)
    
    def test_technical_features_calculation(self):
        """Test technical indicator calculations"""
        tech_features = self.feature_engineer._create_technical_features(self.market_data)
        
        # Check RSI is in valid range
        rsi_values = tech_features['rsi'].dropna()
        self.assertTrue(all(0 <= rsi <= 100 for rsi in rsi_values))
        
        # Check moving averages
        self.assertIn('sma_20', tech_features.columns)
        self.assertIn('sma_50', tech_features.columns)
        
        # Check volatility features
        self.assertIn('volatility_20d', tech_features.columns)
        self.assertIn('atr', tech_features.columns)
    
    def test_options_features_calculation(self):
        """Test options-specific feature calculations"""
        options_features = self.feature_engineer._create_options_features(self.options_data)
        
        # Check Greeks features
        self.assertIn('delta', options_features.columns)
        self.assertIn('gamma', options_features.columns)
        self.assertIn('theta', options_features.columns)
        self.assertIn('vega', options_features.columns)
        
        # Check moneyness calculation
        self.assertIn('moneyness', options_features.columns)
        moneyness = options_features['moneyness'].iloc[0]
        expected = self.options_data['strike'].iloc[0] / self.options_data['underlying_price'].iloc[0]
        self.assertAlmostEqual(moneyness, expected, places=4)
        
        # Check spread features
        self.assertIn('bid_ask_spread', options_features.columns)
        spread = options_features['bid_ask_spread'].iloc[0]
        expected_spread = self.options_data['ask'].iloc[0] - self.options_data['bid'].iloc[0]
        self.assertAlmostEqual(spread, expected_spread, places=4)
    
    def test_sentiment_features_creation(self):
        """Test sentiment feature creation"""
        sentiment_data = {
            'overall_sentiment': 0.6,
            'confidence': 0.8,
            'reddit_sentiment': 0.5,
            'twitter_sentiment': 0.7,
            'news_sentiment': 0.6,
            'bullish_mentions': 150,
            'bearish_mentions': 50,
            'total_mentions': 200
        }
        
        sent_features = self.feature_engineer._create_sentiment_features(sentiment_data)
        
        self.assertEqual(sent_features['sentiment_overall'].iloc[0], 0.6)
        self.assertEqual(sent_features['sentiment_confidence'].iloc[0], 0.8)
        self.assertAlmostEqual(sent_features['bullish_ratio'].iloc[0], 0.75, places=2)
        self.assertAlmostEqual(sent_features['bearish_ratio'].iloc[0], 0.25, places=2)
    
    def test_market_structure_features(self):
        """Test market structure feature creation"""
        market_data = {
            'total_gex': 1000000,
            'call_gex': 600000,
            'put_gex': 400000,
            'max_pain': 105,
            'distance_from_max_pain': 0.02,
            'dark_pool_volume': 500000,
            'dark_pool_ratio': 0.15,
            'vix': 18,
            'vix_percentile': 45
        }
        
        struct_features = self.feature_engineer._create_market_structure_features(market_data)
        
        self.assertEqual(struct_features['total_gex'].iloc[0], 1000000)
        self.assertEqual(struct_features['vix_level'].iloc[0], 18)
        
        # Check GEX tilt calculation
        gex_tilt = struct_features['gex_tilt'].iloc[0]
        expected_tilt = 600000 / (400000 + 1)
        self.assertAlmostEqual(gex_tilt, expected_tilt, places=2)
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        # Create data with missing values
        incomplete_data = self.market_data.copy()
        incomplete_data.loc[incomplete_data.index[:10], 'Close'] = np.nan
        
        features = self.feature_engineer.engineer_features(
            incomplete_data,
            self.options_data
        )
        
        # Should handle missing values without errors
        self.assertFalse(features.isnull().all().any())  # No columns should be all NaN


class TestSignalAggregator(unittest.TestCase):
    """Test signal aggregation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_orchestrator = Mock()
        self.aggregator = SignalAggregator(self.mock_orchestrator)
        
        # Create sample contract
        self.contract = OptionContract(
            symbol='AAPL',
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL,
            bid=5.0,
            ask=5.2,
            last=5.1,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            underlying_price=155.0
        )
    
    def test_aggregate_signals_combines_all_sources(self):
        """Test that all signal sources are aggregated"""
        # Mock market data response
        market_data = pd.DataFrame({
            'Close': [150, 151, 152, 153, 154],
            'High': [151, 152, 153, 154, 155],
            'Low': [149, 150, 151, 152, 153],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        mock_market_response = Mock()
        mock_market_response.data = market_data
        self.mock_orchestrator.get_market_data.return_value = mock_market_response
        
        # Mock options analytics
        self.mock_orchestrator.get_options_analytics.return_value = {
            'chain': [
                {'volume': 1000, 'open_interest': 5000, 'put_call': 'call'},
                {'volume': 800, 'open_interest': 4000, 'put_call': 'put'}
            ],
            'gex': {'total_gamma_exposure': 1000000},
            'max_pain': {'strike': 150}
        }
        
        # Mock sentiment data
        mock_sentiment = Mock()
        mock_sentiment.sentiment_score = 0.6
        self.mock_orchestrator.get_sentiment_data.return_value = {
            'reddit': mock_sentiment
        }
        self.mock_orchestrator.get_advanced_sentiment_data.return_value = mock_sentiment
        
        # Mock market breadth
        mock_breadth = Mock()
        mock_breadth.advancers = 1500
        mock_breadth.decliners = 1000
        self.mock_orchestrator.get_market_breadth.return_value = mock_breadth
        
        # Aggregate signals
        signals = self.aggregator.aggregate_signals('AAPL', self.contract)
        
        self.assertIsInstance(signals, AggregatedSignals)
        self.assertIsInstance(signals.technical, TechnicalSignals)
        self.assertIsInstance(signals.options_flow, OptionsFlowSignals)
        self.assertIsInstance(signals.sentiment, SentimentSignals)
        self.assertIsInstance(signals.market_structure, MarketStructureSignals)
        self.assertGreater(signals.quality_score, 0)
    
    def test_technical_signals_calculation(self):
        """Test technical signal calculations"""
        market_data = pd.DataFrame({
            'Close': np.linspace(100, 110, 50),
            'High': np.linspace(101, 111, 50),
            'Low': np.linspace(99, 109, 50),
            'Volume': np.random.randint(1000000, 2000000, 50)
        })
        
        technical = self.aggregator._get_technical_signals(market_data)
        
        self.assertIsInstance(technical, TechnicalSignals)
        self.assertTrue(0 <= technical.rsi <= 100)
        self.assertIn(technical.volatility_regime, ['low', 'normal', 'high'])
        self.assertIsInstance(technical.momentum_score, float)
    
    def test_options_flow_signals_calculation(self):
        """Test options flow signal calculations"""
        self.mock_orchestrator.get_options_analytics.return_value = {
            'chain': [
                {'volume': 5000, 'open_interest': 10000, 'put_call': 'call'},
                {'volume': 3000, 'open_interest': 8000, 'put_call': 'put'},
                {'volume': 2000, 'open_interest': 5000, 'put_call': 'call'}
            ]
        }
        
        flow_signals = self.aggregator._get_options_flow_signals('AAPL', self.contract)
        
        self.assertIsInstance(flow_signals, OptionsFlowSignals)
        self.assertGreater(flow_signals.put_call_ratio, 0)
        self.assertTrue(0 <= flow_signals.unusual_activity_score <= 1)
        self.assertTrue(0 <= flow_signals.smart_money_indicator <= 1)
    
    def test_sentiment_signals_aggregation(self):
        """Test sentiment signal aggregation"""
        mock_reddit = Mock()
        mock_reddit.sentiment_score = 0.5
        
        mock_twitter = Mock()
        mock_twitter.sentiment_score = 0.7
        
        self.mock_orchestrator.get_sentiment_data.return_value = {
            'reddit': mock_reddit,
            'twitter': mock_twitter
        }
        
        mock_advanced = Mock()
        mock_advanced.sentiment_score = 0.6
        self.mock_orchestrator.get_advanced_sentiment_data.return_value = mock_advanced
        
        sentiment = self.aggregator._get_sentiment_signals('AAPL')
        
        self.assertIsInstance(sentiment, SentimentSignals)
        self.assertEqual(sentiment.overall_sentiment, 0.6)
        self.assertEqual(sentiment.social_sentiment, 0.7)


class TestOptionsPredictionModel(unittest.TestCase):
    """Test the main options prediction model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_orchestrator = Mock()
        self.mock_valuation_engine = Mock()
        
        with patch('data_feeds.options_prediction_model.OptionsValuationEngine') as mock_engine:
            mock_engine.return_value = self.mock_valuation_engine
            self.model = OptionsPredictionModel(
                orchestrator=self.mock_orchestrator,
                valuation_engine=self.mock_valuation_engine
            )
        
        # Create sample contract
        self.contract = OptionContract(
            symbol='AAPL',
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL,
            bid=5.0,
            ask=5.2,
            last=5.1,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            underlying_price=155.0
        )
    
    def test_predict_price_movement(self):
        """Test price movement prediction"""
        # Mock aggregated signals
        mock_signals = Mock(spec=AggregatedSignals)
        mock_signals.technical = Mock()
        mock_signals.technical.rsi = 60
        mock_signals.technical.momentum_score = 0.5
        mock_signals.technical.volatility_regime = 'normal'
        
        mock_signals.options_flow = Mock()
        mock_signals.options_flow.unusual_activity_score = 0.7
        mock_signals.options_flow.smart_money_indicator = 0.6
        
        mock_signals.sentiment = Mock()
        mock_signals.sentiment.overall_sentiment = 0.6
        
        mock_signals.market_structure = Mock()
        mock_signals.market_structure.gex_level = 1000000
        mock_signals.market_structure.vix_regime = 'normal'
        
        mock_signals.valuation_score = 0.7
        mock_signals.quality_score = 80
        
        self.model.signal_aggregator.aggregate_signals = Mock(return_value=mock_signals)
        
        # Mock feature engineering
        mock_features = pd.DataFrame(np.random.randn(1, 50))
        self.model.feature_engineer.engineer_features = Mock(return_value=mock_features)
        
        # Mock ML predictions
        self.model._get_ml_predictions = Mock(return_value=(0.65, {'model1': 0.7, 'model2': 0.6}))
        
        # Make prediction
        result = self.model.predict_price_movement('AAPL', self.contract)
        
        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.contract, self.contract)
        self.assertTrue(0 <= result.price_increase_probability <= 1)
        self.assertIn(result.confidence, [PredictionConfidence.HIGH, 
                                         PredictionConfidence.MEDIUM, 
                                         PredictionConfidence.LOW])
        self.assertIsInstance(result.opportunity_score, float)
        self.assertIn(result.recommendation, ['strong_buy', 'buy', 'hold', 'avoid'])
    
    def test_calculate_opportunity_score(self):
        """Test opportunity score calculation"""
        # Create mock valuation result
        mock_valuation = Mock(spec=ValuationResult)
        mock_valuation.mispricing_ratio = 0.1
        mock_valuation.confidence_score = 80
        mock_valuation.is_undervalued = True
        
        # Create mock signals
        mock_signals = Mock(spec=AggregatedSignals)
        mock_signals.technical = Mock()
        mock_signals.technical.momentum_score = 0.5
        mock_signals.technical.trend_strength = 30
        
        mock_signals.options_flow = Mock()
        mock_signals.options_flow.unusual_activity_score = 0.7
        mock_signals.options_flow.smart_money_indicator = 0.6
        
        mock_signals.sentiment = Mock()
        mock_signals.sentiment.overall_sentiment = 0.6
        
        mock_signals.valuation_score = 0.7
        
        score = self.model.calculate_opportunity_score(
            mock_valuation,
            mock_signals,
            0.65  # prediction probability
        )
        
        self.assertTrue(0 <= score <= 100)
    
    def test_rank_opportunities(self):
        """Test ranking multiple opportunities"""
        # Create multiple prediction results
        predictions = []
        for i in range(5):
            result = Mock(spec=PredictionResult)
            result.opportunity_score = 50 + i * 10
            result.price_increase_probability = 0.5 + i * 0.1
            result.expected_return = 0.1 + i * 0.05
            predictions.append(result)
        
        ranked = self.model.rank_opportunities(predictions)
        
        # Should be sorted by opportunity score descending
        scores = [r.opportunity_score for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        # Mock some trained models
        mock_rf = Mock()
        mock_rf.feature_importances_ = np.random.rand(50)
        
        self.model.models = {'rf_classifier': mock_rf}
        self.model.feature_engineer.feature_names = [f'feature_{i}' for i in range(50)]
        
        importance = self.model.get_feature_importance()
        
        self.assertIsInstance(importance, dict)
        self.assertTrue(all(isinstance(k, str) for k in importance.keys()))
        self.assertTrue(all(isinstance(v, float) for v in importance.values()))
        
        # Top features should sum to 1.0
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        self.assertAlmostEqual(sum(top_features.values()), 1.0, places=5)
    
    def test_model_performance_tracking(self):
        """Test model performance tracking"""
        # Mock some predictions and actuals
        predictions = [0.7, 0.3, 0.8, 0.6, 0.4]
        actuals = [1, 0, 1, 1, 0]
        
        self.model.predictions_history = predictions
        self.model.actuals_history = actuals
        
        performance = self.model.evaluate_model_performance()
        
        self.assertIsInstance(performance, ModelPerformance)
        self.assertTrue(0 <= performance.accuracy <= 1)
        self.assertTrue(0 <= performance.precision <= 1)
        self.assertTrue(0 <= performance.recall <= 1)
        self.assertTrue(0 <= performance.f1_score <= 1)
    
    def test_confidence_determination(self):
        """Test confidence level determination"""
        # High confidence scenario
        confidence = self.model._determine_confidence(
            prediction_prob=0.85,
            signal_quality=90,
            feature_confidence=0.8
        )
        self.assertEqual(confidence, PredictionConfidence.HIGH)
        
        # Medium confidence scenario
        confidence = self.model._determine_confidence(
            prediction_prob=0.65,
            signal_quality=70,
            feature_confidence=0.6
        )
        self.assertEqual(confidence, PredictionConfidence.MEDIUM)
        
        # Low confidence scenario
        confidence = self.model._determine_confidence(
            prediction_prob=0.55,
            signal_quality=50,
            feature_confidence=0.4
        )
        self.assertEqual(confidence, PredictionConfidence.LOW)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete prediction pipeline"""
    
    @patch('data_feeds.options_prediction_model.DataFeedOrchestrator')
    @patch('data_feeds.options_prediction_model.OptionsValuationEngine')
    def test_end_to_end_prediction_pipeline(self, mock_valuation_class, mock_orchestrator_class):
        """Test the complete prediction pipeline"""
        # Set up mocks
        mock_orchestrator = Mock()
        mock_valuation = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        mock_valuation_class.return_value = mock_valuation
        
        # Create model
        model = OptionsPredictionModel(
            orchestrator=mock_orchestrator,
            valuation_engine=mock_valuation
        )
        
        # Mock data responses
        market_data = pd.DataFrame({
            'Close': np.linspace(100, 110, 100),
            'High': np.linspace(101, 111, 100),
            'Low': np.linspace(99, 109, 100),
            'Volume': np.random.randint(1000000, 2000000, 100)
        })
        
        mock_market_response = Mock()
        mock_market_response.data = market_data
        mock_orchestrator.get_market_data.return_value = mock_market_response
        
        mock_orchestrator.get_options_analytics.return_value = {
            'chain': [{'volume': 1000, 'open_interest': 5000, 'put_call': 'call'}],
            'gex': {'total_gamma_exposure': 1000000},
            'max_pain': {'strike': 105}
        }
        
        mock_sentiment = Mock()
        mock_sentiment.sentiment_score = 0.6
        mock_orchestrator.get_sentiment_data.return_value = {'reddit': mock_sentiment}
        mock_orchestrator.get_advanced_sentiment_data.return_value = mock_sentiment
        
        mock_breadth = Mock()
        mock_breadth.advancers = 1500
        mock_breadth.decliners = 1000
        mock_orchestrator.get_market_breadth.return_value = mock_breadth
        
        # Mock valuation
        mock_val_result = Mock()
        mock_val_result.mispricing_ratio = 0.1
        mock_val_result.confidence_score = 80
        mock_val_result.is_undervalued = True
        mock_valuation.detect_mispricing.return_value = mock_val_result
        
        # Create contract
        contract = OptionContract(
            symbol='AAPL',
            strike=105.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL,
            bid=5.0,
            ask=5.2,
            last=5.1,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            underlying_price=110.0
        )
        
        # Run prediction
        result = model.predict_price_movement('AAPL', contract)
        
        # Validate result
        self.assertIsInstance(result, PredictionResult)
        self.assertTrue(0 <= result.price_increase_probability <= 1)
        self.assertTrue(0 <= result.opportunity_score <= 100)
        self.assertIsInstance(result.signals, AggregatedSignals)
        self.assertIsInstance(result.recommendation, str)
        self.assertIsInstance(result.key_drivers, list)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()