#!/usr/bin/env python3
"""
Integration Tests for Oracle-X Options Prediction Pipeline

Tests the complete pipeline functionality including:
- Pipeline initialization
- Single ticker analysis
- Market scanning
- Position monitoring
- Configuration handling
- Error handling
"""

import unittest
import json
import warnings
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Import pipeline components
from oracle_options_pipeline import (
    create_pipeline,
    OracleOptionsPipeline,
    PipelineConfig,
    RiskTolerance,
    OptionStrategy,
    OptionRecommendation,
    PipelineResult,
    OptionContract,
    OptionType,
    OptionStyle
)

# Suppress warnings during tests
warnings.filterwarnings('ignore')


class TestPipelineConfig(unittest.TestCase):
    """Test pipeline configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = PipelineConfig()
        
        self.assertEqual(config.risk_tolerance, RiskTolerance.MODERATE)
        self.assertEqual(config.max_position_size, 0.05)
        self.assertEqual(config.min_opportunity_score, 70.0)
        self.assertEqual(config.min_confidence, 0.6)
        self.assertEqual(config.min_days_to_expiry, 7)
        self.assertEqual(config.max_days_to_expiry, 90)
        self.assertTrue(config.use_advanced_sentiment)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = PipelineConfig(
            risk_tolerance=RiskTolerance.AGGRESSIVE,
            max_position_size=0.1,
            min_opportunity_score=60.0,
            preferred_strategies=[OptionStrategy.LONG_CALL]
        )
        
        self.assertEqual(config.risk_tolerance, RiskTolerance.AGGRESSIVE)
        self.assertEqual(config.max_position_size, 0.1)
        self.assertEqual(config.min_opportunity_score, 60.0)
        self.assertEqual(len(config.preferred_strategies), 1)
        self.assertEqual(config.preferred_strategies[0], OptionStrategy.LONG_CALL)


class TestPipelineInitialization(unittest.TestCase):
    """Test pipeline initialization"""
    
    def test_create_pipeline_default(self):
        """Test creating pipeline with default config"""
        pipeline = create_pipeline()
        
        self.assertIsInstance(pipeline, OracleOptionsPipeline)
        self.assertIsNotNone(pipeline.config)
        self.assertIsNotNone(pipeline.orchestrator)
        self.assertIsNotNone(pipeline.valuation_engine)
        self.assertIsNotNone(pipeline.signal_aggregator)
    
    def test_create_pipeline_custom_config(self):
        """Test creating pipeline with custom config"""
        config = {
            'risk_tolerance': 'conservative',
            'max_position_size': 0.02,
            'min_opportunity_score': 80.0
        }
        
        pipeline = create_pipeline(config)
        
        self.assertEqual(pipeline.config.risk_tolerance, RiskTolerance.CONSERVATIVE)
        self.assertEqual(pipeline.config.max_position_size, 0.02)
        self.assertEqual(pipeline.config.min_opportunity_score, 80.0)


class TestOptionRecommendation(unittest.TestCase):
    """Test OptionRecommendation data class"""
    
    def setUp(self):
        """Set up test recommendation"""
        self.contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL,
            bid=5.0,
            ask=5.2,
            volume=1000,
            open_interest=5000,
            underlying_price=155.0
        )
        
        self.recommendation = OptionRecommendation(
            symbol="AAPL",
            contract=self.contract,
            strategy=OptionStrategy.LONG_CALL,
            opportunity_score=85.0,
            ml_confidence=0.75,
            valuation_score=0.15,
            entry_price=5.1,
            target_price=8.0,
            stop_loss=3.5,
            position_size=0.03,
            max_contracts=10,
            max_loss=510.0,
            expected_return=0.57,
            probability_of_profit=0.65,
            risk_reward_ratio=2.1,
            breakeven_price=155.1,
            key_reasons=["Undervalued by 15%", "High ML confidence"],
            risk_factors=["Time decay risk"],
            entry_signals=["Technical breakout"],
            timestamp=datetime.now(),
            data_quality=85.0
        )
    
    def test_recommendation_to_dict(self):
        """Test converting recommendation to dictionary"""
        rec_dict = self.recommendation.to_dict()
        
        self.assertEqual(rec_dict['symbol'], 'AAPL')
        self.assertEqual(rec_dict['strategy'], 'long_call')
        self.assertEqual(rec_dict['scores']['opportunity'], 85.0)
        self.assertEqual(rec_dict['scores']['ml_confidence'], 0.75)
        self.assertEqual(rec_dict['trade']['entry_price'], 5.1)
        self.assertEqual(rec_dict['trade']['target_price'], 8.0)
        self.assertEqual(rec_dict['risk']['expected_return'], 0.57)
        self.assertIn('Undervalued by 15%', rec_dict['analysis']['key_reasons'])


class TestPipelineAnalysis(unittest.TestCase):
    """Test pipeline analysis methods"""
    
    def setUp(self):
        """Set up test pipeline"""
        self.pipeline = create_pipeline()
    
    @patch('oracle_options_pipeline.OracleOptionsPipeline._fetch_market_data')
    @patch('oracle_options_pipeline.OracleOptionsPipeline._fetch_options_chain')
    def test_analyze_ticker_no_data(self, mock_chain, mock_market):
        """Test analyzing ticker with no data available"""
        mock_market.return_value = None
        mock_chain.return_value = []
        
        recommendations = self.pipeline.analyze_ticker("TEST")
        
        self.assertEqual(len(recommendations), 0)
        mock_market.assert_called_once_with("TEST")
    
    @patch('oracle_options_pipeline.OracleOptionsPipeline._fetch_market_data')
    @patch('oracle_options_pipeline.OracleOptionsPipeline._fetch_options_chain')
    @patch('oracle_options_pipeline.OracleOptionsPipeline._analyze_single_option')
    def test_analyze_ticker_with_data(self, mock_analyze, mock_chain, mock_market):
        """Test analyzing ticker with valid data"""
        # Mock market data
        mock_market.return_value = pd.DataFrame({
            'Close': [150, 151, 152],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        # Mock options chain
        test_contract = OptionContract(
            symbol="TEST",
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL,
            bid=5.0,
            ask=5.2,
            volume=1000,
            open_interest=5000,
            underlying_price=152.0
        )
        mock_chain.return_value = [test_contract]
        
        # Mock analysis result
        mock_recommendation = Mock(spec=OptionRecommendation)
        mock_recommendation.opportunity_score = 75.0
        mock_analyze.return_value = mock_recommendation
        
        recommendations = self.pipeline.analyze_ticker("TEST")
        
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].opportunity_score, 75.0)
    
    def test_filter_options(self):
        """Test options filtering logic"""
        # Create test options
        options = [
            OptionContract(
                symbol="TEST",
                strike=100.0,
                expiry=datetime.now() + timedelta(days=5),  # Too short
                option_type=OptionType.CALL,
                volume=50,  # Too low volume
                open_interest=100,
                underlying_price=100.0
            ),
            OptionContract(
                symbol="TEST",
                strike=100.0,
                expiry=datetime.now() + timedelta(days=30),  # Good
                option_type=OptionType.CALL,
                volume=500,  # Good volume
                open_interest=1000,  # Good OI
                bid=5.0,
                ask=5.1,  # Good spread
                underlying_price=100.0
            ),
            OptionContract(
                symbol="TEST",
                strike=100.0,
                expiry=datetime.now() + timedelta(days=200),  # Too long
                option_type=OptionType.PUT,
                volume=1000,
                open_interest=5000,
                underlying_price=100.0
            )
        ]
        
        filtered = self.pipeline._filter_options(options)
        
        # Only the second option should pass all filters
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].strike, 100.0)
        self.assertEqual(filtered[0].option_type, OptionType.CALL)


class TestMarketScan(unittest.TestCase):
    """Test market scanning functionality"""
    
    def setUp(self):
        """Set up test pipeline"""
        self.pipeline = create_pipeline()
    
    @patch('oracle_options_pipeline.OracleOptionsPipeline.analyze_ticker')
    def test_scan_market_with_symbols(self, mock_analyze):
        """Test scanning specific symbols"""
        # Mock recommendations
        mock_rec = Mock(spec=OptionRecommendation)
        mock_rec.opportunity_score = 80.0
        mock_rec.symbol = "TEST"
        mock_analyze.return_value = [mock_rec]
        
        symbols = ["TEST1", "TEST2", "TEST3"]
        result = self.pipeline.scan_market(symbols, max_symbols=3)
        
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.symbols_analyzed, 3)
        self.assertEqual(result.opportunities_found, 3)
        self.assertEqual(len(result.recommendations), 3)
        self.assertEqual(mock_analyze.call_count, 3)
    
    @patch('oracle_options_pipeline.OracleOptionsPipeline.analyze_ticker')
    def test_scan_market_default_universe(self, mock_analyze):
        """Test scanning default universe"""
        mock_analyze.return_value = []
        
        result = self.pipeline.scan_market(max_symbols=5)
        
        self.assertEqual(result.symbols_analyzed, 5)
        self.assertEqual(result.opportunities_found, 0)
        self.assertEqual(mock_analyze.call_count, 5)


class TestPositionMonitoring(unittest.TestCase):
    """Test position monitoring functionality"""
    
    def setUp(self):
        """Set up test pipeline"""
        self.pipeline = create_pipeline()
    
    @patch('data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_quote')
    @patch('data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_market_data')
    @patch('data_feeds.options_valuation_engine.OptionsValuationEngine.detect_mispricing')
    def test_monitor_positions(self, mock_valuation, mock_market, mock_quote):
        """Test monitoring existing positions"""
        # Mock quote
        mock_quote_obj = Mock()
        mock_quote_obj.price = 155.0
        mock_quote.return_value = mock_quote_obj
        
        # Mock market data
        mock_market_obj = Mock()
        mock_market_obj.data = pd.DataFrame({'Close': [155.0]})
        mock_market.return_value = mock_market_obj
        
        # Mock valuation
        mock_val_result = Mock()
        mock_val_result.market_price = 6.0
        mock_val_result.is_undervalued = False
        mock_val_result.opportunity_score = 65.0
        mock_valuation.return_value = mock_val_result
        
        positions = [
            {
                'symbol': 'TEST',
                'strike': 150.0,
                'expiry': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                'type': 'call',
                'entry_price': 5.0,
                'quantity': 10
            }
        ]
        
        updates = self.pipeline.monitor_positions(positions)
        
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]['position']['symbol'], 'TEST')
        self.assertEqual(updates[0]['current_price'], 6.0)
        self.assertEqual(updates[0]['pnl_percent'], 20.0)  # (6-5)/5 * 100
        self.assertIn(updates[0]['action'], ['hold', 'take_profit', 'stop_loss', 'consider_exit'])


class TestOpportunityScoring(unittest.TestCase):
    """Test opportunity scoring logic"""
    
    def setUp(self):
        """Set up test pipeline"""
        self.pipeline = create_pipeline()
    
    def test_calculate_opportunity_score(self):
        """Test opportunity score calculation"""
        # Create mock valuation result
        mock_valuation = Mock()
        mock_valuation.opportunity_score = 80.0
        
        score = self.pipeline._calculate_opportunity_score(
            valuation=mock_valuation,
            ml_confidence=0.8,
            expected_return=0.25
        )
        
        # Score should be weighted average
        # 80 * 0.4 + 80 * 0.3 + 50 * 0.3 = 32 + 24 + 15 = 71
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_calculate_position_size(self):
        """Test position size calculation"""
        mock_valuation = Mock()
        mock_valuation.confidence_score = 80.0
        
        # Test conservative sizing
        self.pipeline.config.risk_tolerance = RiskTolerance.CONSERVATIVE
        size = self.pipeline._calculate_position_size(
            opportunity_score=80.0,
            ml_confidence=0.8,
            valuation=mock_valuation
        )
        
        self.assertLess(size, 0.025)  # Should be less than 2.5% for conservative
        
        # Test aggressive sizing
        self.pipeline.config.risk_tolerance = RiskTolerance.AGGRESSIVE
        size = self.pipeline._calculate_position_size(
            opportunity_score=90.0,
            ml_confidence=0.9,
            valuation=mock_valuation
        )
        
        self.assertGreater(size, 0.05)  # Should be more than 5% for aggressive


class TestPerformanceStats(unittest.TestCase):
    """Test performance statistics"""
    
    def setUp(self):
        """Set up test pipeline"""
        self.pipeline = create_pipeline()
    
    def test_get_performance_stats_empty(self):
        """Test getting stats with empty cache"""
        stats = self.pipeline.get_performance_stats()
        
        self.assertEqual(stats['cache_size'], 0)
        self.assertEqual(stats['total_recommendations'], 0)
        self.assertEqual(stats['avg_opportunity_score'], 0)
        self.assertEqual(stats['avg_ml_confidence'], 0)
        self.assertEqual(len(stats['top_symbols']), 0)
    
    def test_get_performance_stats_with_data(self):
        """Test getting stats with cached data"""
        # Add mock recommendations to cache
        mock_rec1 = Mock(spec=OptionRecommendation)
        mock_rec1.opportunity_score = 80.0
        mock_rec1.ml_confidence = 0.75
        mock_rec1.symbol = "AAPL"
        
        mock_rec2 = Mock(spec=OptionRecommendation)
        mock_rec2.opportunity_score = 85.0
        mock_rec2.ml_confidence = 0.80
        mock_rec2.symbol = "GOOGL"
        
        self.pipeline._cache['AAPL'] = [mock_rec1]
        self.pipeline._cache['GOOGL'] = [mock_rec2]
        
        stats = self.pipeline.get_performance_stats()
        
        self.assertEqual(stats['cache_size'], 2)
        self.assertEqual(stats['total_recommendations'], 2)
        self.assertAlmostEqual(stats['avg_opportunity_score'], 82.5, places=1)
        self.assertAlmostEqual(stats['avg_ml_confidence'], 0.775, places=3)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in the pipeline"""
    
    def setUp(self):
        """Set up test pipeline"""
        self.pipeline = create_pipeline()
    
    @patch('oracle_options_pipeline.OracleOptionsPipeline._fetch_market_data')
    def test_analyze_ticker_error_handling(self, mock_market):
        """Test error handling in analyze_ticker"""
        mock_market.side_effect = Exception("API Error")
        
        recommendations = self.pipeline.analyze_ticker("ERROR")
        
        # Should return empty list on error
        self.assertEqual(len(recommendations), 0)
    
    @patch('data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_quote')
    def test_monitor_positions_error_handling(self, mock_quote):
        """Test error handling in monitor_positions"""
        mock_quote.side_effect = Exception("Quote Error")
        
        positions = [{
            'symbol': 'ERROR',
            'strike': 100.0,
            'expiry': '2024-12-31',
            'type': 'call',
            'entry_price': 5.0
        }]
        
        updates = self.pipeline.monitor_positions(positions)
        
        # Should return empty list on error
        self.assertEqual(len(updates), 0)


class TestIntegrationFlow(unittest.TestCase):
    """Test complete integration flow"""
    
    @patch('oracle_options_pipeline.DataFeedOrchestrator')
    @patch('oracle_options_pipeline.OptionsValuationEngine')
    def test_complete_workflow(self, mock_valuation_class, mock_orchestrator_class):
        """Test complete workflow from analysis to recommendations"""
        # Setup mocks
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        mock_valuation = Mock()
        mock_valuation_class.return_value = mock_valuation
        
        # Create pipeline
        config = {
            'risk_tolerance': 'moderate',
            'min_opportunity_score': 70.0
        }
        pipeline = create_pipeline(config)
        
        # Mock market data
        mock_market_data = Mock()
        mock_market_data.data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000]
        })
        mock_orchestrator.get_market_data.return_value = mock_market_data
        
        # Mock options analytics
        mock_orchestrator.get_options_analytics.return_value = {
            'chain': [
                {
                    'symbol': 'TEST',
                    'expiry': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                    'strike': 100.0,
                    'put_call': 'call',
                    'bid': 5.0,
                    'ask': 5.2,
                    'volume': 1000,
                    'open_interest': 5000,
                    'underlying': 102.0
                }
            ]
        }
        
        # Mock valuation result
        mock_val_result = Mock()
        mock_val_result.opportunity_score = 75.0
        mock_val_result.is_undervalued = True
        mock_val_result.mispricing_ratio = 0.1
        mock_val_result.confidence_score = 80.0
        mock_val_result.implied_volatility = 0.25
        mock_val_result.iv_rank = 30.0
        mock_valuation.detect_mispricing.return_value = mock_val_result
        
        # Test single ticker analysis
        recommendations = pipeline.analyze_ticker("TEST")
        
        # Verify calls were made
        mock_orchestrator.get_market_data.assert_called()
        mock_orchestrator.get_options_analytics.assert_called()
        
        # Test generate recommendations
        rec_dict = pipeline.generate_recommendations(["TEST"], output_format="dict")
        self.assertIsInstance(rec_dict, list)
        
        # Test JSON output
        rec_json = pipeline.generate_recommendations(["TEST"], output_format="json")
        self.assertIsInstance(rec_json, str)
        parsed = json.loads(rec_json)
        self.assertIsInstance(parsed, list)


def run_tests():
    """Run all tests with verbose output"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
    
    exit(0 if success else 1)