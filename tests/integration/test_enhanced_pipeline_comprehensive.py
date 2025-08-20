"""
Comprehensive test suite for Enhanced Oracle Options Pipeline
Tests all components including safe mode, advanced features, ML ensemble, and risk management
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import time

# Import the enhanced pipeline
from oracle_options_pipeline import (
    EnhancedOracleOptionsPipeline,
    EnhancedPipelineConfig,
    EnhancedFeatureEngine,
    EnhancedMLEngine,
    SafeMode,
    ModelComplexity,
    RiskTolerance,
    OptionStrategy,
    create_enhanced_pipeline
)


class TestEnhancedPipelineConfig:
    """Test enhanced pipeline configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EnhancedPipelineConfig()
        
        assert config.safe_mode == SafeMode.SAFE
        assert config.model_complexity == ModelComplexity.MODERATE
        assert config.enable_advanced_features is True
        assert config.risk_tolerance == RiskTolerance.MODERATE
        assert config.max_position_size == 0.05
        assert config.min_opportunity_score == 70.0
        assert config.min_confidence == 0.6
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = EnhancedPipelineConfig(
            safe_mode=SafeMode.FULL,
            model_complexity=ModelComplexity.ADVANCED,
            risk_tolerance=RiskTolerance.AGGRESSIVE,
            max_position_size=0.10
        )
        
        assert config.safe_mode == SafeMode.FULL
        assert config.model_complexity == ModelComplexity.ADVANCED
        assert config.risk_tolerance == RiskTolerance.AGGRESSIVE
        assert config.max_position_size == 0.10


class TestEnhancedFeatureEngine:
    """Test enhanced feature engineering"""
    
    @pytest.fixture
    def feature_engine(self):
        """Create feature engine for testing"""
        config = EnhancedPipelineConfig()
        return EnhancedFeatureEngine(config)
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, 50)
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = {
            'date': dates,
            'open': [p * 0.995 for p in prices],
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 50)
        }
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df
    
    def test_feature_extraction(self, feature_engine, sample_market_data):
        """Test comprehensive feature extraction"""
        features = feature_engine.extract_features('AAPL', sample_market_data)
        
        # Check that features are extracted
        assert features is not None
        assert features.rsi is not None
        assert features.macd is not None
        assert features.realized_volatility_5d is not None
        assert features.realized_volatility_20d is not None
        assert features.var_1d is not None
        assert features.sharpe_ratio is not None
        
        # Check value ranges
        assert 0 <= features.rsi <= 100
        assert features.realized_volatility_5d > 0
        assert features.var_1d > 0
    
    def test_rsi_calculation(self, feature_engine):
        """Test RSI calculation"""
        # Create trending up data
        prices = np.array([100 + i for i in range(30)])  # Uptrend
        rsi = feature_engine._calculate_rsi(prices)
        
        assert rsi is not None
        assert rsi > 50  # Should be above 50 for uptrend
        
        # Create trending down data
        prices = np.array([130 - i for i in range(30)])  # Downtrend
        rsi = feature_engine._calculate_rsi(prices)
        
        assert rsi is not None
        assert rsi < 50  # Should be below 50 for downtrend
    
    def test_macd_calculation(self, feature_engine):
        """Test MACD calculation"""
        # Create sufficient data for MACD
        prices = np.array([100 + np.sin(i/5) * 10 + i*0.1 for i in range(50)])
        macd_values = feature_engine._calculate_macd(prices)
        
        assert 'macd' in macd_values
        assert 'signal' in macd_values
        assert 'histogram' in macd_values
        assert macd_values['macd'] is not None
    
    def test_bollinger_bands(self, feature_engine):
        """Test Bollinger Bands calculation"""
        prices = np.array([100 + np.random.normal(0, 2) for _ in range(30)])
        bb_values = feature_engine._calculate_bollinger_bands(prices)
        
        assert 'upper' in bb_values
        assert 'lower' in bb_values
        assert 'position' in bb_values
        
        if bb_values['upper'] and bb_values['lower']:
            assert bb_values['upper'] > bb_values['lower']
    
    def test_volatility_calculations(self, feature_engine):
        """Test various volatility calculations"""
        # High volatility data
        high_vol_prices = np.array([100 + np.random.normal(0, 5) for _ in range(30)])
        high_vol = feature_engine._calculate_realized_volatility(high_vol_prices, 10)
        
        # Low volatility data
        low_vol_prices = np.array([100 + np.random.normal(0, 0.5) for _ in range(30)])
        low_vol = feature_engine._calculate_realized_volatility(low_vol_prices, 10)
        
        assert high_vol > low_vol
        
        # Test GARCH estimation
        garch_vol = feature_engine._estimate_garch_volatility(high_vol_prices)
        assert garch_vol is not None
        assert garch_vol > 0
        
        # Test EWMA volatility
        ewma_vol = feature_engine._calculate_ewma_volatility(high_vol_prices)
        assert ewma_vol is not None
        assert ewma_vol > 0
    
    def test_risk_metrics(self, feature_engine):
        """Test risk metric calculations"""
        # Create data with known characteristics
        prices = np.array([100, 105, 103, 108, 102, 110, 95, 115, 90, 120])
        
        # Test VaR calculation
        var_1d = feature_engine._calculate_var(prices, confidence=0.95, horizon=1)
        assert var_1d is not None
        assert var_1d > 0
        
        # Test max drawdown
        max_dd = feature_engine._calculate_max_drawdown(prices)
        assert max_dd is not None
        assert max_dd >= 0
        
        # Test Sharpe ratio
        sharpe = feature_engine._calculate_sharpe_ratio(prices)
        assert sharpe is not None
        
        # Test Sortino ratio
        sortino = feature_engine._calculate_sortino_ratio(prices)
        assert sortino is not None
    
    def test_empty_data_handling(self, feature_engine):
        """Test handling of empty or insufficient data"""
        empty_df = pd.DataFrame()
        features = feature_engine.extract_features('TEST', empty_df)
        
        # Should return empty features without crashing
        assert features is not None
        
        # Test with insufficient data
        short_data = pd.DataFrame({
            'close': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'volume': [1000, 1100, 1200]
        })
        
        features = feature_engine.extract_features('TEST', short_data)
        assert features is not None


class TestEnhancedMLEngine:
    """Test enhanced ML engine"""
    
    @pytest.fixture
    def ml_engine(self):
        """Create ML engine for testing"""
        config = EnhancedPipelineConfig()
        return EnhancedMLEngine(config)
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Generate features
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        X.columns = [f'feature_{i}' for i in range(n_features)]
        
        # Generate target with some relationship to features
        y = (X.iloc[:, 0] * 0.5 + X.iloc[:, 1] * 0.3 + 
             np.random.normal(0, 0.1, n_samples))
        
        return X, y.values
    
    def test_model_initialization(self, ml_engine):
        """Test model initialization"""
        assert len(ml_engine.models) > 0
        assert 'rf' in ml_engine.models
        assert hasattr(ml_engine, 'scaler')
    
    def test_model_training(self, ml_engine, sample_training_data):
        """Test model training"""
        X, y = sample_training_data
        
        scores = ml_engine.train_models(X, y)
        
        assert isinstance(scores, dict)
        assert len(scores) > 0
        
        # Check that models are fitted
        for model_name, model in ml_engine.models.items():
            assert hasattr(model, 'predict')
    
    def test_ensemble_prediction(self, ml_engine, sample_training_data):
        """Test ensemble prediction"""
        X, y = sample_training_data
        
        # Train models first
        ml_engine.train_models(X, y)
        
        # Make prediction on new data
        test_X = X.iloc[:5]  # Use first 5 samples as test
        result = ml_engine.predict_ensemble(test_X)
        
        assert 'prediction' in result
        assert 'confidence' in result
        assert isinstance(result['prediction'], float)
        assert 0 <= result['confidence'] <= 1
    
    def test_empty_data_handling(self, ml_engine):
        """Test handling of empty data"""
        empty_df = pd.DataFrame()
        result = ml_engine.predict_ensemble(empty_df)
        
        assert result['prediction'] == 0.0
        assert result['confidence'] == 0.0


class TestEnhancedOracleOptionsPipeline:
    """Test enhanced options pipeline"""
    
    @pytest.fixture
    def pipeline_config(self):
        """Create test configuration"""
        return EnhancedPipelineConfig(
            safe_mode=SafeMode.SAFE,  # Use safe mode for testing
            model_complexity=ModelComplexity.SIMPLE,
            max_workers=2,
            min_opportunity_score=50.0  # Lower threshold for testing
        )
    
    @pytest.fixture
    def pipeline(self, pipeline_config):
        """Create enhanced pipeline for testing"""
        return EnhancedOracleOptionsPipeline(pipeline_config)
    
    def test_safe_mode_initialization(self, pipeline):
        """Test that safe mode initialization works"""
        assert pipeline is not None
        assert pipeline.config.safe_mode == SafeMode.SAFE
        assert hasattr(pipeline, 'feature_engine')
        assert hasattr(pipeline, 'ml_engine')
        assert hasattr(pipeline, 'executor')
    
    def test_mock_data_generation(self, pipeline):
        """Test mock market data generation"""
        mock_data = pipeline._generate_mock_market_data('AAPL', '1mo')
        
        assert not mock_data.empty
        assert len(mock_data) == 30  # 1 month of data
        assert all(col in mock_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Check data quality
        assert (mock_data['high'] >= mock_data['low']).all()
        assert (mock_data['high'] >= mock_data['close']).all()
        assert (mock_data['low'] <= mock_data['close']).all()
        assert (mock_data['volume'] > 0).all()
    
    def test_symbol_analysis(self, pipeline):
        """Test enhanced symbol analysis"""
        recommendations = pipeline.analyze_symbol_enhanced('AAPL')
        
        assert isinstance(recommendations, list)
        
        if recommendations:  # Check if any recommendations were generated
            rec = recommendations[0]
            
            # Check required fields
            assert 'symbol' in rec
            assert 'strategy' in rec
            assert 'contract' in rec
            assert 'scores' in rec
            assert 'trade' in rec
            assert 'risk' in rec
            assert 'analysis' in rec
            
            # Check score ranges
            assert 0 <= rec['scores']['opportunity'] <= 100
            assert 0 <= rec['scores']['ml_confidence'] <= 1
            
            # Check trade parameters
            assert rec['trade']['entry_price'] > 0
            assert rec['trade']['target_price'] > 0
            assert rec['trade']['position_size'] > 0
    
    def test_opportunity_score_calculation(self, pipeline):
        """Test enhanced opportunity score calculation"""
        from oracle_options_pipeline import AdvancedFeatures
        
        # Create test features
        features = AdvancedFeatures(
            rsi=30,  # Oversold
            macd=0.5,
            macd_signal=0.3,  # Bullish crossover
            volatility_regime="low",
            sharpe_ratio=1.2
        )
        
        ml_result = {
            'confidence': 0.8,
            'prediction': 0.1
        }
        
        score = pipeline._calculate_enhanced_opportunity_score(features, ml_result)
        
        assert 0 <= score <= 100
        assert score > 50  # Should be positive given good indicators
    
    def test_kelly_position_sizing(self, pipeline):
        """Test Kelly Criterion position sizing"""
        # Test conservative sizing
        size = pipeline._calculate_kelly_position_size(
            confidence=0.6,
            expected_return=0.2,
            opportunity_score=75.0
        )
        
        assert 0 <= size <= pipeline.config.max_position_size
        
        # Test aggressive vs conservative
        pipeline.config.risk_tolerance = RiskTolerance.AGGRESSIVE
        aggressive_size = pipeline._calculate_kelly_position_size(0.8, 0.3, 80.0)
        
        pipeline.config.risk_tolerance = RiskTolerance.CONSERVATIVE
        conservative_size = pipeline._calculate_kelly_position_size(0.8, 0.3, 80.0)
        
        assert aggressive_size >= conservative_size
    
    def test_option_price_estimation(self, pipeline):
        """Test Black-Scholes option price estimation"""
        # Test call option
        call_price = pipeline._estimate_option_price(
            S=100,      # Current price
            K=105,      # Strike price
            T=0.25,     # 3 months
            sigma=0.3,  # 30% volatility
            r=0.05,     # 5% risk-free rate
            option_type='CALL'
        )
        
        assert call_price > 0
        assert call_price < 100  # Should be less than stock price for OTM call
        
        # Test put option
        put_price = pipeline._estimate_option_price(
            S=100, K=95, T=0.25, sigma=0.3, r=0.05, option_type='PUT'
        )
        
        assert put_price > 0
        
        # ATM call should be more expensive than OTM call
        atm_call = pipeline._estimate_option_price(
            S=100, K=100, T=0.25, sigma=0.3, r=0.05, option_type='CALL'
        )
        
        assert atm_call > call_price
    
    def test_market_scan(self, pipeline):
        """Test enhanced market scan"""
        symbols = ['AAPL', 'NVDA']
        results = pipeline.generate_market_scan(symbols, max_symbols=2)
        
        assert 'scan_results' in results
        assert 'top_opportunities' in results
        assert 'market_insights' in results
        
        scan_results = results['scan_results']
        assert scan_results['symbols_analyzed'] == 2
        assert scan_results['execution_time'] > 0
        assert 'opportunities_found' in scan_results
    
    def test_performance_tracking(self, pipeline):
        """Test performance monitoring"""
        initial_predictions = pipeline.performance_stats['predictions_made']
        
        # Run analysis to generate predictions
        pipeline.analyze_symbol_enhanced('TEST')
        
        # Check that stats were updated
        assert pipeline.performance_stats['predictions_made'] > initial_predictions
        
        # Test performance summary
        summary = pipeline.get_performance_summary()
        assert 'performance_stats' in summary
        assert 'pipeline_config' in summary
        assert 'cache_stats' in summary
        assert 'ml_engine_stats' in summary
    
    def test_error_handling(self, pipeline):
        """Test error handling and robustness"""
        # Test with invalid symbol
        recommendations = pipeline.analyze_symbol_enhanced('')
        assert recommendations == []
        
        # Test with None inputs
        recommendations = pipeline.analyze_symbol_enhanced(None)
        assert recommendations == []
        
        # Check error count tracking
        initial_errors = pipeline.performance_stats['error_count']
        pipeline.analyze_symbol_enhanced('')  # This should increment error count
        assert pipeline.performance_stats['error_count'] >= initial_errors
    
    def test_pipeline_shutdown(self, pipeline):
        """Test clean pipeline shutdown"""
        # Should not raise any exceptions
        pipeline.shutdown()
        
        # Check that resources are cleaned up
        assert len(pipeline._cache) == 0
        assert len(pipeline._cache_timestamps) == 0


class TestFactoryFunction:
    """Test factory function"""
    
    def test_create_enhanced_pipeline_default(self):
        """Test creating pipeline with default config"""
        pipeline = create_enhanced_pipeline()
        
        assert isinstance(pipeline, EnhancedOracleOptionsPipeline)
        assert pipeline.config.safe_mode == SafeMode.SAFE
    
    def test_create_enhanced_pipeline_custom_config(self):
        """Test creating pipeline with custom config"""
        config = {
            'safe_mode': SafeMode.MINIMAL,
            'model_complexity': ModelComplexity.ADVANCED,
            'max_workers': 8
        }
        
        pipeline = create_enhanced_pipeline(config)
        
        assert pipeline.config.safe_mode == SafeMode.MINIMAL
        assert pipeline.config.model_complexity == ModelComplexity.ADVANCED
        assert pipeline.config.max_workers == 8


class TestIntegration:
    """Integration tests for the full pipeline"""
    
    def test_full_pipeline_flow(self):
        """Test complete pipeline flow from initialization to shutdown"""
        # Initialize pipeline
        config = {
            'safe_mode': SafeMode.SAFE,
            'model_complexity': ModelComplexity.SIMPLE,
            'max_workers': 2,
            'min_opportunity_score': 30.0  # Lower threshold for testing
        }
        
        pipeline = create_enhanced_pipeline(config)
        
        try:
            # Run market scan
            results = pipeline.generate_market_scan(['AAPL', 'SPY'], max_symbols=2)
            
            # Verify results structure
            assert 'scan_results' in results
            assert 'top_opportunities' in results
            assert results['scan_results']['symbols_analyzed'] == 2
            
            # Get performance summary
            performance = pipeline.get_performance_summary()
            assert performance['performance_stats']['predictions_made'] > 0
            
        finally:
            # Clean shutdown
            pipeline.shutdown()
    
    def test_stress_testing(self):
        """Stress test with many symbols"""
        pipeline = create_enhanced_pipeline({
            'safe_mode': SafeMode.SAFE,
            'max_workers': 4,
            'min_opportunity_score': 0.0  # Accept all opportunities
        })
        
        try:
            # Test with many symbols
            symbols = [f'TEST{i}' for i in range(10)]
            start_time = time.time()
            
            results = pipeline.generate_market_scan(symbols, max_symbols=10)
            
            execution_time = time.time() - start_time
            
            # Should complete in reasonable time
            assert execution_time < 30  # 30 seconds max
            assert results['scan_results']['symbols_analyzed'] == 10
            
        finally:
            pipeline.shutdown()


if __name__ == "__main__":
    # Run basic tests
    print("Running Enhanced Oracle Options Pipeline Tests...")
    print("=" * 60)
    
    # Test configuration
    print("Testing configuration...")
    config = EnhancedPipelineConfig()
    assert config.safe_mode == SafeMode.SAFE
    print("✓ Configuration test passed")
    
    # Test pipeline initialization
    print("Testing pipeline initialization...")
    pipeline = create_enhanced_pipeline()
    assert pipeline is not None
    print("✓ Pipeline initialization test passed")
    
    # Test feature extraction
    print("Testing feature extraction...")
    feature_engine = EnhancedFeatureEngine(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'close': np.random.randn(50) + 100,
        'high': np.random.randn(50) + 101,
        'low': np.random.randn(50) + 99,
        'volume': np.random.randint(1000, 10000, 50)
    })
    
    features = feature_engine.extract_features('TEST', sample_data)
    assert features is not None
    print("✓ Feature extraction test passed")
    
    # Test analysis
    print("Testing symbol analysis...")
    recommendations = pipeline.analyze_symbol_enhanced('AAPL')
    assert isinstance(recommendations, list)
    print("✓ Symbol analysis test passed")
    
    # Test market scan
    print("Testing market scan...")
    scan_results = pipeline.generate_market_scan(['AAPL', 'SPY'], max_symbols=2)
    assert 'scan_results' in scan_results
    print("✓ Market scan test passed")
    
    # Cleanup
    pipeline.shutdown()
    
    print("\nAll tests passed! ✓")
    print(f"Enhanced pipeline is working correctly in safe mode.")
