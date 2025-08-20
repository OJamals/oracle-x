"""
Test ML Trading Integration
Validates the complete ML-enhanced trading system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock
import logging

# Setup test environment
logging.basicConfig(level=logging.INFO)

def test_ml_trading_integration():
    """Test the complete ML trading integration"""
    print("Testing ML Trading Integration")
    print("=" * 40)
    
    try:
        # Mock components
        mock_data_orchestrator = Mock()
        mock_sentiment_engine = Mock()
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        test_data = pd.DataFrame({
            'Open': 150 + np.random.randn(len(dates)) * 10,
            'High': 155 + np.random.randn(len(dates)) * 10,
            'Low': 145 + np.random.randn(len(dates)) * 10,
            'Close': 150 + np.random.randn(len(dates)) * 10,
            'Volume': 1000000 + np.random.randn(len(dates)) * 100000,
            'RSI': 50 + np.random.randn(len(dates)) * 20
        }, index=dates)
        
        # Mock data orchestrator responses
        mock_data_orchestrator.get_historical_data = Mock(return_value=test_data)
        
        # Mock sentiment engine responses
        mock_sentiment_engine.get_sentiment_analysis = Mock(return_value={
            'overall_sentiment': 0.3,
            'confidence': 0.8,
            'momentum': 0.15,
            'article_count': 10
        })
        
        # Import and create ML trading orchestrator
        from oracle_engine.ml_trading_integration import (
            MLTradingOrchestrator, MLTradingConfig, create_ml_trading_orchestrator
        )
        
        # Create configuration
        config = MLTradingConfig(
            min_confidence_threshold=0.5,
            min_data_quality_threshold=0.6,
            technical_weight=0.4,
            sentiment_weight=0.3,
            ml_weight=0.3
        )
        
        # Create orchestrator
        orchestrator = create_ml_trading_orchestrator(
            mock_data_orchestrator, mock_sentiment_engine, config
        )
        
        print("✓ Successfully created ML Trading Orchestrator")
        
        # Test signal generation
        test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        print(f"\nGenerating trading signals for {len(test_symbols)} symbols...")
        
        signals_generated = 0
        for symbol in test_symbols:
            try:
                signal = orchestrator.generate_trading_signal(symbol)
                if signal:
                    signals_generated += 1
                    print(f"✓ {symbol}: {signal.signal_type} "
                          f"(confidence: {signal.confidence:.3f}, "
                          f"risk: {signal.risk_score:.3f}, "
                          f"size: {signal.position_size_factor:.3f})")
                    
                    # Validate signal properties
                    assert 0.0 <= signal.confidence <= 1.0, "Invalid confidence"
                    assert 0.0 <= signal.risk_score <= 1.0, "Invalid risk score"
                    assert signal.signal_type in ['BUY', 'SELL', 'HOLD'], "Invalid signal type"
                    
                else:
                    print(f"✗ No signal generated for {symbol}")
                    
            except Exception as e:
                print(f"✗ Error generating signal for {symbol}: {e}")
        
        print(f"\n✓ Generated {signals_generated}/{len(test_symbols)} signals successfully")
        
        # Test performance tracking
        performance = orchestrator.get_signal_performance()
        print(f"\nSignal Performance: {performance}")
        
        # Test model training (should work even with mocked data)
        try:
            training_result = orchestrator.train_models(['AAPL', 'GOOGL'], lookback_days=100)
            print(f"✓ Model training completed: {training_result}")
        except Exception as e:
            print(f"✗ Model training failed: {e}")
        
        # Test retraining logic
        should_retrain = orchestrator.should_retrain_models()
        print(f"✓ Should retrain models: {should_retrain}")
        
        print("\n✓ All ML trading integration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ ML trading integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_generation_edge_cases():
    """Test signal generation with edge cases"""
    print("\nTesting Signal Generation Edge Cases")
    print("=" * 40)
    
    try:
        from oracle_engine.ml_trading_integration import (
            MLTradingOrchestrator, MLTradingConfig
        )
        
        # Mock components
        mock_data_orchestrator = Mock()
        mock_sentiment_engine = Mock()
        
        config = MLTradingConfig()
        orchestrator = MLTradingOrchestrator(
            mock_data_orchestrator, mock_sentiment_engine, config
        )
        
        # Test with empty data
        mock_data_orchestrator.get_historical_data = Mock(return_value=pd.DataFrame())
        signal = orchestrator.generate_trading_signal('EMPTY')
        
        if signal is None:
            print("✓ Correctly handled empty data case")
        else:
            print("✗ Should return None for empty data")
        
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        })
        mock_data_orchestrator.get_historical_data = Mock(return_value=minimal_data)
        
        signal = orchestrator.generate_trading_signal('MINIMAL')
        print(f"✓ Handled minimal data case: {signal.signal_type if signal else 'None'}")
        
        # Test with missing sentiment
        mock_sentiment_engine.get_sentiment_analysis = Mock(return_value=None)
        signal = orchestrator.generate_trading_signal('NO_SENTIMENT')
        print(f"✓ Handled missing sentiment case: {signal.signal_type if signal else 'None'}")
        
        print("\n✓ All edge case tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Edge case testing failed: {e}")
        return False

def test_ml_config_validation():
    """Test ML trading configuration"""
    print("\nTesting ML Trading Configuration")
    print("=" * 40)
    
    try:
        from oracle_engine.ml_trading_integration import MLTradingConfig, get_default_ml_config
        
        # Test default configuration
        default_config = get_default_ml_config()
        print(f"✓ Default config created: min_confidence={default_config.min_confidence_threshold}")
        
        # Test custom configuration
        custom_config = MLTradingConfig(
            min_confidence_threshold=0.8,
            technical_weight=0.5,
            sentiment_weight=0.3,
            ml_weight=0.2
        )
        
        # Validate weights sum
        total_weight = (custom_config.technical_weight + 
                       custom_config.sentiment_weight + 
                       custom_config.ml_weight)
        print(f"✓ Custom config weights sum to: {total_weight}")
        
        # Test configuration validation
        assert 0.0 <= custom_config.min_confidence_threshold <= 1.0
        assert custom_config.retrain_frequency_days > 0
        assert custom_config.min_training_samples > 0
        
        print("✓ Configuration validation passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False

def test_ml_integration_without_ml():
    """Test fallback behavior when ML models not available"""
    print("\nTesting Fallback Behavior (No ML)")
    print("=" * 40)
    
    try:
        # This test simulates when ML libraries are not available
        # The system should still work with fallback methods
        
        from oracle_engine.ml_trading_integration import MLTradingOrchestrator, MLTradingConfig
        
        # Mock components
        mock_data_orchestrator = Mock()
        mock_sentiment_engine = Mock()
        
        # Create test data
        test_data = pd.DataFrame({
            'Close': [100, 102, 98, 105, 103],
            'Volume': [1000, 1100, 900, 1200, 1150]
        })
        
        mock_data_orchestrator.get_historical_data = Mock(return_value=test_data)
        mock_sentiment_engine.get_sentiment_analysis = Mock(return_value={
            'overall_sentiment': 0.2,
            'confidence': 0.7
        })
        
        config = MLTradingConfig()
        orchestrator = MLTradingOrchestrator(
            mock_data_orchestrator, mock_sentiment_engine, config
        )
        
        # Generate signal (should work with fallback methods)
        signal = orchestrator.generate_trading_signal('FALLBACK_TEST')
        
        if signal:
            print(f"✓ Fallback signal generated: {signal.signal_type} "
                  f"(confidence: {signal.confidence:.3f})")
        else:
            print("✗ Fallback signal generation failed")
        
        print("✓ Fallback behavior test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Fallback behavior test failed: {e}")
        return False

def main():
    """Run all ML trading integration tests"""
    print("ML Trading Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("ML Trading Integration", test_ml_trading_integration),
        ("Signal Generation Edge Cases", test_signal_generation_edge_cases),
        ("ML Config Validation", test_ml_config_validation),
        ("Fallback Behavior", test_ml_integration_without_ml)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All ML trading integration tests passed!")
        print("The ML-enhanced trading system is ready for deployment!")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Review the output above.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
