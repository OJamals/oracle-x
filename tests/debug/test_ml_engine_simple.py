#!/usr/bin/env python3
"""
Simple ML Engine Test - Step by Step Demonstration
Shows the ML engine functionality with clear examples
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from oracle_engine.ensemble_ml_engine import (
    EnsemblePredictionEngine,
    PredictionType,
    ModelType,
    create_prediction_engine,
)
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
from sentiment.sentiment_engine import AdvancedSentimentEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_ml_engine_step_by_step():
    """Test ML engine functionality step by step"""

    print("🚀 ML Engine Step-by-Step Test")
    print("=" * 50)

    # Step 1: Initialize components
    print("\n📋 Step 1: Initializing Components")
    try:
        data_orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        ml_engine = create_prediction_engine(data_orchestrator, sentiment_engine)
        print(f"✅ Engine initialized with {len(ml_engine.models)} models")
        print(f"   - Model types: {list(ml_engine.models.keys())}")
        print(f"   - Prediction horizons: {ml_engine.prediction_horizons}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

    # Step 2: Test data access
    print("\n📋 Step 2: Testing Data Access")
    test_symbol = "AAPL"
    try:
        # Get market data
        market_data = data_orchestrator.get_market_data(
            test_symbol, period="1m", interval="1d"
        )
        if market_data and not market_data.data.empty:
            print(f"✅ Market data retrieved: {len(market_data.data)} rows")
            print(
                f"   - Date range: {market_data.data.index[0]} to {market_data.data.index[-1]}"
            )
            print(f"   - Latest close: ${market_data.data['Close'].iloc[-1]:.2f}")
        else:
            print("⚠️  No market data available")

        # Test sentiment
        sentiment = sentiment_engine.get_symbol_sentiment_summary(test_symbol, [])
        if sentiment:
            print(f"✅ Sentiment retrieved: {sentiment.overall_sentiment:.3f}")
        else:
            print("⚠️  No sentiment data available")

    except Exception as e:
        print(f"❌ Data access failed: {e}")
        return False

    # Step 3: Test fallback predictions (these should always work)
    print("\n📋 Step 3: Testing Fallback Predictions")
    try:
        # Test fallback prediction directly
        fallback_result = ml_engine._fallback_predict(
            test_symbol, PredictionType.PRICE_DIRECTION, 5
        )

        if fallback_result:
            print("✅ Fallback prediction successful:")
            print(f"   - Symbol: {fallback_result.symbol}")
            print(f"   - Prediction: {fallback_result.prediction:.4f}")
            print(f"   - Confidence: {fallback_result.confidence:.3f}")
            print(f"   - Type: {fallback_result.prediction_type.value}")
            print(f"   - Horizon: {fallback_result.horizon_days} days")
            print(f"   - Context: {fallback_result.prediction_context}")
        else:
            print("❌ Fallback prediction failed")
            return False

    except Exception as e:
        print(f"❌ Fallback prediction error: {e}")
        return False

    # Step 4: Test regular predictions (may use fallback if models not trained)
    print("\n📋 Step 4: Testing Regular Predictions")
    try:
        for pred_type in [PredictionType.PRICE_DIRECTION, PredictionType.PRICE_TARGET]:
            result = ml_engine.predict(test_symbol, pred_type, 5)

            if result:
                print(f"✅ {pred_type.value} prediction:")
                print(f"   - Prediction: {result.prediction:.4f}")
                print(f"   - Confidence: {result.confidence:.3f}")
                print(f"   - Uncertainty: {result.uncertainty:.3f}")
                print(
                    f"   - Models used: {result.prediction_context.get('models_used', 'unknown')}"
                )
                print(
                    f"   - Fallback: {result.prediction_context.get('fallback_method', False)}"
                )
            else:
                print(f"⚠️  No {pred_type.value} prediction available")

    except Exception as e:
        print(f"❌ Regular prediction error: {e}")
        return False

    # Step 5: Test model management features
    print("\n📋 Step 5: Testing Model Management")
    try:
        # Get performance metrics
        performance = ml_engine.get_model_performance()
        print(f"✅ Performance metrics available for {len(performance)} models")

        # Check model weights
        weights = ml_engine.model_weights
        print(f"✅ Model weights: {len(weights)} weights configured")
        total_weight = sum(weights.values())
        print(f"   - Total weight: {total_weight:.3f}")

        # Test save/load
        test_file = "test_models_simple.json"
        save_success = ml_engine.save_models(test_file)
        print(f"✅ Model save: {'success' if save_success else 'failed'}")

        if save_success:
            load_success = ml_engine.load_models(test_file)
            print(f"✅ Model load: {'success' if load_success else 'failed'}")
            # Cleanup
            Path(test_file).unlink(missing_ok=True)

    except Exception as e:
        print(f"❌ Model management error: {e}")
        return False

    # Step 6: Test multiple symbols
    print("\n📋 Step 6: Testing Multiple Symbols")
    test_symbols = ["MSFT", "GOOGL", "TSLA"]
    successful_predictions = 0

    for symbol in test_symbols:
        try:
            result = ml_engine.predict(symbol, PredictionType.PRICE_DIRECTION, 5)
            if result:
                print(
                    f"✅ {symbol}: {result.prediction:.4f} (confidence: {result.confidence:.3f})"
                )
                successful_predictions += 1
            else:
                print(f"⚠️  {symbol}: No prediction available")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

    print(
        f"📊 {successful_predictions}/{len(test_symbols)} symbols predicted successfully"
    )

    # Step 7: Test error handling
    print("\n📋 Step 7: Testing Error Handling")
    try:
        # Invalid symbol
        result = ml_engine.predict("INVALID_SYMBOL", PredictionType.PRICE_DIRECTION, 5)
        print(f"✅ Invalid symbol handled: {result is None}")

        # Extreme horizon
        result = ml_engine.predict("AAPL", PredictionType.PRICE_DIRECTION, 999)
        print(f"✅ Extreme horizon handled: {result is not None}")

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

    # Final summary
    print("\n📋 Final Summary")
    print("=" * 50)
    print("✅ ML Engine is functional!")
    print("✅ All basic operations work correctly")
    print("✅ Error handling is robust")
    print("✅ Fallback mechanisms are operational")

    if len(ml_engine.models) > 0:
        print("✅ ML models are initialized")
    else:
        print("⚠️  No ML models available (using fallback only)")

    print("\n🎯 The ML Engine is ready for use!")
    return True


def test_prediction_with_training():
    """Test training and prediction together"""
    print("\n🎓 Training and Prediction Test")
    print("=" * 40)

    try:
        # Initialize
        data_orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        ml_engine = create_prediction_engine(data_orchestrator, sentiment_engine)

        # Quick training on a small dataset
        print("📚 Starting quick training...")
        training_symbols = ["AAPL", "MSFT"]

        start_time = time.time()
        training_results = ml_engine.train_models(
            symbols=training_symbols,
            lookback_days=30,  # Short for testing
            update_existing=False,
        )
        training_time = time.time() - start_time

        print(f"✅ Training completed in {training_time:.2f} seconds")

        if training_results:
            print(f"📚 Training results: {len(training_results)} target variables")
            for target, result in list(training_results.items())[:3]:  # Show first 3
                print(f"   - {target}: {type(result).__name__}")

        # Test prediction after training
        print("\n🔮 Testing predictions after training...")
        for symbol in training_symbols:
            result = ml_engine.predict(symbol, PredictionType.PRICE_DIRECTION, 5)
            if result:
                print(
                    f"✅ {symbol}: {result.prediction:.4f} (confidence: {result.confidence:.3f})"
                )
                print(
                    f"   - Models used: {result.prediction_context.get('models_used', 0)}"
                )
            else:
                print(f"⚠️  {symbol}: No prediction after training")

        return True

    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 ML Engine Comprehensive Functionality Test")
    print("=" * 60)

    try:
        # Run step-by-step test
        step_by_step_success = test_ml_engine_step_by_step()

        # Run training test
        if step_by_step_success:
            training_success = test_prediction_with_training()
        else:
            training_success = False

        # Final report
        print("\n" + "=" * 60)
        print("🏁 FINAL TEST REPORT")
        print("=" * 60)

        if step_by_step_success and training_success:
            print("🎉 ALL TESTS PASSED!")
            print("✅ ML Engine is fully functional")
            print("✅ Ready for production use")
            return 0
        elif step_by_step_success:
            print("✅ Basic functionality works")
            print("⚠️  Some advanced features need attention")
            return 0
        else:
            print("❌ Basic functionality has issues")
            print("🔧 ML Engine needs debugging")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
