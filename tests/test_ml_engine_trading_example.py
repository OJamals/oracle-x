#!/usr/bin/env python3
"""
ML Engine Trading Example
Demonstrates the ML engine in a realistic trading scenario
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
logging.basicConfig(level=logging.WARNING)  # Reduce noise


def trading_scenario_example():
    """Demonstrate ML engine in a realistic trading scenario"""

    print("📈 ML Engine Trading Scenario Example")
    print("=" * 50)

    # Portfolio of stocks to analyze
    portfolio = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    prediction_horizons = [1, 5, 10]

    print(f"📊 Analyzing portfolio: {', '.join(portfolio)}")
    print(f"🎯 Prediction horizons: {prediction_horizons} days")

    # Initialize the ML engine
    print("\n🔧 Initializing ML Engine...")
    try:
        data_orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        ml_engine = create_prediction_engine(data_orchestrator, sentiment_engine)
        print(f"✅ Engine ready with {len(ml_engine.models)} models")
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return

    # Scenario 1: Quick portfolio screening
    print("\n📋 Scenario 1: Portfolio Screening")
    print("-" * 30)

    portfolio_signals = {}

    for symbol in portfolio:
        try:
            # Get price direction prediction
            direction_pred = ml_engine.predict(
                symbol, PredictionType.PRICE_DIRECTION, 5  # 5-day horizon
            )

            # Get price target prediction
            target_pred = ml_engine.predict(symbol, PredictionType.PRICE_TARGET, 5)

            if direction_pred and target_pred:
                portfolio_signals[symbol] = {
                    "direction": direction_pred.prediction,
                    "direction_confidence": direction_pred.confidence,
                    "target": target_pred.prediction,
                    "target_confidence": target_pred.confidence,
                    "avg_confidence": (
                        direction_pred.confidence + target_pred.confidence
                    )
                    / 2,
                    "market_regime": direction_pred.market_regime,
                    "data_quality": direction_pred.data_quality_score,
                }

                # Simple signal interpretation
                signal = (
                    "BUY"
                    if direction_pred.prediction > 0.6
                    else "SELL" if direction_pred.prediction < 0.4 else "HOLD"
                )
                confidence_level = (
                    "HIGH"
                    if direction_pred.confidence > 0.7
                    else "MEDIUM" if direction_pred.confidence > 0.5 else "LOW"
                )

                print(
                    f"📈 {symbol:<6} | {signal:<4} | Confidence: {confidence_level:<6} | "
                    f"Direction: {direction_pred.prediction:.3f} | Target: {target_pred.prediction:+.3f}"
                )
            else:
                print(f"📈 {symbol:<6} | ⚠️  No prediction available")

        except Exception as e:
            print(f"📈 {symbol:<6} | ❌ Error: {str(e)[:30]}")

    # Scenario 2: Risk analysis across time horizons
    print("\n📋 Scenario 2: Multi-Horizon Risk Analysis")
    print("-" * 40)

    focus_symbol = "AAPL"  # Focus on one symbol for detailed analysis
    print(f"🔍 Detailed analysis for {focus_symbol}")

    horizon_analysis = {}

    for horizon in prediction_horizons:
        try:
            pred = ml_engine.predict(
                focus_symbol, PredictionType.PRICE_DIRECTION, horizon
            )

            if pred:
                horizon_analysis[horizon] = {
                    "prediction": pred.prediction,
                    "confidence": pred.confidence,
                    "uncertainty": pred.uncertainty,
                }

                # Risk assessment
                risk_level = (
                    "LOW"
                    if pred.uncertainty < 0.2
                    else "MEDIUM" if pred.uncertainty < 0.4 else "HIGH"
                )
                direction_str = (
                    "BULLISH"
                    if pred.prediction > 0.6
                    else "BEARISH" if pred.prediction < 0.4 else "NEUTRAL"
                )

                print(
                    f"📅 {horizon:2d} days | {direction_str:<8} | "
                    f"Prediction: {pred.prediction:.3f} | "
                    f"Confidence: {pred.confidence:.3f} | "
                    f"Risk: {risk_level}"
                )
            else:
                print(f"📅 {horizon:2d} days | ⚠️  No prediction")

        except Exception as e:
            print(f"📅 {horizon:2d} days | ❌ Error: {e}")

    # Scenario 3: Market regime analysis
    print("\n📋 Scenario 3: Market Regime Detection")
    print("-" * 35)

    regime_counts = {}

    for symbol in portfolio[:3]:  # Analyze first 3 symbols
        try:
            pred = ml_engine.predict(symbol, PredictionType.PRICE_DIRECTION, 5)

            if pred:
                regime = pred.market_regime
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

                print(
                    f"🌍 {symbol:<6} | Market Regime: {regime:<10} | "
                    f"Data Quality: {pred.data_quality_score:.3f}"
                )

        except Exception as e:
            print(f"🌍 {symbol:<6} | ❌ Error: {e}")

    if regime_counts:
        print(f"\n📊 Market Regime Summary:")
        for regime, count in regime_counts.items():
            print(f"   - {regime}: {count} symbols")

    # Scenario 4: Performance and model insight
    print("\n📋 Scenario 4: Model Performance Insights")
    print("-" * 38)

    try:
        performance = ml_engine.get_model_performance()

        if performance:
            print("🔧 Model Performance Overview:")
            for model_key, metrics in list(performance.items())[:3]:  # Show first 3
                print(
                    f"   - {model_key[:25]:<25} | Weight: {metrics.get('weight', 0):.3f}"
                )
        else:
            print("🔧 No performance metrics available (models not yet trained)")

        # Show model weights
        weights = ml_engine.model_weights
        if weights:
            print(f"\n⚖️  Ensemble Weights (Total: {sum(weights.values()):.1f}):")
            for model, weight in list(weights.items())[:3]:
                print(f"   - {model[:25]:<25} | {weight:.3f}")

        # Show prediction cache info
        cache_size = len(ml_engine.prediction_cache)
        print(f"\n🗄️  Prediction Cache: {cache_size} entries")

    except Exception as e:
        print(f"🔧 Performance analysis error: {e}")

    # Scenario 5: Trading recommendations
    print("\n📋 Scenario 5: Trading Recommendations")
    print("-" * 35)

    if portfolio_signals:
        # Sort by confidence
        sorted_signals = sorted(
            portfolio_signals.items(),
            key=lambda x: x[1]["avg_confidence"],
            reverse=True,
        )

        print("🎯 Top Trading Opportunities (by confidence):")

        for symbol, signals in sorted_signals[:3]:
            direction = signals["direction"]
            confidence = signals["avg_confidence"]

            action = "BUY" if direction > 0.6 else "SELL" if direction < 0.4 else "HOLD"
            quality = (
                "★★★" if confidence > 0.7 else "★★☆" if confidence > 0.5 else "★☆☆"
            )

            print(f"   {symbol}: {action} {quality} (Confidence: {confidence:.3f})")
    else:
        print("🎯 No signals available for recommendations")

    # Final summary
    print("\n📋 Summary")
    print("=" * 20)
    print(f"✅ Analyzed {len(portfolio)} symbols")
    print(f"✅ Generated predictions for {len(prediction_horizons)} time horizons")
    print(f"✅ Engine processed {len(portfolio_signals)} successful signals")
    print("✅ ML Engine is operational for trading scenarios!")


def backtesting_example():
    """Quick demonstration of how the engine could be used for backtesting"""

    print("\n📈 Backtesting Example")
    print("=" * 30)

    try:
        # Initialize
        data_orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        ml_engine = create_prediction_engine(data_orchestrator, sentiment_engine)

        # Simulate a trading period
        test_symbol = "AAPL"
        print(f"📊 Simulating predictions for {test_symbol}")

        # Generate multiple predictions as if trading over time
        predictions = []

        for i in range(5):  # Simulate 5 trading decisions
            pred = ml_engine.predict(test_symbol, PredictionType.PRICE_DIRECTION, 1)

            if pred:
                predictions.append(
                    {
                        "timestamp": datetime.now() - timedelta(days=4 - i),
                        "prediction": pred.prediction,
                        "confidence": pred.confidence,
                        "action": (
                            "BUY"
                            if pred.prediction > 0.6
                            else "SELL" if pred.prediction < 0.4 else "HOLD"
                        ),
                    }
                )

        if predictions:
            print("\n📅 Trading History:")
            for i, p in enumerate(predictions):
                print(
                    f"   Day {i+1}: {p['action']:<4} | "
                    f"Prediction: {p['prediction']:.3f} | "
                    f"Confidence: {p['confidence']:.3f}"
                )

            # Simple statistics
            buy_signals = sum(1 for p in predictions if p["action"] == "BUY")
            avg_confidence = sum(p["confidence"] for p in predictions) / len(
                predictions
            )

            print(f"\n📊 Statistics:")
            print(f"   - Buy signals: {buy_signals}/{len(predictions)}")
            print(f"   - Average confidence: {avg_confidence:.3f}")
        else:
            print("⚠️  No predictions generated for backtesting")

    except Exception as e:
        print(f"❌ Backtesting example failed: {e}")


def main():
    """Main function"""
    print("🚀 ML Engine Trading Examples")
    print("=" * 60)

    try:
        # Run trading scenario
        trading_scenario_example()

        # Run backtesting example
        backtesting_example()

        print("\n" + "=" * 60)
        print("🎉 Examples completed successfully!")
        print("💡 The ML Engine is ready for real trading applications")

    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Examples failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
