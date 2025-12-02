"""
Deep Training Diagnostic Tool
Examines the exact failure points in model training
"""

import logging
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_feature_engineering():
    """Deep dive into feature engineering process"""
    print("=" * 80)
    print("🔬 DEEP FEATURE ENGINEERING DIAGNOSTIC")
    print("=" * 80)
    
    try:
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine, PredictionType
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        from sentiment.sentiment_engine import AdvancedSentimentEngine
        
        # Initialize components
        orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        engine = EnsemblePredictionEngine(orchestrator, sentiment_engine)
        
        symbol = "AAPL"
        print(f"\n📊 Testing feature engineering for {symbol}")
        
        # Step 1: Get raw market data
        print("\n🔍 Step 1: Raw market data collection")
        market_data = orchestrator.get_market_data(symbol, period="60d", interval="1d")
        
        if market_data and not market_data.data.empty:
            print(f"✅ Market data shape: {market_data.data.shape}")
            print(f"✅ Market data columns: {list(market_data.data.columns)}")
            print(f"✅ Market data types:\n{market_data.data.dtypes}")
            print(f"✅ Market data sample:\n{market_data.data.head()}")
            
            # Check for missing values
            missing_values = market_data.data.isnull().sum()
            print(f"📋 Missing values:\n{missing_values}")
            
            # Check for infinite values
            numeric_cols = market_data.data.select_dtypes(include=[np.number]).columns
            inf_values = {}
            for col in numeric_cols:
                inf_count = np.isinf(market_data.data[col]).sum()
                if inf_count > 0:
                    inf_values[col] = inf_count
            
            if inf_values:
                print(f"⚠️  Infinite values found: {inf_values}")
            else:
                print("✅ No infinite values in market data")
                
        else:
            print("❌ Failed to get market data")
            return None
        
        # Step 2: Test feature engineering
        print("\n🔍 Step 2: Feature engineering process")
        
        if engine.feature_engineer:
            print("✅ Feature engineer available")
            
            # Test with minimal data
            historical_data = {symbol: market_data.data}
            sentiment_data = {}  # Start without sentiment
            
            try:
                features_df = engine.feature_engineer.engineer_features(
                    historical_data, sentiment_data
                )
                
                print(f"✅ Features engineered successfully")
                print(f"✅ Features shape: {features_df.shape}")
                print(f"✅ Features columns: {list(features_df.columns)}")
                
                # Analyze feature quality
                print("\n📊 Feature Quality Analysis")
                
                # Check for missing values
                missing_features = features_df.isnull().sum()
                missing_features = missing_features[missing_features > 0]
                if not missing_features.empty:
                    print(f"⚠️  Features with missing values:\n{missing_features}")
                else:
                    print("✅ No missing values in features")
                
                # Check for infinite values  
                numeric_features = features_df.select_dtypes(include=[np.number]).columns
                inf_features = {}
                for col in numeric_features:
                    inf_count = np.isinf(features_df[col]).sum()
                    if inf_count > 0:
                        inf_features[col] = inf_count
                
                if inf_features:
                    print(f"⚠️  Features with infinite values: {inf_features}")
                else:
                    print("✅ No infinite values in features")
                
                # Check target columns
                target_cols = [col for col in features_df.columns if col.startswith('target_')]
                print(f"\n🎯 Target columns found: {target_cols}")
                
                for target_col in target_cols:
                    target_data = features_df[target_col]
                    valid_targets = target_data.dropna()
                    print(f"  {target_col}: {len(valid_targets)} valid samples")
                    if len(valid_targets) > 0:
                        print(f"    Range: {valid_targets.min():.4f} to {valid_targets.max():.4f}")
                        print(f"    Mean: {valid_targets.mean():.4f}")
                
                return features_df
                
            except Exception as e:
                print(f"❌ Feature engineering failed: {e}")
                print(traceback.format_exc())
                return None
        else:
            print("❌ No feature engineer available")
            return None
    
    except Exception as e:
        print(f"❌ Feature engineering diagnostic failed: {e}")
        print(traceback.format_exc())
        return None

def diagnose_model_training(features_df):
    """Deep dive into individual model training"""
    print("\n" + "=" * 80)
    print("🧠 DEEP MODEL TRAINING DIAGNOSTIC")  
    print("=" * 80)
    
    if features_df is None:
        print("❌ No features available for training diagnostic")
        return
    
    try:
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine, PredictionType
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        from sentiment.sentiment_engine import AdvancedSentimentEngine
        
        # Initialize components
        orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        engine = EnsemblePredictionEngine(orchestrator, sentiment_engine)
        
        print(f"📊 Available models: {list(engine.models.keys())}")
        
        # Test each prediction type
        for prediction_type in [PredictionType.PRICE_DIRECTION, PredictionType.PRICE_TARGET]:
            print(f"\n🎯 Testing {prediction_type.value} models")
            
            # Find target column for this prediction type
            target_prefix = f"target_{'direction' if prediction_type == PredictionType.PRICE_DIRECTION else 'return'}"
            target_cols = [col for col in features_df.columns if col.startswith(target_prefix)]
            
            if not target_cols:
                print(f"❌ No target columns found for {prediction_type.value}")
                continue
            
            # Use first available target
            target_col = target_cols[0]
            print(f"📋 Using target column: {target_col}")
            
            # Prepare training data
            feature_cols = [col for col in features_df.columns 
                          if not col.startswith('target_') and 
                             col not in ['symbol', 'timestamp']]
            
            print(f"📋 Feature columns ({len(feature_cols)}): {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
            
            X = features_df[feature_cols].copy()
            y = features_df[target_col].copy()
            
            print(f"📊 Raw data shapes: X={X.shape}, y={y.shape}")
            
            # Remove NaN values
            initial_samples = len(X)
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            print(f"📊 After NaN removal: X={X.shape}, y={y.shape}")
            print(f"📊 Samples removed: {initial_samples - len(X)}")
            
            if len(X) < 10:
                print(f"❌ Insufficient samples for training: {len(X)}")
                continue
            
            # Check data quality
            print("\n🔍 Data Quality Check")
            
            # Check for infinite values in X
            inf_cols = []
            for col in X.columns:
                if np.isinf(X[col]).any():
                    inf_cols.append(col)
            
            if inf_cols:
                print(f"⚠️  Infinite values in features: {inf_cols}")
                # Replace infinite values
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                print("✅ Replaced infinite values with 0")
            else:
                print("✅ No infinite values in features")
            
            # Check y values
            if np.isinf(y).any():
                print("⚠️  Infinite values in target")
                y = y.replace([np.inf, -np.inf], np.nan).fillna(y.median())
                print("✅ Replaced infinite values in target")
            else:
                print("✅ No infinite values in target")
            
            print(f"📊 Final data shapes: X={X.shape}, y={y.shape}")
            print(f"📊 X data types: {X.dtypes.value_counts()}")
            print(f"📊 Y data type: {y.dtype}")
            print(f"📊 Y value range: {y.min():.4f} to {y.max():.4f}")
            
            # Test individual models
            relevant_models = {k: v for k, v in engine.models.items() 
                             if prediction_type.value in k}
            
            print(f"\n🧠 Testing {len(relevant_models)} models for {prediction_type.value}")
            
            for model_key, model in relevant_models.items():
                print(f"\n🔧 Testing model: {model_key}")
                
                try:
                    # Check model state before training
                    print(f"  Pre-training state: trained={getattr(model, 'is_trained', 'unknown')}")
                    
                    # Attempt training
                    print(f"  Calling model.train(X={X.shape}, y={y.shape})")
                    result = model.train(X, y)
                    
                    print(f"  Training result: {type(result)}")
                    if isinstance(result, dict):
                        print(f"  Result keys: {list(result.keys())}")
                        if 'error' in result:
                            print(f"  ❌ Training error: {result['error']}")
                        if 'validation_metrics' in result:
                            print(f"  📊 Validation metrics: {result['validation_metrics']}")
                    
                    # Check model state after training
                    print(f"  Post-training state: trained={getattr(model, 'is_trained', 'unknown')}")
                    
                    # Test prediction if training succeeded
                    if getattr(model, 'is_trained', False):
                        try:
                            # Use a small sample for prediction test
                            test_X = X.iloc[:5]
                            pred = model.predict(test_X)
                            print(f"  ✅ Prediction test successful: shape={np.array(pred).shape}")
                        except Exception as pred_e:
                            print(f"  ⚠️  Prediction test failed: {pred_e}")
                    else:
                        print(f"  ❌ Model not trained after training call")
                    
                except Exception as e:
                    print(f"  ❌ Model training failed: {e}")
                    print(f"  Error type: {type(e).__name__}")
                    print(f"  Error details: {traceback.format_exc()}")
    
    except Exception as e:
        print(f"❌ Model training diagnostic failed: {e}")
        print(traceback.format_exc())

def diagnose_ml_model_internals():
    """Examine the ML model implementations"""
    print("\n" + "=" * 80)
    print("🔬 ML MODEL INTERNALS DIAGNOSTIC")
    print("=" * 80)
    
    try:
        # Check if ML engine is available
        try:
            import oracle_engine.ml_prediction_engine as ml_engine
            print("✅ ML prediction engine available")
            
            # Test model creation
            model_type = ml_engine.ModelType.RANDOM_FOREST
            prediction_type = ml_engine.PredictionType.PRICE_DIRECTION
            
            print(f"🔧 Testing model creation: {model_type.value}_{prediction_type.value}")
            
            model = ml_engine.create_ml_model(model_type, prediction_type)
            print(f"✅ Model created: {type(model)}")
            print(f"✅ Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            
            # Check training method
            if hasattr(model, 'train'):
                print("✅ Model has train method")
                import inspect
                train_signature = inspect.signature(model.train)
                print(f"✅ Train method signature: {train_signature}")
            else:
                print("❌ Model missing train method")
            
            # Check is_trained attribute
            if hasattr(model, 'is_trained'):
                print(f"✅ Model has is_trained attribute: {model.is_trained}")
            else:
                print("❌ Model missing is_trained attribute")
            
        except ImportError as e:
            print(f"❌ ML prediction engine not available: {e}")
            
    except Exception as e:
        print(f"❌ ML model internals diagnostic failed: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    print("Starting deep training diagnostic...")
    
    # Step 1: Diagnose feature engineering
    features_df = diagnose_feature_engineering()
    
    # Step 2: Diagnose model training
    diagnose_model_training(features_df)
    
    # Step 3: Diagnose ML model internals
    diagnose_ml_model_internals()
    
    print("\n" + "=" * 80)
    print("🏁 DEEP DIAGNOSTIC COMPLETE")
    print("=" * 80)
