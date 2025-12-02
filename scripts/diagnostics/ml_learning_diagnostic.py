"""
Comprehensive ML Learning System Diagnostic and Enhancement Framework
Analyzes current learning capabilities and identifies optimization opportunities
"""

import sys
import os
import logging
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ml_dependencies():
    """Test all ML library dependencies"""
    results = {}
    
    # Test sklearn
    try:
        import sklearn
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        results['sklearn'] = {'available': True, 'version': sklearn.__version__}
    except ImportError as e:
        results['sklearn'] = {'available': False, 'error': str(e)}
    
    # Test xgboost
    try:
        import xgboost as xgb
        results['xgboost'] = {'available': True, 'version': xgb.__version__}
    except ImportError as e:
        results['xgboost'] = {'available': False, 'error': str(e)}
    
    # Test pytorch
    try:
        import torch
        results['pytorch'] = {'available': True, 'version': torch.__version__}
    except ImportError as e:
        results['pytorch'] = {'available': False, 'error': str(e)}
    
    # Test other dependencies
    try:
        import numpy as np
        results['numpy'] = {'available': True, 'version': np.__version__}
    except ImportError as e:
        results['numpy'] = {'available': False, 'error': str(e)}
    
    return results

def test_ml_engine_initialization():
    """Test ML engine initialization and model creation"""
    try:
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        from sentiment.sentiment_engine import AdvancedSentimentEngine
        
        # Initialize required dependencies
        orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        
        # Initialize engine with required parameters
        engine = EnsemblePredictionEngine(
            data_orchestrator=orchestrator,
            sentiment_engine=sentiment_engine
        )
        
        logger.info(f"Engine initialized successfully")
        logger.info(f"Available models: {list(engine.models.keys())}")
        
        # Check model states
        trained_models = []
        for model_key, model in engine.models.items():
            is_trained = hasattr(model, 'is_trained') and model.is_trained
            logger.info(f"Model {model_key}: trained={is_trained}")
            if is_trained:
                trained_models.append(model_key)
        
        return {
            'initialization_success': True,
            'total_models': len(engine.models),
            'trained_models': trained_models,
            'model_keys': list(engine.models.keys())
        }
        
    except Exception as e:
        logger.error(f"ML engine initialization failed: {e}")
        logger.error(traceback.format_exc())
        return {'initialization_success': False, 'error': str(e)}

def test_data_pipeline():
    """Test data collection and feature engineering pipeline"""
    try:
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        
        # Initialize orchestrator
        orchestrator = DataFeedOrchestrator()
        
        # Test data collection
        symbols = ["AAPL"]
        logger.info(f"Testing data collection for {symbols}")
        
        # Collect market data
        for symbol in symbols:
            market_data = orchestrator.get_market_data(symbol, period="30d", interval="1d")
            
            if market_data and hasattr(market_data, 'data') and not market_data.data.empty:
                data = market_data.data
                logger.info(f"Market data collection successful: {data.shape} rows/columns")
                logger.info(f"Columns: {list(data.columns)}")
                
                # Check data quality
                missing_data = data.isnull().sum()
                logger.info(f"Missing data per column: {missing_data[missing_data > 0].to_dict()}")
                
                return {
                    'data_collection_success': True,
                    'data_shape': data.shape,
                    'columns': list(data.columns),
                    'missing_data': missing_data[missing_data > 0].to_dict()
                }
            else:
                logger.warning("Data collection returned empty or None")
                return {'data_collection_success': False, 'error': 'Empty data returned'}
            
    except Exception as e:
        logger.error(f"Data pipeline test failed: {e}")
        logger.error(traceback.format_exc())
        return {'data_collection_success': False, 'error': str(e)}

def test_model_training():
    """Test actual model training process"""
    try:
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        from sentiment.sentiment_engine import AdvancedSentimentEngine
        
        # Initialize engine
        orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        engine = EnsemblePredictionEngine(
            data_orchestrator=orchestrator,
            sentiment_engine=sentiment_engine
        )
        
        symbols = ["AAPL"]
        
        # Attempt training
        logger.info("Starting model training...")
        training_results = engine.train_models(symbols, lookback_days=60)
        
        logger.info(f"Training results: {training_results}")
        
        # Check which models are now trained
        trained_count = 0
        training_details = {}
        
        for model_key, model in engine.models.items():
            is_trained = hasattr(model, 'is_trained') and model.is_trained
            training_details[model_key] = is_trained
            if is_trained:
                trained_count += 1
        
        return {
            'training_success': training_results is not None,
            'training_results': training_results,
            'trained_model_count': trained_count,
            'training_details': training_details
        }
        
    except Exception as e:
        logger.error(f"Model training test failed: {e}")
        logger.error(traceback.format_exc())
        return {'training_success': False, 'error': str(e)}

def test_prediction_pipeline():
    """Test the prediction pipeline end-to-end"""
    try:
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine, PredictionType
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        from sentiment.sentiment_engine import AdvancedSentimentEngine
        
        # Initialize and train engine
        orchestrator = DataFeedOrchestrator()
        sentiment_engine = AdvancedSentimentEngine()
        engine = EnsemblePredictionEngine(
            data_orchestrator=orchestrator,
            sentiment_engine=sentiment_engine
        )
        
        # Try to make a prediction
        prediction = engine.predict("AAPL", PredictionType.PRICE_DIRECTION, horizon_days=5)
        
        logger.info(f"Prediction result: {prediction}")
        
        return {
            'prediction_success': prediction is not None,
            'prediction_result': str(prediction) if prediction else None
        }
        
    except Exception as e:
        logger.error(f"Prediction test failed: {e}")
        logger.error(traceback.format_exc())
        return {'prediction_success': False, 'error': str(e)}

def test_sentiment_integration():
    """Test sentiment data integration with ML pipeline"""
    try:
        from sentiment.sentiment_engine import analyze_symbol_sentiment
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        
        # Initialize orchestrator for getting sentiment data
        orchestrator = DataFeedOrchestrator()
        
        # Test sentiment analysis
        symbols = ["AAPL", "TSLA"]
        sentiment_results = {}
        
        for symbol in symbols:
            try:
                # Get raw sentiment data first
                raw_sentiment = orchestrator.get_sentiment_data(symbol)
                
                # Extract texts for analysis
                all_texts = []
                for source_name, sentiment_data_obj in raw_sentiment.items():
                    if (hasattr(sentiment_data_obj, 'raw_data') and 
                        sentiment_data_obj.raw_data and 
                        'sample_texts' in sentiment_data_obj.raw_data):
                        texts = sentiment_data_obj.raw_data['sample_texts']
                        if texts:
                            all_texts.extend(texts[:5])  # Limit to 5 texts for testing
                
                # Analyze sentiment
                sentiment = analyze_symbol_sentiment(symbol, all_texts)
                
                sentiment_results[symbol] = {
                    'success': sentiment is not None,
                    'sentiment': sentiment.overall_sentiment if sentiment else None,
                    'confidence': sentiment.confidence if sentiment else None,
                    'text_count': len(all_texts)
                }
            except Exception as e:
                sentiment_results[symbol] = {'success': False, 'error': str(e)}
        
        return sentiment_results
        
    except Exception as e:
        logger.error(f"Sentiment integration test failed: {e}")
        logger.error(traceback.format_exc())
        return {'sentiment_integration_success': False, 'error': str(e)}

def analyze_learning_capabilities():
    """Analyze current learning and self-improvement capabilities"""
    capabilities = {
        'online_learning': False,
        'hyperparameter_optimization': False,
        'feature_selection': False,
        'model_ensemble': False,
        'performance_monitoring': False,
        'auto_retraining': False,
        'drift_detection': False,
        'model_versioning': False
    }
    
    try:
        # Check for online learning
        from oracle_engine.ml_prediction_engine import RandomForestPredictor, XGBoostPredictor, NeuralNetworkPredictor, PredictionType
        
        # Test if models support incremental learning
        try:
            rf_model = RandomForestPredictor(PredictionType.PRICE_DIRECTION)
            capabilities['online_learning'] = hasattr(rf_model, 'update') and callable(rf_model.update)
        except:
            capabilities['online_learning'] = False
        
        # Check for ensemble capabilities
        from oracle_engine.ensemble_ml_engine import EnsemblePredictionEngine
        from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
        from sentiment.sentiment_engine import AdvancedSentimentEngine
        
        try:
            orchestrator = DataFeedOrchestrator()
            sentiment_engine = AdvancedSentimentEngine()
            engine = EnsemblePredictionEngine(
                data_orchestrator=orchestrator,
                sentiment_engine=sentiment_engine
            )
            capabilities['model_ensemble'] = hasattr(engine, 'model_weights')
        except:
            capabilities['model_ensemble'] = False
        
        # Check for monitoring
        from oracle_engine.ml_model_manager import ModelMonitor
        try:
            monitor = ModelMonitor()
            capabilities['performance_monitoring'] = hasattr(monitor, 'record_prediction')
            capabilities['auto_retraining'] = hasattr(monitor, '_trigger_retraining')
        except:
            capabilities['performance_monitoring'] = False
            capabilities['auto_retraining'] = False
        
        # Check for drift detection
        from oracle_engine.ml_prediction_engine import ModelPerformance
        try:
            perf = ModelPerformance(model_name="test", prediction_type=PredictionType.PRICE_DIRECTION)
            capabilities['drift_detection'] = hasattr(perf, 'drift_score')
        except:
            capabilities['drift_detection'] = False
        
    except Exception as e:
        logger.error(f"Capability analysis failed: {e}")
    
    return capabilities

def generate_improvement_recommendations():
    """Generate specific recommendations for improving the learning system"""
    recommendations = {
        'critical_fixes': [
            "Fix model training pipeline - models initialize but don't actually train",
            "Resolve empty training results issue",
            "Fix prediction pipeline integration",
            "Ensure proper data flow from feature engineering to training"
        ],
        'learning_enhancements': [
            "Implement true online learning for all model types",
            "Add hyperparameter optimization using Optuna or similar",
            "Create adaptive learning rate scheduling",
            "Implement advanced ensemble methods (stacking, blending)",
            "Add automated feature selection and engineering",
            "Create meta-learning for automatic model selection"
        ],
        'self_improvement_features': [
            "Implement concept drift detection and adaptation",
            "Add automated architecture search for neural networks",
            "Create self-optimizing hyperparameter tuning",
            "Implement performance-based model weight updating",
            "Add automated feature engineering pipeline",
            "Create feedback loops for continuous improvement"
        ],
        'monitoring_improvements': [
            "Build real-time performance monitoring dashboard",
            "Implement comprehensive model validation framework",
            "Add A/B testing capabilities for model improvements",
            "Create automated model rollback mechanisms",
            "Implement model versioning and registry system",
            "Add comprehensive logging and debugging tools"
        ],
        'infrastructure_upgrades': [
            "Implement model checkpointing and persistence",
            "Add distributed training capabilities",
            "Create model serving infrastructure",
            "Implement caching for predictions and features",
            "Add GPU support for neural network training",
            "Create scalable data pipeline architecture"
        ]
    }
    
    return recommendations

def run_comprehensive_diagnostic():
    """Run complete diagnostic of ML learning system"""
    print("="*80)
    print("🚀 COMPREHENSIVE ML LEARNING SYSTEM DIAGNOSTIC")
    print("="*80)
    
    results = {}
    
    # Test 1: Dependencies
    print("\n📦 TESTING ML DEPENDENCIES")
    print("-" * 50)
    dep_results = test_ml_dependencies()
    results['dependencies'] = dep_results
    
    for lib, result in dep_results.items():
        status = "✅" if result['available'] else "❌"
        if result['available']:
            print(f"{status} {lib}: v{result['version']}")
        else:
            print(f"{status} {lib}: {result['error']}")
    
    # Test 2: ML Engine Initialization
    print("\n🔧 TESTING ML ENGINE INITIALIZATION")
    print("-" * 50)
    init_results = test_ml_engine_initialization()
    results['initialization'] = init_results
    
    if init_results.get('initialization_success'):
        print(f"✅ Engine initialized successfully")
        print(f"📊 Total models: {init_results['total_models']}")
        print(f"🎯 Trained models: {len(init_results['trained_models'])}")
        print(f"🔑 Model keys: {init_results['model_keys']}")
    else:
        print(f"❌ Engine initialization failed: {init_results.get('error', 'Unknown error')}")
    
    # Test 3: Data Pipeline
    print("\n💾 TESTING DATA PIPELINE")
    print("-" * 50)
    data_results = test_data_pipeline()
    results['data_pipeline'] = data_results
    
    if data_results and data_results.get('data_collection_success'):
        print(f"✅ Data collection successful")
        print(f"📊 Data shape: {data_results['data_shape']}")
        print(f"📋 Columns: {len(data_results['columns'])}")
        if data_results.get('missing_data'):
            print(f"⚠️  Missing data: {data_results['missing_data']}")
    else:
        error_msg = data_results.get('error', 'Unknown error') if data_results else 'No data results returned'
        print(f"❌ Data pipeline failed: {error_msg}")
    
    # Test 4: Model Training
    print("\n🧠 TESTING MODEL TRAINING")
    print("-" * 50)
    training_results = test_model_training()
    results['training'] = training_results
    
    if training_results.get('training_success'):
        print(f"✅ Training completed")
        print(f"📊 Trained models: {training_results['trained_model_count']}")
        print(f"📋 Training details: {training_results['training_details']}")
    else:
        print(f"❌ Training failed: {training_results.get('error', 'Unknown error')}")
    
    # Test 5: Prediction Pipeline
    print("\n🔮 TESTING PREDICTION PIPELINE")
    print("-" * 50)
    pred_results = test_prediction_pipeline()
    results['prediction'] = pred_results
    
    if pred_results.get('prediction_success'):
        print(f"✅ Prediction successful")
        print(f"📊 Result: {pred_results['prediction_result']}")
    else:
        print(f"❌ Prediction failed: {pred_results.get('error', 'Unknown error')}")
    
    # Test 6: Sentiment Integration
    print("\n💭 TESTING SENTIMENT INTEGRATION")
    print("-" * 50)
    sentiment_results = test_sentiment_integration()
    results['sentiment'] = sentiment_results
    
    for symbol, result in sentiment_results.items():
        if result.get('success'):
            print(f"✅ {symbol}: sentiment={result.get('sentiment', 'N/A'):.3f}, confidence={result.get('confidence', 'N/A'):.3f}")
        else:
            print(f"❌ {symbol}: {result.get('error', 'Unknown error')}")
    
    # Test 7: Learning Capabilities Analysis
    print("\n🎓 ANALYZING LEARNING CAPABILITIES")
    print("-" * 50)
    capabilities = analyze_learning_capabilities()
    results['capabilities'] = capabilities
    
    for capability, available in capabilities.items():
        status = "✅" if available else "❌"
        print(f"{status} {capability.replace('_', ' ').title()}")
    
    # Generate Recommendations
    print("\n💡 IMPROVEMENT RECOMMENDATIONS")
    print("-" * 50)
    recommendations = generate_improvement_recommendations()
    results['recommendations'] = recommendations
    
    for category, items in recommendations.items():
        print(f"\n📋 {category.replace('_', ' ').title()}:")
        for i, item in enumerate(items, 1):
            print(f"   {i}. {item}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*80)
    
    # Calculate overall health score
    health_score = 0
    total_checks = 0
    
    if results.get('dependencies', {}).get('sklearn', {}).get('available'):
        health_score += 1
    total_checks += 1
    
    if results.get('initialization', {}).get('initialization_success'):
        health_score += 1
    total_checks += 1
    
    if results.get('data_pipeline', {}).get('data_collection_success'):
        health_score += 1
    total_checks += 1
    
    if results.get('training', {}).get('training_success'):
        health_score += 1
    total_checks += 1
    
    if results.get('prediction', {}).get('prediction_success'):
        health_score += 1
    total_checks += 1
    
    capability_score = sum(results.get('capabilities', {}).values())
    total_capabilities = len(results.get('capabilities', {}))
    
    overall_score = (health_score / total_checks + capability_score / total_capabilities) / 2 * 100
    
    print(f"🎯 Overall System Health: {overall_score:.1f}%")
    print(f"✅ Core Functions: {health_score}/{total_checks}")
    print(f"🎓 Learning Capabilities: {capability_score}/{total_capabilities}")
    
    if overall_score < 50:
        print("🚨 CRITICAL: System requires immediate attention")
    elif overall_score < 75:
        print("⚠️  WARNING: System needs significant improvements") 
    else:
        print("✅ GOOD: System is functional with room for optimization")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_diagnostic()
    
    # Save results to file
    import json
    with open('ml_learning_diagnostic_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to ml_learning_diagnostic_results.json")
