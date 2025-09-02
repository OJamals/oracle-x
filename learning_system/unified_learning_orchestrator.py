"""
Unified Learning Orchestrator for Oracle-X
Integrates all advanced learning systems with real-time adaptation
"""

import logging
import numpy as np
import pandas as pd
import json
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import all learning system components
from .advanced_rl_framework import MultiAgentRLSystem, RLConfig, TradingEnvironment, TradingAction
from .neural_ensemble_system import NeuralEnsembleSystem, NeuralConfig
from .adaptive_feature_engineering import AdaptiveFeatureEngineering, FeatureConfig
from .advanced_risk_management import AdvancedRiskManager, RiskConfig
from .performance_analytics import PerformanceAnalyzer, ModelInterpreter, PerformanceDashboard, PerformanceMetrics

# Import existing Oracle-X components
try:
    from oracle_engine.advanced_learning_techniques import AdvancedLearningOrchestrator
    from oracle_engine.realtime_learning_engine import RealTimeLearningEngine
    ORACLE_COMPONENTS_AVAILABLE = True
except ImportError:
    ORACLE_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class UnifiedLearningConfig:
    """Configuration for unified learning system"""
    # Component configurations
    rl_config: RLConfig = field(default_factory=RLConfig)
    neural_config: NeuralConfig = field(default_factory=NeuralConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    
    # Integration settings
    enable_rl: bool = True
    enable_neural_ensemble: bool = True
    enable_adaptive_features: bool = True
    enable_risk_management: bool = True
    enable_real_time_learning: bool = True
    enable_performance_analytics: bool = True
    
    # Performance settings
    update_frequency: int = 100  # samples
    retraining_frequency: int = 1000  # samples
    model_selection_frequency: int = 500  # samples
    
    # Ensemble settings
    ensemble_method: str = 'weighted'  # 'weighted', 'stacking', 'voting'
    confidence_threshold: float = 0.6
    
    # Persistence settings
    save_models: bool = True
    model_directory: str = "models"
    checkpoint_frequency: int = 1000

@dataclass
class LearningMetrics:
    """Comprehensive learning performance metrics"""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    avg_return_per_trade: float
    volatility: float
    var_95: float
    model_confidence: float
    feature_importance: Dict[str, float]
    risk_metrics: Dict[str, float]

class UnifiedLearningOrchestrator:
    """Main orchestrator for all advanced learning systems"""
    
    def __init__(self, config: UnifiedLearningConfig = None):
        self.config = config or UnifiedLearningConfig()
        
        # Initialize component systems
        self.rl_system = None
        self.neural_ensemble = None
        self.feature_engineer = None
        self.risk_manager = None
        self.realtime_engine = None
        
        # Performance analytics systems
        self.performance_analyzer = None
        self.model_interpreter = None
        self.performance_dashboard = None
        
        # Integration with existing Oracle-X systems
        self.oracle_orchestrator = None
        
        # State management
        self.is_initialized = False
        self.is_training = False
        self.sample_count = 0
        self.last_update = datetime.now()
        
        # Performance tracking
        self.performance_history = []
        self.model_performance = {}
        self.ensemble_weights = {}
        
        # Data management
        self.feature_cache = {}
        self.prediction_cache = {}
        self.training_data = pd.DataFrame()
        
        # Threading for real-time processing
        self.processing_lock = threading.Lock()
        self.background_thread = None
        self.is_running = False
        
        logger.info("Unified learning orchestrator initialized")
    
    async def initialize_systems(self):
        """Initialize all learning systems"""
        try:
            logger.info("Initializing advanced learning systems...")
            
            # Initialize RL system
            if self.config.enable_rl:
                self.rl_system = MultiAgentRLSystem(self.config.rl_config)
                logger.info("RL system initialized")
            
            # Initialize neural ensemble
            if self.config.enable_neural_ensemble:
                self.neural_ensemble = NeuralEnsembleSystem(self.config.neural_config)
                logger.info("Neural ensemble initialized")
            
            # Initialize feature engineering
            if self.config.enable_adaptive_features:
                self.feature_engineer = AdaptiveFeatureEngineering(self.config.feature_config)
                logger.info("Adaptive feature engineering initialized")
            
            # Initialize risk management
            if self.config.enable_risk_management:
                self.risk_manager = AdvancedRiskManager(self.config.risk_config)
                logger.info("Advanced risk management initialized")
            
            # Initialize real-time learning
            if self.config.enable_real_time_learning:
                from oracle_engine.realtime_learning_engine import RealTimeLearningEngine, OnlineLearningConfig
                self.realtime_engine = RealTimeLearningEngine(OnlineLearningConfig())
                logger.info("Real-time learning engine initialized")
            
            # Initialize performance analytics
            if self.config.enable_performance_analytics:
                self.performance_analyzer = PerformanceAnalyzer()
                self.model_interpreter = ModelInterpreter()
                self.performance_dashboard = PerformanceDashboard()
                logger.info("Performance analytics systems initialized")
            
            # Initialize Oracle-X integration
            if ORACLE_COMPONENTS_AVAILABLE:
                self.oracle_orchestrator = AdvancedLearningOrchestrator()
                logger.info("Oracle-X integration initialized")
            
            self.is_initialized = True
            logger.info("All learning systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing learning systems: {e}")
            raise
    
    def start_real_time_learning(self):
        """Start real-time learning and adaptation"""
        if not self.is_initialized:
            raise RuntimeError("Systems not initialized. Call initialize_systems() first.")
        
        self.is_running = True
        
        # Start real-time learning engine
        if self.realtime_engine:
            self.realtime_engine.start_real_time_learning()
        
        # Start background processing thread
        self.background_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.background_thread.start()
        
        logger.info("Real-time learning started")
    
    def stop_real_time_learning(self):
        """Stop real-time learning"""
        self.is_running = False
        
        if self.realtime_engine:
            self.realtime_engine.stop_real_time_learning()
        
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5.0)
        
        logger.info("Real-time learning stopped")
    
    def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process new market data through all learning systems"""
        with self.processing_lock:
            try:
                start_time = datetime.now()
                results = {
                    'timestamp': start_time,
                    'predictions': {},
                    'risk_assessment': {},
                    'trading_actions': {},
                    'confidence_scores': {},
                    'feature_importance': {},
                    'processing_time_ms': 0
                }
                
                # Convert market data to DataFrame if needed
                if isinstance(market_data, dict):
                    df = pd.DataFrame([market_data])
                else:
                    df = market_data.copy()
                
                # Feature engineering
                if self.feature_engineer:
                    df_features = self.feature_engineer.engineer_features(df)
                    self.feature_cache['latest'] = df_features
                    results['feature_importance'] = self.feature_engineer.get_feature_importance()
                else:
                    df_features = df
                
                # Neural ensemble predictions
                if self.neural_ensemble and hasattr(self.neural_ensemble, 'models') and self.neural_ensemble.models:
                    neural_pred = self.neural_ensemble.predict_ensemble(df_features)
                    results['predictions']['neural_ensemble'] = neural_pred
                
                # RL system decisions
                if self.rl_system:
                    # Convert to trading environment
                    trading_env = self._create_trading_environment(market_data)
                    rl_actions = self.rl_system.coordinate_agents(trading_env)
                    results['trading_actions']['rl_system'] = rl_actions
                
                # Risk assessment
                if self.risk_manager:
                    # Create dummy positions for risk assessment
                    positions = self._create_position_data(market_data)
                    portfolio_risk = self.risk_manager.assess_portfolio_risk(positions)
                    risk_limits = self.risk_manager.check_risk_limits(portfolio_risk)
                    
                    results['risk_assessment'] = {
                        'portfolio_risk': portfolio_risk.__dict__,
                        'risk_limits': risk_limits
                    }
                
                # Real-time learning update
                if self.realtime_engine:
                    # Process sample through real-time engine
                    sample_data = df_features.iloc[-1] if len(df_features) > 0 else pd.Series()
                    rt_result = self.realtime_engine.process_new_sample(sample_data, 0)  # Dummy target
                    results['predictions']['realtime'] = rt_result
                
                # Calculate ensemble prediction
                ensemble_pred = self._calculate_ensemble_prediction(results['predictions'])
                results['ensemble_prediction'] = ensemble_pred
                
                # Performance analytics
                if self.performance_analyzer and len(results['predictions']) > 0:
                    # Analyze current predictions for performance tracking
                    self._update_performance_metrics(results)
                
                # Update sample count
                self.sample_count += 1
                self.last_update = start_time
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                results['processing_time_ms'] = processing_time
                
                logger.debug(f"Processed market data in {processing_time:.2f}ms")
                return results
                
            except Exception as e:
                logger.error(f"Error processing market data: {e}")
                return {'error': str(e), 'timestamp': datetime.now()}
    
    def train_models(self, training_data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Train all learning models"""
        if not self.is_initialized:
            raise RuntimeError("Systems not initialized")
        
        self.is_training = True
        training_results = {}
        
        try:
            logger.info(f"Training models on {len(training_data)} samples")
            
            # Feature engineering
            if self.feature_engineer:
                X_features, y = self.feature_engineer.fit_transform(training_data, target_column)
                training_results['feature_engineering'] = {
                    'original_features': len(training_data.columns) - 1,
                    'engineered_features': len(X_features.columns),
                    'selected_features': len(self.feature_engineer.feature_selector.selected_features)
                }
            else:
                y = training_data[target_column]
                X_features = training_data.drop(columns=[target_column])
            
            # Train neural ensemble
            if self.neural_ensemble:
                # Create models if not exists
                input_size = len(X_features.columns)
                self.neural_ensemble.create_models(input_size=input_size, output_size=1)
                
                # Train models
                neural_results = self.neural_ensemble.train_models(X_features, y)
                training_results['neural_ensemble'] = neural_results
            
            # Train RL agents
            if self.rl_system:
                # Create RL agents for different strategies
                state_size = len(X_features.columns)
                action_size = 3  # buy, hold, sell
                
                self.rl_system.create_agent('trend_follower', state_size, action_size)
                self.rl_system.create_agent('mean_reverter', state_size, action_size)
                self.rl_system.create_agent('momentum_trader', state_size, action_size)
                
                training_results['rl_system'] = {'agents_created': 3}
            
            # Update risk management with historical data
            if self.risk_manager:
                # Extract price data for risk calculations
                price_data = {}
                for col in training_data.columns:
                    if 'price' in col.lower() or 'close' in col.lower():
                        price_data[col] = training_data[col]
                
                if price_data:
                    self.risk_manager.update_market_data(price_data)
                
                training_results['risk_management'] = {'price_series_updated': len(price_data)}
            
            # Register models with real-time engine
            if self.realtime_engine and self.neural_ensemble and hasattr(self.neural_ensemble, 'models'):
                for model_name, model in self.neural_ensemble.models.items():
                    self.realtime_engine.register_model(model, f"neural_{model_name}", "regression")
                
                training_results['realtime_integration'] = {'models_registered': len(self.neural_ensemble.models)}
            
            # Store training data
            self.training_data = training_data.copy()
            
            logger.info("Model training completed successfully")
            training_results['status'] = 'success'
            training_results['training_samples'] = len(training_data)
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            training_results['status'] = 'error'
            training_results['error'] = str(e)
        
        finally:
            self.is_training = False
        
        return training_results
    
    def make_trading_decision(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make comprehensive trading decision using all systems"""
        if not self.is_initialized:
            return {'error': 'Systems not initialized'}
        
        # Process market data
        processing_results = self.process_market_data(market_data)
        
        if 'error' in processing_results:
            return processing_results
        
        # Extract predictions and actions
        predictions = processing_results.get('predictions', {})
        trading_actions = processing_results.get('trading_actions', {})
        risk_assessment = processing_results.get('risk_assessment', {})
        
        # Calculate confidence-weighted decision
        final_decision = self._make_final_decision(predictions, trading_actions, risk_assessment)
        
        # Add metadata
        final_decision.update({
            'timestamp': datetime.now(),
            'processing_time_ms': processing_results.get('processing_time_ms', 0),
            'sample_count': self.sample_count,
            'systems_used': list(predictions.keys()) + list(trading_actions.keys())
        })
        
        return final_decision
    
    def _create_trading_environment(self, market_data: Dict[str, Any]) -> TradingEnvironment:
        """Create trading environment from market data"""
        # Extract relevant data with defaults
        price_history = np.array([market_data.get('close', 100.0)])
        
        technical_indicators = {
            'rsi': market_data.get('rsi', 50.0),
            'macd': market_data.get('macd', 0.0),
            'bb_position': market_data.get('bb_position', 0.5)
        }
        
        risk_metrics = {
            'sharpe_ratio': market_data.get('sharpe_ratio', 0.0),
            'max_drawdown': market_data.get('max_drawdown', 0.0),
            'var_95': market_data.get('var_95', 0.0)
        }
        
        return TradingEnvironment(
            price_history=price_history,
            technical_indicators=technical_indicators,
            market_sentiment=market_data.get('sentiment', 0.0),
            volatility=market_data.get('volatility', 0.2),
            volume=market_data.get('volume', 1000000),
            position=market_data.get('position', 0.0),
            cash=market_data.get('cash', 100000),
            portfolio_value=market_data.get('portfolio_value', 100000),
            risk_metrics=risk_metrics,
            timestamp=datetime.now()
        )
    
    def _create_position_data(self, market_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Create position data for risk assessment"""
        # Create dummy position data
        symbol = market_data.get('symbol', 'UNKNOWN')
        price = market_data.get('close', 100.0)
        
        return {
            symbol: {
                'position_size': 0.1,  # 10% position
                'market_value': price * 100  # 100 shares
            }
        }
    
    def _calculate_ensemble_prediction(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ensemble prediction from all models"""
        if not predictions:
            return {'prediction': 0.0, 'confidence': 0.0}
        
        # Extract numerical predictions
        pred_values = []
        confidence_values = []
        
        for system_name, pred_data in predictions.items():
            if isinstance(pred_data, dict):
                if 'ensemble_prediction' in pred_data:
                    pred_values.append(pred_data['ensemble_prediction'])
                    confidence_values.append(pred_data.get('confidence', 0.5))
                elif 'prediction' in pred_data:
                    pred_values.append(pred_data['prediction'])
                    confidence_values.append(pred_data.get('confidence', 0.5))
            elif isinstance(pred_data, (int, float)):
                pred_values.append(pred_data)
                confidence_values.append(0.5)
        
        if not pred_values:
            return {'prediction': 0.0, 'confidence': 0.0}
        
        # Weighted average based on confidence
        if confidence_values:
            weights = np.array(confidence_values)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            ensemble_pred = np.average(pred_values, weights=weights)
            ensemble_confidence = np.mean(confidence_values)
        else:
            ensemble_pred = np.mean(pred_values)
            ensemble_confidence = 0.5
        
        return {
            'prediction': float(ensemble_pred),
            'confidence': float(ensemble_confidence),
            'individual_predictions': len(pred_values),
            'prediction_range': [float(min(pred_values)), float(max(pred_values))]
        }
    
    def _make_final_decision(self, predictions: Dict[str, Any], 
                           trading_actions: Dict[str, Any],
                           risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Make final trading decision"""
        # Get ensemble prediction
        ensemble_pred = self._calculate_ensemble_prediction(predictions)
        
        # Extract risk limits
        risk_violations = risk_assessment.get('risk_limits', {}).get('violations', [])
        within_limits = len(risk_violations) == 0
        
        # Determine action based on prediction and risk
        prediction = ensemble_pred.get('prediction', 0.0)
        confidence = ensemble_pred.get('confidence', 0.0)
        
        # Decision logic
        if not within_limits:
            action = 'hold'  # Don't trade if risk limits exceeded
            reason = f"Risk limits exceeded: {len(risk_violations)} violations"
        elif confidence < self.config.confidence_threshold:
            action = 'hold'  # Don't trade if confidence too low
            reason = f"Low confidence: {confidence:.2f} < {self.config.confidence_threshold}"
        elif prediction > 0.02:  # 2% threshold
            action = 'buy'
            reason = f"Strong positive prediction: {prediction:.3f}"
        elif prediction < -0.02:
            action = 'sell'
            reason = f"Strong negative prediction: {prediction:.3f}"
        else:
            action = 'hold'
            reason = f"Neutral prediction: {prediction:.3f}"
        
        # Calculate position size if trading
        position_size = 0.0
        if action in ['buy', 'sell'] and self.risk_manager:
            # Use risk manager to calculate optimal position size
            portfolio_value = 100000  # Default portfolio value
            position_size = self.risk_manager.calculate_optimal_position_size(
                'UNKNOWN', 
                {'expected_return': prediction, 'confidence': confidence},
                portfolio_value
            )
        
        return {
            'action': action,
            'position_size': position_size,
            'prediction': prediction,
            'confidence': confidence,
            'reason': reason,
            'risk_violations': len(risk_violations),
            'within_risk_limits': within_limits,
            'ensemble_data': ensemble_pred
        }
    
    def _background_worker(self):
        """Background worker for continuous adaptation"""
        while self.is_running:
            try:
                # Periodic model updates
                if self.sample_count > 0 and self.sample_count % self.config.update_frequency == 0:
                    self._periodic_update()
                
                # Model checkpointing
                if self.config.save_models and self.sample_count % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint()
                
                # Sleep to prevent excessive CPU usage
                threading.Event().wait(1.0)
                
            except Exception as e:
                logger.error(f"Error in background worker: {e}")
    
    def _periodic_update(self):
        """Perform periodic model updates"""
        try:
            logger.info(f"Performing periodic update at sample {self.sample_count}")
            
            # Update ensemble weights based on recent performance
            if self.neural_ensemble and self.model_performance:
                self.neural_ensemble.update_ensemble_weights(self.model_performance)
            
            # Update RL agent weights
            if self.rl_system and self.model_performance:
                rl_performance = {k: v for k, v in self.model_performance.items() if 'rl' in k}
                if rl_performance:
                    self.rl_system.update_agent_weights(rl_performance)
            
            # Trigger real-time learning updates
            if self.realtime_engine:
                status = self.realtime_engine.get_system_status()
                logger.debug(f"Real-time learning status: {status.get('total_samples_processed', 0)} samples processed")
            
        except Exception as e:
            logger.error(f"Error in periodic update: {e}")
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        try:
            checkpoint_dir = Path(self.config.model_directory) / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save neural models
            if self.neural_ensemble:
                self.neural_ensemble.save_models(str(checkpoint_dir / "neural"))
            
            # Save RL models
            if self.rl_system:
                self.rl_system.save_models(str(checkpoint_dir / "rl"))
            
            # Save feature engineering pipeline
            if self.feature_engineer:
                self.feature_engineer.save_pipeline(str(checkpoint_dir / "features.json"))
            
            # Save orchestrator state
            state_data = {
                'config': self.config.__dict__,
                'sample_count': self.sample_count,
                'performance_history': self.performance_history[-100:],  # Last 100 entries
                'model_performance': self.model_performance,
                'ensemble_weights': self.ensemble_weights
            }
            
            with open(checkpoint_dir / "orchestrator_state.json", 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now(),
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'is_training': self.is_training,
            'sample_count': self.sample_count,
            'last_update': self.last_update,
            'systems': {}
        }
        
        # Component system status
        if self.rl_system:
            status['systems']['rl'] = {
                'agents': len(self.rl_system.agents) if hasattr(self.rl_system, 'agents') else 0,
                'active': True
            }
        
        if self.neural_ensemble:
            status['systems']['neural'] = {
                'models': len(self.neural_ensemble.models) if hasattr(self.neural_ensemble, 'models') else 0,
                'active': True
            }
        
        if self.feature_engineer:
            status['systems']['features'] = {
                'selected_features': len(self.feature_engineer.feature_selector.selected_features),
                'active': True
            }
        
        if self.risk_manager:
            status['systems']['risk'] = {
                'active': True,
                'config': self.risk_manager.config.__dict__
            }
        
        if self.realtime_engine:
            rt_status = self.realtime_engine.get_system_status()
            status['systems']['realtime'] = rt_status
        
        # Performance analytics status
        if self.performance_analyzer:
            status['systems']['analytics'] = {
                'performance_history_count': len(self.performance_analyzer.performance_history),
                'active': True
            }
        
        return status
    
    def _update_performance_metrics(self, results: Dict[str, Any]):
        """Update performance metrics with current results"""
        try:
            if not self.performance_analyzer:
                return
            
            # Extract predictions for analysis
            predictions = results.get('predictions', {})
            if not predictions:
                return
            
            # Create dummy actual values for demonstration
            # In real implementation, these would come from market feedback
            actual_values = {}
            for system_name, pred_data in predictions.items():
                if isinstance(pred_data, dict) and 'prediction' in pred_data:
                    # Simulate actual outcome (in real system, this comes from market data)
                    actual_values[system_name] = pred_data['prediction'] + np.random.normal(0, 0.01)
                elif isinstance(pred_data, (int, float)):
                    actual_values[system_name] = pred_data + np.random.normal(0, 0.01)
            
            # Analyze performance for each system
            for system_name in predictions.keys():
                if system_name in actual_values:
                    system_predictions = {system_name: predictions[system_name]}
                    system_actuals = {system_name: actual_values[system_name]}
                    
                    metrics = self.performance_analyzer.analyze_model_performance(
                        system_name, system_predictions, system_actuals
                    )
                    
                    # Store in model performance tracking
                    self.model_performance[system_name] = {
                        'accuracy': metrics.accuracy,
                        'mse': metrics.mse,
                        'timestamp': metrics.timestamp
                    }
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def generate_performance_report(self, time_period: timedelta = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.performance_analyzer:
            return {'error': 'Performance analytics not enabled'}
        
        try:
            # Generate reports for all tracked models
            reports = {}
            
            for model_name in self.model_performance.keys():
                report = self.performance_analyzer.generate_performance_report(
                    model_name, time_period
                )
                reports[model_name] = report
            
            # Generate dashboard if available
            dashboard_data = None
            if self.performance_dashboard:
                models_data = {}
                for model_name, perf_data in self.model_performance.items():
                    models_data[model_name] = {
                        'predictions': {},  # Would be populated with real prediction history
                        'actual': {},       # Would be populated with real outcomes
                        'metrics': perf_data
                    }
                
                dashboard_data = self.performance_dashboard.create_performance_dashboard(models_data)
            
            return {
                'timestamp': datetime.now(),
                'individual_reports': reports,
                'dashboard': dashboard_data,
                'summary': {
                    'total_models': len(reports),
                    'active_models': len([r for r in reports.values() if 'error' not in r]),
                    'sample_count': self.sample_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def get_model_explanations(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get model explanations for current predictions"""
        if not self.model_interpreter:
            return {'error': 'Model interpreter not available'}
        
        try:
            explanations = {}
            
            # Process market data to get features
            if self.feature_engineer:
                df = pd.DataFrame([market_data])
                df_features = self.feature_engineer.engineer_features(df)
            else:
                df_features = pd.DataFrame([market_data])
            
            # Get explanations for each model
            if self.neural_ensemble and hasattr(self.neural_ensemble, 'models'):
                for model_name, model in self.neural_ensemble.models.items():
                    try:
                        prediction = model.predict(df_features.values)
                        explanation = self.model_interpreter.generate_model_explanation(
                            model, df_features, prediction
                        )
                        explanations[f'neural_{model_name}'] = explanation
                    except Exception as e:
                        logger.warning(f"Could not generate explanation for {model_name}: {e}")
            
            return {
                'timestamp': datetime.now(),
                'explanations': explanations,
                'feature_count': len(df_features.columns) if len(df_features) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating model explanations: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Graceful shutdown of all systems"""
        logger.info("Shutting down unified learning orchestrator...")
        
        # Stop real-time learning
        self.stop_real_time_learning()
        
        # Save final checkpoint
        if self.config.save_models:
            self._save_checkpoint()
        
        # Clear caches
        self.feature_cache.clear()
        self.prediction_cache.clear()
        
        logger.info("Unified learning orchestrator shutdown complete")
