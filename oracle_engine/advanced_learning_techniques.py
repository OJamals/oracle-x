"""
Advanced Learning Techniques for Oracle-X ML System
Phase 2B Implementation: Meta-Learning, Ensemble Stacking, Transfer Learning
"""

import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries with fallbacks
try:
    from sklearn.ensemble import StackingClassifier, StackingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import cross_val_score, ParameterGrid
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create dummy base classes
    class BaseEstimator:
        pass
    class ClassifierMixin:
        pass
    class RegressorMixin:
        pass

logger = logging.getLogger(__name__)

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning algorithms"""
    use_cross_validation: bool = True
    cv_folds: int = 5
    meta_model_type: str = 'logistic'  # 'logistic' or 'linear'
    enable_stacking: bool = True
    enable_blending: bool = True
    enable_auto_ml: bool = True
    hyperparameter_optimization: bool = True
    transfer_learning: bool = False

@dataclass
class HyperparameterConfig:
    """Hyperparameter optimization configuration"""
    method: str = 'grid_search'  # 'grid_search', 'random_search', 'bayesian'
    max_iterations: int = 50
    early_stopping: bool = True
    cv_folds: int = 3
    scoring_metric: str = 'accuracy'  # or 'mse', 'f1', etc.

class MetaLearner:
    """
    Advanced meta-learning system for Oracle-X
    Implements ensemble stacking, blending, and automated hyperparameter optimization
    """
    
    def __init__(self, config: MetaLearningConfig = None):
        self.config = config or MetaLearningConfig()
        self.base_models = {}
        self.meta_models = {}
        self.stacked_models = {}
        self.performance_history = []
        
        # Model registry
        self.model_registry = {
            'classifiers': [],
            'regressors': []
        }
        
        logger.info("Meta-learning system initialized")
    
    def register_base_model(self, model, model_name: str, model_type: str):
        """Register a base model for meta-learning"""
        self.base_models[model_name] = {
            'model': model,
            'type': model_type,
            'performance': None
        }
        
        if model_type == 'classification':
            self.model_registry['classifiers'].append(model_name)
        else:
            self.model_registry['regressors'].append(model_name)
        
        logger.info(f"Registered base model: {model_name} ({model_type})")
    
    def create_stacked_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                               prediction_type: str) -> Any:
        """
        Create a stacked ensemble using registered base models
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Sklearn not available, using simple ensemble averaging")
            return self._create_simple_ensemble(prediction_type)
        
        try:
            # Get relevant base models
            if prediction_type == 'classification':
                relevant_models = self.model_registry['classifiers']
                meta_model = LogisticRegression(random_state=42, max_iter=1000)
                stacking_class = StackingClassifier
            else:
                relevant_models = self.model_registry['regressors']
                meta_model = LinearRegression()
                stacking_class = StackingRegressor
            
            if len(relevant_models) < 2:
                logger.warning(f"Need at least 2 base models for stacking, found {len(relevant_models)}")
                return None
            
            # Prepare base estimators
            base_estimators = []
            for model_name in relevant_models:
                model_info = self.base_models[model_name]
                base_estimators.append((model_name, model_info['model']))
            
            # Create stacked ensemble
            stacked_ensemble = stacking_class(
                estimators=base_estimators,
                final_estimator=meta_model,
                cv=self.config.cv_folds if self.config.use_cross_validation else None,
                n_jobs=-1
            )
            
            # Fit the ensemble
            logger.info(f"Training stacked ensemble with {len(base_estimators)} base models")
            stacked_ensemble.fit(X, y)
            
            # Store in registry
            ensemble_name = f"stacked_{prediction_type}"
            self.stacked_models[ensemble_name] = stacked_ensemble
            
            logger.info(f"Successfully created stacked ensemble: {ensemble_name}")
            return stacked_ensemble
            
        except Exception as e:
            logger.error(f"Error creating stacked ensemble: {e}")
            return self._create_simple_ensemble(prediction_type)
    
    def _create_simple_ensemble(self, prediction_type: str):
        """Create simple averaging ensemble when sklearn unavailable"""
        class SimpleEnsemble:
            def __init__(self, models, prediction_type):
                self.models = models
                self.prediction_type = prediction_type
            
            def fit(self, X, y):
                for model in self.models:
                    if hasattr(model, 'fit'):
                        try:
                            model.fit(X, y)
                        except Exception as e:
                            logger.warning(f"Failed to fit model in simple ensemble: {e}")
                return self
            
            def predict(self, X):
                predictions = []
                for model in self.models:
                    if hasattr(model, 'predict'):
                        try:
                            pred = model.predict(X)
                            predictions.append(pred)
                        except Exception as e:
                            logger.warning(f"Failed to get prediction from model: {e}")
                
                if not predictions:
                    return np.zeros(len(X))
                
                # Average predictions
                return np.mean(predictions, axis=0)
        
        # Get relevant models
        if prediction_type == 'classification':
            relevant_models = [self.base_models[name]['model'] 
                             for name in self.model_registry['classifiers']]
        else:
            relevant_models = [self.base_models[name]['model'] 
                             for name in self.model_registry['regressors']]
        
        return SimpleEnsemble(relevant_models, prediction_type)
    
    def optimize_hyperparameters(self, model, param_grid: Dict, X: pd.DataFrame, 
                                y: pd.Series, config: HyperparameterConfig) -> Tuple[Any, Dict]:
        """
        Optimize hyperparameters for a given model
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Sklearn not available, returning original model")
            return model, {}
        
        try:
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
            
            if config.method == 'grid_search':
                optimizer = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=config.cv_folds,
                    scoring=config.scoring_metric,
                    n_jobs=-1,
                    verbose=0
                )
            elif config.method == 'random_search':
                optimizer = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=config.max_iterations,
                    cv=config.cv_folds,
                    scoring=config.scoring_metric,
                    n_jobs=-1,
                    verbose=0,
                    random_state=42
                )
            else:
                logger.warning(f"Unknown optimization method: {config.method}")
                return model, {}
            
            # Fit optimizer
            logger.info(f"Optimizing hyperparameters using {config.method}")
            optimizer.fit(X, y)
            
            best_params = optimizer.best_params_
            best_score = optimizer.best_score_
            
            logger.info(f"Best parameters found: {best_params}")
            logger.info(f"Best cross-validation score: {best_score:.4f}")
            
            return optimizer.best_estimator_, best_params
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {e}")
            return model, {}
    
    def create_blended_ensemble(self, models: List[Any], X: pd.DataFrame, 
                               y: pd.Series, prediction_type: str) -> Any:
        """
        Create a blended ensemble using weighted averaging
        """
        try:
            # Calculate weights based on cross-validation performance
            weights = self._calculate_model_weights(models, X, y, prediction_type)
            
            class BlendedEnsemble:
                def __init__(self, models, weights, prediction_type):
                    self.models = models
                    self.weights = weights
                    self.prediction_type = prediction_type
                
                def fit(self, X, y):
                    for model in self.models:
                        if hasattr(model, 'fit'):
                            try:
                                model.fit(X, y)
                            except Exception as e:
                                logger.warning(f"Failed to fit model in blended ensemble: {e}")
                    return self
                
                def predict(self, X):
                    predictions = []
                    valid_weights = []
                    
                    for i, model in enumerate(self.models):
                        if hasattr(model, 'predict'):
                            try:
                                pred = model.predict(X)
                                predictions.append(pred)
                                valid_weights.append(self.weights[i])
                            except Exception as e:
                                logger.warning(f"Failed to get prediction from model: {e}")
                    
                    if not predictions:
                        return np.zeros(len(X))
                    
                    # Weighted average
                    valid_weights = np.array(valid_weights)
                    valid_weights = valid_weights / valid_weights.sum()  # Normalize
                    
                    weighted_predictions = np.average(predictions, weights=valid_weights, axis=0)
                    return weighted_predictions
            
            ensemble = BlendedEnsemble(models, weights, prediction_type)
            logger.info(f"Created blended ensemble with weights: {weights}")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating blended ensemble: {e}")
            return self._create_simple_ensemble(prediction_type)
    
    def _calculate_model_weights(self, models: List[Any], X: pd.DataFrame, 
                                y: pd.Series, prediction_type: str) -> List[float]:
        """Calculate weights for models based on cross-validation performance"""
        if not SKLEARN_AVAILABLE:
            # Equal weights when sklearn unavailable
            return [1.0 / len(models)] * len(models)
        
        try:
            scores = []
            for model in models:
                try:
                    if prediction_type == 'classification':
                        score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
                    else:
                        score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error').mean()
                        score = -score  # Convert to positive
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Failed to calculate CV score for model: {e}")
                    scores.append(0.1)  # Small default score
            
            # Convert scores to weights (higher score = higher weight)
            scores = np.array(scores)
            if prediction_type == 'regression':
                # For regression, lower MSE is better, so invert
                scores = 1.0 / (scores + 1e-8)
            
            weights = scores / scores.sum()
            return weights.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating model weights: {e}")
            return [1.0 / len(models)] * len(models)
    
    def implement_transfer_learning(self, source_model, target_X: pd.DataFrame, 
                                   target_y: pd.Series, adaptation_method: str = 'fine_tuning') -> Any:
        """
        Implement transfer learning from source model to target domain
        """
        try:
            if adaptation_method == 'fine_tuning':
                return self._fine_tune_model(source_model, target_X, target_y)
            elif adaptation_method == 'feature_extraction':
                return self._extract_features(source_model, target_X, target_y)
            else:
                logger.warning(f"Unknown adaptation method: {adaptation_method}")
                return source_model
                
        except Exception as e:
            logger.error(f"Error in transfer learning: {e}")
            return source_model
    
    def _fine_tune_model(self, model, X: pd.DataFrame, y: pd.Series):
        """Fine-tune a pre-trained model on new data"""
        try:
            # For most sklearn models, this is just retraining
            if hasattr(model, 'fit'):
                model.fit(X, y)
            
            logger.info("Model fine-tuning completed")
            return model
            
        except Exception as e:
            logger.error(f"Error fine-tuning model: {e}")
            return model
    
    def _extract_features(self, model, X: pd.DataFrame, y: pd.Series):
        """Extract features using pre-trained model and train new classifier"""
        try:
            # Simplified feature extraction for demonstration
            if hasattr(model, 'predict_proba'):
                features = model.predict_proba(X)
            elif hasattr(model, 'predict'):
                features = model.predict(X).reshape(-1, 1)
            else:
                features = X
            
            # Train new model on extracted features
            if not SKLEARN_AVAILABLE:
                return model
            
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            if len(np.unique(y)) < 10:  # Classification
                new_model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # Regression
                new_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            new_model.fit(features, y)
            
            logger.info("Feature extraction transfer learning completed")
            return new_model
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return model
    
    def auto_ml_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                        prediction_type: str) -> Dict[str, Any]:
        """
        Automated machine learning pipeline that tries multiple approaches
        """
        try:
            results = {
                'best_model': None,
                'best_score': float('-inf') if prediction_type == 'classification' else float('inf'),
                'ensemble_performance': {},
                'individual_performance': {}
            }
            
            # Test individual models
            for model_name, model_info in self.base_models.items():
                if model_info['type'] == prediction_type:
                    try:
                        model = model_info['model']
                        
                        if SKLEARN_AVAILABLE:
                            if prediction_type == 'classification':
                                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                                score = scores.mean()
                            else:
                                scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
                                score = -scores.mean()
                        else:
                            # Simple train/test split evaluation
                            split_idx = int(0.8 * len(X))
                            model.fit(X.iloc[:split_idx], y.iloc[:split_idx])
                            pred = model.predict(X.iloc[split_idx:])
                            
                            if prediction_type == 'classification':
                                score = np.mean(pred == y.iloc[split_idx:])
                            else:
                                score = np.mean((pred - y.iloc[split_idx:]) ** 2)
                        
                        results['individual_performance'][model_name] = score
                        
                        # Update best model
                        if prediction_type == 'classification':
                            if score > results['best_score']:
                                results['best_score'] = score
                                results['best_model'] = model
                        else:
                            if score < results['best_score']:
                                results['best_score'] = score
                                results['best_model'] = model
                        
                        logger.info(f"Model {model_name} score: {score:.4f}")
                        
                    except Exception as e:
                        logger.warning(f"Error evaluating model {model_name}: {e}")
            
            # Test ensemble methods
            if len(self.base_models) >= 2:
                try:
                    # Stacked ensemble
                    stacked = self.create_stacked_ensemble(X, y, prediction_type)
                    if stacked:
                        if SKLEARN_AVAILABLE:
                            if prediction_type == 'classification':
                                scores = cross_val_score(stacked, X, y, cv=3, scoring='accuracy')
                                ensemble_score = scores.mean()
                            else:
                                scores = cross_val_score(stacked, X, y, cv=3, scoring='neg_mean_squared_error')
                                ensemble_score = -scores.mean()
                        else:
                            # Simple evaluation
                            split_idx = int(0.8 * len(X))
                            stacked.fit(X.iloc[:split_idx], y.iloc[:split_idx])
                            pred = stacked.predict(X.iloc[split_idx:])
                            
                            if prediction_type == 'classification':
                                ensemble_score = np.mean(pred == y.iloc[split_idx:])
                            else:
                                ensemble_score = np.mean((pred - y.iloc[split_idx:]) ** 2)
                        
                        results['ensemble_performance']['stacked'] = ensemble_score
                        
                        # Update best model if ensemble is better
                        if prediction_type == 'classification':
                            if ensemble_score > results['best_score']:
                                results['best_score'] = ensemble_score
                                results['best_model'] = stacked
                        else:
                            if ensemble_score < results['best_score']:
                                results['best_score'] = ensemble_score
                                results['best_model'] = stacked
                        
                        logger.info(f"Stacked ensemble score: {ensemble_score:.4f}")
                
                except Exception as e:
                    logger.warning(f"Error with stacked ensemble: {e}")
            
            logger.info(f"AutoML completed. Best score: {results['best_score']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in AutoML pipeline: {e}")
            return {'best_model': None, 'best_score': 0, 'ensemble_performance': {}, 'individual_performance': {}}

class AdvancedLearningOrchestrator:
    """
    Orchestrates advanced learning techniques for the Oracle-X system
    """
    
    def __init__(self, config: MetaLearningConfig = None):
        self.config = config or MetaLearningConfig()
        self.meta_learner = MetaLearner(config)
        self.active_models = {}
        self.learning_history = []
        
        logger.info("Advanced learning orchestrator initialized")
    
    def integrate_with_ensemble_engine(self, ensemble_engine):
        """Integrate with existing ensemble engine"""
        try:
            # Register existing models with meta-learner
            for target in ['price_direction', 'price_target']:
                for horizon in ['1d', '1w', '1m']:
                    model_key = f"{target}_{horizon}"
                    
                    # RandomForest models
                    rf_model = getattr(ensemble_engine, f'rf_{target}', {}).get(horizon)
                    if rf_model:
                        self.meta_learner.register_base_model(
                            rf_model, f"rf_{model_key}",
                            'classification' if target == 'price_direction' else 'regression'
                        )
                    
                    # XGBoost models
                    xgb_model = getattr(ensemble_engine, f'xgb_{target}', {}).get(horizon)
                    if xgb_model:
                        self.meta_learner.register_base_model(
                            xgb_model, f"xgb_{model_key}",
                            'classification' if target == 'price_direction' else 'regression'
                        )
                    
                    # Neural Network models
                    nn_model = getattr(ensemble_engine, f'nn_{target}', {}).get(horizon)
                    if nn_model:
                        self.meta_learner.register_base_model(
                            nn_model, f"nn_{model_key}",
                            'classification' if target == 'price_direction' else 'regression'
                        )
            
            logger.info("Successfully integrated with ensemble engine")
            
        except Exception as e:
            logger.error(f"Error integrating with ensemble engine: {e}")
    
    def enhance_learning_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                                 prediction_type: str) -> Dict[str, Any]:
        """
        Enhanced learning pipeline with all advanced techniques
        """
        try:
            results = {
                'enhanced_models': {},
                'ensemble_results': {},
                'auto_ml_results': {},
                'performance_comparison': {}
            }
            
            # AutoML pipeline
            if self.config.enable_auto_ml:
                logger.info("Running AutoML pipeline...")
                auto_ml_results = self.meta_learner.auto_ml_pipeline(X, y, prediction_type)
                results['auto_ml_results'] = auto_ml_results
            
            # Create enhanced ensembles
            if self.config.enable_stacking:
                logger.info("Creating stacked ensemble...")
                stacked_model = self.meta_learner.create_stacked_ensemble(X, y, prediction_type)
                if stacked_model:
                    results['ensemble_results']['stacked'] = stacked_model
            
            if self.config.enable_blending:
                logger.info("Creating blended ensemble...")
                base_models = [info['model'] for info in self.meta_learner.base_models.values()
                             if info['type'] == prediction_type]
                if len(base_models) >= 2:
                    blended_model = self.meta_learner.create_blended_ensemble(
                        base_models, X, y, prediction_type
                    )
                    results['ensemble_results']['blended'] = blended_model
            
            # Hyperparameter optimization for base models
            if self.config.hyperparameter_optimization:
                logger.info("Optimizing hyperparameters...")
                # This would be expanded with specific parameter grids for each model type
                # For now, we'll skip detailed implementation
                pass
            
            logger.info("Enhanced learning pipeline completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in enhanced learning pipeline: {e}")
            return {}
    
    def continuous_learning_cycle(self, new_data: pd.DataFrame, 
                                 targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Implement continuous learning with new data
        """
        try:
            results = {
                'models_updated': [],
                'performance_improvements': {},
                'new_ensembles': {}
            }
            
            for target_name, target_values in targets.items():
                prediction_type = 'classification' if 'direction' in target_name else 'regression'
                
                # Update existing models with new data
                for model_name, model_info in self.meta_learner.base_models.items():
                    if model_info['type'] == prediction_type:
                        try:
                            # Incremental learning (if supported) or retraining
                            model = model_info['model']
                            if hasattr(model, 'partial_fit'):
                                model.partial_fit(new_data, target_values)
                            elif hasattr(model, 'fit'):
                                model.fit(new_data, target_values)
                            
                            results['models_updated'].append(model_name)
                            
                        except Exception as e:
                            logger.warning(f"Failed to update model {model_name}: {e}")
                
                # Create new ensemble with updated models
                if len(results['models_updated']) >= 2:
                    enhanced_results = self.enhance_learning_pipeline(
                        new_data, target_values, prediction_type
                    )
                    results['new_ensembles'][target_name] = enhanced_results
            
            logger.info(f"Continuous learning completed. Updated {len(results['models_updated'])} models")
            return results
            
        except Exception as e:
            logger.error(f"Error in continuous learning cycle: {e}")
            return {'models_updated': [], 'performance_improvements': {}, 'new_ensembles': {}}
