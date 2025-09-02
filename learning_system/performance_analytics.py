"""
Performance Analytics and Model Interpretability System
Comprehensive analysis, visualization, and interpretability for Oracle-X learning systems
"""

import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        confusion_matrix, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    timestamp: datetime
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0

class PerformanceAnalyzer:
    """Advanced performance analysis system"""
    
    def __init__(self):
        self.prediction_history = []
        self.performance_history = []
        self.model_comparisons = {}
        
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification performance metrics"""
        if not SKLEARN_AVAILABLE:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression performance metrics"""
        if not SKLEARN_AVAILABLE:
            return {'mse': 0.0, 'mae': 0.0, 'r2_score': 0.0}
        
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2_score': r2_score(y_true, y_pred)
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
            return {'mse': 0.0, 'mae': 0.0, 'r2_score': 0.0}
    
    def calculate_trading_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate trading-specific performance metrics"""
        if len(returns) == 0:
            return {}
        
        try:
            # Basic return metrics
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + returns.mean()) ** 252 - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = 0.02
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # Win rate
            win_rate = len(returns[returns > 0]) / len(returns)
            
            # Profit factor
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # Calmar ratio
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            # VaR and Expected Shortfall
            var_95 = returns.quantile(0.05)
            expected_shortfall = returns[returns <= var_95].mean()
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'calmar_ratio': calmar_ratio,
                'var_95': var_95,
                'expected_shortfall': expected_shortfall
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {}
    
    def analyze_model_performance(self, model_name: str, predictions: Dict[str, Any],
                                actual_values: Dict[str, Any]) -> PerformanceMetrics:
        """Comprehensive model performance analysis"""
        
        metrics = PerformanceMetrics(timestamp=datetime.now())
        
        try:
            # Extract prediction and actual arrays
            pred_array = np.array(list(predictions.values()))
            actual_array = np.array(list(actual_values.values()))
            
            if len(pred_array) == 0 or len(actual_array) == 0:
                return metrics
            
            # Determine if classification or regression
            unique_actual = len(np.unique(actual_array))
            is_classification = unique_actual <= 10
            
            if is_classification:
                # Classification metrics
                class_metrics = self.calculate_classification_metrics(actual_array, pred_array)
                metrics.accuracy = class_metrics['accuracy']
                metrics.precision = class_metrics['precision']
                metrics.recall = class_metrics['recall']
                metrics.f1_score = class_metrics['f1_score']
            else:
                # Regression metrics
                reg_metrics = self.calculate_regression_metrics(actual_array, pred_array)
                metrics.mse = reg_metrics['mse']
                metrics.mae = reg_metrics['mae']
                metrics.r2_score = reg_metrics['r2_score']
            
            # Trading metrics if we have returns
            if 'returns' in predictions:
                returns_series = pd.Series(predictions['returns'])
                trading_metrics = self.calculate_trading_metrics(returns_series)
                
                metrics.sharpe_ratio = trading_metrics.get('sharpe_ratio', 0.0)
                metrics.sortino_ratio = trading_metrics.get('sortino_ratio', 0.0)
                metrics.max_drawdown = trading_metrics.get('max_drawdown', 0.0)
                metrics.total_return = trading_metrics.get('total_return', 0.0)
                metrics.win_rate = trading_metrics.get('win_rate', 0.0)
                metrics.profit_factor = trading_metrics.get('profit_factor', 0.0)
                metrics.calmar_ratio = trading_metrics.get('calmar_ratio', 0.0)
                metrics.var_95 = trading_metrics.get('var_95', 0.0)
                metrics.expected_shortfall = trading_metrics.get('expected_shortfall', 0.0)
            
            # Store in history
            self.performance_history.append(metrics)
            
            logger.info(f"Performance analysis completed for {model_name}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing model performance: {e}")
            return metrics
    
    def generate_performance_report(self, model_name: str, 
                                  time_period: timedelta = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if time_period is None:
            time_period = timedelta(days=30)
        
        cutoff_time = datetime.now() - time_period
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'error': 'No recent performance data available'}
        
        # Aggregate metrics
        report = {
            'model_name': model_name,
            'time_period': str(time_period),
            'sample_count': len(recent_metrics),
            'summary': {},
            'trends': {},
            'recommendations': []
        }
        
        # Calculate summary statistics
        metrics_df = pd.DataFrame([m.__dict__ for m in recent_metrics])
        
        for column in metrics_df.select_dtypes(include=[np.number]).columns:
            if column != 'timestamp':
                report['summary'][column] = {
                    'mean': float(metrics_df[column].mean()),
                    'std': float(metrics_df[column].std()),
                    'min': float(metrics_df[column].min()),
                    'max': float(metrics_df[column].max()),
                    'latest': float(metrics_df[column].iloc[-1])
                }
        
        # Trend analysis
        for metric in ['accuracy', 'sharpe_ratio', 'total_return']:
            if metric in metrics_df.columns:
                values = metrics_df[metric].values
                if len(values) > 1:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    report['trends'][metric] = 'improving' if trend > 0 else 'declining'
        
        # Generate recommendations
        latest_metrics = recent_metrics[-1]
        
        if latest_metrics.accuracy < 0.6:
            report['recommendations'].append("Consider retraining model - accuracy below 60%")
        
        if latest_metrics.sharpe_ratio < 1.0:
            report['recommendations'].append("Risk-adjusted returns are low - review strategy")
        
        if latest_metrics.max_drawdown > 0.15:
            report['recommendations'].append("High drawdown detected - implement risk controls")
        
        return report

class ModelInterpreter:
    """Model interpretability and explainability system"""
    
    def __init__(self):
        self.feature_importance_history = {}
        self.shap_values = {}
        
    def calculate_feature_importance(self, model, X: pd.DataFrame, 
                                   method: str = 'permutation') -> Dict[str, float]:
        """Calculate feature importance using various methods"""
        
        if method == 'permutation' and hasattr(model, 'predict'):
            return self._permutation_importance(model, X)
        elif method == 'coefficients' and hasattr(model, 'coef_'):
            return self._coefficient_importance(model, X)
        elif method == 'tree' and hasattr(model, 'feature_importances_'):
            return dict(zip(X.columns, model.feature_importances_))
        else:
            # Default: equal importance
            return {col: 1.0/len(X.columns) for col in X.columns}
    
    def _permutation_importance(self, model, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate permutation importance"""
        try:
            baseline_score = model.predict(X)
            baseline_mse = np.mean(baseline_score ** 2)
            
            importance_scores = {}
            
            for col in X.columns:
                X_permuted = X.copy()
                X_permuted[col] = np.random.permutation(X_permuted[col].values)
                
                permuted_score = model.predict(X_permuted)
                permuted_mse = np.mean(permuted_score ** 2)
                
                importance_scores[col] = max(0, permuted_mse - baseline_mse)
            
            # Normalize
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                importance_scores = {k: v/total_importance for k, v in importance_scores.items()}
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {e}")
            return {col: 1.0/len(X.columns) for col in X.columns}
    
    def _coefficient_importance(self, model, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate importance from model coefficients"""
        try:
            if hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # Take first row for multi-class
                
                # Use absolute values
                abs_coef = np.abs(coef)
                
                # Normalize
                total = abs_coef.sum()
                if total > 0:
                    normalized_coef = abs_coef / total
                else:
                    normalized_coef = np.ones_like(abs_coef) / len(abs_coef)
                
                return dict(zip(X.columns, normalized_coef))
            
        except Exception as e:
            logger.error(f"Error calculating coefficient importance: {e}")
        
        return {col: 1.0/len(X.columns) for col in X.columns}
    
    def generate_model_explanation(self, model, X: pd.DataFrame, 
                                 prediction: Any) -> Dict[str, Any]:
        """Generate explanation for a specific prediction"""
        
        explanation = {
            'prediction': prediction,
            'timestamp': datetime.now(),
            'feature_contributions': {},
            'top_features': [],
            'confidence': 0.5
        }
        
        try:
            # Calculate feature importance
            importance = self.calculate_feature_importance(model, X)
            
            # Get feature values for this prediction
            if len(X) > 0:
                feature_values = X.iloc[-1].to_dict()
                
                # Calculate contributions (simplified)
                for feature, value in feature_values.items():
                    contrib = importance.get(feature, 0) * value
                    explanation['feature_contributions'][feature] = contrib
                
                # Get top contributing features
                sorted_contrib = sorted(
                    explanation['feature_contributions'].items(),
                    key=lambda x: abs(x[1]), reverse=True
                )
                explanation['top_features'] = sorted_contrib[:5]
            
            # Estimate confidence (simplified)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X.iloc[-1:].values)
                explanation['confidence'] = float(np.max(proba))
            
        except Exception as e:
            logger.error(f"Error generating model explanation: {e}")
        
        return explanation

class PerformanceDashboard:
    """Performance visualization and dashboard system"""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.interpreter = ModelInterpreter()
        
    def create_performance_dashboard(self, models_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive performance dashboard"""
        
        dashboard = {
            'timestamp': datetime.now(),
            'models': {},
            'comparisons': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Analyze each model
        for model_name, model_data in models_data.items():
            try:
                # Performance analysis
                if 'predictions' in model_data and 'actual' in model_data:
                    metrics = self.analyzer.analyze_model_performance(
                        model_name, model_data['predictions'], model_data['actual']
                    )
                    
                    dashboard['models'][model_name] = {
                        'metrics': metrics.__dict__,
                        'status': 'active'
                    }
                
                # Feature importance
                if 'model' in model_data and 'features' in model_data:
                    importance = self.interpreter.calculate_feature_importance(
                        model_data['model'], model_data['features']
                    )
                    dashboard['models'][model_name]['feature_importance'] = importance
                
            except Exception as e:
                logger.error(f"Error analyzing model {model_name}: {e}")
                dashboard['models'][model_name] = {'status': 'error', 'error': str(e)}
        
        # Model comparisons
        if len(dashboard['models']) > 1:
            dashboard['comparisons'] = self._compare_models(dashboard['models'])
        
        # Generate alerts and recommendations
        dashboard['alerts'] = self._generate_alerts(dashboard['models'])
        dashboard['recommendations'] = self._generate_recommendations(dashboard['models'])
        
        return dashboard
    
    def _compare_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across models"""
        
        comparisons = {
            'best_accuracy': {'model': None, 'value': 0},
            'best_sharpe': {'model': None, 'value': -np.inf},
            'lowest_drawdown': {'model': None, 'value': np.inf},
            'rankings': {}
        }
        
        for model_name, model_data in models.items():
            if 'metrics' in model_data:
                metrics = model_data['metrics']
                
                # Track best performers
                if metrics.get('accuracy', 0) > comparisons['best_accuracy']['value']:
                    comparisons['best_accuracy'] = {'model': model_name, 'value': metrics['accuracy']}
                
                if metrics.get('sharpe_ratio', -np.inf) > comparisons['best_sharpe']['value']:
                    comparisons['best_sharpe'] = {'model': model_name, 'value': metrics['sharpe_ratio']}
                
                if metrics.get('max_drawdown', np.inf) < comparisons['lowest_drawdown']['value']:
                    comparisons['lowest_drawdown'] = {'model': model_name, 'value': metrics['max_drawdown']}
        
        return comparisons
    
    def _generate_alerts(self, models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance alerts"""
        
        alerts = []
        
        for model_name, model_data in models.items():
            if 'metrics' in model_data:
                metrics = model_data['metrics']
                
                # Low accuracy alert
                if metrics.get('accuracy', 1.0) < 0.5:
                    alerts.append({
                        'type': 'performance',
                        'severity': 'high',
                        'model': model_name,
                        'message': f"Low accuracy: {metrics['accuracy']:.2%}"
                    })
                
                # High drawdown alert
                if metrics.get('max_drawdown', 0) > 0.2:
                    alerts.append({
                        'type': 'risk',
                        'severity': 'high',
                        'model': model_name,
                        'message': f"High drawdown: {metrics['max_drawdown']:.2%}"
                    })
                
                # Low Sharpe ratio alert
                if metrics.get('sharpe_ratio', 0) < 0.5:
                    alerts.append({
                        'type': 'performance',
                        'severity': 'medium',
                        'model': model_name,
                        'message': f"Low Sharpe ratio: {metrics['sharpe_ratio']:.2f}"
                    })
        
        return alerts
    
    def _generate_recommendations(self, models: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        # Check overall performance
        active_models = [m for m in models.values() if m.get('status') == 'active']
        
        if len(active_models) == 0:
            recommendations.append("No active models detected - check system status")
        
        # Performance-based recommendations
        avg_accuracy = np.mean([m.get('metrics', {}).get('accuracy', 0) for m in active_models])
        if avg_accuracy < 0.6:
            recommendations.append("Consider ensemble methods to improve accuracy")
        
        avg_sharpe = np.mean([m.get('metrics', {}).get('sharpe_ratio', 0) for m in active_models])
        if avg_sharpe < 1.0:
            recommendations.append("Implement risk management to improve risk-adjusted returns")
        
        return recommendations
    
    def export_dashboard(self, dashboard: Dict[str, Any], filepath: str):
        """Export dashboard to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(dashboard, f, indent=2, default=str)
            logger.info(f"Dashboard exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting dashboard: {e}")
