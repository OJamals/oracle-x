"""
Advanced Neural Network Ensemble System for Oracle-X
Deep Learning with Attention Mechanisms, Transformers, and Multi-Modal Learning
"""

import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class NeuralConfig:
    """Configuration for neural network ensemble"""
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    use_attention: bool = True
    use_transformer: bool = True
    use_lstm: bool = True
    ensemble_method: str = 'weighted'  # 'weighted', 'stacking', 'voting'

class MultiHeadAttention(nn.Module if TORCH_AVAILABLE else object):
    """Multi-head attention mechanism for time series"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        if not TORCH_AVAILABLE:
            return
        
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        if not TORCH_AVAILABLE:
            return query
        
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(context)

class TransformerEncoder(nn.Module if TORCH_AVAILABLE else object):
    """Transformer encoder for sequence modeling"""
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 512, dropout: float = 0.1):
        if not TORCH_AVAILABLE:
            return
        
        super(TransformerEncoder, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        if not TORCH_AVAILABLE:
            return x
        
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class LSTMPredictor(nn.Module if TORCH_AVAILABLE else object):
    """LSTM-based predictor with attention"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.3):
        if not TORCH_AVAILABLE:
            return
        
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        self.attention = MultiHeadAttention(hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return torch.zeros(x.size(0), 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_out = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm(lstm_out + attn_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        return output

class TransformerPredictor(nn.Module if TORCH_AVAILABLE else object):
    """Transformer-based predictor"""
    
    def __init__(self, input_size: int, d_model: int = 256, num_heads: int = 8, 
                 num_layers: int = 4, output_size: int = 1, dropout: float = 0.3):
        if not TORCH_AVAILABLE:
            return
        
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._create_positional_encoding(1000, d_model)
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
    
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create positional encoding"""
        if not TORCH_AVAILABLE:
            return None
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return torch.zeros(x.size(0), 1)
        
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.positional_encoding.size(1):
            x += self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling
        pooled = torch.mean(x, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        return output

class CNNPredictor(nn.Module if TORCH_AVAILABLE else object):
    """1D CNN for time series prediction"""
    
    def __init__(self, input_size: int, num_filters: int = 64, 
                 filter_sizes: List[int] = None, output_size: int = 1, dropout: float = 0.3):
        if not TORCH_AVAILABLE:
            return
        
        super(CNNPredictor, self).__init__()
        
        if filter_sizes is None:
            filter_sizes = [3, 5, 7]
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_size, num_filters, kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in filter_sizes
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * len(filter_sizes), num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, output_size)
        )
    
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return torch.zeros(x.size(0), 1)
        
        # Transpose for conv1d (batch, channels, length)
        x = x.transpose(1, 2)
        
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = F.relu(bn(conv(x)))
            pooled = self.global_pool(conv_out).squeeze(-1)
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        combined = torch.cat(conv_outputs, dim=1)
        
        # Classification
        output = self.classifier(combined)
        return output

class NeuralEnsembleSystem:
    """Advanced neural network ensemble system"""
    
    def __init__(self, config: NeuralConfig = None):
        self.config = config or NeuralConfig()
        
        # Models
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.scalers = {}
        
        # Training history
        self.training_history = {}
        self.ensemble_weights = {}
        
        # Performance tracking
        self.performance_metrics = {}
        
        logger.info("Neural ensemble system initialized")
    
    def create_models(self, input_size: int, sequence_length: int = 20, 
                     output_size: int = 1, task_type: str = 'regression'):
        """Create all neural network models"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using dummy models")
            return
        
        # LSTM model
        if self.config.use_lstm:
            self.models['lstm'] = LSTMPredictor(
                input_size=input_size,
                hidden_size=self.config.hidden_sizes[0],
                num_layers=2,
                output_size=output_size,
                dropout=self.config.dropout_rate
            )
        
        # Transformer model
        if self.config.use_transformer:
            self.models['transformer'] = TransformerPredictor(
                input_size=input_size,
                d_model=self.config.hidden_sizes[0],
                num_heads=8,
                num_layers=4,
                output_size=output_size,
                dropout=self.config.dropout_rate
            )
        
        # CNN model
        self.models['cnn'] = CNNPredictor(
            input_size=input_size,
            num_filters=self.config.hidden_sizes[0],
            output_size=output_size,
            dropout=self.config.dropout_rate
        )
        
        # Create optimizers and schedulers
        for name, model in self.models.items():
            self.optimizers[name] = optim.Adam(
                model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=1e-5
            )
            
            self.schedulers[name] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[name], 
                mode='min', 
                factor=0.5, 
                patience=5
            )
            
            # Initialize ensemble weights
            self.ensemble_weights[name] = 1.0
        
        logger.info(f"Created {len(self.models)} neural network models")
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    sequence_length: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for neural network training"""
        if not TORCH_AVAILABLE:
            return None, None
        
        # Scale features
        if 'scaler' not in self.scalers:
            if SKLEARN_AVAILABLE:
                self.scalers['scaler'] = RobustScaler()
                X_scaled = self.scalers['scaler'].fit_transform(X)
            else:
                X_scaled = X.values
        else:
            if SKLEARN_AVAILABLE:
                X_scaled = self.scalers['scaler'].transform(X)
            else:
                X_scaled = X.values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-sequence_length:i])
            y_sequences.append(y.iloc[i])
        
        X_tensor = torch.FloatTensor(np.array(X_sequences))
        y_tensor = torch.FloatTensor(np.array(y_sequences))
        
        return X_tensor, y_tensor
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    validation_split: float = 0.2) -> Dict[str, Any]:
        """Train all neural network models"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping training")
            return {}
        
        # Prepare data
        X_tensor, y_tensor = self.prepare_data(X, y)
        if X_tensor is None:
            return {}
        
        # Train-validation split
        if SKLEARN_AVAILABLE:
            X_train, X_val, y_train, y_val = train_test_split(
                X_tensor, y_tensor, test_size=validation_split, random_state=42
            )
        else:
            split_idx = int(len(X_tensor) * (1 - validation_split))
            X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        training_results = {}
        
        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} model...")
            
            optimizer = self.optimizers[model_name]
            scheduler = self.schedulers[model_name]
            
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    outputs = model(batch_X)
                    loss = F.mse_loss(outputs.squeeze(), batch_y)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = F.mse_loss(outputs.squeeze(), batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    torch.save(model.state_dict(), f'best_{model_name}_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping for {model_name} at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"{model_name} - Epoch {epoch}: "
                              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Load best model
            model.load_state_dict(torch.load(f'best_{model_name}_model.pth'))
            
            training_results[model_name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'final_epoch': epoch
            }
            
            self.training_history[model_name] = training_results[model_name]
        
        logger.info("Neural network training completed")
        return training_results
    
    def predict_ensemble(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Make ensemble predictions"""
        if not TORCH_AVAILABLE or not self.models:
            return {'ensemble_prediction': 0.0, 'individual_predictions': {}}
        
        # Prepare data
        X_tensor, _ = self.prepare_data(X, pd.Series([0] * len(X)))
        if X_tensor is None:
            return {'ensemble_prediction': 0.0, 'individual_predictions': {}}
        
        individual_predictions = {}
        weighted_predictions = []
        total_weight = 0
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor[-1:]).squeeze().item()
                individual_predictions[model_name] = predictions
                
                weight = self.ensemble_weights[model_name]
                weighted_predictions.append(predictions * weight)
                total_weight += weight
        
        # Calculate ensemble prediction
        if total_weight > 0:
            ensemble_prediction = sum(weighted_predictions) / total_weight
        else:
            ensemble_prediction = np.mean(list(individual_predictions.values()))
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': individual_predictions,
            'model_weights': dict(self.ensemble_weights)
        }
    
    def update_ensemble_weights(self, performance_metrics: Dict[str, float]):
        """Update ensemble weights based on performance"""
        total_performance = sum(performance_metrics.values())
        
        if total_performance > 0:
            for model_name in self.ensemble_weights:
                if model_name in performance_metrics:
                    new_weight = performance_metrics[model_name] / total_performance
                    # Smooth update
                    self.ensemble_weights[model_name] = (
                        0.7 * self.ensemble_weights[model_name] + 0.3 * new_weight
                    )
        
        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
    
    def save_models(self, directory: str):
        """Save all models and configurations"""
        if not TORCH_AVAILABLE:
            return
        
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = Path(directory) / f"{model_name}_neural_model.pth"
            torch.save(model.state_dict(), model_path)
        
        # Save scalers
        if SKLEARN_AVAILABLE and self.scalers:
            import joblib
            for scaler_name, scaler in self.scalers.items():
                scaler_path = Path(directory) / f"{scaler_name}.pkl"
                joblib.dump(scaler, scaler_path)
        
        # Save configuration and weights
        config_data = {
            'config': self.config.__dict__,
            'ensemble_weights': self.ensemble_weights,
            'training_history': self.training_history
        }
        
        config_path = Path(directory) / "neural_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        logger.info(f"Saved neural models to {directory}")
    
    def get_model_interpretability(self) -> Dict[str, Any]:
        """Get model interpretability information"""
        interpretability = {
            'model_architectures': {},
            'attention_weights': {},
            'feature_importance': {},
            'training_curves': self.training_history
        }
        
        for model_name, model in self.models.items():
            if hasattr(model, 'attention'):
                # Extract attention weights if available
                interpretability['attention_weights'][model_name] = "Available"
            
            # Model architecture info
            if TORCH_AVAILABLE:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                interpretability['model_architectures'][model_name] = {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_type': type(model).__name__
                }
        
        return interpretability
