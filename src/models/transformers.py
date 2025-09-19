"""
Transformer architectures for time series commodity prediction.
Implements Temporal Fusion Transformer and custom time series transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers library for advanced models
try:
    from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformer
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformerModel(nn.Module):
    """Custom Transformer model for time series prediction."""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 seq_len: int = 50,
                 prediction_horizon: int = 1,
                 num_targets: int = 424):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        self.num_targets = num_targets
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + prediction_horizon)
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection for multi-target prediction
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_targets)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection[-1].weight.data.uniform_(-initrange, initrange)
    
    def create_padding_mask(self, seq_len: int, batch_size: int) -> torch.Tensor:
        """Create padding mask for sequences."""
        return torch.zeros(batch_size, seq_len).bool()
    
    def create_look_ahead_mask(self, size: int) -> torch.Tensor:
        """Create look-ahead mask for decoder."""
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            src: Source sequence [batch_size, seq_len, input_dim]
            tgt: Target sequence for training [batch_size, prediction_horizon, input_dim]
        """
        batch_size, seq_len, _ = src.shape
        
        # Project input to model dimension
        src_embedded = self.input_projection(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoder(src_embedded.transpose(0, 1)).transpose(0, 1)
        
        if tgt is not None:  # Training mode
            tgt_embedded = self.input_projection(tgt) * math.sqrt(self.d_model)
            tgt_embedded = self.pos_encoder(tgt_embedded.transpose(0, 1)).transpose(0, 1)
            
            # Create masks
            tgt_mask = self.create_look_ahead_mask(tgt.size(1)).to(src.device)
            
            # Transformer forward pass
            transformer_out = self.transformer(
                src=src_embedded,
                tgt=tgt_embedded,
                tgt_mask=tgt_mask
            )
        else:  # Inference mode
            # Create dummy target for decoder
            tgt_dummy = torch.zeros(batch_size, self.prediction_horizon, src.size(-1)).to(src.device)
            tgt_embedded = self.input_projection(tgt_dummy) * math.sqrt(self.d_model)
            tgt_embedded = self.pos_encoder(tgt_embedded.transpose(0, 1)).transpose(0, 1)
            
            transformer_out = self.transformer(
                src=src_embedded,
                tgt=tgt_embedded
            )
        
        # Project to target space
        output = self.output_projection(transformer_out)
        
        # Return predictions for all targets
        return output.mean(dim=1)  # Average over sequence dimension


class CommodityTransformerModel(nn.Module):
    """Specialized Transformer for commodity prediction with attention across assets."""
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 seq_len: int = 50,
                 num_targets: int = 424,
                 asset_groups: Optional[Dict[str, List[int]]] = None):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_targets = num_targets
        self.asset_groups = asset_groups or {}
        
        # Feature embedding
        self.feature_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_layers)
        ])
        
        # Cross-asset attention (learn relationships between different assets)
        self.cross_asset_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Target-specific heads
        self.target_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            ) for _ in range(num_targets)
        ])
        
        # Global prediction head (fallback)
        self.global_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_targets)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Feature embedding and positional encoding
        x = self.feature_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Self-attention layers
        for i, (attn, norm, ff) in enumerate(zip(
            self.attention_layers, self.norm_layers, self.feedforward_layers
        )):
            # Multi-head attention
            attn_out, _ = attn(x, x, x)
            x = norm(x + attn_out)
            
            # Feed forward
            ff_out = ff(x)
            x = norm(x + ff_out)
        
        # Cross-asset attention (attend to different time steps)
        cross_attn_out, attention_weights = self.cross_asset_attention(x, x, x)
        x = x + cross_attn_out
        
        # Global average pooling over sequence
        pooled = x.mean(dim=1)  # [batch_size, d_model]
        
        # Target-specific predictions
        target_predictions = []
        for target_head in self.target_heads:
            target_pred = target_head(pooled)  # [batch_size, 1]
            target_predictions.append(target_pred)
        
        # Combine predictions
        predictions = torch.cat(target_predictions, dim=1)  # [batch_size, num_targets]
        
        return predictions


class CommodityDataset(Dataset):
    """Dataset for commodity time series data."""
    
    def __init__(self, 
                 features: pd.DataFrame,
                 targets: Optional[pd.DataFrame] = None,
                 seq_len: int = 50,
                 prediction_horizon: int = 1):
        self.features = features.fillna(0).values.astype(np.float32)
        self.targets = targets.fillna(0).values.astype(np.float32) if targets is not None else None
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        
        # Create sequences
        self.sequences = []
        self.target_sequences = []
        
        for i in range(seq_len, len(self.features) - prediction_horizon + 1):
            # Feature sequence
            seq = self.features[i-seq_len:i]
            self.sequences.append(seq)
            
            # Target sequence (if available)
            if self.targets is not None:
                target_seq = self.targets[i:i+prediction_horizon]
                self.target_sequences.append(target_seq)
        
        self.sequences = np.array(self.sequences)
        if self.target_sequences:
            self.target_sequences = np.array(self.target_sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])
        
        if self.target_sequences is not None and len(self.target_sequences) > idx:
            target = torch.from_numpy(self.target_sequences[idx])
            return seq, target
        else:
            return seq


class TransformerTrainer:
    """Trainer for transformer models."""
    
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            if len(batch) == 2:
                sequences, targets = batch
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Calculate loss (average over prediction horizon)
                if targets.dim() == 3:
                    targets = targets.mean(dim=1)  # Average over prediction horizon
                
                loss = self.criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    sequences, targets = batch
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    
                    predictions = self.model(sequences)
                    
                    if targets.dim() == 3:
                        targets = targets.mean(dim=1)
                    
                    loss = self.criterion(predictions, targets)
                    total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, list) and len(batch) == 1:
                    sequences = batch[0]
                else:
                    sequences = batch
                
                sequences = sequences.to(self.device)
                pred = self.model(sequences)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)


def create_transformer_model(input_dim: int,
                           num_targets: int = 424,
                           model_type: str = 'commodity',
                           **kwargs) -> nn.Module:
    """Create a transformer model for commodity prediction."""
    
    default_params = {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'seq_len': 50
    }
    
    # Update with user parameters
    params = {**default_params, **kwargs}
    
    if model_type == 'commodity':
        model = CommodityTransformerModel(
            input_dim=input_dim,
            num_targets=num_targets,
            **params
        )
    elif model_type == 'vanilla':
        model = TimeSeriesTransformerModel(
            input_dim=input_dim,
            num_targets=num_targets,
            **params
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_transformer_model(train_features: pd.DataFrame,
                          train_targets: pd.DataFrame,
                          val_features: pd.DataFrame = None,
                          val_targets: pd.DataFrame = None,
                          model_params: Dict = None,
                          training_params: Dict = None) -> Tuple[nn.Module, Dict]:
    """Train a transformer model on commodity data."""
    
    # Default parameters
    model_params = model_params or {}
    training_params = training_params or {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'early_stopping_patience': 15,
        'device': 'cpu'
    }
    
    device = training_params['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create datasets
    train_dataset = CommodityDataset(
        train_features, train_targets,
        seq_len=model_params.get('seq_len', 50)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params['batch_size'],
        shuffle=True
    )
    
    val_loader = None
    if val_features is not None and val_targets is not None:
        val_dataset = CommodityDataset(
            val_features, val_targets,
            seq_len=model_params.get('seq_len', 50)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_params['batch_size'],
            shuffle=False
        )
    
    # Create model
    input_dim = train_features.shape[1]
    num_targets = train_targets.shape[1]
    
    model = create_transformer_model(
        input_dim=input_dim,
        num_targets=num_targets,
        **model_params
    )
    
    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        learning_rate=training_params['learning_rate'],
        device=device
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(training_params['epochs']):
        # Train
        train_loss = trainer.train_epoch(train_loader)
        training_history['train_loss'].append(train_loss)
        
        # Validate
        if val_loader is not None:
            val_loss = trainer.validate(val_loader)
            training_history['val_loss'].append(val_loss)
            
            # Training progress
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= training_params['early_stopping_patience']:
                model.load_state_dict(best_model_state)
                break
                
            # Learning rate scheduling
            trainer.scheduler.step(val_loss)
    
    return model, training_history


if __name__ == "__main__":
    
    # Create sample data
    seq_len = 50
    input_dim = 100
    num_targets = 10
    batch_size = 16
    
    # Sample features and targets
    features = torch.randn(200, input_dim)
    targets = torch.randn(200, num_targets)
    
    # Create model
    model = create_transformer_model(
        input_dim=input_dim,
        num_targets=num_targets,
        model_type='commodity',
        seq_len=seq_len
    )
    
    # Model created
    
    # Test forward pass
    sample_input = torch.randn(batch_size, seq_len, input_dim)
    with torch.no_grad():
        output = model(sample_input)
    
    # Test completed