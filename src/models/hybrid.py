"""
Hybrid models combining neural networks and tree-based models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

tf.get_logger().setLevel('ERROR')


class LSTMXGBoostHybrid:
    """Hybrid model combining LSTM and XGBoost."""
    
    def __init__(self, lookback: int = 15, lstm_units: List[int] = [32, 16],
                 xgb_params: Dict[str, Any] = None):
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.xgb_params = xgb_params or {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'verbosity': 0
        }
        
        self.lstm_model = None
        self.xgb_model = None
        self.meta_model = None
        self.feature_scaler = None
        self.target_scaler = None
        
    def _create_lstm_model(self, n_features: int) -> keras.Model:
        """Create LSTM component."""
        model = models.Sequential([
            layers.LSTM(self.lstm_units[0], return_sequences=True, 
                       input_shape=(self.lookback, n_features)),
            layers.Dropout(0.2),
            layers.LSTM(self.lstm_units[1], return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series = None,
                         fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        if fit_scaler:
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            if y is not None:
                self.target_scaler = StandardScaler()
                y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        else:
            X_scaled = self.feature_scaler.transform(X)
            if y is not None:
                y_scaled = self.target_scaler.transform(y.values.reshape(-1, 1)).flatten()
        
        X_sequences = []
        y_sequences = [] if y is not None else None
        
        for i in range(self.lookback, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.lookback:i])
            if y is not None:
                y_sequences.append(y_scaled[i])
        
        return np.array(X_sequences), np.array(y_sequences) if y is not None else None
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            validation_split: float = 0.2, epochs: int = 30) -> 'LSTMXGBoostHybrid':
        """Fit hybrid model."""
        
        valid_mask = ~y.isna()
        if valid_mask.sum() < self.lookback + 20:
            raise ValueError("Insufficient data for hybrid model")
        
        X_valid = X[valid_mask].reset_index(drop=True)
        y_valid = y[valid_mask].reset_index(drop=True)
        
        # Train LSTM component
        X_seq, y_seq = self._create_sequences(X_valid, y_valid)
        n_features = X_seq.shape[2]
        
        self.lstm_model = self._create_lstm_model(n_features)
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        self.lstm_model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Get LSTM predictions for XGBoost training
        lstm_train_pred = self.lstm_model.predict(X_seq, verbose=0)
        lstm_train_pred = self.target_scaler.inverse_transform(lstm_train_pred).flatten()
        
        # Prepare data for XGBoost (aligned with LSTM predictions)
        X_xgb = X_valid.iloc[self.lookback:].reset_index(drop=True)
        y_xgb = y_valid.iloc[self.lookback:].reset_index(drop=True)
        
        # Add LSTM predictions as features for XGBoost
        X_xgb_enhanced = X_xgb.copy()
        X_xgb_enhanced['lstm_pred'] = lstm_train_pred
        
        # Train XGBoost component
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(X_xgb_enhanced, y_xgb)
        
        # Train meta-model (simple weighted average)
        xgb_pred = self.xgb_model.predict(X_xgb_enhanced)
        
        # Find optimal weights
        best_weight = 0.5
        best_rmse = np.inf
        
        for weight in np.linspace(0.1, 0.9, 9):
            ensemble_pred = weight * lstm_train_pred + (1 - weight) * xgb_pred
            rmse = np.sqrt(mean_squared_error(y_xgb, ensemble_pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weight = weight
        
        self.meta_weight = best_weight
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        if self.lstm_model is None or self.xgb_model is None:
            raise ValueError("Model not fitted")
        
        X_seq, _ = self._create_sequences(X, fit_scaler=False)
        
        if len(X_seq) == 0:
            return np.array([])
        
        # LSTM predictions
        lstm_pred_scaled = self.lstm_model.predict(X_seq, verbose=0)
        lstm_pred = self.target_scaler.inverse_transform(lstm_pred_scaled).flatten()
        
        # XGBoost predictions
        X_xgb = X.iloc[self.lookback:].reset_index(drop=True)
        X_xgb_enhanced = X_xgb.copy()
        X_xgb_enhanced['lstm_pred'] = lstm_pred
        
        xgb_pred = self.xgb_model.predict(X_xgb_enhanced)
        
        # Ensemble prediction
        ensemble_pred = self.meta_weight * lstm_pred + (1 - self.meta_weight) * xgb_pred
        
        return ensemble_pred


class TemporalConvolutionalNetwork:
    """Temporal Convolutional Network implementation."""
    
    def __init__(self, filters: List[int] = [32, 64, 32], 
                 kernel_size: int = 3, dropout: float = 0.2,
                 dilations: List[int] = [1, 2, 4]):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilations = dilations
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
    
    def _create_tcn_block(self, x, filters: int, kernel_size: int, 
                         dilation_rate: int, dropout: float):
        """Create TCN residual block."""
        # Dilated causal convolution
        conv1 = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                             padding='causal', activation='relu')(x)
        conv1 = layers.Dropout(dropout)(conv1)
        
        conv2 = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                             padding='causal', activation='relu')(conv1)
        conv2 = layers.Dropout(dropout)(conv2)
        
        # Residual connection
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding='same')(x)
        
        output = layers.Add()([x, conv2])
        return layers.Activation('relu')(output)
    
    def build_model(self, sequence_length: int, n_features: int) -> keras.Model:
        """Build TCN model."""
        inputs = layers.Input(shape=(sequence_length, n_features))
        x = inputs
        
        for i, (filters, dilation) in enumerate(zip(self.filters, self.dilations)):
            x = self._create_tcn_block(x, filters, self.kernel_size, dilation, self.dropout)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            lookback: int = 20, validation_split: float = 0.2,
            epochs: int = 30) -> 'TemporalConvolutionalNetwork':
        """Fit TCN model."""
        
        valid_mask = ~y.isna()
        if valid_mask.sum() < lookback + 20:
            raise ValueError("Insufficient data for TCN")
        
        X_valid = X[valid_mask].reset_index(drop=True)
        y_valid = y[valid_mask].reset_index(drop=True)
        
        # Scale features and target
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X_valid)
        
        self.target_scaler = StandardScaler()
        y_scaled = self.target_scaler.fit_transform(y_valid.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(lookback, len(X_scaled)):
            X_sequences.append(X_scaled[i-lookback:i])
            y_sequences.append(y_scaled[i])
        
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences)
        
        # Build and train model
        self.model = self.build_model(lookback, X_seq.shape[2])
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self
    
    def predict(self, X: pd.DataFrame, lookback: int = 20) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        X_scaled = self.feature_scaler.transform(X)
        
        X_sequences = []
        for i in range(lookback, len(X_scaled)):
            X_sequences.append(X_scaled[i-lookback:i])
        
        if len(X_sequences) == 0:
            return np.array([])
        
        X_seq = np.array(X_sequences)
        predictions_scaled = self.model.predict(X_seq, verbose=0)
        predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()
        
        return predictions


class AttentionLSTM:
    """LSTM with attention mechanism."""
    
    def __init__(self, lookback: int = 20, lstm_units: int = 64,
                 attention_units: int = 32):
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
    
    def _attention_layer(self, lstm_output):
        """Create attention mechanism."""
        # lstm_output shape: (batch_size, timesteps, features)
        
        # Attention weights
        attention = layers.Dense(self.attention_units, activation='tanh')(lstm_output)
        attention = layers.Dense(1, activation='softmax')(attention)
        
        # Apply attention weights
        context = layers.Multiply()([lstm_output, attention])
        context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
        
        return context
    
    def build_model(self, n_features: int) -> keras.Model:
        """Build LSTM model with attention."""
        inputs = layers.Input(shape=(self.lookback, n_features))
        
        lstm_out = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
        lstm_out = layers.Dropout(0.2)(lstm_out)
        
        # Apply attention
        context = self._attention_layer(lstm_out)
        
        # Final layers
        x = layers.Dense(32, activation='relu')(context)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            validation_split: float = 0.2, epochs: int = 30) -> 'AttentionLSTM':
        """Fit attention LSTM model."""
        
        valid_mask = ~y.isna()
        if valid_mask.sum() < self.lookback + 20:
            raise ValueError("Insufficient data for attention LSTM")
        
        X_valid = X[valid_mask].reset_index(drop=True)
        y_valid = y[valid_mask].reset_index(drop=True)
        
        # Scale data
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X_valid)
        
        self.target_scaler = StandardScaler()
        y_scaled = self.target_scaler.fit_transform(y_valid.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.lookback, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.lookback:i])
            y_sequences.append(y_scaled[i])
        
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences)
        
        # Build and train model
        self.model = self.build_model(X_seq.shape[2])
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        X_scaled = self.feature_scaler.transform(X)
        
        X_sequences = []
        for i in range(self.lookback, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.lookback:i])
        
        if len(X_sequences) == 0:
            return np.array([])
        
        X_seq = np.array(X_sequences)
        predictions_scaled = self.model.predict(X_seq, verbose=0)
        predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()
        
        return predictions


class HybridEnsemble:
    """Ensemble of hybrid models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def fit_all_models(self, X: pd.DataFrame, target_df: pd.DataFrame,
                      target_cols: List[str] = None,
                      max_targets: int = 3,
                      epochs: int = 20) -> Dict[str, Any]:
        """Train all hybrid models."""
        
        if target_cols is None:
            target_cols = [col for col in target_df.columns 
                          if col.startswith('target_')][:max_targets]
        
        print(f"Training hybrid models on {len(target_cols)} targets...")
        
        # Reduce feature set for hybrid models
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'date_id'][:25]
        X_features = X[feature_cols].fillna(0)
        
        models_to_train = {
            'lstm_xgboost': None,  # Will be instantiated per target
            'tcn': TemporalConvolutionalNetwork(),
            'attention_lstm': AttentionLSTM()
        }
        
        results = {}
        
        for model_name in models_to_train.keys():
            print(f"\n=== Training {model_name.upper()} ===")
            model_results = {}
            
            for target_col in target_cols:
                print(f"Training {model_name} for {target_col}...")
                
                if target_col not in target_df.columns:
                    continue
                
                y = target_df[target_col]
                valid_mask = ~y.isna()
                
                if valid_mask.sum() < 50:
                    model_results[target_col] = {
                        'status': 'insufficient_data',
                        'n_valid': valid_mask.sum()
                    }
                    continue
                
                try:
                    X_valid = X_features[valid_mask].reset_index(drop=True)
                    y_valid = y[valid_mask].reset_index(drop=True)
                    
                    n_train = int(0.8 * len(X_valid))
                    X_train, X_val = X_valid[:n_train], X_valid[n_train:]
                    y_train, y_val = y_valid[:n_train], y_valid[n_train:]
                    
                    if model_name == 'lstm_xgboost':
                        model = LSTMXGBoostHybrid()
                    elif model_name == 'tcn':
                        model = TemporalConvolutionalNetwork()
                    elif model_name == 'attention_lstm':
                        model = AttentionLSTM()
                    
                    model.fit(X_train, y_train, epochs=epochs)
                    
                    # Evaluate
                    if len(X_val) > 20:
                        val_pred = model.predict(X_val)
                        
                        if len(val_pred) > 0:
                            if model_name == 'lstm_xgboost':
                                val_actual = y_val.iloc[model.lookback:].values
                            else:
                                val_actual = y_val.iloc[20:].values
                            
                            if len(val_actual) == len(val_pred):
                                val_rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
                                val_mae = mean_absolute_error(val_actual, val_pred)
                            else:
                                val_rmse, val_mae = np.inf, np.inf
                        else:
                            val_rmse, val_mae = np.inf, np.inf
                    else:
                        val_rmse, val_mae = np.inf, np.inf
                    
                    model_results[target_col] = {
                        'status': 'success',
                        'model': model,
                        'val_rmse': val_rmse,
                        'val_mae': val_mae,
                        'n_samples': len(X_valid)
                    }
                    
                except Exception as e:
                    model_results[target_col] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            results[model_name] = model_results
        
        self.results = results
        
        summary = self._create_summary(target_cols)
        best_models = self._get_best_models(target_cols)
        
        return {
            'model_results': results,
            'summary': summary,
            'best_models': best_models
        }
    
    def _create_summary(self, target_cols: List[str]) -> Dict[str, Any]:
        """Create summary statistics."""
        summary = {}
        
        for model_name, model_results in self.results.items():
            successful_results = [r for r in model_results.values() 
                                if r.get('status') == 'success']
            
            if successful_results:
                rmse_values = [r['val_rmse'] for r in successful_results 
                             if r['val_rmse'] != np.inf]
                
                if rmse_values:
                    summary[model_name] = {
                        'n_successful': len(successful_results),
                        'avg_rmse': np.mean(rmse_values),
                        'best_rmse': np.min(rmse_values),
                        'std_rmse': np.std(rmse_values)
                    }
        
        return summary
    
    def _get_best_models(self, target_cols: List[str]) -> Dict[str, str]:
        """Get best model for each target."""
        best_models = {}
        
        for target_col in target_cols:
            best_rmse = np.inf
            best_model = None
            
            for model_name, model_results in self.results.items():
                if (target_col in model_results and 
                    model_results[target_col].get('status') == 'success'):
                    
                    val_rmse = model_results[target_col]['val_rmse']
                    if val_rmse < best_rmse:
                        best_rmse = val_rmse
                        best_model = model_name
            
            if best_model:
                best_models[target_col] = best_model
        
        return best_models


def run_hybrid_evaluation(train_df: pd.DataFrame, target_df: pd.DataFrame,
                         max_targets: int = 2, epochs: int = 15) -> Dict[str, Any]:
    """Run comprehensive hybrid model evaluation."""
    
    ensemble = HybridEnsemble()
    results = ensemble.fit_all_models(train_df, target_df, 
                                     max_targets=max_targets, epochs=epochs)
    
    print(f"\n=== HYBRID MODEL RESULTS ===")
    for model_name, metrics in results['summary'].items():
        print(f"{model_name.upper()}:")
        print(f"  Successful targets: {metrics['n_successful']}/{max_targets}")
        print(f"  Average RMSE: {metrics['avg_rmse']:.6f}")
        print(f"  Best RMSE: {metrics['best_rmse']:.6f}")
    
    print(f"\nBest hybrid models by target:")
    for target, best_model in results['best_models'].items():
        print(f"  {target}: {best_model}")
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import load_competition_data
    
    loader, data = load_competition_data()
    train_df = data['train']
    target_df = data['train_labels']
    
    print("Running hybrid model evaluation...")
    results = run_hybrid_evaluation(train_df, target_df, 
                                   max_targets=2, epochs=10)
    
    print(f"\nHybrid model evaluation completed!")