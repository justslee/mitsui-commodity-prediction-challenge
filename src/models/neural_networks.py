"""
Neural network models for commodity prediction using TensorFlow/Keras.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Generator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

tf.get_logger().setLevel('ERROR')


class TimeSeriesDataGenerator:
    """Data generator for time series neural networks."""
    
    def __init__(self, lookback: int = 20, batch_size: int = 32, 
                 scaler_type: str = 'standard'):
        self.lookback = lookback
        self.batch_size = batch_size
        self.scaler_type = scaler_type
        self.scalers = {}
    
    def create_sequences(self, X: pd.DataFrame, y: pd.Series = None,
                        fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling."""
        
        if fit_scaler:
            if self.scaler_type == 'minmax':
                self.scalers['features'] = MinMaxScaler()
                X_scaled = self.scalers['features'].fit_transform(X)
            else:
                self.scalers['features'] = StandardScaler()
                X_scaled = self.scalers['features'].fit_transform(X)
            
            if y is not None:
                self.scalers['target'] = StandardScaler()
                y_scaled = self.scalers['target'].fit_transform(y.values.reshape(-1, 1)).flatten()
        else:
            X_scaled = self.scalers['features'].transform(X)
            if y is not None:
                y_scaled = self.scalers['target'].transform(y.values.reshape(-1, 1)).flatten()
        
        X_sequences = []
        y_sequences = [] if y is not None else None
        
        for i in range(self.lookback, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.lookback:i])
            if y is not None:
                y_sequences.append(y_scaled[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences) if y is not None else None
        
        return X_sequences, y_sequences
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform predictions to original scale."""
        if 'target' in self.scalers:
            return self.scalers['target'].inverse_transform(predictions.reshape(-1, 1)).flatten()
        return predictions


class LSTMModel:
    """LSTM model for time series prediction."""
    
    def __init__(self, lookback: int = 20, lstm_units: List[int] = [50, 25],
                 dropout: float = 0.2, dense_units: int = 25,
                 learning_rate: float = 0.001):
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.model = None
        self.data_generator = None
        self.history = None
    
    def build_model(self, n_features: int) -> keras.Model:
        """Build LSTM model architecture."""
        
        model = models.Sequential()
        
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            if i == 0:
                model.add(layers.LSTM(units, return_sequences=return_sequences,
                                    input_shape=(self.lookback, n_features)))
            else:
                model.add(layers.LSTM(units, return_sequences=return_sequences))
            model.add(layers.Dropout(self.dropout))
        
        if self.dense_units > 0:
            model.add(layers.Dense(self.dense_units, activation='relu'))
            model.add(layers.Dropout(self.dropout))
        
        model.add(layers.Dense(1, activation='linear'))
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            validation_split: float = 0.2, epochs: int = 50,
            patience: int = 10) -> 'LSTMModel':
        """Fit LSTM model."""
        
        valid_mask = ~y.isna()
        if valid_mask.sum() < self.lookback + 10:
            raise ValueError("Insufficient valid data for LSTM training")
        
        X_valid = X[valid_mask].reset_index(drop=True)
        y_valid = y[valid_mask].reset_index(drop=True)
        
        self.data_generator = TimeSeriesDataGenerator(lookback=self.lookback)
        X_seq, y_seq = self.data_generator.create_sequences(X_valid, y_valid)
        
        if len(X_seq) < 10:
            raise ValueError("Insufficient sequences for LSTM training")
        
        n_features = X_seq.shape[2]
        self.model = self.build_model(n_features)
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-7
        )
        
        self.history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None or self.data_generator is None:
            raise ValueError("Model not fitted")
        
        X_seq, _ = self.data_generator.create_sequences(X, fit_scaler=False)
        
        if len(X_seq) == 0:
            return np.array([])
        
        predictions_scaled = self.model.predict(X_seq, verbose=0)
        predictions = self.data_generator.inverse_transform_predictions(predictions_scaled)
        
        return predictions


class BidirectionalLSTMModel(LSTMModel):
    """Bidirectional LSTM model."""
    
    def build_model(self, n_features: int) -> keras.Model:
        """Build Bidirectional LSTM model architecture."""
        
        model = models.Sequential()
        
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            if i == 0:
                model.add(layers.Bidirectional(
                    layers.LSTM(units, return_sequences=return_sequences),
                    input_shape=(self.lookback, n_features)
                ))
            else:
                model.add(layers.Bidirectional(
                    layers.LSTM(units, return_sequences=return_sequences)
                ))
            model.add(layers.Dropout(self.dropout))
        
        if self.dense_units > 0:
            model.add(layers.Dense(self.dense_units, activation='relu'))
            model.add(layers.Dropout(self.dropout))
        
        model.add(layers.Dense(1, activation='linear'))
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model


class GRUModel:
    """GRU model for time series prediction."""
    
    def __init__(self, lookback: int = 20, gru_units: List[int] = [50, 25],
                 dropout: float = 0.2, dense_units: int = 25,
                 learning_rate: float = 0.001):
        self.lookback = lookback
        self.gru_units = gru_units
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.model = None
        self.data_generator = None
        self.history = None
    
    def build_model(self, n_features: int) -> keras.Model:
        """Build GRU model architecture."""
        
        model = models.Sequential()
        
        for i, units in enumerate(self.gru_units):
            return_sequences = i < len(self.gru_units) - 1
            if i == 0:
                model.add(layers.GRU(units, return_sequences=return_sequences,
                                   input_shape=(self.lookback, n_features)))
            else:
                model.add(layers.GRU(units, return_sequences=return_sequences))
            model.add(layers.Dropout(self.dropout))
        
        if self.dense_units > 0:
            model.add(layers.Dense(self.dense_units, activation='relu'))
            model.add(layers.Dropout(self.dropout))
        
        model.add(layers.Dense(1, activation='linear'))
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            validation_split: float = 0.2, epochs: int = 50,
            patience: int = 10) -> 'GRUModel':
        """Fit GRU model."""
        
        valid_mask = ~y.isna()
        if valid_mask.sum() < self.lookback + 10:
            raise ValueError("Insufficient valid data for GRU training")
        
        X_valid = X[valid_mask].reset_index(drop=True)
        y_valid = y[valid_mask].reset_index(drop=True)
        
        self.data_generator = TimeSeriesDataGenerator(lookback=self.lookback)
        X_seq, y_seq = self.data_generator.create_sequences(X_valid, y_valid)
        
        if len(X_seq) < 10:
            raise ValueError("Insufficient sequences for GRU training")
        
        n_features = X_seq.shape[2]
        self.model = self.build_model(n_features)
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-7
        )
        
        self.history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None or self.data_generator is None:
            raise ValueError("Model not fitted")
        
        X_seq, _ = self.data_generator.create_sequences(X, fit_scaler=False)
        
        if len(X_seq) == 0:
            return np.array([])
        
        predictions_scaled = self.model.predict(X_seq, verbose=0)
        predictions = self.data_generator.inverse_transform_predictions(predictions_scaled)
        
        return predictions


class NeuralNetworkEnsemble:
    """Ensemble of neural network models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def fit_all_models(self, X: pd.DataFrame, target_df: pd.DataFrame,
                      target_cols: List[str] = None,
                      max_targets: int = 3,
                      epochs: int = 30) -> Dict[str, Any]:
        """Train all neural network models."""
        
        if target_cols is None:
            target_cols = [col for col in target_df.columns 
                          if col.startswith('target_')][:max_targets]
        
        print(f"Training neural networks on {len(target_cols)} targets...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'date_id'][:30]
        X_features = X[feature_cols].fillna(0)
        
        models_to_train = {
            'lstm': LSTMModel(lookback=15, lstm_units=[32, 16]),
            'bidirectional_lstm': BidirectionalLSTMModel(lookback=15, lstm_units=[32, 16]),
            'gru': GRUModel(lookback=15, gru_units=[32, 16])
        }
        
        results = {}
        
        for model_name, model_class in models_to_train.items():
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
                    
                    if model_name == 'lstm':
                        model = LSTMModel(lookback=15, lstm_units=[32, 16])
                    elif model_name == 'bidirectional_lstm':
                        model = BidirectionalLSTMModel(lookback=15, lstm_units=[32, 16])
                    elif model_name == 'gru':
                        model = GRUModel(lookback=15, gru_units=[32, 16])
                    
                    model.fit(X_train, y_train, epochs=epochs, patience=5)
                    
                    if len(X_val) > model.lookback:
                        val_pred = model.predict(X_val)
                        
                        val_actual = y_val.iloc[model.lookback:].values
                        val_rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
                        val_mae = mean_absolute_error(val_actual, val_pred)
                    else:
                        val_rmse = np.inf
                        val_mae = np.inf
                    
                    train_pred = model.predict(X_train)
                    train_actual = y_train.iloc[model.lookback:].values
                    train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
                    
                    model_results[target_col] = {
                        'status': 'success',
                        'model': model,
                        'train_rmse': train_rmse,
                        'val_rmse': val_rmse,
                        'val_mae': val_mae,
                        'n_samples': len(X_valid),
                        'epochs_trained': len(model.history.history['loss']) if model.history else epochs
                    }
                    
                except Exception as e:
                    model_results[target_col] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            results[model_name] = model_results
        
        self.models = models_to_train
        self.results = results
        
        summary = self._create_summary(target_cols)
        best_models = self._get_best_models(target_cols)
        
        return {
            'model_results': results,
            'summary': summary,
            'best_models': best_models
        }
    
    def _create_summary(self, target_cols: List[str]) -> Dict[str, Any]:
        """Create summary statistics across models and targets."""
        
        summary = {}
        
        for model_name, model_results in self.results.items():
            successful_results = [r for r in model_results.values() 
                                if r.get('status') == 'success']
            
            if successful_results:
                val_rmse_values = [r['val_rmse'] for r in successful_results 
                                 if r['val_rmse'] != np.inf]
                
                if val_rmse_values:
                    summary[model_name] = {
                        'n_successful': len(successful_results),
                        'avg_val_rmse': np.mean(val_rmse_values),
                        'median_val_rmse': np.median(val_rmse_values),
                        'std_val_rmse': np.std(val_rmse_values),
                        'best_val_rmse': np.min(val_rmse_values)
                    }
        
        return summary
    
    def _get_best_models(self, target_cols: List[str]) -> Dict[str, str]:
        """Identify best neural network model for each target."""
        
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


def run_neural_network_evaluation(train_df: pd.DataFrame, target_df: pd.DataFrame,
                                 max_targets: int = 3, epochs: int = 20) -> Dict[str, Any]:
    """Run comprehensive neural network evaluation."""
    
    ensemble = NeuralNetworkEnsemble()
    results = ensemble.fit_all_models(train_df, target_df, 
                                     max_targets=max_targets, epochs=epochs)
    
    print(f"\n=== NEURAL NETWORK RESULTS ===")
    for model_name, metrics in results['summary'].items():
        print(f"{model_name.upper()}:")
        print(f"  Successful targets: {metrics['n_successful']}/{max_targets}")
        print(f"  Average val RMSE: {metrics['avg_val_rmse']:.6f}")
        print(f"  Best val RMSE: {metrics['best_val_rmse']:.6f}")
    
    print(f"\nBest neural network models by target:")
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
    
    print("Running neural network evaluation...")
    results = run_neural_network_evaluation(train_df, target_df, 
                                           max_targets=2, epochs=15)
    
    print(f"\nNeural network evaluation completed!")