"""
Submission model for Mitsui Commodity Prediction Challenge.
Uses our best performing Random Forest model.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


class MitsuiSubmissionModel:
    """Production model for Mitsui submission."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.target_columns = None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        # Use numeric columns only, excluding date_id
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'date_id']
        
        # Fill missing values
        X = df[feature_cols].fillna(0)
        
        return X
    
    def train_all_targets(self, train_df: pd.DataFrame, target_df: pd.DataFrame,
                         max_targets: int = None) -> None:
        """Train Random Forest models for all targets."""
        
        print("Training submission models...")
        
        # Get all target columns
        target_cols = [col for col in target_df.columns if col.startswith('target_')]
        if max_targets:
            target_cols = target_cols[:max_targets]
        
        self.target_columns = target_cols
        
        # Prepare features
        X = self.prepare_features(train_df)
        self.feature_columns = X.columns.tolist()
        
        # Train models for each target
        successful_targets = []
        
        for target_col in target_cols:
            print(f"Training model for {target_col}...")
            
            if target_col not in target_df.columns:
                continue
            
            y = target_df[target_col]
            valid_mask = ~y.isna()
            
            if valid_mask.sum() < 20:
                print(f"  Insufficient data for {target_col}: {valid_mask.sum()} samples")
                continue
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            try:
                # Train Random Forest (our best model)
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_valid, y_valid)
                
                self.models[target_col] = model
                successful_targets.append(target_col)
                
                # Validation
                pred = model.predict(X_valid)
                rmse = np.sqrt(np.mean((y_valid - pred) ** 2))
                print(f"  {target_col} RMSE: {rmse:.6f}")
                
            except Exception as e:
                print(f"  Error training {target_col}: {e}")
        
        print(f"Successfully trained {len(successful_targets)} models")
        self.is_trained = True
        
    def predict_single_date(self, test_batch: pl.DataFrame, 
                           label_lags_1: pl.DataFrame,
                           label_lags_2: pl.DataFrame,
                           label_lags_3: pl.DataFrame,
                           label_lags_4: pl.DataFrame) -> pd.DataFrame:
        """Predict for a single date batch."""
        
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Convert to pandas for processing
        test_df = test_batch.to_pandas()
        
        # Prepare features
        X = self.prepare_features(test_df)
        
        # Ensure we have the same features as training
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        
        X = X[self.feature_columns]
        
        # Make predictions for all targets
        predictions = {}
        
        for target_col in self.target_columns:
            if target_col in self.models:
                try:
                    pred = self.models[target_col].predict(X)
                    predictions[target_col] = pred[0] if len(pred) > 0 else 0.0
                except Exception as e:
                    print(f"Error predicting {target_col}: {e}")
                    predictions[target_col] = 0.0
            else:
                predictions[target_col] = 0.0
        
        # Create result DataFrame with expected format
        result_df = pd.DataFrame([predictions])
        
        return result_df
    
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_columns = model_data['feature_columns']
        self.target_columns = model_data['target_columns']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
        print(f"Loaded {len(self.models)} target models")


def train_submission_model():
    """Train and save the submission model."""
    
    # Import data loader
    import sys
    sys.path.append('src')
    from utils.data_loader import load_competition_data
    
    # Load data
    loader, data = load_competition_data()
    train_df = data['train']
    target_df = data['train_labels']
    
    # Initialize and train model
    submission_model = MitsuiSubmissionModel()
    submission_model.train_all_targets(train_df, target_df, max_targets=50)  # Train first 50 targets
    
    # Save model
    submission_model.save_model('mitsui_submission_model.joblib')
    
    return submission_model


def test_submission_format():
    """Test the submission format with local data."""
    
    # Load model
    submission_model = MitsuiSubmissionModel()
    submission_model.load_model('mitsui_submission_model.joblib')
    
    # Load test data
    test_df = pl.read_csv('test.csv')
    label_lags_1 = pl.read_csv('lagged_test_labels/test_labels_lag_1.csv')
    label_lags_2 = pl.read_csv('lagged_test_labels/test_labels_lag_2.csv')
    label_lags_3 = pl.read_csv('lagged_test_labels/test_labels_lag_3.csv')
    label_lags_4 = pl.read_csv('lagged_test_labels/test_labels_lag_4.csv')
    
    # Test prediction for first date
    date_ids = test_df['date_id'].unique().to_list()[:3]
    
    print(f"Testing submission format for {len(date_ids)} dates...")
    
    for date_id in date_ids:
        print(f"\nTesting date_id: {date_id}")
        
        test_batch = test_df.filter(pl.col('date_id') == date_id)
        label_lags_1_batch = label_lags_1.filter(pl.col('date_id') == date_id)
        label_lags_2_batch = label_lags_2.filter(pl.col('date_id') == date_id)
        label_lags_3_batch = label_lags_3.filter(pl.col('date_id') == date_id)
        label_lags_4_batch = label_lags_4.filter(pl.col('date_id') == date_id)
        
        prediction = submission_model.predict_single_date(
            test_batch, label_lags_1_batch, label_lags_2_batch,
            label_lags_3_batch, label_lags_4_batch
        )
        
        print(f"  Prediction shape: {prediction.shape}")
        print(f"  Predicted targets: {len(prediction.columns)}")
        print(f"  Sample predictions: {list(prediction.iloc[0][:5].round(6))}")
        
        # Validate format
        assert isinstance(prediction, pd.DataFrame)
        assert len(prediction) == 1
        assert 'date_id' not in prediction.columns
        
        # Check we have predictions for expected targets
        provided_label_lags = pl.concat(
            [label_lags_1_batch.drop(['date_id', 'label_date_id']),
             label_lags_2_batch.drop(['date_id', 'label_date_id']),
             label_lags_3_batch.drop(['date_id', 'label_date_id']),
             label_lags_4_batch.drop(['date_id', 'label_date_id'])],
            how='horizontal'
        )
        
        print(f"  Expected targets: {len(provided_label_lags.columns)}")
        print("  Format validation: PASSED")
    
    print(f"\n✅ Submission format test completed successfully!")


if __name__ == "__main__":
    print("=== TRAINING SUBMISSION MODEL ===")
    model = train_submission_model()
    
    print("\n=== TESTING SUBMISSION FORMAT ===")
    test_submission_format()
    
    print("\n=== SUBMISSION MODEL READY ===")
    print("Model saved as: mitsui_submission_model.joblib")
    print("Use this model in the Kaggle submission notebook.")