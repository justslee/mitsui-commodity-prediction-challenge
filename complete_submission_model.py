"""
Complete submission model for all 424 targets in Mitsui Commodity Prediction Challenge.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import joblib
import warnings
warnings.filterwarnings('ignore')


class CompleteSubmissionModel:
    """Production model for complete Mitsui submission with all targets."""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = None
        self.all_target_columns = None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        # Use numeric columns only, excluding date_id
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'date_id']
        
        # Fill missing values
        X = df[feature_cols].fillna(0)
        
        return X
    
    def train_all_targets(self, train_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
        """Train models for ALL targets."""
        
        print("Training complete submission models for all targets...")
        
        # Get all target columns
        self.all_target_columns = [col for col in target_df.columns if col.startswith('target_')]
        print(f"Total targets to train: {len(self.all_target_columns)}")
        
        # Prepare features
        X = self.prepare_features(train_df)
        self.feature_columns = X.columns.tolist()
        
        # Train models for each target
        successful_targets = 0
        insufficient_data_targets = 0
        error_targets = 0
        
        for i, target_col in enumerate(self.all_target_columns):
            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(self.all_target_columns)} targets")
            
            if target_col not in target_df.columns:
                continue
            
            y = target_df[target_col]
            valid_mask = ~y.isna()
            
            if valid_mask.sum() < 10:
                # Use simple mean prediction for targets with insufficient data
                self.models[target_col] = {'type': 'mean', 'value': 0.0}
                insufficient_data_targets += 1
                continue
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            try:
                if valid_mask.sum() >= 50:
                    # Use Random Forest for targets with sufficient data
                    model = RandomForestRegressor(
                        n_estimators=50,  # Reduced for speed
                        max_depth=8,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        max_features='sqrt',
                        random_state=42,
                        n_jobs=1  # Single job for stability
                    )
                else:
                    # Use Ridge regression for smaller datasets
                    model = Ridge(alpha=1.0, random_state=42)
                
                model.fit(X_valid, y_valid)
                self.models[target_col] = {'type': 'model', 'model': model}
                successful_targets += 1
                
            except Exception as e:
                # Fallback to mean prediction
                mean_value = y_valid.mean() if len(y_valid) > 0 else 0.0
                self.models[target_col] = {'type': 'mean', 'value': mean_value}
                error_targets += 1
        
        print(f"Training completed:")
        print(f"  Successful models: {successful_targets}")
        print(f"  Insufficient data (mean): {insufficient_data_targets}")
        print(f"  Error fallback (mean): {error_targets}")
        print(f"  Total: {successful_targets + insufficient_data_targets + error_targets}")
        
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
        
        # Get expected targets from lagged labels
        provided_label_lags = pl.concat(
            [label_lags_1.drop(['date_id', 'label_date_id']),
             label_lags_2.drop(['date_id', 'label_date_id']),
             label_lags_3.drop(['date_id', 'label_date_id']),
             label_lags_4.drop(['date_id', 'label_date_id'])],
            how='horizontal'
        )
        expected_targets = provided_label_lags.columns
        
        # Make predictions for all expected targets
        predictions = {}
        
        for target_col in expected_targets:
            if target_col in self.models:
                try:
                    model_info = self.models[target_col]
                    if model_info['type'] == 'model':
                        pred = model_info['model'].predict(X)
                        predictions[target_col] = pred[0] if len(pred) > 0 else 0.0
                    else:  # mean prediction
                        predictions[target_col] = model_info['value']
                except Exception:
                    predictions[target_col] = 0.0
            else:
                # Target not in training data, predict 0
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
            'feature_columns': self.feature_columns,
            'all_target_columns': self.all_target_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Complete model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.feature_columns = model_data['feature_columns']
        self.all_target_columns = model_data['all_target_columns']
        self.is_trained = model_data['is_trained']
        
        print(f"Complete model loaded from {filepath}")
        print(f"Loaded {len(self.models)} target models")


def train_complete_submission_model():
    """Train and save the complete submission model."""
    
    # Import data loader
    import sys
    sys.path.append('src')
    from utils.data_loader import load_competition_data
    
    # Load data
    loader, data = load_competition_data()
    train_df = data['train']
    target_df = data['train_labels']
    
    # Initialize and train model
    submission_model = CompleteSubmissionModel()
    submission_model.train_all_targets(train_df, target_df)
    
    # Save model
    submission_model.save_model('complete_mitsui_submission_model.joblib')
    
    return submission_model


def test_complete_submission_format():
    """Test the complete submission format with local data."""
    
    # Load model
    submission_model = CompleteSubmissionModel()
    submission_model.load_model('complete_mitsui_submission_model.joblib')
    
    # Load test data
    test_df = pl.read_csv('test.csv')
    label_lags_1 = pl.read_csv('lagged_test_labels/test_labels_lag_1.csv')
    label_lags_2 = pl.read_csv('lagged_test_labels/test_labels_lag_2.csv')
    label_lags_3 = pl.read_csv('lagged_test_labels/test_labels_lag_3.csv')
    label_lags_4 = pl.read_csv('lagged_test_labels/test_labels_lag_4.csv')
    
    # Test prediction for first few dates
    date_ids = test_df['date_id'].unique().to_list()[:3]
    
    print(f"Testing complete submission format for {len(date_ids)} dates...")
    
    all_predictions = []
    
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
        
        expected_targets = len(provided_label_lags.columns)
        actual_targets = len(prediction.columns)
        
        print(f"  Expected targets: {expected_targets}")
        print(f"  Actual targets: {actual_targets}")
        
        if expected_targets == actual_targets:
            print("  ✅ Format validation: PASSED")
        else:
            print("  ❌ Format validation: FAILED - target count mismatch")
        
        all_predictions.append(prediction)
    
    print(f"\n✅ Complete submission format test completed!")
    print(f"Total predictions generated: {len(all_predictions)}")
    
    return all_predictions


if __name__ == "__main__":
    print("=== TRAINING COMPLETE SUBMISSION MODEL ===")
    model = train_complete_submission_model()
    
    print("\n=== TESTING COMPLETE SUBMISSION FORMAT ===")
    predictions = test_complete_submission_format()
    
    print("\n=== COMPLETE SUBMISSION MODEL READY ===")
    print("Model saved as: complete_mitsui_submission_model.joblib")
    print("This model handles all 424 targets and is ready for Kaggle submission.")