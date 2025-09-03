"""
Fast submission model using Ridge regression for all targets.
"""

import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import Ridge
import joblib
import warnings
warnings.filterwarnings('ignore')


class FastSubmissionModel:
    """Fast production model using Ridge regression."""
    
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
        
        # Limit features for speed and take only top features
        feature_cols = feature_cols[:100]  # Use first 100 features
        
        # Fill missing values
        X = df[feature_cols].fillna(0)
        
        return X
    
    def train_all_targets(self, train_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
        """Train Ridge regression models for ALL targets."""
        
        print("Training fast submission models for all targets...")
        
        # Get all target columns
        self.all_target_columns = [col for col in target_df.columns if col.startswith('target_')]
        print(f"Total targets to train: {len(self.all_target_columns)}")
        
        # Prepare features
        X = self.prepare_features(train_df)
        self.feature_columns = X.columns.tolist()
        print(f"Using {len(self.feature_columns)} features")
        
        # Train models for each target
        successful_targets = 0
        mean_predictions = 0
        
        for i, target_col in enumerate(self.all_target_columns):
            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(self.all_target_columns)} targets")
            
            if target_col not in target_df.columns:
                continue
            
            y = target_df[target_col]
            valid_mask = ~y.isna()
            
            if valid_mask.sum() < 5:
                # Use zero prediction for targets with insufficient data
                self.models[target_col] = {'type': 'mean', 'value': 0.0}
                mean_predictions += 1
                continue
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            try:
                # Use Ridge regression for all targets
                model = Ridge(alpha=1.0, random_state=42)
                model.fit(X_valid, y_valid)
                self.models[target_col] = {'type': 'model', 'model': model}
                successful_targets += 1
                
            except Exception as e:
                # Fallback to mean prediction
                mean_value = y_valid.mean() if len(y_valid) > 0 else 0.0
                self.models[target_col] = {'type': 'mean', 'value': mean_value}
                mean_predictions += 1
        
        print(f"Training completed:")
        print(f"  Ridge models: {successful_targets}")
        print(f"  Mean/zero predictions: {mean_predictions}")
        print(f"  Total: {successful_targets + mean_predictions}")
        
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
        print(f"Fast model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.feature_columns = model_data['feature_columns']
        self.all_target_columns = model_data['all_target_columns']
        self.is_trained = model_data['is_trained']
        
        print(f"Fast model loaded from {filepath}")
        print(f"Loaded {len(self.models)} target models")


def train_fast_submission_model():
    """Train and save the fast submission model."""
    
    # Import data loader
    import sys
    sys.path.append('src')
    from utils.data_loader import load_competition_data
    
    # Load data
    loader, data = load_competition_data()
    train_df = data['train']
    target_df = data['train_labels']
    
    # Initialize and train model
    submission_model = FastSubmissionModel()
    submission_model.train_all_targets(train_df, target_df)
    
    # Save model
    submission_model.save_model('fast_mitsui_submission_model.joblib')
    
    return submission_model


def test_fast_submission_format():
    """Test the fast submission format."""
    
    # Load model
    submission_model = FastSubmissionModel()
    submission_model.load_model('fast_mitsui_submission_model.joblib')
    
    # Load test data
    test_df = pl.read_csv('test.csv')
    label_lags_1 = pl.read_csv('lagged_test_labels/test_labels_lag_1.csv')
    label_lags_2 = pl.read_csv('lagged_test_labels/test_labels_lag_2.csv')
    label_lags_3 = pl.read_csv('lagged_test_labels/test_labels_lag_3.csv')
    label_lags_4 = pl.read_csv('lagged_test_labels/test_labels_lag_4.csv')
    
    # Test prediction for first date
    date_ids = test_df['date_id'].unique().to_list()[:1]
    
    print(f"Testing fast submission format for {len(date_ids)} dates...")
    
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
        
        # Check target count
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
        print(f"  ✅ Format validation: {'PASSED' if expected_targets == actual_targets else 'FAILED'}")
    
    print(f"\n✅ Fast submission format test completed!")


if __name__ == "__main__":
    print("=== TRAINING FAST SUBMISSION MODEL ===")
    model = train_fast_submission_model()
    
    print("\n=== TESTING FAST SUBMISSION FORMAT ===")
    test_fast_submission_format()
    
    print("\n=== FAST SUBMISSION MODEL READY ===")
    print("Model saved as: fast_mitsui_submission_model.joblib")
    print("This model handles all targets using Ridge regression and is ready for Kaggle submission.")