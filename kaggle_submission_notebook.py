"""
Kaggle Submission for Mitsui Commodity Prediction Challenge
"""

import os
import pandas as pd
import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.mitsui_inference_server

NUM_TARGET_COLUMNS = 424

# Global model storage
model = None
feature_columns = None

def load_and_train_model():
    """Load data and train models - called once during first prediction."""
    global model, feature_columns
    
    print("Loading data and training models...")
    
    # Load training data
    train_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
    target_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')
    
    # Prepare features (use first 100 numeric features)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'date_id'][:100]
    X = train_df[feature_cols].fillna(0)
    feature_columns = X.columns.tolist()
    
    # Get all target columns
    target_columns = [col for col in target_df.columns if col.startswith('target_')]
    
    # Train models for each target
    models = {}
    successful_targets = 0
    
    for target_col in target_columns:
        if target_col not in target_df.columns:
            continue
        
        y = target_df[target_col]
        valid_mask = ~y.isna()
        
        if valid_mask.sum() < 5:
            models[target_col] = {'type': 'mean', 'value': 0.0}
            continue
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        try:
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_valid, y_valid)
            models[target_col] = {'type': 'model', 'model': ridge_model}
            successful_targets += 1
        except Exception:
            mean_value = y_valid.mean() if len(y_valid) > 0 else 0.0
            models[target_col] = {'type': 'mean', 'value': mean_value}
    
    model = models
    print(f"Training completed: {successful_targets} Ridge models")


def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    """Replace this function with your inference code."""
    global model, feature_columns
    
    # Load and train model on first call
    if model is None:
        load_and_train_model()
    
    # Convert to pandas for processing
    test_df = test.to_pandas()
    
    # Prepare features
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns
    available_features = [col for col in numeric_cols if col != 'date_id'][:100]
    X = test_df[available_features].fillna(0)
    
    # Ensure we have the same features as training
    missing_cols = set(feature_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    
    X = X[feature_columns]
    
    # Get expected targets from lagged labels
    provided_label_lags = pl.concat(
        [label_lags_1_batch.drop(['date_id', 'label_date_id']),
         label_lags_2_batch.drop(['date_id', 'label_date_id']),
         label_lags_3_batch.drop(['date_id', 'label_date_id']),
         label_lags_4_batch.drop(['date_id', 'label_date_id'])],
        how='horizontal'
    )
    expected_targets = provided_label_lags.columns
    
    # Make predictions for all expected targets
    predictions = {}
    
    for target_col in expected_targets:
        if target_col in model:
            try:
                model_info = model[target_col]
                if model_info['type'] == 'model':
                    pred = model_info['model'].predict(X)
                    predictions[target_col] = pred[0] if len(pred) > 0 else 0.0
                else:
                    predictions[target_col] = model_info['value']
            except Exception:
                predictions[target_col] = 0.0
        else:
            predictions[target_col] = 0.0
    
    # Create result DataFrame
    result_df = pl.DataFrame([predictions])
    
    assert isinstance(result_df, (pd.DataFrame, pl.DataFrame))
    assert len(result_df) == 1
    return result_df


# When your notebook is run on the hidden test set, inference_server.serve must be called within 15 minutes of the notebook starting
# or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very
# first `predict` call, which does not have the usual 1 minute response deadline.
inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))