"""
Generate submission.parquet file for Kaggle submission.
"""

import pandas as pd
import polars as pl
import numpy as np
import joblib
from pathlib import Path


def generate_kaggle_submission():
    """Generate submission.parquet file with predictions for all test data."""
    
    print("Loading trained model...")
    
    # Load our trained model
    try:
        from fast_submission_model import FastSubmissionModel
        model = FastSubmissionModel()
        model.load_model('fast_mitsui_submission_model.joblib')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Loading test data...")
    
    # Load test data
    test_df = pl.read_csv('test.csv')
    label_lags_1 = pl.read_csv('lagged_test_labels/test_labels_lag_1.csv')
    label_lags_2 = pl.read_csv('lagged_test_labels/test_labels_lag_2.csv')
    label_lags_3 = pl.read_csv('lagged_test_labels/test_labels_lag_3.csv')
    label_lags_4 = pl.read_csv('lagged_test_labels/test_labels_lag_4.csv')
    
    # Get all unique date_ids
    date_ids = test_df['date_id'].unique().to_list()
    print(f"Generating predictions for {len(date_ids)} dates...")
    
    all_predictions = []
    
    for i, date_id in enumerate(date_ids):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(date_ids)} dates")
        
        # Filter data for current date
        test_batch = test_df.filter(pl.col('date_id') == date_id)
        label_lags_1_batch = label_lags_1.filter(pl.col('date_id') == date_id)
        label_lags_2_batch = label_lags_2.filter(pl.col('date_id') == date_id)
        label_lags_3_batch = label_lags_3.filter(pl.col('date_id') == date_id)
        label_lags_4_batch = label_lags_4.filter(pl.col('date_id') == date_id)
        
        # Generate predictions for this date
        try:
            prediction = model.predict_single_date(
                test_batch, label_lags_1_batch, label_lags_2_batch,
                label_lags_3_batch, label_lags_4_batch
            )
            
            # Add date_id to predictions
            prediction['date_id'] = date_id
            all_predictions.append(prediction)
            
        except Exception as e:
            print(f"Error predicting for date_id {date_id}: {e}")
            # Create empty prediction with zeros
            provided_label_lags = pl.concat(
                [label_lags_1_batch.drop(['date_id', 'label_date_id']),
                 label_lags_2_batch.drop(['date_id', 'label_date_id']),
                 label_lags_3_batch.drop(['date_id', 'label_date_id']),
                 label_lags_4_batch.drop(['date_id', 'label_date_id'])],
                how='horizontal'
            )
            expected_targets = provided_label_lags.columns
            
            predictions = {target: 0.0 for target in expected_targets}
            predictions['date_id'] = date_id
            
            prediction_df = pd.DataFrame([predictions])
            all_predictions.append(prediction_df)
    
    print("Combining all predictions...")
    
    # Combine all predictions
    if all_predictions:
        submission_df = pd.concat(all_predictions, ignore_index=True)
        
        # Reorder columns to have date_id first
        cols = ['date_id'] + [col for col in submission_df.columns if col != 'date_id']
        submission_df = submission_df[cols]
        
        print(f"Submission shape: {submission_df.shape}")
        print(f"Date range: {submission_df['date_id'].min()} to {submission_df['date_id'].max()}")
        print(f"Target columns: {len([col for col in submission_df.columns if col.startswith('target_')])}")
        
        # Save as parquet file
        submission_df.to_parquet('submission.parquet', index=False)
        
        print("✅ Successfully generated submission.parquet!")
        print(f"File size: {Path('submission.parquet').stat().st_size / 1024:.1f} KB")
        
        # Display sample predictions
        print("\nSample predictions:")
        sample_cols = ['date_id'] + [col for col in submission_df.columns if col.startswith('target_')][:5]
        print(submission_df[sample_cols].head())
        
        return submission_df
        
    else:
        print("❌ No predictions generated!")
        return None


def validate_submission():
    """Validate the generated submission file."""
    
    print("Validating submission.parquet...")
    
    try:
        # Load submission file
        submission = pd.read_parquet('submission.parquet')
        
        # Basic validation
        print(f"✅ File loads successfully")
        print(f"✅ Shape: {submission.shape}")
        print(f"✅ Columns: {len(submission.columns)}")
        
        # Check for required columns
        if 'date_id' in submission.columns:
            print(f"✅ date_id column present")
        else:
            print(f"❌ date_id column missing")
        
        # Check target columns
        target_cols = [col for col in submission.columns if col.startswith('target_')]
        print(f"✅ Target columns: {len(target_cols)}")
        
        # Check for missing values
        missing_values = submission.isnull().sum().sum()
        if missing_values == 0:
            print(f"✅ No missing values")
        else:
            print(f"⚠️ Missing values: {missing_values}")
        
        # Check date_id range
        date_range = f"{submission['date_id'].min()} to {submission['date_id'].max()}"
        print(f"✅ Date range: {date_range}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


if __name__ == "__main__":
    print("=== GENERATING KAGGLE SUBMISSION ===")
    
    # Generate submission
    submission_df = generate_kaggle_submission()
    
    if submission_df is not None:
        print("\n=== VALIDATING SUBMISSION ===")
        validate_submission()
        
        print("\n=== SUBMISSION READY ===")
        print("📁 File: submission.parquet")
        print("🚀 Ready to upload to Kaggle!")
    else:
        print("❌ Failed to generate submission")