"""
Kaggle Submission for Mitsui Commodity Prediction Challenge
"""

import pandas as pd
import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
import os
import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.core.templates
from kaggle_evaluation import mitsui_gateway


class MitsuiSubmissionModel:
    def __init__(self):
        self.models = {}
        self.feature_columns = None
        self.all_target_columns = None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'date_id'][:100]
        X = df[feature_cols].fillna(0)
        return X
    
    def train_on_kaggle_data(self):
        print("Training models...")
        
        train_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
        target_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')
        
        self.all_target_columns = [col for col in target_df.columns if col.startswith('target_')]
        
        X = self.prepare_features(train_df)
        self.feature_columns = X.columns.tolist()
        
        successful_targets = 0
        
        for target_col in self.all_target_columns:
            if target_col not in target_df.columns:
                continue
            
            y = target_df[target_col]
            valid_mask = ~y.isna()
            
            if valid_mask.sum() < 5:
                self.models[target_col] = {'type': 'mean', 'value': 0.0}
                continue
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            try:
                model = Ridge(alpha=1.0, random_state=42)
                model.fit(X_valid, y_valid)
                self.models[target_col] = {'type': 'model', 'model': model}
                successful_targets += 1
            except Exception:
                mean_value = y_valid.mean() if len(y_valid) > 0 else 0.0
                self.models[target_col] = {'type': 'mean', 'value': mean_value}
        
        print(f"Trained {successful_targets} Ridge models")
        self.is_trained = True
        
    def predict_single_date(self, test_batch: pl.DataFrame, 
                           label_lags_1: pl.DataFrame,
                           label_lags_2: pl.DataFrame,
                           label_lags_3: pl.DataFrame,
                           label_lags_4: pl.DataFrame) -> pd.DataFrame:
        
        test_df = test_batch.to_pandas()
        X = self.prepare_features(test_df)
        
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        
        X = X[self.feature_columns]
        
        provided_label_lags = pl.concat(
            [label_lags_1.drop(['date_id', 'label_date_id']),
             label_lags_2.drop(['date_id', 'label_date_id']),
             label_lags_3.drop(['date_id', 'label_date_id']),
             label_lags_4.drop(['date_id', 'label_date_id'])],
            how='horizontal'
        )
        expected_targets = provided_label_lags.columns
        
        predictions = {}
        
        for target_col in expected_targets:
            if target_col in self.models:
                try:
                    model_info = self.models[target_col]
                    if model_info['type'] == 'model':
                        pred = model_info['model'].predict(X)
                        predictions[target_col] = pred[0] if len(pred) > 0 else 0.0
                    else:
                        predictions[target_col] = model_info['value']
                except Exception:
                    predictions[target_col] = 0.0
            else:
                predictions[target_col] = 0.0
        
        return pd.DataFrame([predictions])


class MitsuiInferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    def __init__(self):
        super().__init__()
        self.model = MitsuiSubmissionModel()
        self.model.train_on_kaggle_data()
    
    def predict(self, test_batch, label_lags_1, label_lags_2, label_lags_3, label_lags_4):
        return self.model.predict_single_date(
            test_batch, label_lags_1, label_lags_2, label_lags_3, label_lags_4
        )
    
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None):
        return mitsui_gateway.MitsuiGateway(data_paths)


def generate_predictions():
    """Generate predictions and save as submission.parquet"""
    model = MitsuiSubmissionModel()
    model.train_on_kaggle_data()
    
    # Load test data
    test_df = pl.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/test.csv')
    label_lags_1 = pl.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/lagged_test_labels/test_labels_lag_1.csv')
    label_lags_2 = pl.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/lagged_test_labels/test_labels_lag_2.csv')
    label_lags_3 = pl.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/lagged_test_labels/test_labels_lag_3.csv')
    label_lags_4 = pl.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/lagged_test_labels/test_labels_lag_4.csv')
    
    date_ids = test_df['date_id'].unique().to_list()
    all_predictions = []
    
    for date_id in date_ids:
        test_batch = test_df.filter(pl.col('date_id') == date_id)
        label_lags_1_batch = label_lags_1.filter(pl.col('date_id') == date_id)
        label_lags_2_batch = label_lags_2.filter(pl.col('date_id') == date_id)
        label_lags_3_batch = label_lags_3.filter(pl.col('date_id') == date_id)
        label_lags_4_batch = label_lags_4.filter(pl.col('date_id') == date_id)
        
        prediction = model.predict_single_date(
            test_batch, label_lags_1_batch, label_lags_2_batch,
            label_lags_3_batch, label_lags_4_batch
        )
        
        prediction['date_id'] = date_id
        all_predictions.append(prediction)
    
    submission_df = pd.concat(all_predictions, ignore_index=True)
    cols = ['date_id'] + [col for col in submission_df.columns if col != 'date_id']
    submission_df = submission_df[cols]
    
    submission_df.to_parquet('submission.parquet', index=False)
    print(f"Generated submission.parquet with shape {submission_df.shape}")


def main():
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        server = MitsuiInferenceServer()
        server.serve()
    else:
        generate_predictions()


if __name__ == "__main__":
    main()