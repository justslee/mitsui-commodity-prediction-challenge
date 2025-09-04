"""
High-Performance Kaggle Submission for Mitsui Commodity Prediction Challenge
"""

import os

import pandas as pd

import polars as pl  # type: ignore

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.mitsui_inference_server
NUM_TARGET_COLUMNS = 424
# Global model storage
models = None
feature_columns = None
feature_categories = None

def create_technical_indicators(df: pd.DataFrame, price_cols: list, max_cols: int = 10) -> pd.DataFrame:
    df_result = df.copy()
    price_cols = price_cols[:max_cols]  # Limit for performance
    for col in price_cols:
        if col in df.columns:
            # Rolling averages
            for window in [5, 10, 20]:
                df_result[f"{col}_ma_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
            # RSI
            delta = df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_result[f"{col}_rsi"] = 100 - (100 / (1 + rs))
            # Volatility
            returns = df[col].pct_change()
            df_result[f"{col}_volatility"] = returns.rolling(window=20).std()
            # Momentum
            df_result[f"{col}_momentum"] = df[col] - df[col].shift(10)
    return df_result

def create_cross_market_features(df: pd.DataFrame, feature_categories: dict) -> pd.DataFrame:
    df_result = df.copy()
    # Currency-adjusted prices
    lme_cols = [col for col in feature_categories.get('LME', []) if col in df.columns][:3]
    jpx_cols = [col for col in feature_categories.get('JPX', []) if col in df.columns][:3]
    if 'FX_USDJPY' in df.columns:
        for col in jpx_cols:
            if col in df.columns:
                df_result[f"{col}_usd_adj"] = df[col] / df['FX_USDJPY']
    # Cross-market correlations (simplified)
    for i, lme_col in enumerate(lme_cols[:2]):
        for j, jpx_col in enumerate(jpx_cols[:2]):
            if lme_col in df.columns and jpx_col in df.columns:
                df_result[f"corr_{i}_{j}"] = df[lme_col].rolling(50).corr(df[jpx_col])
    return df_result

def create_time_series_features(df: pd.DataFrame, important_cols: list, max_cols: int = 5) -> pd.DataFrame:
    df_result = df.copy()
    important_cols = important_cols[:max_cols]
    for col in important_cols:
        if col in df.columns:
            # Lag features
            for lag in [1, 2, 5]:
                df_result[f"{col}_lag_{lag}"] = df[col].shift(lag)
            # Rolling statistics
            for window in [10, 20]:
                df_result[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window).mean()
                df_result[f"{col}_roll_std_{window}"] = df[col].rolling(window=window).std()
    # Seasonal features
    if 'date_id' in df.columns:
        df_result['date_sin_weekly'] = np.sin(2 * np.pi * df['date_id'] / 7)
        df_result['date_cos_weekly'] = np.cos(2 * np.pi * df['date_id'] / 7)
        df_result['date_sin_monthly'] = np.sin(2 * np.pi * df['date_id'] / 30)
        df_result['date_cos_monthly'] = np.cos(2 * np.pi * df['date_id'] / 30)
    return df_result

def get_feature_categories():
    return {
        'LME': ['LME_AH_Close', 'LME_CA_Close', 'LME_PB_Close', 'LME_ZS_Close'],
        'JPX': ['JPX_Gold_Standard_Futures_Close', 'JPX_Platinum_Standard_Futures_Close'],
        'FX': ['FX_USDJPY', 'FX_EURUSD', 'FX_GBPUSD'],
        'US_Stock': ['US_Stock_VT_adj_close', 'US_Stock_VYM_adj_close', 'US_Stock_IEMG_adj_close']
    }

def load_and_train_models():
    global models, feature_columns, feature_categories
    # Load training data
    train_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
    target_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')
    # Get feature categories
    feature_categories = get_feature_categories()
    # Feature Engineering
    price_cols = []
    for category, cols in feature_categories.items():
        price_cols.extend([col for col in cols if col in train_df.columns])
    train_df = create_technical_indicators(train_df, price_cols, max_cols=15)
    train_df = create_cross_market_features(train_df, feature_categories)
    important_cols = price_cols[:10]
    train_df = create_time_series_features(train_df, important_cols, max_cols=8)
    # Feature selection
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'date_id']
    # Remove highly correlated features
    X_temp = train_df[feature_cols].fillna(0)
    correlation_matrix = X_temp.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    # Find features with correlation > 0.95
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    feature_cols = [col for col in feature_cols if col not in high_corr_features]
    X = train_df[feature_cols].fillna(0)
    feature_columns = X.columns.tolist()
    # Get target columns
    target_columns = [col for col in target_df.columns if col.startswith('target_')]
    # Train models for each target
    trained_models = {}
    rf_targets = 0
    xgb_targets = 0
    ridge_targets = 0
    for i, target_col in enumerate(target_columns):
        if (i + 1) % 50 == 0:
        if target_col not in target_df.columns:
            continue
        y = target_df[target_col]
        valid_mask = ~y.isna()
        if valid_mask.sum() < 10:
            trained_models[target_col] = {'type': 'mean', 'value': 0.0}
            continue
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        try:
            # Choose model based on data size and target characteristics
            if valid_mask.sum() >= 500:  # Use Random Forest for targets with lots of data
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
                trained_models[target_col] = {'type': 'rf', 'model': model}
                rf_targets += 1
            elif valid_mask.sum() >= 100:  # Use XGBoost for medium-sized targets
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0,
                    n_jobs=1
                )
                model.fit(X_valid, y_valid)
                trained_models[target_col] = {'type': 'xgb', 'model': model}
                xgb_targets += 1
            else:  # Use Ridge for small targets
                model = Ridge(alpha=1.0, random_state=42)
                model.fit(X_valid, y_valid)
                trained_models[target_col] = {'type': 'ridge', 'model': model}
                ridge_targets += 1
        except Exception:
            mean_value = y_valid.mean() if len(y_valid) > 0 else 0.0
            trained_models[target_col] = {'type': 'mean', 'value': mean_value}
    models = trained_models

def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    global models, feature_columns, feature_categories
    # Load and train models on first call
    if models is None:
        load_and_train_models()
    # Convert to pandas
    test_df = test.to_pandas()
    # Apply same feature engineering as training
    price_cols = []
    for category, cols in feature_categories.items():
        price_cols.extend([col for col in cols if col in test_df.columns])
    test_df = create_technical_indicators(test_df, price_cols, max_cols=15)
    test_df = create_cross_market_features(test_df, feature_categories)
    important_cols = price_cols[:10]
    test_df = create_time_series_features(test_df, important_cols, max_cols=8)
    # Prepare features
    X = test_df[feature_columns].fillna(0) if all(col in test_df.columns for col in feature_columns) else pd.DataFrame()
    # Handle missing features
    if X.empty:
        # Fallback: use available numeric features
        numeric_cols = test_df.select_dtypes(include=[np.number]).columns
        available_features = [col for col in numeric_cols if col != 'date_id']
        X = test_df[available_features].fillna(0)
        # Pad or trim to match expected features
        missing_cols = set(feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[feature_columns] if all(col in X.columns for col in feature_columns) else X.iloc[:, :len(feature_columns)]
    # Get expected targets
    provided_label_lags = pl.concat(
        [label_lags_1_batch.drop(['date_id', 'label_date_id']),
         label_lags_2_batch.drop(['date_id', 'label_date_id']),
         label_lags_3_batch.drop(['date_id', 'label_date_id']),
         label_lags_4_batch.drop(['date_id', 'label_date_id'])],
        how='horizontal'
    )
    expected_targets = provided_label_lags.columns
    # Make predictions
    predictions = {}
    for target_col in expected_targets:
        if target_col in models:
            try:
                model_info = models[target_col]
                if model_info['type'] in ['rf', 'xgb', 'ridge']:
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
# Initialize inference server

inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))