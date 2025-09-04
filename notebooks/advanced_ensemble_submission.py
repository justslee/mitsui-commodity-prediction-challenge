"""
Advanced Ensemble Submission for Mitsui Commodity Prediction Challenge
Uses multiple model types and ensemble methods for maximum performance
"""

import os

import pandas as pd

import polars as pl  # type: ignore

import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso

import xgboost as xgb

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.mitsui_inference_server
NUM_TARGET_COLUMNS = 424
# Global storage
ensemble_models = None
feature_columns = None
scalers = None

def create_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    # Get price columns
    price_cols = []
    for col in df.columns:
        if any(keyword in col for keyword in ['Close', 'close', 'price', 'Price']):
            price_cols.append(col)
    price_cols = price_cols[:20]  # Limit for performance
    # Technical indicators
    for col in price_cols:
        if col in df.columns:
            # Multiple timeframe moving averages
            for window in [5, 10, 20, 50]:
                df_result[f"{col}_sma_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
                df_result[f"{col}_ema_{window}"] = df[col].ewm(span=window).mean()
            # RSI with multiple periods
            for period in [14, 21]:
                delta = df[col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df_result[f"{col}_rsi_{period}"] = 100 - (100 / (1 + rs))
            # Bollinger Bands
            rolling_mean = df[col].rolling(window=20).mean()
            rolling_std = df[col].rolling(window=20).std()
            df_result[f"{col}_bb_upper"] = rolling_mean + (rolling_std * 2)
            df_result[f"{col}_bb_lower"] = rolling_mean - (rolling_std * 2)
            df_result[f"{col}_bb_width"] = df_result[f"{col}_bb_upper"] - df_result[f"{col}_bb_lower"]
            # MACD
            ema_12 = df[col].ewm(span=12).mean()
            ema_26 = df[col].ewm(span=26).mean()
            df_result[f"{col}_macd"] = ema_12 - ema_26
            df_result[f"{col}_macd_signal"] = df_result[f"{col}_macd"].ewm(span=9).mean()
            # Volatility and momentum
            returns = df[col].pct_change()
            for window in [10, 20, 50]:
                df_result[f"{col}_vol_{window}"] = returns.rolling(window=window).std()
                df_result[f"{col}_mom_{window}"] = df[col] - df[col].shift(window)
            # Price position
            for window in [20, 50]:
                rolling_min = df[col].rolling(window=window).min()
                rolling_max = df[col].rolling(window=window).max()
                df_result[f"{col}_pos_{window}"] = (df[col] - rolling_min) / (rolling_max - rolling_min + 1e-8)
    # Time series features
    important_cols = price_cols[:10]
    for col in important_cols:
        if col in df.columns:
            # Multiple lags
            for lag in [1, 2, 3, 5, 10, 20]:
                df_result[f"{col}_lag_{lag}"] = df[col].shift(lag)
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                df_result[f"{col}_mean_{window}"] = df[col].rolling(window=window).mean()
                df_result[f"{col}_std_{window}"] = df[col].rolling(window=window).std()
                df_result[f"{col}_min_{window}"] = df[col].rolling(window=window).min()
                df_result[f"{col}_max_{window}"] = df[col].rolling(window=window).max()
                df_result[f"{col}_skew_{window}"] = df[col].rolling(window=window).skew()
    # Cross-asset features
    if len(price_cols) >= 2:
        for i in range(min(5, len(price_cols))):
            for j in range(i+1, min(5, len(price_cols))):
                col1, col2 = price_cols[i], price_cols[j]
                if col1 in df.columns and col2 in df.columns:
                    # Price ratios
                    df_result[f"ratio_{i}_{j}"] = df[col1] / (df[col2] + 1e-8)
                    # Price differences
                    df_result[f"diff_{i}_{j}"] = df[col1] - df[col2]
                    # Rolling correlations
                    df_result[f"corr_{i}_{j}"] = df[col1].rolling(30).corr(df[col2])
    # Seasonal features
    if 'date_id' in df.columns:
        df_result['date_sin_weekly'] = np.sin(2 * np.pi * df['date_id'] / 7)
        df_result['date_cos_weekly'] = np.cos(2 * np.pi * df['date_id'] / 7)
        df_result['date_sin_monthly'] = np.sin(2 * np.pi * df['date_id'] / 30)
        df_result['date_cos_monthly'] = np.cos(2 * np.pi * df['date_id'] / 30)
        df_result['date_sin_quarterly'] = np.sin(2 * np.pi * df['date_id'] / 90)
        df_result['date_cos_quarterly'] = np.cos(2 * np.pi * df['date_id'] / 90)
    return df_result

def create_ensemble_model(X_valid: pd.DataFrame, y_valid: pd.Series, target_name: str):
    n_samples = len(y_valid)
    if n_samples >= 1000:
        # Large dataset: use sophisticated ensemble
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
        )
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0, n_jobs=1
        )
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            feature_fraction=0.8, bagging_fraction=0.8,
            random_state=42, verbosity=-1, n_jobs=1
        )
        # Create voting ensemble
        ensemble = VotingRegressor([
            ('rf', rf),
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ])
        ensemble.fit(X_valid, y_valid)
        return {'type': 'ensemble', 'model': ensemble}
    elif n_samples >= 200:
        # Medium dataset: use single best model
        rf = RandomForestRegressor(
            n_estimators=50, max_depth=8, min_samples_split=3,
            min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
        )
        rf.fit(X_valid, y_valid)
        return {'type': 'rf', 'model': rf}
    elif n_samples >= 50:
        # Small dataset: use XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=0
        )
        xgb_model.fit(X_valid, y_valid)
        return {'type': 'xgb', 'model': xgb_model}
    else:
        # Very small dataset: use Ridge
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_valid, y_valid)
        return {'type': 'ridge', 'model': ridge}

def load_and_train_ensemble_models():
    global ensemble_models, feature_columns, scalers
    # Load data
    train_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
    target_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')
    # Create comprehensive features
    train_df = create_comprehensive_features(train_df)
    # Feature selection
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'date_id']
    # Remove features with too many NaNs
    X_temp = train_df[feature_cols]
    nan_ratio = X_temp.isnull().sum() / len(X_temp)
    feature_cols = [col for col in feature_cols if nan_ratio[col] < 0.5]
    # Remove highly correlated features
    X_temp = train_df[feature_cols].fillna(0)
    correlation_matrix = X_temp.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    feature_cols = [col for col in feature_cols if col not in high_corr_features]
    X = train_df[feature_cols].fillna(0)
    feature_columns = X.columns.tolist()
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    scalers = {'feature_scaler': scaler}
    # Train models
    target_columns = [col for col in target_df.columns if col.startswith('target_')]
    trained_models = {}
    ensemble_count = 0
    rf_count = 0
    xgb_count = 0
    ridge_count = 0
    for i, target_col in enumerate(target_columns):
        if (i + 1) % 25 == 0:
            print(f"Progress: {i + 1}/{len(target_columns)} targets")
        if target_col not in target_df.columns:
            continue
        y = target_df[target_col]
        valid_mask = ~y.isna()
        if valid_mask.sum() < 10:
            trained_models[target_col] = {'type': 'mean', 'value': 0.0}
            continue
        X_valid = X_scaled[valid_mask]
        y_valid = y[valid_mask]
        try:
            model_info = create_ensemble_model(X_valid, y_valid, target_col)
            trained_models[target_col] = model_info
            # Count model types
            if model_info['type'] == 'ensemble':
                ensemble_count += 1
            elif model_info['type'] == 'rf':
                rf_count += 1
            elif model_info['type'] == 'xgb':
                xgb_count += 1
            elif model_info['type'] == 'ridge':
                ridge_count += 1
        except Exception:
            mean_value = y_valid.mean() if len(y_valid) > 0 else 0.0
            trained_models[target_col] = {'type': 'mean', 'value': mean_value}
    ensemble_models = trained_models

def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    global ensemble_models, feature_columns, scalers
    # Load models on first call
    if ensemble_models is None:
        load_and_train_ensemble_models()
    # Convert and process features
    test_df = test.to_pandas()
    test_df = create_comprehensive_features(test_df)
    # Prepare features
    try:
        X = test_df[feature_columns].fillna(0)
        X_scaled = pd.DataFrame(
            scalers['feature_scaler'].transform(X),
            columns=X.columns,
            index=X.index
        )
    except Exception:
        # Fallback for missing features
        available_cols = [col for col in feature_columns if col in test_df.columns]
        X = test_df[available_cols].fillna(0) if available_cols else pd.DataFrame([[0] * len(feature_columns)])
        # Pad missing columns
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]
        X_scaled = pd.DataFrame(
            scalers['feature_scaler'].transform(X),
            columns=X.columns,
            index=X.index
        )
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
        if target_col in ensemble_models:
            try:
                model_info = ensemble_models[target_col]
                if model_info['type'] in ['ensemble', 'rf', 'xgb', 'ridge']:
                    pred = model_info['model'].predict(X_scaled)
                    predictions[target_col] = pred[0] if len(pred) > 0 else 0.0
                else:
                    predictions[target_col] = model_info['value']
            except Exception:
                predictions[target_col] = 0.0
        else:
            predictions[target_col] = 0.0
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