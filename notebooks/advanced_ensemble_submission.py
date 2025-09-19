"""
Advanced Ensemble Submission for Mitsui Commodity Prediction Challenge
Uses multiple model types and ensemble methods for maximum performance
"""

import os
from pathlib import Path
import sys

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

# Allow importing project feature modules when running locally
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_PATH = PROJECT_ROOT / 'src'
    if SRC_PATH.exists():
        sys.path.append(str(SRC_PATH))
        from features.technical_indicators import create_technical_features  # type: ignore
        from features.time_series import create_time_series_features  # type: ignore
        from features.cross_market import create_cross_market_features  # type: ignore
        from features.global_factors import create_enhanced_features  # type: ignore
        from features.feature_selection import select_best_features  # type: ignore
        HAVE_ADVANCED_FEATURES = True
    else:
        HAVE_ADVANCED_FEATURES = False
except Exception:
    HAVE_ADVANCED_FEATURES = False

def create_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    # Optionally enrich with advanced feature modules (kept lightweight by internal limits)
    if HAVE_ADVANCED_FEATURES:
        try:
            # Derive simple feature categories from columns
            cols = df_result.columns
            feature_categories = {
                'lme': [c for c in cols if isinstance(c, str) and c.startswith('LME_')],
                'jpx': [c for c in cols if isinstance(c, str) and c.startswith('JPX_')],
                'us_stock': [c for c in cols if isinstance(c, str) and c.startswith('US_Stock_')],
                'fx': [c for c in cols if isinstance(c, str) and c.startswith('FX_')],
            }
            # Apply a small set of advanced features first
            df_result = create_technical_features(df_result, feature_categories)
            df_result = create_time_series_features(df_result, feature_categories)
            # Cross-market interactions can add leakage risk if overly broad; keep minimal
            df_result = create_cross_market_features(df_result, feature_categories)
            # PCA/ICA factors to denoise; small components for speed
            df_result = create_enhanced_features(df_result, n_pca_components=2, use_ica=False)
        except Exception:
            # If anything fails, continue with basic engineered features below
            pass
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


def append_train_label_lag_features(train_df: pd.DataFrame, target_df: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    """Append lagged label features (from train_labels) to the training feature frame.

    For each target column, we add target_{...}_lag_{k} for k in [1..max_lag].
    """
    if target_df is None or train_df is None:
        return train_df
    result_df = train_df.copy()
    label_cols = [c for c in target_df.columns if c.startswith('target_')]
    if not label_cols:
        return result_df
    for lag in range(1, max_lag + 1):
        lagged = target_df[label_cols].shift(lag)
        lagged.columns = [f"{c}_lag_{lag}" for c in label_cols]
        result_df = pd.concat([result_df, lagged], axis=1)
    return result_df


def add_label_lag_features_from_batches(df: pd.DataFrame,
                                        lag1: pl.DataFrame,
                                        lag2: pl.DataFrame,
                                        lag3: pl.DataFrame,
                                        lag4: pl.DataFrame) -> pd.DataFrame:
    """For inference, add provided lagged label values as features on the single-row test frame.

    Creates columns named target_{...}_lag_{k} to match training-time feature names.
    """
    result_df = df.copy()
    lag_batches = [(1, lag1), (2, lag2), (3, lag3), (4, lag4)]
    for lag_num, pl_df in lag_batches:
        try:
            pd_df = pl_df.to_pandas()
        except Exception:
            continue
        # drop meta columns if present
        for drop_col in ['date_id', 'label_date_id']:
            if drop_col in pd_df.columns:
                pd_df = pd_df.drop(columns=[drop_col])
        # single-row expected
        if len(pd_df) == 0:
            continue
        for col in pd_df.columns:
            if not str(col).startswith('target_'):
                continue
            result_df[f"{col}_lag_{lag_num}"] = pd_df.iloc[0][col]
    return result_df

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
    data_dir = os.getenv('LOCAL_DATA_DIR', '/kaggle/input/mitsui-commodity-prediction-challenge')
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    target_df = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
    # Append lagged label features before creating engineered features
    train_df = append_train_label_lag_features(train_df, target_df, max_lag=4)
    # Create comprehensive features
    train_df = create_comprehensive_features(train_df)
    # Global prefiltering
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    base_feature_cols = [col for col in numeric_cols if col != 'date_id']
    # Remove features with too many NaNs (stricter)
    X_temp = train_df[base_feature_cols]
    nan_ratio = X_temp.isnull().sum() / len(X_temp)
    base_feature_cols = [col for col in base_feature_cols if nan_ratio[col] < 0.4]
    # Remove highly correlated features
    X_temp = train_df[base_feature_cols].fillna(0)
    correlation_matrix = X_temp.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]
    base_feature_cols = [col for col in base_feature_cols if col not in high_corr_features]
    X_full = train_df[base_feature_cols].fillna(0)
    # Scale using global scaler so we can subset per-target later
    scaler = StandardScaler()
    X_full_scaled = pd.DataFrame(scaler.fit_transform(X_full), columns=X_full.columns, index=X_full.index)
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
        # Per-target lightweight feature selection (Spearman)
        try:
            if HAVE_ADVANCED_FEATURES:
                selected_features, _ = select_best_features(
                    X_full, y, max_features=150, selection_method='spearman'
                )
                # Ensure selected features are in our scaled matrix
                selected_features = [c for c in selected_features if c in X_full_scaled.columns]
                if len(selected_features) < 20:
                    selected_features = X_full_scaled.columns.tolist()[:200]
            else:
                selected_features = X_full_scaled.columns.tolist()[:200]
        except Exception:
            selected_features = X_full_scaled.columns.tolist()[:200]

        X_valid = X_full_scaled.loc[valid_mask, selected_features]
        y_valid = y[valid_mask]
        try:
            model_info = create_ensemble_model(X_valid, y_valid, target_col)
            model_info['feature_columns'] = selected_features
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
    # Add provided lagged label features to align with training-time features
    test_df = add_label_lag_features_from_batches(
        test_df,
        label_lags_1_batch,
        label_lags_2_batch,
        label_lags_3_batch,
        label_lags_4_batch,
    )
    # Then add engineered features on top
    test_df = create_comprehensive_features(test_df)
    # Prepare global scaled features; we'll subset per target
    all_model_features = set()
    for mi in (ensemble_models or {}).values():
        if isinstance(mi, dict) and 'feature_columns' in mi:
            for c in mi['feature_columns']:
                all_model_features.add(c)
    if not all_model_features:
        all_model_features = set(feature_columns or [])

    try:
        X_all = test_df[list(all_model_features)].fillna(0)
    except Exception:
        available_cols = [c for c in all_model_features if c in test_df.columns]
        X_all = test_df[available_cols].fillna(0) if available_cols else pd.DataFrame([[0]])
        for c in all_model_features:
            if c not in X_all.columns:
                X_all[c] = 0
        X_all = X_all[list(all_model_features)]

    X_all_scaled = pd.DataFrame(
        scalers['feature_scaler'].transform(X_all.reindex(columns=scalers['feature_scaler'].feature_names_in_, fill_value=0)),
        columns=scalers['feature_scaler'].feature_names_in_,
        index=X_all.index
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
                    cols = model_info.get('feature_columns', list(X_all_scaled.columns))
                    # Ensure columns exist and order matches training
                    cols_existing = [c for c in cols if c in X_all_scaled.columns]
                    X_target = X_all_scaled[cols_existing]
                    if X_target.shape[1] == 0:
                        X_target = X_all_scaled
                    pred = model_info['model'].predict(X_target)
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
    local_dir = os.getenv('LOCAL_DATA_DIR')
    if local_dir:
        inference_server.run_local_gateway((local_dir,))
    else:
        inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))