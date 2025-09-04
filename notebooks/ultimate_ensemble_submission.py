"""
Ultimate Ensemble Submission for Mitsui Commodity Prediction Challenge
Combines multiple model types with ranking optimization for maximum performance
"""

import os

import pandas as pd

import polars as pl  # type: ignore

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import xgboost as xgb

import lightgbm as lgb
from scipy.stats import rankdata

import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.mitsui_inference_server
NUM_TARGET_COLUMNS = 424
# Global storage
ultimate_models = None
feature_columns = None
scalers = None
target_statistics = None

def create_ultimate_features(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    # Get price columns
    price_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['close', 'price', 'value', 'futures']):
            price_cols.append(col)
    price_cols = price_cols[:25]  # Use more for ultimate version
    # Technical indicators
    for col in price_cols:
        if col in df.columns:
            # Multiple timeframe moving averages
            for window in [3, 5, 10, 20, 50, 100]:
                df_result[f"{col}_sma_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
                if window <= 50:  # EMA for shorter windows
                    df_result[f"{col}_ema_{window}"] = df[col].ewm(span=window).mean()
            # RSI with multiple periods
            for period in [7, 14, 21, 28]:
                delta = df[col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df_result[f"{col}_rsi_{period}"] = 100 - (100 / (1 + rs))
            # Bollinger Bands with multiple periods
            for period in [10, 20, 30]:
                rolling_mean = df[col].rolling(window=period).mean()
                rolling_std = df[col].rolling(window=period).std()
                df_result[f"{col}_bb_upper_{period}"] = rolling_mean + (rolling_std * 2)
                df_result[f"{col}_bb_lower_{period}"] = rolling_mean - (rolling_std * 2)
                df_result[f"{col}_bb_position_{period}"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            # MACD variations
            for fast, slow in [(12, 26), (8, 21), (5, 10)]:
                ema_fast = df[col].ewm(span=fast).mean()
                ema_slow = df[col].ewm(span=slow).mean()
                macd = ema_fast - ema_slow
                df_result[f"{col}_macd_{fast}_{slow}"] = macd
                df_result[f"{col}_macd_signal_{fast}_{slow}"] = macd.ewm(span=9).mean()
            # Momentum and volatility
            returns = df[col].pct_change()
            for window in [5, 10, 20, 50]:
                df_result[f"{col}_momentum_{window}"] = df[col] - df[col].shift(window)
                df_result[f"{col}_volatility_{window}"] = returns.rolling(window=window).std()
                df_result[f"{col}_sharpe_{window}"] = returns.rolling(window=window).mean() / (returns.rolling(window=window).std() + 1e-8)
            # Price position and extremes
            for window in [10, 20, 50, 100]:
                rolling_min = df[col].rolling(window=window).min()
                rolling_max = df[col].rolling(window=window).max()
                df_result[f"{col}_position_{window}"] = (df[col] - rolling_min) / (rolling_max - rolling_min + 1e-8)
                df_result[f"{col}_extreme_high_{window}"] = (df[col] == rolling_max).astype(float)
                df_result[f"{col}_extreme_low_{window}"] = (df[col] == rolling_min).astype(float)
            # Ranking features
            for window in [10, 20, 50]:
                df_result[f"{col}_percentile_{window}"] = df[col].rolling(window=window).rank(pct=True)
                df_result[f"{col}_zscore_{window}"] = (df[col] - df[col].rolling(window=window).mean()) / (df[col].rolling(window=window).std() + 1e-8)
    # Cross-asset features
    if len(price_cols) >= 2:
        # Correlation matrix features
        for window in [30, 50, 100]:
            corr_sum = 0
            corr_count = 0
            for i in range(min(10, len(price_cols))):
                for j in range(i+1, min(10, len(price_cols))):
                    col1, col2 = price_cols[i], price_cols[j]
                    if col1 in df.columns and col2 in df.columns:
                        # Rolling correlation
                        corr = df[col1].rolling(window).corr(df[col2])
                        df_result[f"corr_{i}_{j}_{window}"] = corr
                        # Relative strength
                        df_result[f"rel_strength_{i}_{j}"] = df[col1] / (df[col2] + 1e-8)
                        # Spread
                        df_result[f"spread_{i}_{j}"] = df[col1] - df[col2]
                        corr_sum += corr.fillna(0)
                        corr_count += 1
            # Market coherence
            if corr_count > 0:
                df_result[f"market_coherence_{window}"] = corr_sum / corr_count
        # Market index from top assets
        market_index = df[price_cols[:5]].mean(axis=1)
        df_result['market_index'] = market_index
        for col in price_cols[:15]:
            if col in df.columns:
                # Beta to market
                returns_asset = df[col].pct_change()
                returns_market = market_index.pct_change()
                for window in [30, 50, 100]:
                    cov_window = returns_asset.rolling(window).cov(returns_market)
                    var_window = returns_market.rolling(window).var()
                    df_result[f"{col}_beta_{window}"] = cov_window / (var_window + 1e-8)
                    # Tracking error
                    tracking_error = (returns_asset - returns_market).rolling(window).std()
                    df_result[f"{col}_tracking_error_{window}"] = tracking_error
                    # Information ratio
                    excess_return = returns_asset.rolling(window).mean() - returns_market.rolling(window).mean()
                    df_result[f"{col}_info_ratio_{window}"] = excess_return / (tracking_error + 1e-8)
    # Time series features
    important_cols = price_cols[:15]
    for col in important_cols:
        if col in df.columns:
            # Extended lag features
            for lag in [1, 2, 3, 5, 10, 20, 50]:
                df_result[f"{col}_lag_{lag}"] = df[col].shift(lag)
                if lag <= 10:  # Lag differences
                    df_result[f"{col}_lag_diff_{lag}"] = df[col] - df[col].shift(lag)
            # Seasonal decomposition approximation
            for period in [7, 30, 90]:  # Weekly, monthly, quarterly
                df_result[f"{col}_seasonal_{period}"] = df[col].rolling(period).mean()
                df_result[f"{col}_detrended_{period}"] = df[col] - df[col].rolling(period * 2).mean()
    # Advanced time features
    if 'date_id' in df.columns:
        # Multiple seasonal cycles
        for period, name in [(7, 'weekly'), (14, 'biweekly'), (30, 'monthly'), (90, 'quarterly'), (365, 'yearly')]:
            df_result[f'date_sin_{name}'] = np.sin(2 * np.pi * df['date_id'] / period)
            df_result[f'date_cos_{name}'] = np.cos(2 * np.pi * df['date_id'] / period)
        # Trend features
        df_result['date_trend'] = df['date_id'] / df['date_id'].max()
        df_result['date_squared'] = (df['date_id'] / df['date_id'].max()) ** 2
        df_result['date_cubed'] = (df['date_id'] / df['date_id'].max()) ** 3
    return df_result

def create_ultimate_ensemble(X_valid: pd.DataFrame, y_valid: pd.Series, target_name: str):
    n_samples = len(y_valid)
    # Prepare multiple scaled versions
    scalers = {}
    X_versions = {}
    # Standard scaling
    scaler_std = StandardScaler()
    X_std = pd.DataFrame(scaler_std.fit_transform(X_valid), columns=X_valid.columns, index=X_valid.index)
    scalers['std'] = scaler_std
    X_versions['std'] = X_std
    # Robust scaling
    scaler_robust = RobustScaler()
    X_robust = pd.DataFrame(scaler_robust.fit_transform(X_valid), columns=X_valid.columns, index=X_valid.index)
    scalers['robust'] = scaler_robust
    X_versions['robust'] = X_robust
    # MinMax scaling
    scaler_minmax = MinMaxScaler()
    X_minmax = pd.DataFrame(scaler_minmax.fit_transform(X_valid), columns=X_valid.columns, index=X_valid.index)
    scalers['minmax'] = scaler_minmax
    X_versions['minmax'] = X_minmax
    # Create target ranks for ranking-aware training
    y_ranks = rankdata(y_valid, method='average') / len(y_valid)
    models = []
    model_info = {'scalers': scalers, 'models': []}
    if n_samples >= 1000:
        # Large dataset: multiple sophisticated models
        # Random Forest variants
        rf1 = RandomForestRegressor(
            n_estimators=150, max_depth=15, min_samples_split=3, min_samples_leaf=1,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        rf2 = RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_split=5, min_samples_leaf=2,
            max_features='log2', random_state=123, n_jobs=-1
        )
        # XGBoost variants
        xgb1 = xgb.XGBRegressor(
            n_estimators=150, max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbosity=0
        )
        xgb2 = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, reg_alpha=0.05, reg_lambda=0.05,
            random_state=123, verbosity=0
        )
        # LightGBM variants
        lgb1 = lgb.LGBMRegressor(
            n_estimators=150, max_depth=8, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbosity=-1
        )
        # Linear models on different scalings
        ridge_robust = Ridge(alpha=1.0, random_state=42)
        lasso_std = Lasso(alpha=0.1, random_state=42, max_iter=2000)
        elastic_minmax = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=2000)
        # Train models on appropriate data
        rf1.fit(X_std, y_valid * 0.7 + y_ranks * 0.3)  # Ranking-aware
        rf2.fit(X_robust, y_valid)
        xgb1.fit(X_std, y_ranks)  # Ranking-focused
        xgb2.fit(X_robust, y_valid)
        lgb1.fit(X_std, y_valid * 0.8 + y_ranks * 0.2)
        ridge_robust.fit(X_robust, y_valid)
        lasso_std.fit(X_std, y_valid)
        elastic_minmax.fit(X_minmax, y_valid)
        model_info['models'] = [
            {'model': rf1, 'scaler': 'std', 'weight': 0.2},
            {'model': rf2, 'scaler': 'robust', 'weight': 0.15},
            {'model': xgb1, 'scaler': 'std', 'weight': 0.2},
            {'model': xgb2, 'scaler': 'robust', 'weight': 0.15},
            {'model': lgb1, 'scaler': 'std', 'weight': 0.15},
            {'model': ridge_robust, 'scaler': 'robust', 'weight': 0.05},
            {'model': lasso_std, 'scaler': 'std', 'weight': 0.05},
            {'model': elastic_minmax, 'scaler': 'minmax', 'weight': 0.05}
        ]
        return {'type': 'ultimate_ensemble', 'model_info': model_info}
    elif n_samples >= 300:
        # Medium dataset: focused ensemble
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=3,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0
        )
        ridge = Ridge(alpha=1.0, random_state=42)
        rf.fit(X_std, y_valid * 0.8 + y_ranks * 0.2)
        xgb_model.fit(X_std, y_ranks)
        ridge.fit(X_robust, y_valid)
        model_info['models'] = [
            {'model': rf, 'scaler': 'std', 'weight': 0.5},
            {'model': xgb_model, 'scaler': 'std', 'weight': 0.3},
            {'model': ridge, 'scaler': 'robust', 'weight': 0.2}
        ]
        return {'type': 'medium_ensemble', 'model_info': model_info}
    elif n_samples >= 50:
        # Small dataset: single best model with ranking
        xgb_model = xgb.XGBRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=0
        )
        xgb_model.fit(X_std, y_ranks)
        model_info['models'] = [
            {'model': xgb_model, 'scaler': 'std', 'weight': 1.0}
        ]
        return {'type': 'single_xgb', 'model_info': model_info}
    else:
        # Very small dataset: regularized linear model
        ridge = Ridge(alpha=5.0, random_state=42)
        ridge.fit(X_robust, y_ranks)
        model_info['models'] = [
            {'model': ridge, 'scaler': 'robust', 'weight': 1.0}
        ]
        return {'type': 'single_ridge', 'model_info': model_info}

def load_and_train_ultimate_models():
    global ultimate_models, feature_columns, scalers, target_statistics
    # Load data
    train_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
    target_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')
    # Create ultimate features
    train_df = create_ultimate_features(train_df)
    # Feature selection
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'date_id']
    # Remove features with too many NaNs
    X_temp = train_df[feature_cols]
    nan_ratio = X_temp.isnull().sum() / len(X_temp)
    feature_cols = [col for col in feature_cols if nan_ratio[col] < 0.7]
    # Remove highly correlated features
    X_temp = train_df[feature_cols].fillna(0)
    if len(feature_cols) > 1:
        correlation_matrix = X_temp.corr().abs()
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.98)]
        feature_cols = [col for col in feature_cols if col not in high_corr_features]
    X = train_df[feature_cols].fillna(0)
    feature_columns = X.columns.tolist()
    # Collect target statistics
    target_columns = [col for col in target_df.columns if col.startswith('target_')]
    target_stats = {}
    for target_col in target_columns:
        if target_col in target_df.columns:
            y = target_df[target_col].dropna()
            if len(y) > 0:
                target_stats[target_col] = {
                    'mean': y.mean(),
                    'std': y.std(),
                    'median': y.median(),
                    'q25': y.quantile(0.25),
                    'q75': y.quantile(0.75),
                    'min': y.min(),
                    'max': y.max()
                }
    target_statistics = target_stats
    # Train ultimate ensemble models
    trained_models = {}
    model_type_counts = {'ultimate_ensemble': 0, 'medium_ensemble': 0, 'single_xgb': 0, 'single_ridge': 0}
    for i, target_col in enumerate(target_columns):
        if target_col not in target_df.columns:
            continue
        y = target_df[target_col]
        valid_mask = ~y.isna()
        if valid_mask.sum() < 10:
            median_val = target_stats.get(target_col, {}).get('median', 0.0)
            trained_models[target_col] = {'type': 'median', 'value': median_val}
            continue
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        try:
            model_info = create_ultimate_ensemble(X_valid, y_valid, target_col)
            trained_models[target_col] = model_info
            model_type_counts[model_info['type']] += 1
        except Exception:
            median_val = target_stats.get(target_col, {}).get('median', 0.0)
            trained_models[target_col] = {'type': 'median', 'value': median_val}
    ultimate_models = trained_models

def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    global ultimate_models, feature_columns, scalers, target_statistics
    # Load models on first call
    if ultimate_models is None:
        load_and_train_ultimate_models()
    # Convert and create features
    test_df = test.to_pandas()
    test_df = create_ultimate_features(test_df)
    # Prepare features
    try:
        X = test_df[feature_columns].fillna(0)
    except Exception:
        # Fallback for missing features
        available_cols = [col for col in feature_columns if col in test_df.columns]
        X = test_df[available_cols].fillna(0) if available_cols else pd.DataFrame([[0] * len(feature_columns)])
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]
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
    raw_predictions = {}
    for target_col in expected_targets:
        if target_col in ultimate_models:
            try:
                model_info = ultimate_models[target_col]
                if model_info['type'] in ['ultimate_ensemble', 'medium_ensemble', 'single_xgb', 'single_ridge']:
                    ensemble_pred = 0.0
                    total_weight = 0.0
                    for model_data in model_info['model_info']['models']:
                        model = model_data['model']
                        scaler_name = model_data['scaler']
                        weight = model_data['weight']
                        # Scale features
                        scaler = model_info['model_info']['scalers'][scaler_name]
                        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
                        # Get prediction
                        pred = model.predict(X_scaled)
                        ensemble_pred += pred[0] * weight
                        total_weight += weight
                    raw_predictions[target_col] = ensemble_pred / total_weight if total_weight > 0 else 0.0
                else:
                    raw_predictions[target_col] = model_info['value']
            except Exception:
                raw_predictions[target_col] = 0.0
        else:
            raw_predictions[target_col] = 0.0
    # Ranking-based post-processing
    if len(raw_predictions) > 1:
        pred_values = np.array(list(raw_predictions.values()))
        # Convert to percentile ranks
        pred_ranks = rankdata(pred_values, method='average') / len(pred_values)
        # Scale using target statistics
        processed_predictions = {}
        for i, target_col in enumerate(expected_targets):
            if target_col in raw_predictions:
                rank = pred_ranks[i]
                if target_col in target_statistics:
                    stats = target_statistics[target_col]
                    # Enhanced quantile mapping
                    if rank <= 0.1:
                        value = stats['min'] + (stats['q25'] - stats['min']) * (rank / 0.1)
                    elif rank <= 0.25:
                        value = stats['q25']
                    elif rank <= 0.5:
                        value = stats['q25'] + (stats['median'] - stats['q25']) * ((rank - 0.25) / 0.25)
                    elif rank <= 0.75:
                        value = stats['median'] + (stats['q75'] - stats['median']) * ((rank - 0.5) / 0.25)
                    elif rank <= 0.9:
                        value = stats['q75'] + (stats['max'] - stats['q75']) * ((rank - 0.75) / 0.15)
                    else:
                        value = stats['max']
                    processed_predictions[target_col] = value
                else:
                    processed_predictions[target_col] = (rank - 0.5) * 0.02
        final_predictions = processed_predictions
    else:
        final_predictions = raw_predictions
    result_df = pl.DataFrame([final_predictions])
    assert isinstance(result_df, (pd.DataFrame, pl.DataFrame))
    assert len(result_df) == 1
    return result_df
# Initialize inference server

inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))