"""
Ranking-Optimized Submission for Mitsui Commodity Prediction Challenge
Optimized for Sharpe ratio of Spearman rank correlations
"""

import os

import pandas as pd

import polars as pl  # type: ignore

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler

import xgboost as xgb
from scipy.stats import rankdata

import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.mitsui_inference_server
NUM_TARGET_COLUMNS = 424
# Global storage
models = None
feature_columns = None
scalers = None
target_statistics = None

def create_ranking_features(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    # Get price/value columns
    price_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['close', 'price', 'value', 'futures']):
            price_cols.append(col)
    price_cols = price_cols[:20]  # Focus on most important
    # Ranking-focused features
    for col in price_cols:
        if col in df.columns:
            # Relative performance vs rolling mean
            for window in [5, 10, 20, 50]:
                rolling_mean = df[col].rolling(window=window).mean()
                df_result[f"{col}_rel_perf_{window}"] = (df[col] - rolling_mean) / (rolling_mean + 1e-8)
            # Percentile position within rolling window
            for window in [10, 20, 50]:
                df_result[f"{col}_percentile_{window}"] = df[col].rolling(window=window).rank(pct=True)
            # Momentum strength
            for period in [1, 5, 10, 20]:
                momentum = df[col] - df[col].shift(period)
                df_result[f"{col}_momentum_{period}"] = momentum / (df[col].shift(period) + 1e-8)
            # Volatility-adjusted returns
            returns = df[col].pct_change()
            for window in [10, 20]:
                vol = returns.rolling(window=window).std()
                df_result[f"{col}_vol_adj_ret_{window}"] = returns / (vol + 1e-8)
            # RSI as ranking indicator
            delta = df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_result[f"{col}_rsi"] = 100 - (100 / (1 + rs))
            # Price strength relative to cross-section
            if len(price_cols) > 1:
                # Calculate rank within all assets at each time point
                temp_ranks = df[price_cols].rank(axis=1, pct=True)
                if col in temp_ranks.columns:
                    df_result[f"{col}_cross_rank"] = temp_ranks[col]
    # Cross-asset relative features
    if len(price_cols) >= 2:
        # Pairwise relative performance
        for i in range(min(5, len(price_cols))):
            for j in range(i+1, min(5, len(price_cols))):
                col1, col2 = price_cols[i], price_cols[j]
                if col1 in df.columns and col2 in df.columns:
                    # Relative strength
                    df_result[f"rel_strength_{i}_{j}"] = df[col1] / (df[col2] + 1e-8)
                    # Rolling correlation (ranking stability indicator)
                    df_result[f"rolling_corr_{i}_{j}"] = df[col1].rolling(30).corr(df[col2])
    # Market regime indicators
    if len(price_cols) >= 3:
        # Create market index from top assets
        market_index = df[price_cols[:3]].mean(axis=1)
        for col in price_cols[:10]:
            if col in df.columns:
                # Beta to market
                returns_asset = df[col].pct_change()
                returns_market = market_index.pct_change()
                # Rolling beta calculation
                cov_window = returns_asset.rolling(50).cov(returns_market)
                var_window = returns_market.rolling(50).var()
                df_result[f"{col}_beta"] = cov_window / (var_window + 1e-8)
                # Relative performance to market
                df_result[f"{col}_vs_market"] = df[col] / (market_index + 1e-8)
    # Temporal features for ranking consistency
    if 'date_id' in df.columns:
        # Cyclical patterns
        df_result['date_sin_weekly'] = np.sin(2 * np.pi * df['date_id'] / 7)
        df_result['date_cos_weekly'] = np.cos(2 * np.pi * df['date_id'] / 7)
        df_result['date_sin_monthly'] = np.sin(2 * np.pi * df['date_id'] / 30)
        df_result['date_cos_monthly'] = np.cos(2 * np.pi * df['date_id'] / 30)
        # Trend components
        df_result['date_trend'] = df['date_id'] / df['date_id'].max()
        df_result['date_squared'] = (df['date_id'] / df['date_id'].max()) ** 2
    return df_result

def create_ranking_optimized_model(X_valid: pd.DataFrame, y_valid: pd.Series, target_name: str):
    n_samples = len(y_valid)
    # Convert targets to rankings for training (ranking-aware training)
    y_ranks = rankdata(y_valid, method='average') / len(y_valid)
    if n_samples >= 500:
        # Large dataset: use ensemble optimized for ranking
        # Random Forest with ranking focus
        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        # Train on both raw values and ranks
        rf.fit(X_valid, y_valid * 0.7 + y_ranks * 0.3)  # Weighted combination
        return {'type': 'rf_ranking', 'model': rf}
    elif n_samples >= 100:
        # Medium dataset: XGBoost with ranking objective
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,  # L1 regularization for feature selection
            reg_lambda=0.1,  # L2 regularization
            random_state=42,
            verbosity=0
        )
        # Use ranking-aware training
        xgb_model.fit(X_valid, y_ranks)  # Train directly on ranks
        return {'type': 'xgb_ranking', 'model': xgb_model}
    else:
        # Small dataset: Ridge with ranking preprocessing
        ridge = Ridge(alpha=2.0, random_state=42)  # Higher regularization
        ridge.fit(X_valid, y_ranks)  # Train on ranks
        return {'type': 'ridge_ranking', 'model': ridge}

def load_and_train_ranking_models():
    global models, feature_columns, scalers, target_statistics
    # Load data
    train_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
    target_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')
    # Create ranking-focused features
    train_df = create_ranking_features(train_df)
    # Feature selection focused on ranking predictors
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'date_id']
    # Prefer relative/ranking features
    ranking_features = [col for col in feature_cols if any(keyword in col.lower() 
                       for keyword in ['rel_', 'rank', 'percentile', 'momentum', 'rsi', 'beta', 'corr'])]
    other_features = [col for col in feature_cols if col not in ranking_features]
    # Prioritize ranking features, then add others
    selected_features = ranking_features + other_features[:200]  # Limit total features
    # Remove highly correlated features
    X_temp = train_df[selected_features].fillna(0)
    correlation_matrix = X_temp.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    feature_cols = [col for col in selected_features if col not in high_corr_features]
    X = train_df[feature_cols].fillna(0)
    feature_columns = X.columns.tolist()
    # Use robust scaling for ranking stability
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    scalers = {'feature_scaler': scaler}
    # Collect target statistics for ranking normalization
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
                    'q75': y.quantile(0.75)
                }
    target_statistics = target_stats
    # Train ranking-optimized models
    trained_models = {}
    rf_count = 0
    xgb_count = 0
    ridge_count = 0
    for i, target_col in enumerate(target_columns):
        if target_col not in target_df.columns:
            continue
        y = target_df[target_col]
        valid_mask = ~y.isna()
        if valid_mask.sum() < 10:
            # Use median for consistency
            median_val = target_stats.get(target_col, {}).get('median', 0.0)
            trained_models[target_col] = {'type': 'median', 'value': median_val}
            continue
        X_valid = X_scaled[valid_mask]
        y_valid = y[valid_mask]
        try:
            model_info = create_ranking_optimized_model(X_valid, y_valid, target_col)
            trained_models[target_col] = model_info
            if model_info['type'] == 'rf_ranking':
                rf_count += 1
            elif model_info['type'] == 'xgb_ranking':
                xgb_count += 1
            elif model_info['type'] == 'ridge_ranking':
                ridge_count += 1
        except Exception:
            median_val = target_stats.get(target_col, {}).get('median', 0.0)
            trained_models[target_col] = {'type': 'median', 'value': median_val}
    models = trained_models

def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    global models, feature_columns, scalers, target_statistics
    # Load models on first call
    if models is None:
        load_and_train_ranking_models()
    # Convert and create features
    test_df = test.to_pandas()
    test_df = create_ranking_features(test_df)
    # Prepare features
    try:
        X = test_df[feature_columns].fillna(0)
        X_scaled = pd.DataFrame(
            scalers['feature_scaler'].transform(X),
            columns=X.columns,
            index=X.index
        )
    except Exception:
        # Fallback
        available_cols = [col for col in feature_columns if col in test_df.columns]
        X = test_df[available_cols].fillna(0) if available_cols else pd.DataFrame([[0] * len(feature_columns)])
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
    raw_predictions = {}
    for target_col in expected_targets:
        if target_col in models:
            try:
                model_info = models[target_col]
                if model_info['type'] in ['rf_ranking', 'xgb_ranking', 'ridge_ranking']:
                    pred = model_info['model'].predict(X_scaled)
                    raw_predictions[target_col] = pred[0] if len(pred) > 0 else 0.0
                else:
                    raw_predictions[target_col] = model_info['value']
            except Exception:
                raw_predictions[target_col] = 0.0
        else:
            raw_predictions[target_col] = 0.0
    # Post-process predictions for ranking consistency
    if len(raw_predictions) > 1:
        # Apply ranking-based normalization
        pred_values = np.array(list(raw_predictions.values()))
        # Convert to percentile ranks for consistency
        pred_ranks = rankdata(pred_values, method='average') / len(pred_values)
        # Scale back to reasonable ranges using target statistics
        processed_predictions = {}
        for i, target_col in enumerate(expected_targets):
            if target_col in raw_predictions:
                rank = pred_ranks[i]
                # Use target statistics to scale appropriately
                if target_col in target_statistics:
                    stats = target_statistics[target_col]
                    # Map rank to value using quantiles
                    if rank <= 0.25:
                        value = stats['q25']
                    elif rank <= 0.75:
                        value = stats['median']
                    else:
                        value = stats['q75']
                    processed_predictions[target_col] = value
                else:
                    # Fallback: center around 0 with rank-based scaling
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