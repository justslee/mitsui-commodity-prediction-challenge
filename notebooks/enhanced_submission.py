"""
Enhanced Submission for Mitsui Commodity Prediction Challenge
Combines PCA factors, automated feature selection, interaction features, and transformer models
"""

import os
import sys
sys.path.append('/kaggle/working')

import pandas as pd
import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.mitsui_inference_server

# Try to import our custom modules
try:
    from src.features.global_factors import GlobalFactorExtractor, InteractionFeatureCreator
    from src.features.feature_selection import AdvancedFeatureSelector
    CUSTOM_MODULES_AVAILABLE = True
except ImportError:
    CUSTOM_MODULES_AVAILABLE = False

# Try to import transformer modules
try:
    import torch
    import torch.nn as nn
    from src.models.transformers import create_transformer_model, CommodityDataset, TransformerTrainer
    TRANSFORMER_AVAILABLE = torch.cuda.is_available()  # Only use if GPU available
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    TRANSFORMER_AVAILABLE = False
    DEVICE = 'cpu'

NUM_TARGET_COLUMNS = 424

# Global storage
enhanced_models = None
feature_columns = None
scalers = None
factor_extractor = None
feature_selector = None


class BuiltInGlobalFactors:
    """Built-in implementation of global factors when custom modules not available."""
    
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.feature_groups = {
            'lme_metals': ['LME_AH_Close', 'LME_CA_Close', 'LME_PB_Close', 'LME_ZS_Close'],
            'fx_majors': ['FX_USDJPY', 'FX_EURUSD', 'FX_GBPUSD', 'FX_USDCAD'],
            'us_equities': ['US_Stock_VT_adj_close', 'US_Stock_VTI_adj_close', 'US_Stock_QQQ_adj_close'],
            'precious_metals': ['JPX_Gold_Standard_Futures_Close', 'JPX_Platinum_Standard_Futures_Close']
        }
        self.pca_models = {}
        self.scalers = {}
    
    def fit_transform(self, df):
        result_df = df.copy()
        
        for group_name, features in self.feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) < 2:
                continue
            
            group_data = df[available_features].fillna(method='ffill').fillna(0)
            
            if group_data.shape[0] < 10:
                continue
            
            # Fit PCA
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(group_data)
            
            n_comp = min(self.n_components, len(available_features))
            pca = PCA(n_components=n_comp, random_state=42)
            factors = pca.fit_transform(scaled_data)
            
            # Store models
            self.pca_models[group_name] = pca
            self.scalers[group_name] = scaler
            
            # Add factors to dataframe
            for i in range(factors.shape[1]):
                result_df[f"{group_name}_factor_{i+1}"] = factors[:, i]
        
        return result_df


class BuiltInInteractionFeatures:
    """Built-in implementation of interaction features."""
    
    def create_features(self, df):
        result_df = df.copy()
        
        # Key interaction features
        interactions = [
            ('JPX_Gold_Standard_Futures_Close', 'JPX_Silver_Standard_Futures_Close', 'ratio'),
            ('LME_CA_Close', 'FX_USDJPY', 'currency_adjusted'),
            ('US_Stock_VIX_adj_close', 'JPX_Gold_Standard_Futures_Close', 'product'),
        ]
        
        for f1, f2, interaction_type in interactions:
            if f1 in df.columns and f2 in df.columns:
                if interaction_type == 'ratio':
                    result_df[f"{f1}_{f2}_ratio"] = df[f1] / (df[f2] + 1e-8)
                elif interaction_type == 'currency_adjusted':
                    result_df[f"{f1}_fx_adjusted"] = df[f1] / (df[f2] + 1e-8)
                elif interaction_type == 'product':
                    result_df[f"{f1}_{f2}_product"] = df[f1] * df[f2]
        
        # Lead/lag features
        lead_lag_pairs = [
            ('FX_USDJPY', 'LME_CA_Close', 1),
            ('US_Stock_VTI_adj_close', 'LME_AL_Close', 1),
        ]
        
        for leader, follower, lag in lead_lag_pairs:
            if leader in df.columns and follower in df.columns:
                result_df[f"{leader}_leads_{follower}_lag{lag}"] = df[leader].shift(lag)
        
        return result_df


def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced features using available implementations."""
    
    if CUSTOM_MODULES_AVAILABLE:
        # Use custom implementations
        print("Using custom feature engineering modules...")
        
        # Global factors
        factor_extractor = GlobalFactorExtractor(n_components=3, use_ica=True)
        df_with_factors = factor_extractor.fit_transform(df)
        
        # Interaction features
        interaction_creator = InteractionFeatureCreator()
        df_with_interactions = interaction_creator.create_interactions(df_with_factors)
        df_final = interaction_creator.create_lead_lag_features(df_with_interactions)
        
    else:
        # Use built-in implementations
        print("Using built-in feature engineering...")
        
        # Global factors
        factor_creator = BuiltInGlobalFactors(n_components=3)
        df_with_factors = factor_creator.fit_transform(df)
        
        # Interaction features
        interaction_creator = BuiltInInteractionFeatures()
        df_final = interaction_creator.create_features(df_with_factors)
    
    # Additional technical indicators
    df_final = create_additional_technical_features(df_final)
    
    return df_final


def create_additional_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional technical indicators."""
    result_df = df.copy()
    
    # Get price columns
    price_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['close', 'price', 'value', 'futures']):
            price_cols.append(col)
    
    price_cols = price_cols[:15]  # Limit for performance
    
    for col in price_cols:
        if col in df.columns:
            # Enhanced technical indicators
            for window in [5, 10, 20, 50]:
                # Moving averages
                result_df[f"{col}_sma_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Relative strength
                result_df[f"{col}_rel_strength_{window}"] = df[col] / result_df[f"{col}_sma_{window}"]
                
                # Percentile position
                result_df[f"{col}_percentile_{window}"] = df[col].rolling(window=window).rank(pct=True)
            
            # RSI
            delta = df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result_df[f"{col}_rsi"] = 100 - (100 / (1 + rs))
            
            # Momentum features
            for period in [5, 10, 20]:
                result_df[f"{col}_momentum_{period}"] = df[col] - df[col].shift(period)
                result_df[f"{col}_pct_change_{period}"] = df[col].pct_change(periods=period)
            
            # Volatility
            returns = df[col].pct_change()
            for window in [10, 20]:
                result_df[f"{col}_volatility_{window}"] = returns.rolling(window=window).std()
    
    # Cross-market features
    if len(price_cols) >= 3:
        # Market index from top assets
        market_index = df[price_cols[:3]].mean(axis=1)
        result_df['market_index'] = market_index
        
        for col in price_cols[:5]:
            if col in df.columns:
                # Beta to market
                returns_asset = df[col].pct_change()
                returns_market = market_index.pct_change()
                
                cov_30d = returns_asset.rolling(30).cov(returns_market)
                var_30d = returns_market.rolling(30).var()
                result_df[f"{col}_beta_30d"] = cov_30d / (var_30d + 1e-8)
                
                # Relative performance
                result_df[f"{col}_rel_to_market"] = df[col] / (market_index + 1e-8)
    
    # Time-based features
    if 'date_id' in df.columns:
        result_df['date_sin_weekly'] = np.sin(2 * np.pi * df['date_id'] / 7)
        result_df['date_cos_weekly'] = np.cos(2 * np.pi * df['date_id'] / 7)
        result_df['date_sin_monthly'] = np.sin(2 * np.pi * df['date_id'] / 30)
        result_df['date_cos_monthly'] = np.cos(2 * np.pi * df['date_id'] / 30)
    
    return result_df


def create_time_series_sequences(X: pd.DataFrame, y: pd.Series, seq_len: int = 20):
    """Create time series sequences for transformer input."""
    if len(X) < seq_len + 1:
        return X.values, y.values
    
    X_sequences = []
    y_sequences = []
    
    for i in range(seq_len, len(X)):
        X_sequences.append(X.iloc[i-seq_len:i].values)
        y_sequences.append(y.iloc[i])
    
    return np.array(X_sequences), np.array(y_sequences)


def select_best_features(X: pd.DataFrame, y: pd.Series, max_features: int = 250) -> list:
    """Select the best features using multiple methods."""
    
    if CUSTOM_MODULES_AVAILABLE:
        try:
            from src.features.feature_selection import select_best_features
            selected_features, _ = select_best_features(X, y, max_features, 'ensemble')
            return selected_features
        except Exception as e:
            print(f"Custom feature selection failed: {e}, using fallback")
    
    # Fallback feature selection
    X_clean = X.select_dtypes(include=[np.number]).fillna(0)
    y_clean = y.fillna(y.median())
    
    # Variance threshold
    from sklearn.feature_selection import VarianceThreshold
    var_selector = VarianceThreshold(threshold=0.01)
    X_var = var_selector.fit_transform(X_clean)
    var_features = X_clean.columns[var_selector.get_support()].tolist()
    
    # Correlation filter
    if len(var_features) > max_features:
        X_var_df = X_clean[var_features]
        corr_matrix = X_var_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        var_features = [f for f in var_features if f not in to_drop]
    
    # Model-based selection
    if len(var_features) > max_features:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        selector = SelectFromModel(rf, threshold='median', max_features=max_features)
        X_selected = X_clean[var_features]
        selector.fit(X_selected, y_clean)
        selected_features = X_selected.columns[selector.get_support()].tolist()
    else:
        selected_features = var_features
    
    print(f"Feature selection: {len(selected_features)} features selected from {X.shape[1]} original")
    return selected_features


def create_enhanced_model(X_valid: pd.DataFrame, y_valid: pd.Series, target_name: str):
    """Create enhanced model with multiple algorithms."""
    n_samples = len(y_valid)
    
    if n_samples >= 500:
        # Large dataset: use sophisticated ensemble
        models = []
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=0
        )
        models.append(('xgb', xgb_model))
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        models.append(('rf', rf_model))
        
        # Ridge regression for stability
        ridge_model = Ridge(alpha=1.0, random_state=42)
        models.append(('ridge', ridge_model))
        
        # Add transformer model if available and GPU enabled
        if TRANSFORMER_AVAILABLE and n_samples >= 1000:  # Only for very large datasets
            try:
                # Create sequences for transformer
                seq_len = min(20, n_samples // 10)  # Adaptive sequence length
                
                # Convert to sequences
                X_seq, y_seq = create_time_series_sequences(X_valid, y_valid, seq_len)
                
                if len(X_seq) >= 50:  # Minimum sequences needed
                    transformer_model = create_transformer_model(
                        input_dim=X_valid.shape[1],
                        num_targets=1,  # Single target per model
                        model_type='commodity',
                        d_model=128,  # Smaller for speed
                        nhead=4,
                        num_layers=3,
                        seq_len=seq_len
                    )
                    
                    # Quick training (limited epochs for speed)
                    trainer = TransformerTrainer(transformer_model, device=DEVICE)
                    
                    # Train for just a few epochs in production
                    dataset = CommodityDataset(X_seq, y_seq.reshape(-1, 1), seq_len=seq_len)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
                    
                    for epoch in range(5):  # Quick training
                        trainer.train_epoch(dataloader)
                    
                    models.append(('transformer', transformer_model, trainer))
            except Exception:
                pass  # Skip transformer if it fails
        
        # Train traditional models
        trained_models = []
        for item in models:
            if len(item) == 2:  # Traditional model
                name, model = item
                try:
                    model.fit(X_valid, y_valid)
                    trained_models.append((name, model))
                except Exception:
                    pass
            else:  # Transformer
                trained_models.append(item)
        
        return {'type': 'ensemble', 'models': trained_models}
        
    elif n_samples >= 100:
        # Medium dataset: use single best model
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_valid, y_valid)
        return {'type': 'xgb', 'model': xgb_model}
        
    else:
        # Small dataset: use regularized linear model
        ridge = Ridge(alpha=2.0, random_state=42)
        ridge.fit(X_valid, y_valid)
        return {'type': 'ridge', 'model': ridge}


def load_and_train_enhanced_models():
    """Load data and train enhanced models."""
    global enhanced_models, feature_columns, scalers
    
    print("Loading training data...")
    train_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
    target_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')
    
    enhanced_df = create_enhanced_features(train_df)
    
    # Get numeric features
    numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
    base_features = [col for col in numeric_cols if col != 'date_id']
    
    # Feature selection for each target
    target_columns = [col for col in target_df.columns if col.startswith('target_')]
    trained_models = {}
    
    # Use representative target for global feature selection
    sample_targets = target_columns[:5]  # Use first 5 targets for feature selection
    
    print("Performing global feature selection...")
    global_features = []
    for target_col in sample_targets:
        if target_col in target_df.columns:
            y = target_df[target_col].dropna()
            if len(y) > 20:
                X = enhanced_df.loc[y.index][base_features]
                selected = select_best_features(X, y, max_features=50)
                global_features.extend(selected)
    
    # Remove duplicates and limit features
    global_features = list(set(global_features))[:200]
    
    if len(global_features) < 50:
        print("Using all base features due to insufficient selection")
        global_features = base_features[:200]
    
    print(f"Global feature selection: {len(global_features)} features selected")
    
    # Prepare final feature set
    X = enhanced_df[global_features].fillna(0)
    feature_columns = X.columns.tolist()
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    scalers = {'feature_scaler': scaler}
    
    # Train models for each target
    print("Training enhanced models...")
    ensemble_count = 0
    xgb_count = 0
    ridge_count = 0
    
    for i, target_col in enumerate(target_columns):
        if (i + 1) % 50 == 0:
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
            model_info = create_enhanced_model(X_valid, y_valid, target_col)
            trained_models[target_col] = model_info
            
            # Count model types
            if model_info['type'] == 'ensemble':
                ensemble_count += 1
            elif model_info['type'] == 'xgb':
                xgb_count += 1
            elif model_info['type'] == 'ridge':
                ridge_count += 1
                
        except Exception as e:
            print(f"Failed to train model for {target_col}: {e}")
            mean_value = y_valid.mean() if len(y_valid) > 0 else 0.0
            trained_models[target_col] = {'type': 'mean', 'value': mean_value}
    
    enhanced_models = trained_models
    
    print(f"Model training completed:")
    print(f"  Ensemble models: {ensemble_count}")
    print(f"  XGBoost models: {xgb_count}")
    print(f"  Ridge models: {ridge_count}")
    print(f"  Mean fallback: {len(trained_models) - ensemble_count - xgb_count - ridge_count}")


def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    """Enhanced prediction function."""
    global enhanced_models, feature_columns, scalers
    
    # Load models on first call
    if enhanced_models is None:
        load_and_train_enhanced_models()
    
    # Convert and create enhanced features
    test_df = test.to_pandas()
    test_enhanced = create_enhanced_features(test_df)
    
    # Prepare features
    try:
        X = test_enhanced[feature_columns].fillna(0)
        X_scaled = pd.DataFrame(
            scalers['feature_scaler'].transform(X),
            columns=X.columns,
            index=X.index
        )
    except Exception:
        # Fallback
        available_cols = [col for col in feature_columns if col in test_enhanced.columns]
        if available_cols:
            X = test_enhanced[available_cols].fillna(0)
            # Pad missing columns
            for col in feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_columns]
        else:
            X = pd.DataFrame([[0] * len(feature_columns)], columns=feature_columns)
        
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
        if target_col in enhanced_models:
            try:
                model_info = enhanced_models[target_col]
                
                if model_info['type'] == 'ensemble':
                    # Ensemble prediction (average)
                    preds = []
                    for item in model_info['models']:
                        try:
                            if len(item) == 2:  # Traditional model
                                _, model = item
                                pred = model.predict(X_scaled)[0]
                                preds.append(pred)
                            elif len(item) == 3:  # Transformer model
                                _, transformer_model, _ = item
                                if TRANSFORMER_AVAILABLE:
                                    # Create sequence for transformer prediction
                                    seq_len = 20  # Use same as training
                                    if len(X_scaled) >= seq_len:
                                        X_seq = X_scaled.iloc[-seq_len:].values.reshape(1, seq_len, -1)
                                        X_tensor = torch.FloatTensor(X_seq).to(DEVICE)
                                        
                                        transformer_model.eval()
                                        with torch.no_grad():
                                            pred_tensor = transformer_model(X_tensor)
                                            pred = pred_tensor.cpu().numpy()[0, 0]
                                            preds.append(pred)
                        except:
                            continue
                    
                    if preds:
                        predictions[target_col] = np.mean(preds)
                    else:
                        predictions[target_col] = 0.0
                        
                elif model_info['type'] in ['xgb', 'ridge']:
                    pred = model_info['model'].predict(X_scaled)
                    predictions[target_col] = pred[0] if len(pred) > 0 else 0.0
                    
                else:  # mean fallback
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