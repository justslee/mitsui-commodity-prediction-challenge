"""
Time series specific feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from scipy import stats
from sklearn.decomposition import PCA


class TimeSeriesFeatures:
    """Time series feature engineering."""
    
    def __init__(self):
        self.feature_names = []
    
    def add_lag_features(self, df: pd.DataFrame,
                        columns: List[str],
                        lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged features."""
        df_result = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    lag_name = f"{col}_lag_{lag}"
                    df_result[lag_name] = df[col].shift(lag)
                    self.feature_names.append(lag_name)
        
        return df_result
    
    def add_rolling_stats(self, df: pd.DataFrame,
                         columns: List[str],
                         windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add rolling statistical features."""
        df_result = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    mean_name = f"{col}_roll_mean_{window}"
                    std_name = f"{col}_roll_std_{window}"
                    min_name = f"{col}_roll_min_{window}"
                    max_name = f"{col}_roll_max_{window}"
                    skew_name = f"{col}_roll_skew_{window}"
                    
                    df_result[mean_name] = df[col].rolling(window=window).mean()
                    df_result[std_name] = df[col].rolling(window=window).std()
                    df_result[min_name] = df[col].rolling(window=window).min()
                    df_result[max_name] = df[col].rolling(window=window).max()
                    df_result[skew_name] = df[col].rolling(window=window).skew()
                    
                    self.feature_names.extend([mean_name, std_name, min_name, max_name, skew_name])
        
        return df_result
    
    def add_seasonal_features(self, df: pd.DataFrame,
                            date_col: str = 'date_id') -> pd.DataFrame:
        """Add seasonal and cyclical features."""
        df_result = df.copy()
        
        if date_col in df.columns:
            df_result[f'{date_col}_sin_weekly'] = np.sin(2 * np.pi * df[date_col] / 7)
            df_result[f'{date_col}_cos_weekly'] = np.cos(2 * np.pi * df[date_col] / 7)
            df_result[f'{date_col}_sin_monthly'] = np.sin(2 * np.pi * df[date_col] / 30)
            df_result[f'{date_col}_cos_monthly'] = np.cos(2 * np.pi * df[date_col] / 30)
            df_result[f'{date_col}_sin_quarterly'] = np.sin(2 * np.pi * df[date_col] / 90)
            df_result[f'{date_col}_cos_quarterly'] = np.cos(2 * np.pi * df[date_col] / 90)
            
            seasonal_features = [
                f'{date_col}_sin_weekly', f'{date_col}_cos_weekly',
                f'{date_col}_sin_monthly', f'{date_col}_cos_monthly',
                f'{date_col}_sin_quarterly', f'{date_col}_cos_quarterly'
            ]
            self.feature_names.extend(seasonal_features)
        
        return df_result
    
    def add_regime_features(self, df: pd.DataFrame,
                          columns: List[str],
                          windows: List[int] = [20, 50]) -> pd.DataFrame:
        """Add market regime detection features."""
        df_result = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    volatility_regime_name = f"{col}_vol_regime_{window}"
                    trend_regime_name = f"{col}_trend_regime_{window}"
                    momentum_regime_name = f"{col}_momentum_regime_{window}"
                    
                    returns = df[col].pct_change(fill_method=None)
                    rolling_vol = returns.rolling(window=window).std()
                    vol_threshold = rolling_vol.rolling(window=window*2).median()
                    df_result[volatility_regime_name] = (rolling_vol > vol_threshold).astype(int)
                    
                    rolling_mean = df[col].rolling(window=window).mean()
                    df_result[trend_regime_name] = (df[col] > rolling_mean).astype(int)
                    
                    momentum = df[col] - df[col].shift(window//4)
                    df_result[momentum_regime_name] = (momentum > 0).astype(int)
                    
                    self.feature_names.extend([volatility_regime_name, trend_regime_name, momentum_regime_name])
        
        return df_result
    
    def add_fourier_features(self, df: pd.DataFrame,
                           columns: List[str],
                           n_components: int = 5) -> pd.DataFrame:
        """Add Fourier transform features for cyclical patterns."""
        df_result = df.copy()
        
        for col in columns:
            if col in df.columns and len(df[col].dropna()) > 50:
                values = df[col].fillna(method='ffill').fillna(method='bfill')
                
                fft_values = np.fft.fft(values)
                freqs = np.fft.fftfreq(len(values))
                
                for i in range(1, min(n_components + 1, len(fft_values) // 2)):
                    real_name = f"{col}_fft_real_{i}"
                    imag_name = f"{col}_fft_imag_{i}"
                    mag_name = f"{col}_fft_mag_{i}"
                    
                    df_result[real_name] = np.real(fft_values[i])
                    df_result[imag_name] = np.imag(fft_values[i])
                    df_result[mag_name] = np.abs(fft_values[i])
                    
                    self.feature_names.extend([real_name, imag_name, mag_name])
        
        return df_result
    
    def add_statistical_features(self, df: pd.DataFrame,
                               columns: List[str],
                               windows: List[int] = [10, 20]) -> pd.DataFrame:
        """Add statistical time series features."""
        df_result = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    percentile_25_name = f"{col}_p25_{window}"
                    percentile_75_name = f"{col}_p75_{window}"
                    iqr_name = f"{col}_iqr_{window}"
                    zscore_name = f"{col}_zscore_{window}"
                    
                    df_result[percentile_25_name] = df[col].rolling(window=window).quantile(0.25)
                    df_result[percentile_75_name] = df[col].rolling(window=window).quantile(0.75)
                    df_result[iqr_name] = df_result[percentile_75_name] - df_result[percentile_25_name]
                    
                    rolling_mean = df[col].rolling(window=window).mean()
                    rolling_std = df[col].rolling(window=window).std()
                    df_result[zscore_name] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
                    
                    self.feature_names.extend([percentile_25_name, percentile_75_name, iqr_name, zscore_name])
        
        return df_result
    
    def add_change_point_features(self, df: pd.DataFrame,
                                columns: List[str],
                                window: int = 20) -> pd.DataFrame:
        """Add change point detection features."""
        df_result = df.copy()
        
        for col in columns:
            if col in df.columns:
                change_point_name = f"{col}_change_point_{window}"
                
                rolling_mean_before = df[col].rolling(window=window).mean()
                rolling_mean_after = df[col].shift(-window).rolling(window=window).mean()
                
                change_magnitude = np.abs(rolling_mean_after - rolling_mean_before)
                rolling_std = df[col].rolling(window=window*2).std()
                
                df_result[change_point_name] = change_magnitude / (rolling_std + 1e-8)
                self.feature_names.append(change_point_name)
        
        return df_result
    
    def create_all_features(self, df: pd.DataFrame,
                          feature_categories: Dict[str, List[str]],
                          max_features_per_category: int = 3) -> pd.DataFrame:
        """Create all time series features."""
        df_result = df.copy()
        
        important_cols = []
        for category, features in feature_categories.items():
            close_cols = [f for f in features if 'Close' in f and f in df.columns]
            important_cols.extend(close_cols[:max_features_per_category])
        
        if not important_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            important_cols = [c for c in numeric_cols if c != 'date_id'][:10]
        
        df_result = self.add_lag_features(df_result, important_cols[:5], [1, 2, 5])
        df_result = self.add_rolling_stats(df_result, important_cols[:3], [5, 20])
        df_result = self.add_seasonal_features(df_result)
        df_result = self.add_regime_features(df_result, important_cols[:3])
        df_result = self.add_statistical_features(df_result, important_cols[:3])
        df_result = self.add_change_point_features(df_result, important_cols[:2])
        
        return df_result
    
    def get_feature_names(self) -> List[str]:
        """Get list of created feature names."""
        return self.feature_names.copy()


def create_time_series_features(df: pd.DataFrame,
                              feature_categories: Dict[str, List[str]]) -> pd.DataFrame:
    """Create time series features for the dataset."""
    
    ts_features = TimeSeriesFeatures()
    df_with_features = ts_features.create_all_features(df, feature_categories)
    
    return df_with_features


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import load_competition_data
    
    loader, data = load_competition_data()
    train_df = data['train']
    feature_categories = loader.get_feature_categories()
    
    print("Creating time series features...")
    df_with_ts = create_time_series_features(train_df, feature_categories)
    
    print(f"Original features: {len(train_df.columns)}")
    print(f"With time series features: {len(df_with_ts.columns)}")
    print(f"New features added: {len(df_with_ts.columns) - len(train_df.columns)}")
    
    new_features = [col for col in df_with_ts.columns if col not in train_df.columns]
    print(f"\nSample new features: {new_features[:15]}")