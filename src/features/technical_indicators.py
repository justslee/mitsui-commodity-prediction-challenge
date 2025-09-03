"""
Technical indicators for time series feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """Technical indicators for financial time series."""
    
    def __init__(self):
        self.feature_names = []
    
    def add_rolling_averages(self, df: pd.DataFrame, 
                           price_cols: List[str],
                           windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Add rolling average features."""
        df_result = df.copy()
        
        for col in price_cols:
            if col in df.columns:
                for window in windows:
                    feature_name = f"{col}_ma_{window}"
                    df_result[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                    self.feature_names.append(feature_name)
        
        return df_result
    
    def add_rsi(self, df: pd.DataFrame, price_cols: List[str], period: int = 14) -> pd.DataFrame:
        """Add RSI indicator."""
        df_result = df.copy()
        
        for col in price_cols:
            if col in df.columns:
                delta = df[col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                feature_name = f"{col}_rsi_{period}"
                df_result[feature_name] = 100 - (100 / (1 + rs))
                self.feature_names.append(feature_name)
        
        return df_result
    
    def add_macd(self, df: pd.DataFrame, price_cols: List[str],
                 fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """Add MACD indicator."""
        df_result = df.copy()
        
        for col in price_cols:
            if col in df.columns:
                ema_fast = df[col].ewm(span=fast_period).mean()
                ema_slow = df[col].ewm(span=slow_period).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal_period).mean()
                
                macd_name = f"{col}_macd"
                signal_name = f"{col}_macd_signal"
                hist_name = f"{col}_macd_hist"
                
                df_result[macd_name] = macd_line
                df_result[signal_name] = signal_line
                df_result[hist_name] = macd_line - signal_line
                
                self.feature_names.extend([macd_name, signal_name, hist_name])
        
        return df_result
    
    def add_bollinger_bands(self, df: pd.DataFrame, price_cols: List[str],
                          period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Add Bollinger Bands."""
        df_result = df.copy()
        
        for col in price_cols:
            if col in df.columns:
                rolling_mean = df[col].rolling(window=period).mean()
                rolling_std = df[col].rolling(window=period).std()
                
                upper_name = f"{col}_bb_upper"
                lower_name = f"{col}_bb_lower"
                width_name = f"{col}_bb_width"
                position_name = f"{col}_bb_position"
                
                df_result[upper_name] = rolling_mean + (rolling_std * std_dev)
                df_result[lower_name] = rolling_mean - (rolling_std * std_dev)
                df_result[width_name] = df_result[upper_name] - df_result[lower_name]
                df_result[position_name] = (df[col] - df_result[lower_name]) / df_result[width_name]
                
                self.feature_names.extend([upper_name, lower_name, width_name, position_name])
        
        return df_result
    
    def add_momentum_features(self, df: pd.DataFrame, price_cols: List[str],
                            periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Add momentum and rate of change features."""
        df_result = df.copy()
        
        for col in price_cols:
            if col in df.columns:
                for period in periods:
                    momentum_name = f"{col}_momentum_{period}"
                    roc_name = f"{col}_roc_{period}"
                    
                    df_result[momentum_name] = df[col] - df[col].shift(period)
                    df_result[roc_name] = (df[col] / df[col].shift(period) - 1) * 100
                    
                    self.feature_names.extend([momentum_name, roc_name])
        
        return df_result
    
    def add_volatility_indicators(self, df: pd.DataFrame, price_cols: List[str],
                                windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add volatility indicators."""
        df_result = df.copy()
        
        for col in price_cols:
            if col in df.columns:
                returns = df[col].pct_change()
                
                for window in windows:
                    vol_name = f"{col}_volatility_{window}"
                    df_result[vol_name] = returns.rolling(window=window).std()
                    self.feature_names.append(vol_name)
        
        return df_result
    
    def add_return_features(self, df: pd.DataFrame, price_cols: List[str],
                          periods: List[int] = [1, 2, 5, 10]) -> pd.DataFrame:
        """Add return and differencing features."""
        df_result = df.copy()
        
        for col in price_cols:
            if col in df.columns:
                for period in periods:
                    return_name = f"{col}_return_{period}"
                    log_return_name = f"{col}_log_return_{period}"
                    diff_name = f"{col}_diff_{period}"
                    
                    df_result[return_name] = df[col].pct_change(periods=period)
                    df_result[log_return_name] = np.log(df[col] / df[col].shift(period))
                    df_result[diff_name] = df[col].diff(periods=period)
                    
                    self.feature_names.extend([return_name, log_return_name, diff_name])
        
        return df_result
    
    def create_all_features(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """Create all technical indicators at once."""
        df_result = df.copy()
        
        df_result = self.add_rolling_averages(df_result, price_cols)
        df_result = self.add_rsi(df_result, price_cols)
        df_result = self.add_macd(df_result, price_cols)
        df_result = self.add_bollinger_bands(df_result, price_cols)
        df_result = self.add_momentum_features(df_result, price_cols)
        df_result = self.add_volatility_indicators(df_result, price_cols)
        df_result = self.add_return_features(df_result, price_cols)
        
        return df_result
    
    def get_feature_names(self) -> List[str]:
        """Get list of created feature names."""
        return self.feature_names.copy()


def create_technical_features(df: pd.DataFrame, 
                            feature_categories: Dict[str, List[str]],
                            date_col: str = 'date_id') -> pd.DataFrame:
    """Create technical features for all price columns."""
    
    price_cols = []
    for category, features in feature_categories.items():
        price_cols.extend([f for f in features if f in df.columns and 'Close' in f])
    
    if not price_cols:
        price_cols = [col for col in df.columns if col != date_col and df[col].dtype in ['float64', 'int64']]
    
    indicators = TechnicalIndicators()
    df_with_features = indicators.create_all_features(df, price_cols[:10])  # Limit to first 10 to avoid memory issues
    
    return df_with_features


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import load_competition_data
    
    loader, data = load_competition_data()
    train_df = data['train']
    feature_categories = loader.get_feature_categories()
    
    print("Creating technical indicators...")
    df_with_tech = create_technical_features(train_df, feature_categories)
    
    print(f"Original features: {len(train_df.columns)}")
    print(f"With technical indicators: {len(df_with_tech.columns)}")
    print(f"New features added: {len(df_with_tech.columns) - len(train_df.columns)}")
    
    new_features = [col for col in df_with_tech.columns if col not in train_df.columns]
    print(f"\nSample new features: {new_features[:10]}")