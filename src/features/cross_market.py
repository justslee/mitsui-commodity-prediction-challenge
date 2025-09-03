"""
Cross-market features for multi-asset analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from itertools import combinations


class CrossMarketFeatures:
    """Cross-market and cross-asset feature engineering."""
    
    def __init__(self):
        self.feature_names = []
    
    def add_currency_adjusted_prices(self, df: pd.DataFrame, 
                                   price_cols: List[str],
                                   fx_mapping: Dict[str, str] = None) -> pd.DataFrame:
        """Add currency-adjusted prices using FX rates."""
        df_result = df.copy()
        
        if fx_mapping is None:
            fx_mapping = {
                'JPX': 'FX_USDJPY',
                'LME': 'FX_GBPUSD'
            }
        
        for col in price_cols:
            if col in df.columns:
                if 'JPX' in col and 'FX_USDJPY' in df.columns:
                    adj_name = f"{col}_usd_adj"
                    df_result[adj_name] = df[col] / df['FX_USDJPY']
                    self.feature_names.append(adj_name)
                
                elif 'LME' in col and 'FX_GBPUSD' in df.columns:
                    adj_name = f"{col}_usd_adj"
                    df_result[adj_name] = df[col] * df['FX_GBPUSD']
                    self.feature_names.append(adj_name)
        
        return df_result
    
    def add_correlation_features(self, df: pd.DataFrame,
                               feature_categories: Dict[str, List[str]],
                               window: int = 20) -> pd.DataFrame:
        """Add rolling correlation features between asset classes."""
        df_result = df.copy()
        
        category_representatives = {}
        for category, features in feature_categories.items():
            close_cols = [f for f in features if 'Close' in f and f in df.columns]
            if close_cols:
                category_representatives[category] = close_cols[0]
        
        for cat1, cat2 in combinations(category_representatives.keys(), 2):
            col1 = category_representatives[cat1]
            col2 = category_representatives[cat2]
            
            if col1 in df.columns and col2 in df.columns:
                corr_name = f"corr_{cat1}_{cat2}_{window}"
                df_result[corr_name] = df[col1].rolling(window=window).corr(df[col2])
                self.feature_names.append(corr_name)
        
        return df_result
    
    def add_spread_features(self, df: pd.DataFrame,
                          asset_pairs: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """Add spread features between related assets."""
        df_result = df.copy()
        
        if asset_pairs is None:
            asset_pairs = []
            lme_cols = [c for c in df.columns if 'LME_' in c and 'Close' in c]
            jpx_gold_cols = [c for c in df.columns if 'JPX_Gold' in c and 'Close' in c]
            
            for lme_col in lme_cols[:2]:
                for jpx_col in jpx_gold_cols[:2]:
                    asset_pairs.append((lme_col, jpx_col))
        
        for col1, col2 in asset_pairs:
            if col1 in df.columns and col2 in df.columns:
                spread_name = f"spread_{col1}_{col2}".replace('_Close', '')
                ratio_name = f"ratio_{col1}_{col2}".replace('_Close', '')
                
                df_result[spread_name] = df[col1] - df[col2]
                df_result[ratio_name] = df[col1] / (df[col2] + 1e-8)
                
                self.feature_names.extend([spread_name, ratio_name])
        
        return df_result
    
    def add_sector_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add US market sector-based indicators."""
        df_result = df.copy()
        
        sector_mapping = {
            'metals': ['US_Stock_FCX', 'US_Stock_NEM', 'US_Stock_GOLD'],
            'energy': ['US_Stock_XLE', 'US_Stock_XOM', 'US_Stock_CVX'],
            'materials': ['US_Stock_XLB', 'US_Stock_CAT'],
            'bonds': ['US_Stock_AGG', 'US_Stock_TIP', 'US_Stock_IEF']
        }
        
        for sector, tickers in sector_mapping.items():
            available_tickers = [f"{t}_adj_close" for t in tickers if f"{t}_adj_close" in df.columns]
            
            if available_tickers:
                sector_avg_name = f"sector_{sector}_avg"
                sector_vol_name = f"sector_{sector}_volatility"
                
                sector_values = df[available_tickers].mean(axis=1)
                df_result[sector_avg_name] = sector_values
                df_result[sector_vol_name] = df[available_tickers].std(axis=1)
                
                self.feature_names.extend([sector_avg_name, sector_vol_name])
        
        return df_result
    
    def add_cross_correlations(self, df: pd.DataFrame,
                             base_assets: List[str] = None,
                             target_assets: List[str] = None,
                             lags: List[int] = [1, 2, 5]) -> pd.DataFrame:
        """Add lagged cross-correlations between assets."""
        df_result = df.copy()
        
        if base_assets is None:
            base_assets = [c for c in df.columns if 'LME_' in c and 'Close' in c]
        
        if target_assets is None:
            target_assets = [c for c in df.columns if 'US_Stock_GLD' in c and 'close' in c]
        
        for base_asset in base_assets[:2]:
            for target_asset in target_assets[:2]:
                if base_asset in df.columns and target_asset in df.columns:
                    for lag in lags:
                        xcorr_name = f"xcorr_{base_asset}_{target_asset}_lag_{lag}".replace('_Close', '').replace('_adj_close', '')
                        
                        base_shifted = df[base_asset].shift(lag)
                        correlation = base_shifted.rolling(window=20, min_periods=10).corr(df[target_asset])
                        df_result[xcorr_name] = correlation
                        
                        self.feature_names.append(xcorr_name)
        
        return df_result
    
    def add_market_regime_features(self, df: pd.DataFrame,
                                 benchmark_col: str = None,
                                 windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """Add market regime indicators."""
        df_result = df.copy()
        
        if benchmark_col is None:
            benchmark_candidates = ['US_Stock_VT_adj_close', 'US_Stock_ACWI_adj_close']
            benchmark_col = next((col for col in benchmark_candidates if col in df.columns), None)
        
        if benchmark_col and benchmark_col in df.columns:
            for window in windows:
                returns = df[benchmark_col].pct_change()
                
                vol_regime_name = f"vol_regime_{window}"
                trend_regime_name = f"trend_regime_{window}"
                
                rolling_vol = returns.rolling(window=window).std()
                vol_threshold = rolling_vol.rolling(window=window*2).median()
                df_result[vol_regime_name] = (rolling_vol > vol_threshold).astype(int)
                
                rolling_mean = df[benchmark_col].rolling(window=window).mean()
                df_result[trend_regime_name] = (df[benchmark_col] > rolling_mean).astype(int)
                
                self.feature_names.extend([vol_regime_name, trend_regime_name])
        
        return df_result
    
    def create_all_features(self, df: pd.DataFrame,
                          feature_categories: Dict[str, List[str]]) -> pd.DataFrame:
        """Create all cross-market features."""
        df_result = df.copy()
        
        price_cols = []
        for features in feature_categories.values():
            price_cols.extend([f for f in features if 'Close' in f and f in df.columns])
        
        df_result = self.add_currency_adjusted_prices(df_result, price_cols)
        df_result = self.add_correlation_features(df_result, feature_categories)
        df_result = self.add_spread_features(df_result)
        df_result = self.add_sector_indicators(df_result)
        df_result = self.add_cross_correlations(df_result)
        df_result = self.add_market_regime_features(df_result)
        
        return df_result
    
    def get_feature_names(self) -> List[str]:
        """Get list of created feature names."""
        return self.feature_names.copy()


def create_cross_market_features(df: pd.DataFrame,
                               feature_categories: Dict[str, List[str]]) -> pd.DataFrame:
    """Create cross-market features for the dataset."""
    
    cross_market = CrossMarketFeatures()
    df_with_features = cross_market.create_all_features(df, feature_categories)
    
    return df_with_features


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import load_competition_data
    
    loader, data = load_competition_data()
    train_df = data['train']
    feature_categories = loader.get_feature_categories()
    
    print("Creating cross-market features...")
    df_with_cross_market = create_cross_market_features(train_df, feature_categories)
    
    print(f"Original features: {len(train_df.columns)}")
    print(f"With cross-market features: {len(df_with_cross_market.columns)}")
    print(f"New features added: {len(df_with_cross_market.columns) - len(train_df.columns)}")
    
    new_features = [col for col in df_with_cross_market.columns if col not in train_df.columns]
    print(f"\nSample new features: {new_features[:10]}")