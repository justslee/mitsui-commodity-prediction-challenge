"""
Global Factor Engineering using PCA and Factor Analysis
Creates interpretable global factors from commodity groups to reduce noise and improve generalization.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GlobalFactorExtractor:
    """Extract global factors from commodity groups using PCA and ICA."""
    
    def __init__(self, n_components: int = 3, use_ica: bool = True):
        self.n_components = n_components
        self.use_ica = use_ica
        self.factor_models = {}
        self.scalers = {}
        self.feature_groups = self._define_feature_groups()
        
    def _define_feature_groups(self) -> Dict[str, List[str]]:
        """Define commodity groups for factor extraction."""
        return {
            'lme_metals': [
                'LME_AH_Close', 'LME_CA_Close', 'LME_PB_Close', 'LME_ZS_Close',
                'LME_NI_Close', 'LME_SN_Close', 'LME_AL_Close'
            ],
            'precious_metals': [
                'JPX_Gold_Standard_Futures_Close', 'JPX_Platinum_Standard_Futures_Close',
                'JPX_Silver_Standard_Futures_Close', 'JPX_Palladium_Standard_Futures_Close'
            ],
            'fx_majors': [
                'FX_USDJPY', 'FX_EURUSD', 'FX_GBPUSD', 'FX_USDCAD',
                'FX_AUDUSD', 'FX_NZDUSD', 'FX_USDCHF'
            ],
            'us_equities': [
                'US_Stock_VT_adj_close', 'US_Stock_VTI_adj_close', 'US_Stock_VYM_adj_close',
                'US_Stock_IEMG_adj_close', 'US_Stock_VEA_adj_close', 'US_Stock_QQQ_adj_close'
            ],
            'energy': [
                'US_Stock_XLE_adj_close', 'US_Stock_VDE_adj_close', 
                'US_Stock_ICLN_adj_close'  # Clean energy
            ],
            'volatility': [
                'US_Stock_VIX_adj_close', 'US_Stock_UVXY_adj_close'
            ]
        }
    
    def fit(self, df: pd.DataFrame) -> 'GlobalFactorExtractor':
        """Fit PCA/ICA models on each commodity group."""
        for group_name, features in self.feature_groups.items():
            # Get available features for this group
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) < 2:
                continue
                
            # Extract data and handle missing values
            group_data = df[available_features].fillna(method='ffill').fillna(0)
            
            if group_data.shape[0] < 10:  # Need minimum samples
                continue
                
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(group_data)
            
            # Fit PCA
            n_comp = min(self.n_components, len(available_features), scaled_data.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=42)
            pca.fit(scaled_data)
            
            # Optionally fit ICA for independent components
            ica = None
            if self.use_ica and n_comp >= 2:
                try:
                    ica = FastICA(n_components=n_comp, random_state=42, max_iter=1000)
                    ica.fit(scaled_data)
                except:
                    ica = None
            
            # Store models
            self.factor_models[group_name] = {
                'pca': pca,
                'ica': ica,
                'features': available_features,
                'explained_variance': pca.explained_variance_ratio_
            }
            self.scalers[group_name] = scaler
            
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted factor models."""
        result_df = df.copy()
        
        for group_name, model_info in self.factor_models.items():
            features = model_info['features']
            pca = model_info['pca']
            ica = model_info['ica']
            scaler = self.scalers[group_name]
            
            # Get available features and scale
            group_data = df[features].fillna(method='ffill').fillna(0)
            scaled_data = scaler.transform(group_data)
            
            # PCA factors
            pca_factors = pca.transform(scaled_data)
            for i in range(pca_factors.shape[1]):
                factor_name = f"{group_name}_pca_factor_{i+1}"
                result_df[factor_name] = pca_factors[:, i]
            
            # ICA factors if available
            if ica is not None:
                try:
                    ica_factors = ica.transform(scaled_data)
                    for i in range(ica_factors.shape[1]):
                        factor_name = f"{group_name}_ica_factor_{i+1}"
                        result_df[factor_name] = ica_factors[:, i]
                except:
                    pass
                    
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def get_factor_interpretation(self) -> Dict[str, Dict]:
        """Get interpretation of factors based on loadings."""
        interpretations = {}
        
        for group_name, model_info in self.factor_models.items():
            pca = model_info['pca']
            features = model_info['features']
            
            factor_loadings = pd.DataFrame(
                pca.components_,
                columns=features,
                index=[f'Factor_{i+1}' for i in range(pca.components_.shape[0])]
            )
            
            # Interpret each factor
            factor_meanings = {}
            for i, factor in enumerate(factor_loadings.index):
                loadings = factor_loadings.loc[factor]
                top_positive = loadings.nlargest(3).index.tolist()
                top_negative = loadings.nsmallest(3).index.tolist()
                
                factor_meanings[factor] = {
                    'explained_variance': model_info['explained_variance'][i],
                    'top_positive_loadings': top_positive,
                    'top_negative_loadings': top_negative,
                    'interpretation': self._interpret_factor(group_name, top_positive, top_negative)
                }
            
            interpretations[group_name] = {
                'factors': factor_meanings,
                'total_explained_variance': sum(model_info['explained_variance'])
            }
            
        return interpretations
    
    def _interpret_factor(self, group_name: str, positive_features: List[str], 
                         negative_features: List[str]) -> str:
        """Provide human-readable interpretation of factors."""
        if group_name == 'lme_metals':
            if any('AL' in f or 'CA' in f for f in positive_features):
                return "Industrial metals strength (aluminum, copper driven)"
            elif any('AU' in f or 'AG' in f for f in positive_features):
                return "Precious metals factor"
            else:
                return "General metals complex factor"
                
        elif group_name == 'fx_majors':
            if 'FX_USDJPY' in positive_features and 'FX_EURUSD' in negative_features:
                return "USD strength factor (risk-off)"
            elif 'FX_AUDUSD' in positive_features:
                return "Risk-on currency factor (commodity currencies)"
            else:
                return "General USD factor"
                
        elif group_name == 'us_equities':
            if 'QQQ' in str(positive_features):
                return "Tech/Growth factor"
            elif 'VYM' in str(positive_features):
                return "Value/Dividend factor"
            else:
                return "General equity market factor"
                
        elif group_name == 'volatility':
            return "Market stress/volatility factor"
            
        else:
            return f"General {group_name} factor"


class InteractionFeatureCreator:
    """Create interaction features and lead/lag relationships."""
    
    def __init__(self):
        self.interaction_pairs = self._define_interactions()
        self.lead_lag_pairs = self._define_lead_lag_relationships()
    
    def _define_interactions(self) -> List[Tuple[str, str, str]]:
        """Define meaningful interaction pairs (feature1, feature2, interaction_type)."""
        return [
            # Precious metals ratios
            ('JPX_Gold_Standard_Futures_Close', 'JPX_Silver_Standard_Futures_Close', 'ratio'),
            ('JPX_Gold_Standard_Futures_Close', 'JPX_Platinum_Standard_Futures_Close', 'ratio'),
            
            # Currency-adjusted metal prices
            ('LME_CA_Close', 'FX_USDJPY', 'currency_adjusted'),
            ('LME_AL_Close', 'FX_USDJPY', 'currency_adjusted'),
            
            # Energy spreads
            ('US_Stock_XLE_adj_close', 'US_Stock_VDE_adj_close', 'ratio'),
            
            # Volatility interactions
            ('US_Stock_VIX_adj_close', 'JPX_Gold_Standard_Futures_Close', 'product'),
            ('US_Stock_VIX_adj_close', 'FX_USDJPY', 'product'),
            
            # Cross-market momentum
            ('US_Stock_VTI_adj_close', 'LME_CA_Close', 'correlation'),
        ]
    
    def _define_lead_lag_relationships(self) -> List[Tuple[str, str, int]]:
        """Define lead/lag relationships (leader, follower, lag_days)."""
        return [
            # USD leading commodity prices
            ('FX_USDJPY', 'LME_CA_Close', 1),
            ('FX_USDJPY', 'JPX_Gold_Standard_Futures_Close', 1),
            
            # Equity leading commodities
            ('US_Stock_VTI_adj_close', 'LME_AL_Close', 1),
            ('US_Stock_XLE_adj_close', 'LME_CA_Close', 2),  # Energy leading industrial metals
            
            # VIX leading safe havens
            ('US_Stock_VIX_adj_close', 'JPX_Gold_Standard_Futures_Close', 1),
            ('US_Stock_VIX_adj_close', 'FX_USDJPY', 1),
            
            # Cross-metal relationships
            ('LME_CA_Close', 'LME_AL_Close', 1),  # Copper leading aluminum
        ]
    
    def create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        result_df = df.copy()
        
        for feature1, feature2, interaction_type in self.interaction_pairs:
            if feature1 in df.columns and feature2 in df.columns:
                
                if interaction_type == 'ratio':
                    result_df[f"{feature1}_{feature2}_ratio"] = (
                        df[feature1] / (df[feature2] + 1e-8)
                    )
                    
                elif interaction_type == 'currency_adjusted':
                    # Adjust commodity price by FX rate
                    result_df[f"{feature1}_fx_adjusted"] = (
                        df[feature1] / (df[feature2] + 1e-8)
                    )
                    
                elif interaction_type == 'product':
                    result_df[f"{feature1}_{feature2}_product"] = (
                        df[feature1] * df[feature2]
                    )
                    
                elif interaction_type == 'correlation':
                    # Rolling correlation
                    result_df[f"{feature1}_{feature2}_corr_30d"] = (
                        df[feature1].rolling(30).corr(df[feature2])
                    )
                    
        return result_df
    
    def create_lead_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lead/lag relationship features."""
        result_df = df.copy()
        
        for leader, follower, lag_days in self.lead_lag_pairs:
            if leader in df.columns and follower in df.columns:
                # Add lagged leader as predictor for follower
                feature_name = f"{leader}_leads_{follower}_lag{lag_days}"
                result_df[feature_name] = df[leader].shift(lag_days)
                
                # Also add the interaction term
                interaction_name = f"{leader}_lag{lag_days}_{follower}_interaction"
                result_df[interaction_name] = (
                    df[leader].shift(lag_days) * df[follower]
                )
                
        return result_df


def create_enhanced_features(df: pd.DataFrame, 
                           n_pca_components: int = 3,
                           use_ica: bool = True) -> pd.DataFrame:
    """Main function to create all enhanced features."""
    
    factor_extractor = GlobalFactorExtractor(n_components=n_pca_components, use_ica=use_ica)
    df_with_factors = factor_extractor.fit_transform(df)
    
    interaction_creator = InteractionFeatureCreator()
    df_with_interactions = interaction_creator.create_interactions(df_with_factors)
    df_final = interaction_creator.create_lead_lag_features(df_with_interactions)
    
    return df_final


if __name__ == "__main__":
    # Test with sample data
    train_df = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
    enhanced_df = create_enhanced_features(train_df)