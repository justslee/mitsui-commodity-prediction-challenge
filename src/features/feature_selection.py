"""
Automated Feature Selection for Commodity Prediction
Uses multiple methods to select the most predictive features and reduce overfitting.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    RFECV, SelectKBest, f_regression, mutual_info_regression,
    VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    print("Boruta not available. Install with: pip install Boruta")


class AdvancedFeatureSelector:
    """Advanced feature selection using multiple methods."""
    
    def __init__(self, 
                 max_features: int = 200,
                 cv_folds: int = 5,
                 random_state: int = 42):
        self.max_features = max_features
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.selected_features = {}
        self.feature_scores = {}
        self.selection_methods = {}
        
    def variance_threshold_selection(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Remove features with low variance."""
        selector = VarianceThreshold(threshold=threshold)
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        selector.fit(X_numeric)
        
        selected_features = X_numeric.columns[selector.get_support()].tolist()
        print(f"Variance threshold: {len(selected_features)}/{X_numeric.shape[1]} features selected")
        return selected_features
    
    def correlation_filter(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features."""
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        corr_matrix = X_numeric.corr().abs()
        
        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > threshold)]
        
        selected_features = [col for col in X_numeric.columns if col not in to_drop]
        print(f"Correlation filter: {len(selected_features)}/{X_numeric.shape[1]} features selected")
        return selected_features
    
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, 
                           k: int = 100, method: str = 'f_regression') -> List[str]:
        """Select features based on univariate statistical tests."""
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        y_clean = y.fillna(y.median())
        
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, X_clean.shape[1]))
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X_clean.shape[1]))
        else:
            raise ValueError("Method must be 'f_regression' or 'mutual_info'")
            
        selector.fit(X_clean, y_clean)
        selected_features = X_clean.columns[selector.get_support()].tolist()
        
        # Store scores for analysis
        scores = dict(zip(X_clean.columns, selector.scores_))
        self.feature_scores[f'univariate_{method}'] = scores
        
        print(f"Univariate selection ({method}): {len(selected_features)} features selected")
        return selected_features
    
    def model_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                            model_type: str = 'random_forest') -> List[str]:
        """Select features based on model importance."""
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        y_clean = y.fillna(y.median())
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
            )
        elif model_type == 'extra_trees':
            model = ExtraTreesRegressor(
                n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
            )
        elif model_type == 'lasso':
            model = LassoCV(cv=3, random_state=self.random_state, max_iter=1000)
        elif model_type == 'elastic_net':
            model = ElasticNetCV(cv=3, random_state=self.random_state, max_iter=1000)
        else:
            raise ValueError("Invalid model_type")
            
        # Use SelectFromModel to choose features
        selector = SelectFromModel(
            model, threshold='median', max_features=self.max_features
        )
        selector.fit(X_clean, y_clean)
        
        selected_features = X_clean.columns[selector.get_support()].tolist()
        
        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(X_clean.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            importance = dict(zip(X_clean.columns, np.abs(model.coef_)))
        else:
            importance = {}
            
        self.feature_scores[f'{model_type}_importance'] = importance
        
        print(f"Model-based selection ({model_type}): {len(selected_features)} features selected")
        return selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                    min_features: int = 50) -> List[str]:
        """RFECV - Recursive Feature Elimination with Cross-Validation."""
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        y_clean = y.fillna(y.median())
        
        # Use a fast estimator for RFE
        estimator = RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=self.random_state, n_jobs=-1
        )
        
        # Time series cross-validation
        cv = TimeSeriesSplit(n_splits=min(3, len(y_clean) // 50))
        
        selector = RFECV(
            estimator=estimator,
            min_features_to_select=min_features,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        try:
            selector.fit(X_clean, y_clean)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            
            # Store ranking scores
            ranking = dict(zip(X_clean.columns, selector.ranking_))
            self.feature_scores['rfecv_ranking'] = ranking
            
            print(f"RFECV: {len(selected_features)} features selected "
                  f"(optimal: {selector.n_features_})")
            
        except Exception as e:
            print(f"RFECV failed: {e}, falling back to top features")
            # Fallback to model-based selection
            selected_features = self.model_based_selection(X, y, 'random_forest')
            
        return selected_features
    
    def boruta_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Boruta algorithm for feature selection."""
        if not BORUTA_AVAILABLE:
            print("Boruta not available, skipping...")
            return self.model_based_selection(X, y, 'random_forest')
            
        X_clean = X.select_dtypes(include=[np.number]).fillna(0).values
        y_clean = y.fillna(y.median()).values
        
        # Initialize Boruta
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
        )
        
        boruta_selector = BorutaPy(
            rf, n_estimators='auto', verbose=0, random_state=self.random_state, max_iter=50
        )
        
        try:
            boruta_selector.fit(X_clean, y_clean)
            selected_mask = boruta_selector.support_
            feature_names = X.select_dtypes(include=[np.number]).columns
            selected_features = feature_names[selected_mask].tolist()
            
            # Store Boruta rankings
            ranking = dict(zip(feature_names, boruta_selector.ranking_))
            self.feature_scores['boruta_ranking'] = ranking
            
            print(f"Boruta: {len(selected_features)} features selected")
            
        except Exception as e:
            print(f"Boruta failed: {e}, falling back to model-based selection")
            selected_features = self.model_based_selection(X, y, 'random_forest')
            
        return selected_features
    
    def spearman_correlation_selection(self, X: pd.DataFrame, y: pd.Series, 
                                     top_k: int = 100) -> List[str]:
        """Select features based on Spearman correlation with target (good for ranking metrics)."""
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        y_clean = y.fillna(y.median())
        
        correlations = {}
        for col in X_clean.columns:
            try:
                corr, p_value = spearmanr(X_clean[col], y_clean)
                correlations[col] = abs(corr) if not np.isnan(corr) else 0
            except:
                correlations[col] = 0
                
        # Sort by absolute correlation and take top k
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, corr in sorted_features[:top_k]]
        
        self.feature_scores['spearman_correlation'] = correlations
        
        print(f"Spearman correlation: {len(selected_features)} features selected")
        return selected_features
    
    def stability_selection(self, X: pd.DataFrame, y: pd.Series, 
                          n_bootstrap: int = 10, threshold: float = 0.6) -> List[str]:
        """Select features that are consistently selected across bootstrap samples."""
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        y_clean = y.fillna(y.median())
        
        feature_selection_counts = {col: 0 for col in X_clean.columns}
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            n_samples = len(X_clean)
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_clean.iloc[bootstrap_idx]
            y_boot = y_clean.iloc[bootstrap_idx]
            
            # Select features using LASSO
            try:
                lasso = LassoCV(cv=3, random_state=self.random_state + i, max_iter=1000)
                selector = SelectFromModel(lasso, threshold='mean')
                selector.fit(X_boot, y_boot)
                
                selected_in_boot = X_boot.columns[selector.get_support()].tolist()
                for feature in selected_in_boot:
                    feature_selection_counts[feature] += 1
                    
            except Exception:
                continue
        
        # Select features that appear in at least threshold% of bootstrap samples
        min_count = int(n_bootstrap * threshold)
        stable_features = [feat for feat, count in feature_selection_counts.items() 
                          if count >= min_count]
        
        # Store stability scores
        stability_scores = {feat: count / n_bootstrap for feat, count in feature_selection_counts.items()}
        self.feature_scores['stability'] = stability_scores
        
        print(f"Stability selection: {len(stable_features)} features selected "
              f"(threshold: {threshold})")
        
        return stable_features
    
    def ensemble_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Combine multiple selection methods using ensemble voting."""
        print("Running ensemble feature selection...")
        
        # Clean data once
        X_clean = X.select_dtypes(include=[np.number])
        
        # Remove features with too many missing values
        missing_ratio = X_clean.isnull().sum() / len(X_clean)
        X_clean = X_clean.loc[:, missing_ratio < 0.5]
        X_clean = X_clean.fillna(0)
        
        y_clean = y.fillna(y.median())
        
        if len(y_clean) < 50:
            print("Insufficient data for ensemble selection, using correlation method")
            return self.spearman_correlation_selection(X_clean, y_clean)
        
        # Apply preliminary filters
        features_var = self.variance_threshold_selection(X_clean, threshold=0.001)
        X_filtered = X_clean[features_var]
        
        features_corr = self.correlation_filter(X_filtered, threshold=0.95)
        X_filtered = X_filtered[features_corr]
        
        # Run different selection methods
        selection_results = {}
        
        # 1. Spearman correlation (good for ranking metrics)
        selection_results['spearman'] = set(
            self.spearman_correlation_selection(X_filtered, y_clean, top_k=150)
        )
        
        # 2. Random Forest importance
        selection_results['random_forest'] = set(
            self.model_based_selection(X_filtered, y_clean, 'random_forest')
        )
        
        # 3. LASSO regularization
        selection_results['lasso'] = set(
            self.model_based_selection(X_filtered, y_clean, 'lasso')
        )
        
        # 4. Univariate F-test
        selection_results['univariate'] = set(
            self.univariate_selection(X_filtered, y_clean, k=100, method='f_regression')
        )
        
        # 5. Mutual Information
        selection_results['mutual_info'] = set(
            self.univariate_selection(X_filtered, y_clean, k=100, method='mutual_info')
        )
        
        # 6. RFECV (if enough data)
        if len(y_clean) >= 100:
            selection_results['rfecv'] = set(
                self.recursive_feature_elimination(X_filtered, y_clean, min_features=30)
            )
        
        # 7. Boruta (if available)
        if BORUTA_AVAILABLE and len(y_clean) >= 100:
            selection_results['boruta'] = set(
                self.boruta_selection(X_filtered, y_clean)
            )
        
        # 8. Stability selection
        if len(y_clean) >= 100:
            selection_results['stability'] = set(
                self.stability_selection(X_filtered, y_clean, n_bootstrap=5)
            )
        
        # Ensemble voting: select features that appear in multiple methods
        all_features = set()
        for method_features in selection_results.values():
            all_features.update(method_features)
        
        feature_votes = {}
        for feature in all_features:
            votes = sum(1 for method_features in selection_results.values() 
                       if feature in method_features)
            feature_votes[feature] = votes
        
        # Select features with at least 2 votes (adjust threshold as needed)
        min_votes = max(2, len(selection_results) // 3)  # At least 1/3 of methods
        ensemble_features = [feat for feat, votes in feature_votes.items() 
                           if votes >= min_votes]
        
        # If too few features, take top by vote count
        if len(ensemble_features) < 50:
            sorted_by_votes = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            ensemble_features = [feat for feat, votes in sorted_by_votes[:100]]
        
        # Store results
        self.selected_features = selection_results
        self.feature_scores['ensemble_votes'] = feature_votes
        
        print(f"Ensemble selection: {len(ensemble_features)} features selected")
        print(f"Selection methods used: {list(selection_results.keys())}")
        print(f"Vote threshold: {min_votes}/{len(selection_results)}")
        
        return ensemble_features
    
    def get_feature_analysis(self) -> Dict:
        """Get detailed analysis of feature selection results."""
        analysis = {
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'selection_summary': {}
        }
        
        if 'ensemble_votes' in self.feature_scores:
            votes = self.feature_scores['ensemble_votes']
            analysis['selection_summary'] = {
                'top_features_by_votes': sorted(votes.items(), key=lambda x: x[1], reverse=True)[:20],
                'vote_distribution': {
                    str(v): sum(1 for vote in votes.values() if vote == v) 
                    for v in range(1, max(votes.values()) + 1)
                }
            }
        
        return analysis


def select_best_features(X: pd.DataFrame, y: pd.Series, 
                        max_features: int = 200, 
                        selection_method: str = 'ensemble') -> Tuple[List[str], Dict]:
    """Main function to select the best features."""
    
    selector = AdvancedFeatureSelector(max_features=max_features)
    
    if selection_method == 'ensemble':
        selected_features = selector.ensemble_selection(X, y)
    elif selection_method == 'spearman':
        selected_features = selector.spearman_correlation_selection(X, y, top_k=max_features)
    elif selection_method == 'rfecv':
        selected_features = selector.recursive_feature_elimination(X, y)
    elif selection_method == 'boruta':
        selected_features = selector.boruta_selection(X, y)
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    analysis = selector.get_feature_analysis()
    
    return selected_features, analysis


if __name__ == "__main__":
    print("Testing feature selection...")
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples, n_features = 1000, 300
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    
    # Create target with some features being predictive
    y = (X.iloc[:, :10].sum(axis=1) + 
         0.5 * X.iloc[:, 10:20].sum(axis=1) + 
         np.random.randn(n_samples) * 0.1)
    y = pd.Series(y, name='target')
    
    selected_features, analysis = select_best_features(X, y, max_features=50)