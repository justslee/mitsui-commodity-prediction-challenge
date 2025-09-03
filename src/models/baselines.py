"""
Baseline models for commodity prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class BaselineModels:
    """Collection of baseline models for comparison."""
    
    def __init__(self):
        self.models = {}
        self.performance = {}
    
    def naive_last_value(self, train_df: pd.DataFrame, 
                        target_df: pd.DataFrame,
                        target_col: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """Naive forecast using last available value."""
        if target_col not in target_df.columns:
            return np.array([]), {}
        
        target_values = target_df[target_col].dropna()
        if len(target_values) == 0:
            return np.array([]), {}
        
        last_value = target_values.iloc[-1]
        n_predictions = len(target_df)
        predictions = np.full(n_predictions, last_value)
        
        actual_values = target_df[target_col].values
        valid_mask = ~np.isnan(actual_values)
        
        if valid_mask.sum() > 0:
            rmse = np.sqrt(mean_squared_error(actual_values[valid_mask], predictions[valid_mask]))
            mae = mean_absolute_error(actual_values[valid_mask], predictions[valid_mask])
            metrics = {'rmse': rmse, 'mae': mae}
        else:
            metrics = {'rmse': np.inf, 'mae': np.inf}
        
        return predictions, metrics
    
    def naive_mean_reversion(self, train_df: pd.DataFrame,
                           target_df: pd.DataFrame,
                           target_col: str,
                           window: int = 50) -> Tuple[np.ndarray, Dict[str, float]]:
        """Mean reversion baseline."""
        if target_col not in target_df.columns:
            return np.array([]), {}
        
        target_values = target_df[target_col]
        rolling_mean = target_values.rolling(window=window, min_periods=10).mean()
        
        predictions = rolling_mean.fillna(0).values
        actual_values = target_values.values
        valid_mask = ~np.isnan(actual_values)
        
        if valid_mask.sum() > 0:
            rmse = np.sqrt(mean_squared_error(actual_values[valid_mask], predictions[valid_mask]))
            mae = mean_absolute_error(actual_values[valid_mask], predictions[valid_mask])
            metrics = {'rmse': rmse, 'mae': mae}
        else:
            metrics = {'rmse': np.inf, 'mae': np.inf}
        
        return predictions, metrics
    
    def linear_regression_baseline(self, train_df: pd.DataFrame,
                                 target_df: pd.DataFrame,
                                 target_col: str,
                                 feature_cols: List[str] = None,
                                 n_splits: int = 3) -> Tuple[np.ndarray, Dict[str, float]]:
        """Linear regression baseline with time series CV."""
        if target_col not in target_df.columns:
            return np.array([]), {}
        
        if feature_cols is None:
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != 'date_id'][:10]
        
        feature_cols = [col for col in feature_cols if col in train_df.columns]
        if not feature_cols:
            return self.naive_last_value(train_df, target_df, target_col)
        
        X = train_df[feature_cols].fillna(0)
        y = target_df[target_col]
        
        valid_mask = ~y.isna()
        if valid_mask.sum() < 10:
            return self.naive_last_value(train_df, target_df, target_col)
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        model = LinearRegression()
        
        if len(y_valid) > n_splits * 2:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_valid):
                X_train, X_val = X_valid.iloc[train_idx], X_valid.iloc[val_idx]
                y_train, y_val = y_valid.iloc[train_idx], y_valid.iloc[val_idx]
                
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                cv_scores.append(np.sqrt(mean_squared_error(y_val, val_pred)))
            
            cv_rmse = np.mean(cv_scores)
        else:
            cv_rmse = np.inf
        
        model.fit(X_valid, y_valid)
        predictions = model.predict(X)
        
        actual_values = y.values
        valid_mask_all = ~np.isnan(actual_values)
        
        if valid_mask_all.sum() > 0:
            rmse = np.sqrt(mean_squared_error(actual_values[valid_mask_all], predictions[valid_mask_all]))
            mae = mean_absolute_error(actual_values[valid_mask_all], predictions[valid_mask_all])
            metrics = {'rmse': rmse, 'mae': mae, 'cv_rmse': cv_rmse}
        else:
            metrics = {'rmse': np.inf, 'mae': np.inf, 'cv_rmse': cv_rmse}
        
        return predictions, metrics
    
    def ridge_regression_baseline(self, train_df: pd.DataFrame,
                                target_df: pd.DataFrame,
                                target_col: str,
                                feature_cols: List[str] = None,
                                alpha: float = 1.0) -> Tuple[np.ndarray, Dict[str, float]]:
        """Ridge regression with regularization."""
        if target_col not in target_df.columns:
            return np.array([]), {}
        
        if feature_cols is None:
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != 'date_id'][:20]
        
        feature_cols = [col for col in feature_cols if col in train_df.columns]
        if not feature_cols:
            return self.naive_last_value(train_df, target_df, target_col)
        
        X = train_df[feature_cols].fillna(0)
        y = target_df[target_col]
        
        valid_mask = ~y.isna()
        if valid_mask.sum() < 10:
            return self.naive_last_value(train_df, target_df, target_col)
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        model = Ridge(alpha=alpha)
        model.fit(X_valid, y_valid)
        predictions = model.predict(X)
        
        actual_values = y.values
        valid_mask_all = ~np.isnan(actual_values)
        
        if valid_mask_all.sum() > 0:
            rmse = np.sqrt(mean_squared_error(actual_values[valid_mask_all], predictions[valid_mask_all]))
            mae = mean_absolute_error(actual_values[valid_mask_all], predictions[valid_mask_all])
            
            feature_importance = np.abs(model.coef_)
            top_features = np.argsort(feature_importance)[-5:][::-1]
            
            metrics = {
                'rmse': rmse, 
                'mae': mae,
                'top_features': [feature_cols[i] for i in top_features]
            }
        else:
            metrics = {'rmse': np.inf, 'mae': np.inf, 'top_features': []}
        
        return predictions, metrics
    
    def lasso_regression_baseline(self, train_df: pd.DataFrame,
                                target_df: pd.DataFrame,
                                target_col: str,
                                feature_cols: List[str] = None,
                                alpha: float = 0.01) -> Tuple[np.ndarray, Dict[str, float]]:
        """Lasso regression for feature selection."""
        if target_col not in target_df.columns:
            return np.array([]), {}
        
        if feature_cols is None:
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != 'date_id'][:50]
        
        feature_cols = [col for col in feature_cols if col in train_df.columns]
        if not feature_cols:
            return self.naive_last_value(train_df, target_df, target_col)
        
        X = train_df[feature_cols].fillna(0)
        y = target_df[target_col]
        
        valid_mask = ~y.isna()
        if valid_mask.sum() < 10:
            return self.naive_last_value(train_df, target_df, target_col)
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        model = Lasso(alpha=alpha, max_iter=2000)
        model.fit(X_valid, y_valid)
        predictions = model.predict(X)
        
        actual_values = y.values
        valid_mask_all = ~np.isnan(actual_values)
        
        if valid_mask_all.sum() > 0:
            rmse = np.sqrt(mean_squared_error(actual_values[valid_mask_all], predictions[valid_mask_all]))
            mae = mean_absolute_error(actual_values[valid_mask_all], predictions[valid_mask_all])
            
            selected_features = [feature_cols[i] for i, coef in enumerate(model.coef_) if abs(coef) > 1e-6]
            n_selected = len(selected_features)
            
            metrics = {
                'rmse': rmse, 
                'mae': mae,
                'n_features_selected': n_selected,
                'selected_features': selected_features[:10]
            }
        else:
            metrics = {'rmse': np.inf, 'mae': np.inf, 'n_features_selected': 0, 'selected_features': []}
        
        return predictions, metrics
    
    def evaluate_all_baselines(self, train_df: pd.DataFrame,
                             target_df: pd.DataFrame,
                             target_cols: List[str] = None,
                             max_targets: int = 10) -> Dict[str, Dict[str, Any]]:
        """Evaluate all baseline models on multiple targets."""
        if target_cols is None:
            target_cols = [col for col in target_df.columns if col.startswith('target_')]
        
        target_cols = target_cols[:max_targets]
        results = {}
        
        for target_col in target_cols:
            print(f"Evaluating baselines for {target_col}...")
            
            target_results = {}
            
            pred, metrics = self.naive_last_value(train_df, target_df, target_col)
            target_results['naive_last'] = metrics
            
            pred, metrics = self.naive_mean_reversion(train_df, target_df, target_col)
            target_results['mean_reversion'] = metrics
            
            pred, metrics = self.linear_regression_baseline(train_df, target_df, target_col)
            target_results['linear_regression'] = metrics
            
            pred, metrics = self.ridge_regression_baseline(train_df, target_df, target_col)
            target_results['ridge_regression'] = metrics
            
            pred, metrics = self.lasso_regression_baseline(train_df, target_df, target_col)
            target_results['lasso_regression'] = metrics
            
            results[target_col] = target_results
        
        return results
    
    def get_best_baseline(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Identify best baseline model for each target."""
        best_models = {}
        
        for target_col, target_results in results.items():
            best_rmse = np.inf
            best_model = 'naive_last'
            
            for model_name, metrics in target_results.items():
                if 'rmse' in metrics and metrics['rmse'] < best_rmse:
                    best_rmse = metrics['rmse']
                    best_model = model_name
            
            best_models[target_col] = best_model
        
        return best_models


def run_baseline_evaluation(train_df: pd.DataFrame, 
                          target_df: pd.DataFrame,
                          max_targets: int = 5) -> Dict[str, Any]:
    """Run complete baseline evaluation."""
    
    baseline_models = BaselineModels()
    results = baseline_models.evaluate_all_baselines(train_df, target_df, max_targets=max_targets)
    best_models = baseline_models.get_best_baseline(results)
    
    summary = {
        'results': results,
        'best_models': best_models,
        'avg_metrics': {}
    }
    
    for model_type in ['naive_last', 'mean_reversion', 'linear_regression', 'ridge_regression', 'lasso_regression']:
        rmse_values = [target_results[model_type]['rmse'] 
                      for target_results in results.values() 
                      if model_type in target_results and target_results[model_type]['rmse'] != np.inf]
        
        if rmse_values:
            summary['avg_metrics'][model_type] = {
                'avg_rmse': np.mean(rmse_values),
                'median_rmse': np.median(rmse_values),
                'std_rmse': np.std(rmse_values)
            }
    
    return summary


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import load_competition_data
    
    loader, data = load_competition_data()
    train_df = data['train']
    target_df = data['train_labels']
    
    print("Running baseline model evaluation...")
    results = run_baseline_evaluation(train_df, target_df, max_targets=3)
    
    print(f"\n=== BASELINE RESULTS ===")
    print(f"\nAverage Metrics Across Targets:")
    for model_type, metrics in results['avg_metrics'].items():
        print(f"  {model_type}: RMSE = {metrics['avg_rmse']:.6f} ± {metrics['std_rmse']:.6f}")
    
    print(f"\nBest Models by Target:")
    for target, best_model in results['best_models'].items():
        target_rmse = results['results'][target][best_model]['rmse']
        print(f"  {target}: {best_model} (RMSE = {target_rmse:.6f})")