"""
Tree-based models for commodity prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class XGBoostModel:
    """XGBoost implementation for time series prediction."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_estimators': 100,
            'early_stopping_rounds': 10,
            'verbosity': 0
        }
        self.params = {**self.default_params, **(params or {})}
        self.models = {}
        self.feature_importance = {}
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> 'XGBoostModel':
        """Fit XGBoost model."""
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model = xgb.XGBRegressor(**self.params)
        
        fit_params = {}
        if eval_set:
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = False
        
        self.model.fit(X_train, y_train, **fit_params)
        self.feature_importance['gain'] = self.model.feature_importances_
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)
    
    def fit_multiple_targets(self, X: pd.DataFrame, target_df: pd.DataFrame,
                           target_cols: List[str] = None) -> Dict[str, Any]:
        """Fit separate XGBoost models for multiple targets."""
        
        if target_cols is None:
            target_cols = [col for col in target_df.columns 
                          if col.startswith('target_')]
        
        results = {}
        
        for target_col in target_cols:
            print(f"Training XGBoost for {target_col}...")
            
            if target_col not in target_df.columns:
                continue
            
            y = target_df[target_col]
            valid_mask = ~y.isna()
            
            if valid_mask.sum() < 20:
                results[target_col] = {'status': 'insufficient_data'}
                continue
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            n_train = int(0.8 * len(X_valid))
            X_train, X_val = X_valid[:n_train], X_valid[n_train:]
            y_train, y_val = y_valid[:n_train], y_valid[n_train:]
            
            try:
                model = XGBoostModel(self.params)
                if len(X_val) > 0:
                    model.fit(X_train, y_train, X_val, y_val)
                else:
                    model.fit(X_train, y_train)
                
                train_pred = model.predict(X_train)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                
                if len(X_val) > 0:
                    val_pred = model.predict(X_val)
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                else:
                    val_rmse = train_rmse
                
                top_features = X.columns[np.argsort(model.feature_importance['gain'])[-10:][::-1]]
                
                results[target_col] = {
                    'status': 'success',
                    'model': model,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'n_samples': len(X_valid),
                    'top_features': list(top_features)
                }
                
            except Exception as e:
                results[target_col] = {'status': 'error', 'error': str(e)}
        
        self.models = results
        return results


class LightGBMModel:
    """LightGBM implementation for time series prediction."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'max_depth': 6,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'random_state': 42,
            'n_estimators': 100,
            'verbosity': -1,
            'early_stopping_rounds': 10
        }
        self.params = {**self.default_params, **(params or {})}
        self.models = {}
        self.feature_importance = {}
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> 'LightGBMModel':
        """Fit LightGBM model."""
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model = lgb.LGBMRegressor(**self.params)
        
        fit_params = {}
        if eval_set:
            fit_params['eval_set'] = eval_set
            fit_params['callbacks'] = [lgb.log_evaluation(0)]
        
        self.model.fit(X_train, y_train, **fit_params)
        self.feature_importance['split'] = self.model.feature_importances_
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)
    
    def fit_multiple_targets(self, X: pd.DataFrame, target_df: pd.DataFrame,
                           target_cols: List[str] = None) -> Dict[str, Any]:
        """Fit separate LightGBM models for multiple targets."""
        
        if target_cols is None:
            target_cols = [col for col in target_df.columns 
                          if col.startswith('target_')]
        
        results = {}
        
        for target_col in target_cols:
            print(f"Training LightGBM for {target_col}...")
            
            if target_col not in target_df.columns:
                continue
            
            y = target_df[target_col]
            valid_mask = ~y.isna()
            
            if valid_mask.sum() < 20:
                results[target_col] = {'status': 'insufficient_data'}
                continue
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            n_train = int(0.8 * len(X_valid))
            X_train, X_val = X_valid[:n_train], X_valid[n_train:]
            y_train, y_val = y_valid[:n_train], y_valid[n_train:]
            
            try:
                model = LightGBMModel(self.params)
                if len(X_val) > 0:
                    model.fit(X_train, y_train, X_val, y_val)
                else:
                    model.fit(X_train, y_train)
                
                train_pred = model.predict(X_train)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                
                if len(X_val) > 0:
                    val_pred = model.predict(X_val)
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                else:
                    val_rmse = train_rmse
                
                top_features = X.columns[np.argsort(model.feature_importance['split'])[-10:][::-1]]
                
                results[target_col] = {
                    'status': 'success',
                    'model': model,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'n_samples': len(X_valid),
                    'top_features': list(top_features)
                }
                
            except Exception as e:
                results[target_col] = {'status': 'error', 'error': str(e)}
        
        self.models = results
        return results


class RandomForestModel:
    """Random Forest implementation with time series considerations."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        self.params = {**self.default_params, **(params or {})}
        self.models = {}
        self.feature_importance = {}
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'RandomForestModel':
        """Fit Random Forest model."""
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)
        self.feature_importance['importance'] = self.model.feature_importances_
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)
    
    def fit_multiple_targets(self, X: pd.DataFrame, target_df: pd.DataFrame,
                           target_cols: List[str] = None) -> Dict[str, Any]:
        """Fit separate Random Forest models for multiple targets."""
        
        if target_cols is None:
            target_cols = [col for col in target_df.columns 
                          if col.startswith('target_')]
        
        results = {}
        
        for target_col in target_cols:
            print(f"Training Random Forest for {target_col}...")
            
            if target_col not in target_df.columns:
                continue
            
            y = target_df[target_col]
            valid_mask = ~y.isna()
            
            if valid_mask.sum() < 20:
                results[target_col] = {'status': 'insufficient_data'}
                continue
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            try:
                model = RandomForestModel(self.params)
                model.fit(X_valid, y_valid)
                
                pred = model.predict(X_valid)
                rmse = np.sqrt(mean_squared_error(y_valid, pred))
                mae = mean_absolute_error(y_valid, pred)
                
                top_features = X.columns[np.argsort(model.feature_importance['importance'])[-10:][::-1]]
                
                results[target_col] = {
                    'status': 'success',
                    'model': model,
                    'rmse': rmse,
                    'mae': mae,
                    'n_samples': len(X_valid),
                    'top_features': list(top_features)
                }
                
            except Exception as e:
                results[target_col] = {'status': 'error', 'error': str(e)}
        
        self.models = results
        return results


class CatBoostModel:
    """CatBoost implementation for time series prediction."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 10
        }
        self.params = {**self.default_params, **(params or {})}
        self.models = {}
        self.feature_importance = {}
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> 'CatBoostModel':
        """Fit CatBoost model."""
        
        self.model = CatBoostRegressor(**self.params)
        
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = (X_val, y_val)
        
        self.model.fit(X_train, y_train, **fit_params)
        self.feature_importance['importance'] = self.model.feature_importances_
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)
    
    def fit_multiple_targets(self, X: pd.DataFrame, target_df: pd.DataFrame,
                           target_cols: List[str] = None) -> Dict[str, Any]:
        """Fit separate CatBoost models for multiple targets."""
        
        if target_cols is None:
            target_cols = [col for col in target_df.columns 
                          if col.startswith('target_')]
        
        results = {}
        
        for target_col in target_cols:
            print(f"Training CatBoost for {target_col}...")
            
            if target_col not in target_df.columns:
                continue
            
            y = target_df[target_col]
            valid_mask = ~y.isna()
            
            if valid_mask.sum() < 20:
                results[target_col] = {'status': 'insufficient_data'}
                continue
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            n_train = int(0.8 * len(X_valid))
            X_train, X_val = X_valid[:n_train], X_valid[n_train:]
            y_train, y_val = y_valid[:n_train], y_valid[n_train:]
            
            try:
                model = CatBoostModel(self.params)
                if len(X_val) > 0:
                    model.fit(X_train, y_train, X_val, y_val)
                else:
                    model.fit(X_train, y_train)
                
                train_pred = model.predict(X_train)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                
                if len(X_val) > 0:
                    val_pred = model.predict(X_val)
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                else:
                    val_rmse = train_rmse
                
                top_features = X.columns[np.argsort(model.feature_importance['importance'])[-10:][::-1]]
                
                results[target_col] = {
                    'status': 'success',
                    'model': model,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'n_samples': len(X_valid),
                    'top_features': list(top_features)
                }
                
            except Exception as e:
                results[target_col] = {'status': 'error', 'error': str(e)}
        
        self.models = results
        return results


class TreeModelEnsemble:
    """Ensemble of tree-based models."""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.results = {}
    
    def fit_all_models(self, X: pd.DataFrame, target_df: pd.DataFrame,
                      target_cols: List[str] = None,
                      max_targets: int = 5) -> Dict[str, Any]:
        """Train all tree-based models."""
        
        if target_cols is None:
            target_cols = [col for col in target_df.columns 
                          if col.startswith('target_')][:max_targets]
        
        print(f"Training tree models on {len(target_cols)} targets...")
        
        xgb_model = XGBoostModel()
        lgb_model = LightGBMModel()
        rf_model = RandomForestModel()
        cb_model = CatBoostModel()
        
        xgb_results = xgb_model.fit_multiple_targets(X, target_df, target_cols)
        lgb_results = lgb_model.fit_multiple_targets(X, target_df, target_cols)
        rf_results = rf_model.fit_multiple_targets(X, target_df, target_cols)
        cb_results = cb_model.fit_multiple_targets(X, target_df, target_cols)
        
        self.models = {
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'random_forest': rf_model,
            'catboost': cb_model
        }
        
        self.results = {
            'xgboost': xgb_results,
            'lightgbm': lgb_results,
            'random_forest': rf_results,
            'catboost': cb_results
        }
        
        summary = self._create_summary(target_cols)
        
        return {
            'model_results': self.results,
            'summary': summary,
            'best_models': self._get_best_models(target_cols)
        }
    
    def _create_summary(self, target_cols: List[str]) -> Dict[str, Any]:
        """Create summary statistics across models and targets."""
        
        summary = {}
        
        for model_name, model_results in self.results.items():
            successful_results = [r for r in model_results.values() 
                                if r.get('status') == 'success']
            
            if successful_results:
                if model_name == 'random_forest':
                    rmse_values = [r['rmse'] for r in successful_results]
                else:
                    rmse_values = [r['val_rmse'] for r in successful_results]
                
                summary[model_name] = {
                    'n_successful': len(successful_results),
                    'avg_rmse': np.mean(rmse_values),
                    'median_rmse': np.median(rmse_values),
                    'std_rmse': np.std(rmse_values),
                    'best_rmse': np.min(rmse_values),
                    'worst_rmse': np.max(rmse_values)
                }
        
        return summary
    
    def _get_best_models(self, target_cols: List[str]) -> Dict[str, str]:
        """Identify best model for each target."""
        
        best_models = {}
        
        for target_col in target_cols:
            best_rmse = np.inf
            best_model = None
            
            for model_name, model_results in self.results.items():
                if target_col in model_results and model_results[target_col].get('status') == 'success':
                    if model_name == 'random_forest':
                        rmse = model_results[target_col]['rmse']
                    else:
                        rmse = model_results[target_col]['val_rmse']
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model_name
            
            best_models[target_col] = best_model
        
        return best_models


def run_tree_model_evaluation(train_df: pd.DataFrame, target_df: pd.DataFrame,
                             max_targets: int = 5) -> Dict[str, Any]:
    """Run comprehensive tree model evaluation."""
    
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'date_id']
    X = train_df[feature_cols].fillna(0)
    
    ensemble = TreeModelEnsemble()
    results = ensemble.fit_all_models(X, target_df, max_targets=max_targets)
    
    print(f"\n=== TREE MODEL RESULTS ===")
    for model_name, metrics in results['summary'].items():
        print(f"{model_name.upper()}:")
        print(f"  Successful targets: {metrics['n_successful']}/{max_targets}")
        print(f"  Average RMSE: {metrics['avg_rmse']:.6f}")
        print(f"  Best RMSE: {metrics['best_rmse']:.6f}")
    
    print(f"\nBest models by target:")
    for target, best_model in results['best_models'].items():
        print(f"  {target}: {best_model}")
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import load_competition_data
    
    loader, data = load_competition_data()
    train_df = data['train']
    target_df = data['train_labels']
    
    print("Running tree model evaluation...")
    results = run_tree_model_evaluation(train_df, target_df, max_targets=3)
    
    print(f"\nOverall best performing model: {max(results['summary'].keys(), key=lambda k: -results['summary'][k]['avg_rmse'])}")