"""
Hyperparameter tuning for tree-based models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """Optuna-based hyperparameter tuning for tree models."""
    
    def __init__(self, model_type: str, n_trials: int = 50, cv_folds: int = 3):
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = {}
        self.study = None
        
    def _objective_xgboost(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Objective function for XGBoost hyperparameter optimization."""
        
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'verbosity': 0
        }
        
        model = xgb.XGBRegressor(**params)
        
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        if len(X_val) == 0:
            return float('inf')
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        
        return np.sqrt(mean_squared_error(y_val, y_pred))
    
    def _objective_lightgbm(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Objective function for LightGBM hyperparameter optimization."""
        
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'verbosity': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        if len(X_val) == 0:
            return float('inf')
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
        y_pred = model.predict(X_val)
        
        return np.sqrt(mean_squared_error(y_val, y_pred))
    
    def _objective_catboost(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Objective function for CatBoost hyperparameter optimization."""
        
        params = {
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'iterations': trial.suggest_int('iterations', 50, 300),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': 42,
            'verbose': False
        }
        
        model = CatBoostRegressor(**params)
        
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        if len(X_val) == 0:
            return float('inf')
        
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        y_pred = model.predict(X_val)
        
        return np.sqrt(mean_squared_error(y_val, y_pred))
    
    def _objective_random_forest(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Objective function for Random Forest hyperparameter optimization."""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        return np.sqrt(mean_squared_error(y, y_pred))
    
    def tune_single_target(self, X: pd.DataFrame, y: pd.Series,
                          target_name: str) -> Dict[str, Any]:
        """Tune hyperparameters for a single target."""
        
        print(f"Tuning {self.model_type} for {target_name}...")
        
        valid_mask = ~y.isna()
        if valid_mask.sum() < 50:
            return {
                'status': 'insufficient_data',
                'target': target_name,
                'n_valid': valid_mask.sum()
            }
        
        X_valid = X[valid_mask].reset_index(drop=True)
        y_valid = y[valid_mask].reset_index(drop=True)
        
        objective_funcs = {
            'xgboost': self._objective_xgboost,
            'lightgbm': self._objective_lightgbm,
            'catboost': self._objective_catboost,
            'random_forest': self._objective_random_forest
        }
        
        if self.model_type not in objective_funcs:
            return {
                'status': 'unsupported_model',
                'target': target_name,
                'model_type': self.model_type
            }
        
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            study = optuna.create_study(
                direction='minimize',
                study_name=f'{self.model_type}_{target_name}'
            )
            
            objective_func = objective_funcs[self.model_type]
            study.optimize(
                lambda trial: objective_func(trial, X_valid, y_valid),
                n_trials=self.n_trials
            )
            
            self.study = study
            self.best_params[target_name] = study.best_params
            
            return {
                'status': 'success',
                'target': target_name,
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'n_samples': len(X_valid)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'target': target_name,
                'error': str(e)
            }
    
    def tune_multiple_targets(self, X: pd.DataFrame, target_df: pd.DataFrame,
                            target_cols: List[str] = None,
                            max_targets: int = 5) -> Dict[str, Any]:
        """Tune hyperparameters for multiple targets."""
        
        if target_cols is None:
            target_cols = [col for col in target_df.columns 
                          if col.startswith('target_')][:max_targets]
        
        results = {}
        
        for target_col in target_cols:
            if target_col in target_df.columns:
                result = self.tune_single_target(X, target_df[target_col], target_col)
                results[target_col] = result
        
        summary = self._create_tuning_summary(results)
        
        return {
            'target_results': results,
            'summary': summary,
            'best_params': self.best_params
        }
    
    def _create_tuning_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of tuning results."""
        
        successful_results = [r for r in results.values() 
                            if r.get('status') == 'success']
        
        if not successful_results:
            return {'status': 'no_successful_tuning'}
        
        scores = [r['best_score'] for r in successful_results]
        
        return {
            'n_successful': len(successful_results),
            'n_total': len(results),
            'avg_best_score': np.mean(scores),
            'median_best_score': np.median(scores),
            'std_best_score': np.std(scores),
            'best_overall_score': np.min(scores),
            'best_target': min(successful_results, key=lambda x: x['best_score'])['target']
        }


class TargetSpecificTuner:
    """Target-specific hyperparameter tuning coordinator."""
    
    def __init__(self, n_trials_per_model: int = 30):
        self.n_trials_per_model = n_trials_per_model
        self.tuners = {}
        self.results = {}
    
    def tune_all_models(self, X: pd.DataFrame, target_df: pd.DataFrame,
                       target_cols: List[str] = None,
                       model_types: List[str] = None,
                       max_targets: int = 3) -> Dict[str, Any]:
        """Tune all model types for all targets."""
        
        if target_cols is None:
            target_cols = [col for col in target_df.columns 
                          if col.startswith('target_')][:max_targets]
        
        if model_types is None:
            model_types = ['xgboost', 'lightgbm', 'random_forest', 'catboost']
        
        print(f"Tuning {len(model_types)} models for {len(target_cols)} targets...")
        
        for model_type in model_types:
            print(f"\n=== TUNING {model_type.upper()} ===")
            
            tuner = HyperparameterTuner(
                model_type=model_type, 
                n_trials=self.n_trials_per_model
            )
            
            model_results = tuner.tune_multiple_targets(X, target_df, target_cols)
            
            self.tuners[model_type] = tuner
            self.results[model_type] = model_results
            
            print(f"Completed {model_type}: {model_results['summary'].get('n_successful', 0)} successful")
        
        best_configs = self._get_best_configurations()
        overall_summary = self._create_overall_summary()
        
        return {
            'model_results': self.results,
            'best_configurations': best_configs,
            'overall_summary': overall_summary
        }
    
    def _get_best_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get best model configuration for each target."""
        
        best_configs = {}
        
        all_targets = set()
        for model_results in self.results.values():
            all_targets.update(model_results['target_results'].keys())
        
        for target in all_targets:
            best_score = np.inf
            best_config = None
            
            for model_type, model_results in self.results.items():
                target_result = model_results['target_results'].get(target, {})
                
                if (target_result.get('status') == 'success' and 
                    target_result['best_score'] < best_score):
                    best_score = target_result['best_score']
                    best_config = {
                        'model_type': model_type,
                        'params': target_result['best_params'],
                        'score': best_score
                    }
            
            if best_config:
                best_configs[target] = best_config
        
        return best_configs
    
    def _create_overall_summary(self) -> Dict[str, Any]:
        """Create overall summary across all models and targets."""
        
        summary = {}
        
        for model_type, model_results in self.results.items():
            model_summary = model_results.get('summary', {})
            if model_summary.get('status') != 'no_successful_tuning':
                summary[model_type] = {
                    'avg_score': model_summary.get('avg_best_score', np.inf),
                    'best_score': model_summary.get('best_overall_score', np.inf),
                    'n_successful': model_summary.get('n_successful', 0)
                }
        
        if summary:
            best_overall_model = min(summary.keys(), 
                                   key=lambda k: summary[k]['avg_score'])
            summary['best_overall_model'] = best_overall_model
        
        return summary


def run_comprehensive_tuning(train_df: pd.DataFrame, target_df: pd.DataFrame,
                           max_targets: int = 3, n_trials: int = 20) -> Dict[str, Any]:
    """Run comprehensive hyperparameter tuning."""
    
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'date_id'][:50]
    X = train_df[feature_cols].fillna(0)
    
    tuner = TargetSpecificTuner(n_trials_per_model=n_trials)
    results = tuner.tune_all_models(X, target_df, max_targets=max_targets)
    
    print(f"\n=== TUNING SUMMARY ===")
    for model_type, summary in results['overall_summary'].items():
        if model_type != 'best_overall_model':
            print(f"{model_type.upper()}:")
            print(f"  Average best score: {summary['avg_score']:.6f}")
            print(f"  Best score: {summary['best_score']:.6f}")
            print(f"  Successful targets: {summary['n_successful']}")
    
    print(f"\nBest overall model: {results['overall_summary'].get('best_overall_model', 'N/A')}")
    
    print(f"\nBest configurations by target:")
    for target, config in results['best_configurations'].items():
        print(f"  {target}: {config['model_type']} (RMSE: {config['score']:.6f})")
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import load_competition_data
    
    loader, data = load_competition_data()
    train_df = data['train']
    target_df = data['train_labels']
    
    print("Running hyperparameter tuning...")
    results = run_comprehensive_tuning(train_df, target_df, max_targets=2, n_trials=10)
    
    print(f"\nTuning completed for {len(results['best_configurations'])} targets")