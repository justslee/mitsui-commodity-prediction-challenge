"""
Time series validation framework for commodity prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesValidator:
    """Time series cross-validation with purged splits."""
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None, 
                 gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.splits_info = []
    
    def get_splits(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series cross-validation splits."""
        n_samples = len(X)
        
        if self.test_size is None:
            self.test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * self.test_size
            test_end = test_start + self.test_size
            train_end = test_start - self.gap
            
            if train_end <= 0:
                continue
                
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, min(test_end, n_samples))
            
            split_info = {
                'split': i,
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                'train_end': train_end,
                'test_start': test_start,
                'gap': self.gap
            }
            self.splits_info.append(split_info)
            
            yield train_indices, test_indices
    
    def purged_cross_val_score(self, model, X: pd.DataFrame, y: pd.Series,
                              scoring: str = 'rmse') -> Dict[str, Any]:
        """Perform purged cross-validation."""
        scores = []
        predictions = {}
        
        for i, (train_idx, test_idx) in enumerate(self.get_splits(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if scoring == 'rmse':
                score = np.sqrt(mean_squared_error(y_test, y_pred))
            elif scoring == 'mae':
                score = mean_absolute_error(y_test, y_pred)
            elif scoring == 'r2':
                score = r2_score(y_test, y_pred)
            else:
                score = mean_squared_error(y_test, y_pred)
            
            scores.append(score)
            predictions[f'fold_{i}'] = {
                'test_idx': test_idx,
                'y_true': y_test.values,
                'y_pred': y_pred,
                'score': score
            }
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'predictions': predictions,
            'splits_info': self.splits_info
        }


class ModelEvaluator:
    """Comprehensive model evaluation for time series."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_single_target(self, model, X: pd.DataFrame, y: pd.Series,
                             target_name: str, validator: TimeSeriesValidator) -> Dict[str, Any]:
        """Evaluate model on single target."""
        
        valid_mask = ~y.isna()
        if valid_mask.sum() < 10:
            return {
                'target': target_name,
                'status': 'insufficient_data',
                'n_valid': valid_mask.sum(),
                'metrics': {}
            }
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        cv_results = validator.purged_cross_val_score(model, X_valid, y_valid)
        
        model.fit(X_valid, y_valid)
        y_pred_full = model.predict(X_valid)
        
        metrics = {
            'cv_rmse_mean': cv_results['mean_score'],
            'cv_rmse_std': cv_results['std_score'],
            'full_rmse': np.sqrt(mean_squared_error(y_valid, y_pred_full)),
            'full_mae': mean_absolute_error(y_valid, y_pred_full),
            'full_r2': r2_score(y_valid, y_pred_full),
            'n_samples': len(y_valid),
            'cv_scores': cv_results['scores']
        }
        
        return {
            'target': target_name,
            'status': 'success',
            'metrics': metrics,
            'cv_results': cv_results
        }
    
    def evaluate_multiple_targets(self, model, X: pd.DataFrame, 
                                target_df: pd.DataFrame,
                                target_cols: List[str] = None,
                                n_splits: int = 5, gap: int = 0) -> Dict[str, Any]:
        """Evaluate model on multiple targets."""
        
        if target_cols is None:
            target_cols = [col for col in target_df.columns 
                          if col.startswith('target_')]
        
        validator = TimeSeriesValidator(n_splits=n_splits, gap=gap)
        results = {}
        
        for target_col in target_cols:
            print(f"Evaluating {target_col}...")
            
            if target_col not in target_df.columns:
                continue
                
            result = self.evaluate_single_target(
                model, X, target_df[target_col], target_col, validator
            )
            results[target_col] = result
        
        summary = self._create_summary(results)
        
        return {
            'target_results': results,
            'summary': summary,
            'validation_config': {
                'n_splits': n_splits,
                'gap': gap,
                'n_targets': len(target_cols)
            }
        }
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics across targets."""
        
        successful_results = [r for r in results.values() 
                            if r['status'] == 'success']
        
        if not successful_results:
            return {'status': 'no_successful_evaluations'}
        
        cv_rmse_values = [r['metrics']['cv_rmse_mean'] for r in successful_results]
        full_rmse_values = [r['metrics']['full_rmse'] for r in successful_results]
        
        return {
            'n_successful': len(successful_results),
            'n_total': len(results),
            'cv_rmse_mean': np.mean(cv_rmse_values),
            'cv_rmse_median': np.median(cv_rmse_values),
            'cv_rmse_std': np.std(cv_rmse_values),
            'full_rmse_mean': np.mean(full_rmse_values),
            'full_rmse_median': np.median(full_rmse_values),
            'full_rmse_std': np.std(full_rmse_values),
            'best_target': min(successful_results, 
                             key=lambda x: x['metrics']['cv_rmse_mean'])['target'],
            'worst_target': max(successful_results, 
                              key=lambda x: x['metrics']['cv_rmse_mean'])['target']
        }


class WalkForwardValidator:
    """Walk-forward analysis for time series models."""
    
    def __init__(self, initial_train_size: int, step_size: int = 1):
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.results = []
    
    def validate(self, model, X: pd.DataFrame, y: pd.Series,
                refit_freq: int = 1) -> Dict[str, Any]:
        """Perform walk-forward validation."""
        
        n_samples = len(X)
        predictions = []
        actuals = []
        
        for i in range(self.initial_train_size, n_samples, self.step_size):
            train_end = i
            test_start = i
            test_end = min(i + self.step_size, n_samples)
            
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            if len(X_test) == 0 or y_test.isna().all():
                continue
            
            train_valid_mask = ~y_train.isna()
            if train_valid_mask.sum() < 5:
                continue
                
            X_train_valid = X_train[train_valid_mask]
            y_train_valid = y_train[train_valid_mask]
            
            if i == self.initial_train_size or (i - self.initial_train_size) % refit_freq == 0:
                model.fit(X_train_valid, y_train_valid)
            
            y_pred = model.predict(X_test)
            
            valid_mask = ~y_test.isna()
            if valid_mask.sum() > 0:
                predictions.extend(y_pred[valid_mask])
                actuals.extend(y_test[valid_mask])
                
                step_rmse = np.sqrt(mean_squared_error(
                    y_test[valid_mask], y_pred[valid_mask]
                ))
                
                self.results.append({
                    'step': i,
                    'train_size': train_valid_mask.sum(),
                    'test_size': valid_mask.sum(),
                    'rmse': step_rmse
                })
        
        if not predictions:
            return {'status': 'no_valid_predictions'}
        
        overall_rmse = np.sqrt(mean_squared_error(actuals, predictions))
        overall_mae = mean_absolute_error(actuals, predictions)
        
        return {
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'n_predictions': len(predictions),
            'step_results': self.results,
            'rmse_trend': [r['rmse'] for r in self.results]
        }


def validate_model_comprehensive(model, X: pd.DataFrame, target_df: pd.DataFrame,
                               target_cols: List[str] = None,
                               validation_type: str = 'cv',
                               **kwargs) -> Dict[str, Any]:
    """Comprehensive model validation."""
    
    if target_cols is None:
        target_cols = [col for col in target_df.columns 
                      if col.startswith('target_')][:5]
    
    results = {}
    
    if validation_type == 'cv':
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_multiple_targets(
            model, X, target_df, target_cols, **kwargs
        )
    
    elif validation_type == 'walk_forward':
        initial_size = kwargs.get('initial_train_size', len(X) // 2)
        step_size = kwargs.get('step_size', 1)
        
        wf_results = {}
        for target_col in target_cols:
            if target_col in target_df.columns:
                validator = WalkForwardValidator(initial_size, step_size)
                wf_results[target_col] = validator.validate(
                    model, X, target_df[target_col]
                )
        
        results = {'walk_forward_results': wf_results}
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import load_competition_data
    from sklearn.linear_model import Ridge
    
    loader, data = load_competition_data()
    train_df = data['train']
    target_df = data['train_labels']
    
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    target_cols = [col for col in target_df.columns if col.startswith('target_')][:3]
    
    model = Ridge(alpha=1.0)
    
    print("Testing time series validation...")
    results = validate_model_comprehensive(
        model, X, target_df, target_cols, 
        validation_type='cv', n_splits=3
    )
    
    print(f"Summary: {results['summary']['cv_rmse_mean']:.6f} CV RMSE")
    print(f"Best target: {results['summary']['best_target']}")
    print(f"Successful evaluations: {results['summary']['n_successful']}/{results['summary']['n_total']}")