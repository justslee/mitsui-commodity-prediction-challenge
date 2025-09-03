"""
Data preprocessing utilities for the MITSUI Commodity Prediction Challenge.

This module provides comprehensive preprocessing functionality including
missing data handling, scaling, time series alignment, and data validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingDataHandler:
    """
    Handles missing data with multiple strategies appropriate for time series data.
    """
    
    def __init__(self):
        self.imputer_dict = {}
        self.strategy_used = {}
    
    def fit(self, df: pd.DataFrame, strategy: str = 'auto') -> 'MissingDataHandler':
        """
        Fit missing data handling strategy.
        
        Args:
            df: DataFrame with missing values
            strategy: 'forward_fill', 'interpolate', 'mean', 'median', 'auto'
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting missing data handler with strategy: {strategy}")
        
        for column in df.columns:
            if df[column].isnull().any():
                missing_pct = df[column].isnull().mean()
                
                if strategy == 'auto':
                    # Auto-select strategy based on missing percentage and data type
                    if missing_pct > 0.5:
                        # High missing rate - use median for robustness
                        selected_strategy = 'median'
                    elif missing_pct > 0.1:
                        # Medium missing rate - use interpolation
                        selected_strategy = 'interpolate'
                    else:
                        # Low missing rate - use forward fill for time series
                        selected_strategy = 'forward_fill'
                else:
                    selected_strategy = strategy
                
                self.strategy_used[column] = selected_strategy
                
                # Prepare imputers for non-time-series strategies
                if selected_strategy in ['mean', 'median']:
                    self.imputer_dict[column] = SimpleImputer(strategy=selected_strategy)
                    self.imputer_dict[column].fit(df[[column]])
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing data handling transformations.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        for column in df_processed.columns:
            if column in self.strategy_used:
                strategy = self.strategy_used[column]
                
                if strategy == 'forward_fill':
                    df_processed[column] = df_processed[column].ffill()
                    # Handle remaining NaNs at the beginning with backward fill
                    df_processed[column] = df_processed[column].bfill()
                
                elif strategy == 'interpolate':
                    df_processed[column] = df_processed[column].interpolate(method='linear')
                    # Handle edge cases
                    df_processed[column] = df_processed[column].ffill()
                    df_processed[column] = df_processed[column].bfill()
                
                elif strategy in ['mean', 'median']:
                    df_processed[column] = self.imputer_dict[column].transform(df_processed[[column]]).flatten()
        
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, strategy).transform(df)
    
    def get_strategy_summary(self) -> Dict[str, str]:
        """Get summary of strategies used for each feature."""
        return self.strategy_used.copy()


class TimeSeriesAlignment:
    """
    Handles time series alignment and date processing.
    """
    
    def __init__(self):
        self.date_range = None
        self.missing_dates = None
    
    def analyze_time_series(self, df: pd.DataFrame, date_col: str = 'date_id') -> Dict[str, Any]:
        """
        Analyze time series characteristics.
        
        Args:
            df: DataFrame with time series data
            date_col: Name of date column
            
        Returns:
            Dictionary with time series analysis
        """
        analysis = {
            'total_periods': len(df),
            'date_range': (df[date_col].min(), df[date_col].max()),
            'expected_periods': df[date_col].max() - df[date_col].min() + 1,
            'missing_periods': [],
            'duplicate_periods': []
        }
        
        # Check for missing periods
        expected_range = range(df[date_col].min(), df[date_col].max() + 1)
        actual_dates = set(df[date_col])
        missing = [d for d in expected_range if d not in actual_dates]
        analysis['missing_periods'] = missing
        
        # Check for duplicates
        duplicates = df[df[date_col].duplicated()][date_col].tolist()
        analysis['duplicate_periods'] = duplicates
        
        # Continuity check
        analysis['is_continuous'] = len(missing) == 0 and len(duplicates) == 0
        
        return analysis
    
    def align_time_series(self, dfs: List[pd.DataFrame], date_col: str = 'date_id') -> List[pd.DataFrame]:
        """
        Align multiple time series DataFrames to common date range.
        
        Args:
            dfs: List of DataFrames to align
            date_col: Name of date column
            
        Returns:
            List of aligned DataFrames
        """
        if not dfs:
            return []
        
        # Find common date range
        min_date = max(df[date_col].min() for df in dfs)
        max_date = min(df[date_col].max() for df in dfs)
        
        logger.info(f"Aligning {len(dfs)} DataFrames to date range: {min_date} - {max_date}")
        
        # Filter all DataFrames to common range
        aligned_dfs = []
        for df in dfs:
            aligned_df = df[(df[date_col] >= min_date) & (df[date_col] <= max_date)].copy()
            aligned_df = aligned_df.sort_values(date_col).reset_index(drop=True)
            aligned_dfs.append(aligned_df)
        
        return aligned_dfs


class FeatureScaler:
    """
    Handles feature scaling with multiple strategies.
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize scaler.
        
        Args:
            method: 'standard', 'robust', 'minmax'
        """
        self.method = method
        self.scalers = {}
        self.feature_categories = {}
        
        if method == 'standard':
            self.scaler_class = StandardScaler
        elif method == 'robust':
            self.scaler_class = RobustScaler
        elif method == 'minmax':
            self.scaler_class = MinMaxScaler
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def fit(self, df: pd.DataFrame, feature_categories: Dict[str, List[str]] = None) -> 'FeatureScaler':
        """
        Fit scalers to data.
        
        Args:
            df: DataFrame to fit scalers on
            feature_categories: Optional feature categories for category-specific scaling
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.method} scalers")
        
        if feature_categories:
            # Scale each category separately
            self.feature_categories = feature_categories
            for category, features in feature_categories.items():
                category_features = [f for f in features if f in df.columns and f != 'date_id']
                if category_features:
                    self.scalers[category] = self.scaler_class()
                    self.scalers[category].fit(df[category_features])
        else:
            # Scale all numeric features together
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [c for c in numeric_cols if c != 'date_id']
            if feature_cols:
                self.scalers['all'] = self.scaler_class()
                self.scalers['all'].fit(df[feature_cols])
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling transformations.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Scaled DataFrame
        """
        df_scaled = df.copy()
        
        if self.feature_categories:
            # Transform each category separately
            for category, features in self.feature_categories.items():
                if category in self.scalers:
                    category_features = [f for f in features if f in df.columns and f != 'date_id']
                    if category_features:
                        df_scaled[category_features] = self.scalers[category].transform(df[category_features])
        else:
            # Transform all features together
            if 'all' in self.scalers:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                feature_cols = [c for c in numeric_cols if c != 'date_id']
                if feature_cols:
                    df_scaled[feature_cols] = self.scalers['all'].transform(df[feature_cols])
        
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame, feature_categories: Dict[str, List[str]] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, feature_categories).transform(df)


class DataPreprocessor:
    """
    Main preprocessing pipeline that combines all preprocessing steps.
    """
    
    def __init__(self, 
                 missing_strategy: str = 'auto',
                 scaling_method: str = 'robust',
                 handle_outliers: bool = True):
        """
        Initialize preprocessing pipeline.
        
        Args:
            missing_strategy: Strategy for handling missing data
            scaling_method: Method for feature scaling
            handle_outliers: Whether to handle outliers
        """
        self.missing_strategy = missing_strategy
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        
        # Components
        self.missing_handler = MissingDataHandler()
        self.time_aligner = TimeSeriesAlignment()
        self.scaler = FeatureScaler(scaling_method)
        
        # State tracking
        self.is_fitted = False
        self.feature_categories = None
        self.outlier_bounds = {}
        
    def analyze_data(self, train_df: pd.DataFrame, 
                    target_df: pd.DataFrame = None,
                    feature_categories: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive data analysis before preprocessing.
        
        Args:
            train_df: Training features
            target_df: Training targets (optional)
            feature_categories: Feature categories
            
        Returns:
            Analysis results
        """
        analysis = {
            'train_shape': train_df.shape,
            'missing_summary': train_df.isnull().sum().to_dict(),
            'data_types': train_df.dtypes.to_dict()
        }
        
        # Time series analysis
        if 'date_id' in train_df.columns:
            analysis['time_series'] = self.time_aligner.analyze_time_series(train_df)
        
        # Target analysis if provided
        if target_df is not None:
            analysis['target_shape'] = target_df.shape
            analysis['target_missing'] = target_df.isnull().sum().to_dict()
        
        # Category analysis if provided
        if feature_categories:
            analysis['category_sizes'] = {cat: len(features) for cat, features in feature_categories.items()}
            
            category_missing = {}
            for cat, features in feature_categories.items():
                cat_features = [f for f in features if f in train_df.columns]
                if cat_features:
                    category_missing[cat] = train_df[cat_features].isnull().sum().sum()
            analysis['category_missing'] = category_missing
        
        return analysis
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       factor: float = 3.0) -> Dict[str, Tuple[float, float]]:
        """
        Detect outliers and calculate bounds.
        
        Args:
            df: DataFrame to analyze
            method: 'iqr' or 'zscore'
            factor: Multiplier for outlier detection
            
        Returns:
            Dictionary of outlier bounds per feature
        """
        bounds = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'date_id':
                values = df[col].dropna()
                
                if method == 'iqr':
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                
                elif method == 'zscore':
                    mean_val = values.mean()
                    std_val = values.std()
                    lower_bound = mean_val - factor * std_val
                    upper_bound = mean_val + factor * std_val
                
                bounds[col] = (lower_bound, upper_bound)
        
        return bounds
    
    def handle_outliers_data(self, df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]] = None,
                            method: str = 'clip') -> pd.DataFrame:
        """
        Handle outliers in the data.
        
        Args:
            df: DataFrame with potential outliers
            bounds: Outlier bounds (if None, will detect)
            method: 'clip', 'remove', or 'winsorize'
            
        Returns:
            DataFrame with outliers handled
        """
        if bounds is None:
            bounds = self.outlier_bounds
        
        df_processed = df.copy()
        
        for col, (lower, upper) in bounds.items():
            if col in df_processed.columns:
                if method == 'clip':
                    df_processed[col] = df_processed[col].clip(lower=lower, upper=upper)
                elif method == 'winsorize':
                    # Similar to clip but preserves more of the distribution
                    df_processed[col] = np.where(df_processed[col] < lower, 
                                               df_processed[col].quantile(0.05), df_processed[col])
                    df_processed[col] = np.where(df_processed[col] > upper, 
                                               df_processed[col].quantile(0.95), df_processed[col])
        
        return df_processed
    
    def fit(self, train_df: pd.DataFrame, 
           target_df: pd.DataFrame = None,
           feature_categories: Dict[str, List[str]] = None) -> 'DataPreprocessor':
        """
        Fit preprocessing pipeline on training data.
        
        Args:
            train_df: Training features
            target_df: Training targets (optional)
            feature_categories: Feature categories for category-specific processing
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting preprocessing pipeline")
        
        self.feature_categories = feature_categories
        
        # Handle missing values
        self.missing_handler.fit(train_df, self.missing_strategy)
        train_df_imputed = self.missing_handler.transform(train_df)
        
        # Detect outliers
        if self.handle_outliers:
            self.outlier_bounds = self.detect_outliers(train_df_imputed)
            train_df_clean = self.handle_outliers_data(train_df_imputed, self.outlier_bounds)
        else:
            train_df_clean = train_df_imputed
        
        # Fit scaler
        self.scaler.fit(train_df_clean, feature_categories)
        
        self.is_fitted = True
        logger.info("Preprocessing pipeline fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Preprocessed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Apply transformations in order
        df_processed = self.missing_handler.transform(df)
        
        if self.handle_outliers:
            df_processed = self.handle_outliers_data(df_processed, self.outlier_bounds)
        
        df_processed = self.scaler.transform(df_processed)
        
        return df_processed
    
    def fit_transform(self, train_df: pd.DataFrame, 
                     target_df: pd.DataFrame = None,
                     feature_categories: Dict[str, List[str]] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(train_df, target_df, feature_categories).transform(train_df)
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps applied.
        
        Returns:
            Dictionary with preprocessing summary
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        summary = {
            "status": "fitted",
            "missing_strategy": self.missing_strategy,
            "scaling_method": self.scaling_method,
            "handle_outliers": self.handle_outliers,
            "missing_strategies_used": self.missing_handler.get_strategy_summary(),
            "outlier_bounds_count": len(self.outlier_bounds) if self.handle_outliers else 0
        }
        
        if self.feature_categories:
            summary["feature_categories"] = {cat: len(features) 
                                           for cat, features in self.feature_categories.items()}
        
        return summary


def create_preprocessing_pipeline(missing_strategy: str = 'auto',
                                scaling_method: str = 'robust',
                                handle_outliers: bool = True) -> DataPreprocessor:
    """
    Factory function to create preprocessing pipeline.
    
    Args:
        missing_strategy: Strategy for handling missing data
        scaling_method: Method for feature scaling  
        handle_outliers: Whether to handle outliers
        
    Returns:
        Configured DataPreprocessor instance
    """
    return DataPreprocessor(
        missing_strategy=missing_strategy,
        scaling_method=scaling_method,
        handle_outliers=handle_outliers
    )


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import load_competition_data
    
    # Load data
    loader, data = load_competition_data()
    train_df = data['train']
    
    # Get feature categories
    feature_categories = loader.get_feature_categories()
    
    # Create and test preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Analyze data before preprocessing
    analysis = preprocessor.analyze_data(train_df, feature_categories=feature_categories)
    print("Data Analysis:")
    for key, value in analysis.items():
        if isinstance(value, dict) and len(value) > 10:
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")
    
    # Fit and transform
    train_processed = preprocessor.fit_transform(train_df, feature_categories=feature_categories)
    
    print(f"\nPreprocessing Results:")
    print(f"Original shape: {train_df.shape}")
    print(f"Processed shape: {train_processed.shape}")
    print(f"Missing values before: {train_df.isnull().sum().sum()}")
    print(f"Missing values after: {train_processed.isnull().sum().sum()}")
    
    # Get preprocessing summary
    summary = preprocessor.get_preprocessing_summary()
    print(f"\nPreprocessing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")