"""
Data loading utilities for the MITSUI Commodity Prediction Challenge.

This module provides consistent data loading functionality with proper
error handling and data validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Centralized data loader for the MITSUI competition.
    
    Handles loading of all competition files with consistent error handling,
    data validation, and preprocessing options.
    """
    
    def __init__(self, data_dir: Union[str, Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory. If None, assumes current directory.
        """
        if data_dir is None:
            self.data_dir = Path.cwd()
        else:
            self.data_dir = Path(data_dir)
        
        self.train_df = None
        self.train_labels_df = None
        self.test_df = None
        self.target_pairs_df = None
        self.lagged_test_labels = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all competition data files.
        
        Returns:
            Dictionary containing all loaded dataframes
        """
        logger.info("Loading all competition data...")
        
        data = {}
        data['train'] = self.load_train_data()
        data['train_labels'] = self.load_train_labels()
        data['test'] = self.load_test_data()
        data['target_pairs'] = self.load_target_pairs()
        data['lagged_test_labels'] = self.load_lagged_test_labels()
        
        logger.info("All data loaded successfully!")
        return data
    
    def load_train_data(self) -> pd.DataFrame:
        """Load training features data."""
        file_path = self.data_dir / "train.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Train data file not found: {file_path}")
        
        logger.info(f"Loading training data from {file_path}")
        self.train_df = pd.read_csv(file_path)
        
        # Basic validation
        expected_cols = 558  # Based on our analysis
        if len(self.train_df.columns) != expected_cols:
            logger.warning(f"Expected {expected_cols} columns, got {len(self.train_df.columns)}")
        
        logger.info(f"Training data loaded: {self.train_df.shape}")
        return self.train_df
    
    def load_train_labels(self) -> pd.DataFrame:
        """Load training labels data."""
        file_path = self.data_dir / "train_labels.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Train labels file not found: {file_path}")
        
        logger.info(f"Loading training labels from {file_path}")
        self.train_labels_df = pd.read_csv(file_path)
        
        # Basic validation
        expected_cols = 425  # Based on our analysis
        if len(self.train_labels_df.columns) != expected_cols:
            logger.warning(f"Expected {expected_cols} columns, got {len(self.train_labels_df.columns)}")
        
        logger.info(f"Training labels loaded: {self.train_labels_df.shape}")
        return self.train_labels_df
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test features data."""
        file_path = self.data_dir / "test.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")
        
        logger.info(f"Loading test data from {file_path}")
        self.test_df = pd.read_csv(file_path)
        logger.info(f"Test data loaded: {self.test_df.shape}")
        return self.test_df
    
    def load_target_pairs(self) -> pd.DataFrame:
        """Load target pairs mapping."""
        file_path = self.data_dir / "target_pairs.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Target pairs file not found: {file_path}")
        
        logger.info(f"Loading target pairs from {file_path}")
        self.target_pairs_df = pd.read_csv(file_path)
        logger.info(f"Target pairs loaded: {self.target_pairs_df.shape}")
        return self.target_pairs_df
    
    def load_lagged_test_labels(self) -> Dict[int, pd.DataFrame]:
        """Load all lagged test labels."""
        lagged_dir = self.data_dir / "lagged_test_labels"
        if not lagged_dir.exists():
            logger.warning(f"Lagged test labels directory not found: {lagged_dir}")
            return {}
        
        logger.info("Loading lagged test labels...")
        
        for lag in range(1, 5):  # lag_1 to lag_4
            file_path = lagged_dir / f"test_labels_lag_{lag}.csv"
            if file_path.exists():
                self.lagged_test_labels[lag] = pd.read_csv(file_path)
                logger.info(f"Lagged test labels lag_{lag} loaded: {self.lagged_test_labels[lag].shape}")
            else:
                logger.warning(f"Lagged test labels file not found: {file_path}")
        
        return self.lagged_test_labels
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """
        Categorize features by their source.
        
        Returns:
            Dictionary with feature categories
        """
        if self.train_df is None:
            self.load_train_data()
        
        cols = self.train_df.columns[1:]  # Exclude date_id
        
        categories = {
            'lme': [c for c in cols if c.startswith('LME_')],
            'jpx': [c for c in cols if c.startswith('JPX_')],
            'us_stock': [c for c in cols if c.startswith('US_Stock_')],
            'fx': [c for c in cols if c.startswith('FX_')],
        }
        
        # Log category sizes
        for cat, features in categories.items():
            logger.info(f"{cat.upper()} features: {len(features)}")
        
        return categories
    
    def get_missing_data_summary(self) -> Dict[str, pd.Series]:
        """
        Get missing data summary for all datasets.
        
        Returns:
            Dictionary with missing data summaries
        """
        summaries = {}
        
        if self.train_df is not None:
            summaries['train'] = self.train_df.isnull().sum()
        
        if self.train_labels_df is not None:
            summaries['train_labels'] = self.train_labels_df.isnull().sum()
        
        if self.test_df is not None:
            summaries['test'] = self.test_df.isnull().sum()
        
        return summaries
    
    def validate_data_consistency(self) -> Dict[str, bool]:
        """
        Validate data consistency across different files.
        
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        if self.train_df is not None and self.train_labels_df is not None:
            # Check if number of rows match
            train_rows = len(self.train_df)
            labels_rows = len(self.train_labels_df)
            results['train_labels_rows_match'] = train_rows == labels_rows
            
            # Check if date_ids match
            if 'date_id' in self.train_df.columns and 'date_id' in self.train_labels_df.columns:
                train_dates = set(self.train_df['date_id'])
                label_dates = set(self.train_labels_df['date_id'])
                results['train_labels_dates_match'] = train_dates == label_dates
        
        if self.train_df is not None and self.test_df is not None:
            # Check if feature columns match (excluding date_id)
            train_features = set(self.train_df.columns) - {'date_id'}
            test_features = set(self.test_df.columns) - {'date_id'}
            results['train_test_features_match'] = train_features == test_features
        
        return results
    
    def get_data_info(self) -> Dict[str, Dict]:
        """
        Get comprehensive information about all loaded datasets.
        
        Returns:
            Dictionary with detailed information about each dataset
        """
        info = {}
        
        for name, df in [
            ('train', self.train_df),
            ('train_labels', self.train_labels_df), 
            ('test', self.test_df),
            ('target_pairs', self.target_pairs_df)
        ]:
            if df is not None:
                info[name] = {
                    'shape': df.shape,
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                    'dtypes': df.dtypes.value_counts().to_dict(),
                    'missing_values': df.isnull().sum().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / df.size) * 100
                }
        
        return info


def load_competition_data(data_dir: Union[str, Path] = None) -> Tuple[DataLoader, Dict[str, pd.DataFrame]]:
    """
    Convenience function to load all competition data.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Tuple of (DataLoader instance, dictionary of loaded dataframes)
    """
    loader = DataLoader(data_dir)
    data = loader.load_all_data()
    return loader, data


if __name__ == "__main__":
    # Example usage
    loader, data = load_competition_data()
    
    # Print basic info
    print("\n=== Data Loading Summary ===")
    for name, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"{name}: {df.shape}")
        elif isinstance(df, dict):
            print(f"{name}: {len(df)} files")
    
    # Print feature categories
    print("\n=== Feature Categories ===")
    categories = loader.get_feature_categories()
    for cat, features in categories.items():
        print(f"{cat.upper()}: {len(features)} features")
    
    # Validation
    print("\n=== Data Validation ===")
    validation = loader.validate_data_consistency()
    for check, result in validation.items():
        print(f"{check}: {'✓' if result else '✗'}")