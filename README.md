# MITSUI Commodity Prediction Challenge

Competition submission for the Kaggle MITSUI Commodity Prediction Challenge.

## Project Structure

```
├── data/
│   ├── raw/                    # Original data files
│   │   ├── train.csv
│   │   ├── train_labels.csv  
│   │   ├── test.csv
│   │   └── lagged_test_labels/
│   └── processed/              # Generated outputs
│       ├── submission.parquet
│       └── target_pairs.csv
├── notebooks/                  # Submission notebooks
│   ├── ranking_optimized_submission.py     # Ranking-focused (recommended)
│   ├── advanced_ensemble_submission.py     # Multi-model ensemble
│   ├── high_performance_submission.py      # Performance-optimized
│   ├── ultimate_ensemble_submission.py     # Ultimate ensemble
│   └── 01_data_exploration.ipynb          # Data exploration
├── src/                       # Core implementation
│   ├── features/              # Feature engineering
│   ├── models/               # Model implementations
│   └── utils/                # Utilities
├── kaggle_evaluation/        # Kaggle evaluation framework
├── generate_submission.py    # Submission generator
└── todo.md                  # Project progress
```

## Submissions

Multiple submission approaches optimized for different aspects:

- **ranking_optimized_submission.py**: Optimized for Sharpe ratio of Spearman rank correlations (recommended)
- **advanced_ensemble_submission.py**: Multi-model ensemble (RF, XGBoost, LightGBM)
- **high_performance_submission.py**: Performance-optimized approach
- **ultimate_ensemble_submission.py**: Ultimate ensemble with weighted model combinations

## Usage

For Kaggle submission, use any of the notebook files in the `notebooks/` directory. Each implements the required `predict()` function compatible with the competition's streaming evaluation API.

## Competition Metric

Sharpe ratio of Spearman rank correlations - higher scores are better.