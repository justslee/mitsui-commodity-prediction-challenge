# MITSUI Commodity Prediction Challenge - Implementation Plan

## Project Overview
- **Training Data**: 1,917 rows × 558 columns (557 features + date_id)
- **Training Labels**: 1,917 rows × 425 columns (424 targets + date_id)
- **Test Data**: 91 rows × 558 columns
- **Target Pairs**: 424 unique target-feature combinations with lag information

## Phase 1: Foundation & Data Pipeline (Week 1)

### Step 1.1: Project Setup
- [ ] Create project directory structure
  - [ ] `mkdir -p {data,src/{features,models,utils},notebooks,results,configs}`
- [ ] Setup virtual environment
- [ ] Install dependencies: `pandas numpy scikit-learn xgboost tensorflow optuna matplotlib seaborn`
- [ ] Initialize git repository
- [ ] Create .gitignore file

### Step 1.2: Data Loading & Exploration
- [ ] Create `src/utils/data_loader.py` for consistent data loading
- [ ] Build `notebooks/01_data_exploration.ipynb` for comprehensive EDA
- [ ] Implement missing data analysis and visualization
- [ ] Create data quality reports
- [ ] Analyze target distribution patterns
- [ ] Document data structure findings

### Step 1.3: Basic Data Preprocessing
- [ ] Handle missing values (forward-fill for time series, interpolation)
- [ ] Create `src/features/preprocessing.py` for cleaning functions
- [ ] Implement date handling and time series alignment
- [ ] Basic feature scaling and normalization
- [ ] Create preprocessing pipeline class

## Phase 2: Feature Engineering (Week 2)

### Step 2.1: Technical Indicators
- [ ] Create `src/features/technical_indicators.py`
- [ ] Implement rolling averages (5, 10, 20, 50 days)
- [ ] Add RSI calculation
- [ ] Add MACD calculation
- [ ] Add Bollinger Bands
- [ ] Implement price momentum measures
- [ ] Add volatility indicators
- [ ] Create return calculations and differencing

### Step 2.2: Cross-Market Features
- [ ] Create `src/features/cross_market.py`
- [ ] Implement currency-adjusted prices using FX rates
- [ ] Add correlation features between asset classes
- [ ] Create LME-JPX spread relationships
- [ ] Add US market sector indicators
- [ ] Implement cross-correlation features

### Step 2.3: Lag Features & Time Series
- [ ] Create `src/features/time_series.py`
- [ ] Implement multi-period lags (1, 2, 3, 5, 10 days)
- [ ] Add rolling statistics windows
- [ ] Create seasonal decomposition features
- [ ] Implement market regime detection
- [ ] Add Fourier transform features for cyclical patterns

## Phase 3: Baseline Models (Week 3)

### Step 3.1: Simple Baselines
- [x] Create `src/models/baselines.py`
- [x] Implement naive forecasts (last value, mean reversion)
- [x] Add linear regression with basic features
- [x] Implement Ridge/Lasso regression for feature selection
- [x] Create target-specific simple models
- [x] Document baseline performance metrics

### Step 3.2: Tree-Based Models
- [x] Create `src/models/tree_models.py`
- [x] Implement XGBoost individual target models
- [x] Add LightGBM for faster iteration
- [x] Implement Random Forest ensemble
- [x] Create target-specific hyperparameter tuning
- [x] Add CatBoost implementation

### Step 3.3: Validation Framework
- [x] Create `src/utils/validation.py`
- [x] Implement time series cross-validation
- [x] Add target-specific evaluation metrics
- [x] Implement purged cross-validation to prevent leakage
- [x] Create performance tracking and logging system
- [x] Add walk-forward analysis

## Phase 4: Advanced Models (Week 4)

### Step 4.1: Deep Learning Setup
- [x] Create `src/models/neural_networks.py`
- [x] Implement LSTM architecture for time series
- [x] Add multi-target prediction setup
- [x] Implement Bidirectional LSTM
- [x] Add GRU alternative architecture
- [x] Create proper data generators for neural networks

### Step 4.2: Hybrid Models
- [x] Create `src/models/hybrid.py`
- [x] Implement LSTM-XGBoost ensemble
- [x] Add TCN (Temporal Convolutional Network)
- [x] Create multi-modal architecture for different data types
- [x] Implement attention mechanisms for feature importance
- [x] Add weighted ensemble methods

### Step 4.3: Target-Specific Modeling
- [ ] Create `src/models/target_specific.py`
- [ ] Implement separate models for single assets vs spreads
- [ ] Add custom loss functions for different target types
- [ ] Create separate models for high vs low missing data targets
- [ ] Implement target-type ensemble methods

## Phase 5: Model Optimization (Week 5)

### Step 5.1: Hyperparameter Tuning
- [ ] Create `src/optimization/hyperparameter_tuning.py`
- [ ] Implement Optuna-based optimization
- [ ] Add target-specific parameter spaces
- [ ] Implement multi-objective optimization (accuracy + stability)
- [ ] Add Bayesian optimization for neural networks
- [ ] Create automated tuning pipelines

### Step 5.2: Feature Selection & Engineering
- [ ] Create `src/features/selection.py`
- [ ] Implement recursive feature elimination
- [ ] Add target-specific feature importance analysis
- [ ] Implement correlation-based feature removal
- [ ] Add stability-based feature selection
- [ ] Create feature interaction terms

### Step 5.3: Ensemble Methods
- [ ] Create `src/models/ensemble.py`
- [ ] Implement weighted ensemble across models
- [ ] Add stacking with meta-learners
- [ ] Create target-specific ensemble weights
- [ ] Implement dynamic weighting based on market conditions
- [ ] Add ensemble diversity measures

## Phase 4: Advanced Models - CURRENT PHASE (Extended)

### Step 4.2: Transformer & Attention Models
- [ ] **Transformer Architecture for Time Series**
  - [ ] Implement Temporal Fusion Transformer (TFT)
  - [ ] Add positional encoding for time series data
  - [ ] Multi-head attention across features and time
  - [ ] Create encoder-decoder setup for multi-step prediction
- [ ] **Time Series Transformers**
  - [ ] Implement Informer model for long sequence prediction
  - [ ] Add sparse self-attention for efficiency
  - [ ] Cross-asset attention mechanisms
  - [ ] Seasonal-trend decomposition attention

### Step 4.3: Enhanced RNN Architectures
- [ ] **Advanced RNN Variants**
  - [ ] Implement GRU with attention mechanisms  
  - [ ] Add Transformer-RNN hybrid architecture
  - [ ] Create bidirectional RNN with skip connections
  - [ ] Multi-scale RNN (different frequencies)
- [ ] **Sequence-to-Sequence Models**
  - [ ] Encoder-decoder RNN for multi-target prediction
  - [ ] Add beam search for prediction sequences
  - [ ] Implement teacher forcing strategies

## Phase 5: Advanced Feature Engineering - CURRENT PRIORITY

### Step 5.1: Dimensionality Reduction & Global Factors
- [ ] **PCA-based Feature Engineering**
  - [ ] Apply PCA to metal price groups (LME metals)
  - [ ] Create FX strength factors from currency pairs
  - [ ] Extract global growth factors from equity indices
  - [ ] Generate commodity sector factors (energy, metals, agriculture)
- [ ] **Factor Analysis & ICA**
  - [ ] Independent Component Analysis for uncorrelated factors
  - [ ] Create interpretable factors (USD strength, risk-on/off, growth)
  - [ ] Rolling PCA for time-varying factors

### Step 5.2: Automated Feature Selection
- [ ] **Model-based Feature Selection**
  - [ ] RFECV (Recursive Feature Elimination with Cross-Validation)
  - [ ] Feature importance aggregation across multiple models
  - [ ] Stability-based feature selection across time folds
  - [ ] Boruta algorithm for feature selection
- [ ] **Statistical Feature Selection**
  - [ ] Mutual information for non-linear relationships
  - [ ] LASSO regularization for automatic selection
  - [ ] Forward/backward stepwise selection

### Step 5.3: Interaction & Lead/Lag Features
- [ ] **Cross-Market Interactions**
  - [ ] Gold/Silver ratio, Oil/Gas spreads
  - [ ] Currency-adjusted metal prices (LME/FX interactions)
  - [ ] Equity sector rotations vs commodity performance
  - [ ] VIX interactions with commodity volatility
- [ ] **Lead/Lag Relationships**
  - [ ] Oil leading gasoline prices (t-1 to t relationships)
  - [ ] USD strength leading EM commodity performance
  - [ ] Equity market leading commodity sentiment
  - [ ] Create cross-correlation analysis for optimal lags

## Phase 6: Validation & Submission (Week 6)

### Step 6.1: Comprehensive Validation
- [ ] Create `src/validation/comprehensive.py`
- [ ] Implement out-of-sample backtesting
- [ ] Add rolling window validation
- [ ] Calculate stability metrics across time periods
- [ ] Perform error analysis and diagnostics
- [ ] Create performance comparison reports

### Step 6.2: Model Interpretation
- [ ] Create `src/interpretation/explainability.py`
- [ ] Implement SHAP values for model interpretation
- [ ] Add feature importance analysis
- [ ] Create target-specific contribution analysis
- [ ] Add market regime impact assessment
- [ ] Generate interpretation reports

### Step 6.3: Submission Pipeline
- [ ] Create `src/submission/pipeline.py`
- [ ] Implement automated prediction generation
- [ ] Add missing value handling for test set
- [ ] Create submission format validation
- [ ] Implement multiple model submission strategy
- [ ] Add final quality checks

## CURRENT FOCUS AREAS

### Immediate Priority: Advanced Feature Engineering
- **PCA & Factor Analysis**: Extract global factors from commodity groups
- **Automated Feature Selection**: RFECV, Boruta, mutual information
- **Interaction Features**: Cross-market ratios, lead/lag relationships
- **Goal**: Reduce noise, improve generalization

### Secondary Priority: Transformer Models  
- **Temporal Fusion Transformer**: State-of-the-art for time series
- **Cross-attention**: Learn relationships across different assets
- **Goal**: Capture complex temporal patterns better than RNNs

### COMPLETED PHASES
- ✅ **Phases 1-3**: Foundation, feature engineering, baseline models
- ✅ **Phase 4 (Partial)**: Basic neural networks and hybrid models
- ✅ **Current baseline**: ranking_optimized_submission.py achieving Sharpe score of 1.7

## Current Status & Key Performance Targets
- [x] **COMPLETED**: Basic project structure and data pipeline
- [x] **COMPLETED**: Baseline models (RMSE validation framework)
- [x] **COMPLETED**: Tree-based models with validation
- [x] **COMPLETED**: Neural networks and hybrid models
- [x] **CURRENT BASELINE**: Ranking-optimized submission with Sharpe score of 1.7
- [ ] **CURRENT GOAL**: Improve Sharpe score from 1.7 to 2.5+ through ranking optimization

## Quality Checkpoints
- [ ] All code follows consistent style and documentation
- [ ] Unit tests for critical functions
- [ ] Proper error handling and logging
- [ ] Reproducible results with fixed random seeds
- [ ] Version control with meaningful commit messages
- [ ] Performance monitoring and tracking

## Final Deliverables
- [ ] Complete model training pipeline
- [ ] Submission file generation
- [ ] Model performance report
- [ ] Code documentation and README
- [ ] Jupyter notebooks for analysis
- [ ] Model interpretation reports

---

**Competition Deadline**: September 29, 2025
**Implementation Timeline**: 6 weeks (42 days)
**Target Performance**: RMSE < 0.012, Consistent across validation periods