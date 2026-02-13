# 🏦 Bank Marketing Campaign Predictor

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project for predicting bank marketing campaign success using the UCI Bank Marketing dataset. This project implements a complete ML pipeline from exploratory data analysis to model deployment, featuring multiple model architectures, hyperparameter optimization, and an interactive web interface.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Web Interface](#web-interface)
- [WandB Integration](#wandb-integration)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## 🎯 Overview

This project addresses the challenge of predicting whether a bank client will subscribe to a term deposit based on customer characteristics and campaign information. The dataset contains **45,211 instances** with **16 input features** and exhibits significant **class imbalance** (88% negative, 12% positive), making it a realistic industrial machine learning problem.

### Key Objectives

- **Exploratory Data Analysis**: Comprehensive analysis with 6+ visualizations
- **Data Preprocessing**: Feature engineering, encoding, and train/val/test splitting
- **Model Development**: Multiple algorithms from baseline to production-ready models
- **Hyperparameter Optimization**: Bayesian optimization using Optuna
- **Model Evaluation**: Comprehensive metrics, ROC curves, and SHAP explainability
- **Deployment**: Interactive Streamlit web interface for real-time predictions

### Dataset

- **Source**: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Task**: Binary Classification
- **Instances**: 45,211 (`bank-full.csv`)
- **Features**: 16 input variables + 1 target variable (`y`)

## ✨ Features

### 🔬 Data Science Pipeline
- **Comprehensive EDA** with automated visualization generation
- **Robust preprocessing** with handling of missing values and categorical encoding
- **Data leakage prevention** (duration variable excluded)
- **Stratified train/validation/test splits** (70/15/15)

### 🤖 Machine Learning Models
- **Baseline**: Logistic Regression with balanced classes
- **Random Forest**: Tree-based ensemble with SMOTE
- **XGBoost Variants**: Multiple approaches including SMOTE and class weighting
- **Champion Model**: Optimized weighted XGBoost (ROC-AUC: 0.789)

### 🎛️ Advanced Features
- **Hyperparameter Optimization**: Optuna-based Bayesian optimization
- **Threshold Tuning**: Optimal decision boundary for F1-score maximization
- **Model Explainability**: SHAP plots for global and local interpretability
- **Model Comparison**: Comprehensive evaluation across all models

### 🚀 Deployment & Infrastructure
- **Interactive Web UI**: Streamlit-based prediction interface
- **WandB Integration**: Experiment tracking with online/offline modes
- **CI/CD Pipeline**: Automated testing and code quality checks
- **Comprehensive Testing**: Unit tests and smoke tests

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd AI-FinalProject-MHM-POWER
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download Dataset

The dataset will be automatically downloaded during preprocessing, or you can manually download it:

```bash
mkdir -p data/raw/bank
wget -O /tmp/bank-marketing.zip "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
unzip -j /tmp/bank-marketing.zip "bank-full.csv" -d data/raw/bank/
rm /tmp/bank-marketing.zip
```

Alternatively, you can use the Python package:
```bash
pip install ucimlrepo
python -c "from ucimlrepo import fetch_ucirepo; bank = fetch_ucirepo(id=222); bank.data.features.to_csv('data/raw/bank/bank-full.csv', index=False, sep=';')"
```

## 🚀 Quick Start

### 1. Run Exploratory Data Analysis

```bash
python run_eda.py
```

This generates visualizations in `reports/figures/` and displays statistics in the console.

### 2. Preprocess Data

```bash
python -m src.preprocessing.main
```

This creates train/val/test splits and saves the preprocessor to `src/models/preprocessor.pkl`.

### 3. Train Models

```bash
python -m src.training.train_baseline
python -m src.training.train_rf
python -m src.training.train_xgboost
python -m src.training.train_weighted_xgboost
python -m src.training.optimize_xgboost
python -m src.training.optimize_weighted_xgboost
```

### 4. Launch Web Interface

```bash
python run_ui.py
# or
streamlit run src/ui/streamlit.py
```

The interface will open in your default browser at `http://localhost:8501`.

## 📁 Project Structure

```
AI-FinalProject-MHM-POWER/
│
├── config/                          # Configuration files
│   └── config.yaml                  # Main configuration (paths, thresholds, WandB)
│
├── data/                            # Data directory
│   ├── raw/                         # Raw, immutable data
│   │   └── bank/
│   │       └── bank-full.csv       # Original dataset
│   └── processed/                   # Processed data splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── docs/  
│   └── AI_FinalProject_Report.pdf
├── src/                             # Source code
│   ├── eda/                         # Exploratory Data Analysis
│   │   ├── data_loader.py          # Data loading utilities
│   │   ├── visualizations.py        # Plotting functions
│   │   └── main.py                 # Main EDA script
│   │
│   ├── preprocessing/               # Data preprocessing
│   │   ├── main.py                 # Preprocessing pipeline
│   │   ├── *.pkl                   # Trained models and preprocessor
│   │
│   ├── training/                    # Model training scripts
│   │   ├── train_baseline.py       # Logistic Regression baseline
│   │   ├── train_rf.py             # Random Forest with SMOTE
│   │   ├── train_xgboost.py        # XGBoost with SMOTE
│   │   ├── train_weighted_xgboost.py  # Weighted XGBoost
│   │   ├── optimize_xgboost.py     # Optuna optimization (SMOTE)
│   │   ├── optimize_weighted_xgboost.py  # Optuna optimization (Weighted)
│   │   └── wandb_utils.py         # WandB utility functions
│   │
│   ├── evaluation/                  # Model evaluation
│   │   ├── compare_models.py       # Compare all models
│   │   ├── evaluate_final_model.py # Champion model evaluation
│   │   ├── explainability_shap.py  # SHAP explanations
│   │   └── tune_threshold.py      # Threshold optimization
│   │
│   ├── models/                      # Trained models (gitignored)
│   │   ├── baseline_logreg.pkl
│   │   ├── random_forest_model_smote.pkl
│   │   ├── xgboost_model_smote.pkl
│   │   ├── xgboost_weighted.pkl
│   │   ├── xgboost_optimized.pkl
│   │   └── xgboost_weighted_optimized.pkl  # Champion Model
│   │
│   └── ui/                          # Web interface
│       └── streamlit.py             # Streamlit application
│
├── results/                         # Evaluation results
│   ├── charts/                     # Performance plots
│   ├── tuning/                     # Threshold tuning results
│   └── optimization/               # Optuna optimization plots
│
├── tests/                           # Test suite
│   ├── test_data_loader.py        # Unit tests
│   └── test_smoke.py              # Smoke tests
│
├── notebooks/                       # Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb
│
├── .github/
│   └── workflows/                   # CI/CD pipelines
│       ├── ci.yml                  # Main CI pipeline
│       └── smoke-test.yml          # Smoke test pipeline
│
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata
├── Makefile                         # Common commands
├── run_eda.py                       # EDA entry point
├── run_ui.py                        # UI entry point
└── README.md                        # This file
```

## 📖 Usage Guide

### Data Preparation

#### 1. Exploratory Data Analysis

```bash
python run_eda.py
# or
python -m src.eda.main
```

**Output**: Visualizations saved to `reports/figures/`:
- Class imbalance analysis
- Categorical conversion rates
- Numerical variable distributions
- Correlation heatmap
- Seasonality analysis
- Duration analysis

#### 2. Data Preprocessing

```bash
python -m src.preprocessing.main
```

**Output**:
- Processed train/val/test splits in `data/processed/`
- Preprocessor saved to `src/models/preprocessor.pkl`

**Features**:
- Removes `duration` variable (data leakage prevention)
- Handles missing values (`unknown` → NaN)
- Standardizes numerical features
- One-hot encodes categorical features
- Stratified splitting (70/15/15)

### Model Training

#### Phase 1: Baseline Model

```bash
python -m src.training.train_baseline
```

Trains a Logistic Regression model with balanced class weights.

#### Phase 2: Advanced Models

```bash
# Random Forest with SMOTE
python -m src.training.train_rf

# XGBoost with SMOTE
python -m src.training.train_xgboost

# Weighted XGBoost (No SMOTE)
python -m src.training.train_weighted_xgboost
```

#### Phase 3: Hyperparameter Optimization

```bash
# Optimize XGBoost with SMOTE
python -m src.training.optimize_xgboost

# Optimize Weighted XGBoost (Champion Model)
python -m src.training.optimize_weighted_xgboost
```

### Model Evaluation

#### Compare All Models

```bash
python -m src.evaluation.compare_models
```

Generates comparison charts showing performance metrics across all models.

#### Evaluate Champion Model

```bash
python -m src.evaluation.evaluate_final_model
```

Creates ROC curve and confusion matrix for the production model.

#### SHAP Explainability

```bash
python -m src.evaluation.explainability_shap
```

Generates SHAP plots for model interpretability:
- Global feature importance
- Local explanations for individual predictions

#### Threshold Tuning

```bash
python -m src.evaluation.tune_threshold
```

Finds optimal decision threshold and updates `config/config.yaml`.

### Web Interface

Launch the interactive prediction interface:

```bash
python run_ui.py
```

**Features**:
- Input customer information via sidebar
- Real-time predictions from all models
- Visual comparison of model probabilities
- Production threshold-based recommendations
- Model agreement analysis

## 📊 Model Performance

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Baseline (Logistic Regression) | 76.0% | 26.6% | 59.9% | 0.36 | 0.749 |
| Random Forest (SMOTE) | 88.3% | 50.6% | 30.3% | 0.37 | 0.755 |
| XGBoost (SMOTE) | 89.2% | 58.4% | 29.7% | 0.39 | 0.773 |
| XGBoost Optimized (SMOTE) | 87.0% | 44.0% | 39.1% | 0.41 | 0.750 |
| XGBoost Weighted | 82.6% | 35.4% | 58.1% | 0.44 | 0.787 |
| **Champion (Weighted + Optimized)** | **86.0%** | **43.0%** | **52.0%** | **0.47** | **0.789** |

### Key Insights

1. **SMOTE vs Weighted Approach**: While SMOTE improved upon Random Forest, it hit a performance ceiling. The weighted approach (using `scale_pos_weight`) proved superior for this dataset.

2. **Champion Model**: The optimized weighted XGBoost achieves the best balance with:
   - Highest ROC-AUC: **0.789**
   - Best F1-Score: **0.47**
   - Balanced precision/recall trade-off

3. **Threshold Optimization**: The production threshold of **0.597** maximizes F1-score while balancing false positives and false negatives.

## ⚙️ Configuration

All configuration is managed through `config/config.yaml`:

```yaml
# Data paths
data:
  raw: "data/raw/bank/bank-full.csv"
  delimiter: ";"

# Model settings
model:
  path: "src/models/xgboost_weighted_optimized.pkl"
  threshold: 0.5974526405334473  # Production threshold

# WandB settings
wandb:
  enabled: true
  mode: "online"  # Options: "online", "offline", "disabled"
  project: "ai-finalproject-mhm-power"
```

### WandB Configuration

The project supports flexible WandB integration:

- **Online Mode**: Logs to WandB cloud (requires authentication)
- **Offline Mode**: Saves logs locally (`WANDB_MODE=offline`)
- **Disabled**: No WandB logging (`enabled: false`)

Set via config file or environment variable:
```bash
export WANDB_MODE=offline
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Smoke Tests

```bash
pytest -v -m smoke tests/test_smoke.py
```

Smoke tests verify:
- Configuration file structure
- Data preprocessing pipeline
- Model training (all models)
- Model inference capability

### Test Coverage

- **Unit Tests**: `tests/test_data_loader.py`
- **Smoke Tests**: `tests/test_smoke.py`
- **CI Integration**: Automated testing on push/PR

## 🔄 CI/CD

The project uses GitHub Actions for continuous integration:

### CI Pipeline (`.github/workflows/ci.yml`)

1. **Code Quality Checks**
   - Black formatting check
   - Flake8 linting
   - Mypy type checking

2. **Test Execution**
   - Downloads dataset
   - Runs pytest test suite

3. **Script Verification**
   - Runs EDA script
   - Runs preprocessing script
   - Verifies artifact generation

### Smoke Test Pipeline (`.github/workflows/smoke-test.yml`)

Quick validation pipeline that runs smoke tests on every push/PR.

For detailed CI/CD documentation, see [CI_GUIDE.md](CI_GUIDE.md).

## 🌐 Web Interface

The Streamlit web interface provides an intuitive way to make predictions:

### Features

- **Customer Input Form**: Sidebar with all required features
- **Multi-Model Predictions**: Predictions from all trained models
- **Visual Comparisons**: Interactive charts comparing model probabilities
- **Production Recommendations**: Threshold-based contact recommendations
- **Model Information**: Detailed descriptions and performance metrics

### Usage

1. Launch the interface: `python run_ui.py`
2. Fill in customer information in the sidebar
3. Click "Predict with All Models"
4. Review predictions and recommendations

### Model Support

The UI automatically loads all available models:
- Baseline Logistic Regression
- Random Forest (SMOTE)
- XGBoost (SMOTE)
- XGBoost Weighted
- XGBoost Optimized (SMOTE)
- Champion Model (Weighted Optimized)

## 📈 WandB Integration

The project includes comprehensive WandB integration for experiment tracking:

### Features

- **Automatic Logging**: Hyperparameters, metrics, and artifacts
- **Flexible Modes**: Online, offline, or disabled
- **Model Artifacts**: Automatic model versioning
- **Visualizations**: Confusion matrices, feature importance, optimization history

### Usage

All training scripts automatically log to WandB. Configure in `config/config.yaml`:

```yaml
wandb:
  enabled: true
  mode: "online"  # or "offline" or set enabled: false
```

### Viewing Results

- **Online**: View in WandB web interface
- **Offline**: Check `src/training/wandb/` directory

## 🛠️ Development

### Using Makefile

```bash
make install      # Install dependencies
make install-dev  # Install with dev tools
make run          # Run EDA
make test         # Run tests
make clean        # Clean generated files
make format       # Format code with black
make lint         # Run linters
```

### Code Style

- **Formatter**: Black (line length: 100)
- **Linter**: Flake8
- **Type Checker**: Mypy

### Adding New Models

1. Create training script in `src/training/`
2. Use `wandb_utils.py` for consistent logging
3. Save model to `src/models/`
4. Add to UI model list in `src/ui/streamlit.py`
5. Update evaluation scripts if needed

## ⚠️ Important Notes

### Data Leakage Prevention

The `duration` variable is **strictly removed** during preprocessing. This variable represents call duration, which is only known after a call is completed. Including it would create unrealistic performance metrics.

### Model Compatibility

- Models trained with SMOTE are **pipelines** that expect preprocessed input
- Regular models expect **preprocessed input** (from ColumnTransformer)
- All models use the same preprocessor saved in `src/models/preprocessor.pkl`

### File Paths

All paths in the project are relative to the project root. Ensure you run commands from the project root directory.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Run code quality checks (`make format lint`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd AI-FinalProject-MHM-POWER

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in development mode

# Run tests
pytest tests/ -v
```

## 📝 License

This project is for educational purposes as part of an AI Final Project.

## 📚 References

### Dataset

- **UCI Machine Learning Repository**: [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Citation**: Moro, S., Laureano, R., & Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*.

### Key Technologies

- **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
- **SMOTE**: Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
- **Optuna**: Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework.
- **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions.

### Documentation

- [Project Structure Guide](docs/PROJECT_STRUCTURE.md)
- [CI/CD Guide](CI_GUIDE.md)
- [Quick Start Guide](QUICKSTART.md)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Open-source ML community for excellent tools and libraries
- Project contributors and reviewers

---

**Built with ❤️ using Python, XGBoost, Streamlit, and modern ML best practices.**

For questions or issues, please open an issue on GitHub.
