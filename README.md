# Bank Marketing Dataset - EDA Project

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project performs comprehensive Exploratory Data Analysis (EDA) on the UCI Bank Marketing dataset to prepare for a classification task. The goal is to predict whether a client will subscribe to a term deposit (variable `y`) based on marketing campaign data.

**Dataset**: UCI Bank Marketing Dataset  
**Task**: Binary Classification (Yes/No subscription prediction)  
**Instances**: 45,211 (bank-full.csv)  
**Features**: 16 input variables + 1 target variable

## 📁 Project Structure

This project follows industry-standard Data Science project structure:

```
AI-FinalProject-MHM-POWER/
│
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration file
│
├── data/                   # Data directory
│   ├── raw/               # Raw, immutable data
│   ├── external/          # External data sources
│   └── processed/         # Processed data
│
├── docs/                   # Documentation
│   └── PROJECT_STRUCTURE.md
│
├── models/                 # Trained models
│
├── notebooks/              # Jupyter notebooks for exploration
│
├── reports/                # Generated reports and figures
│   └── figures/           # Visualization outputs
│
├── src/                    # Source code
│   ├── __init__.py
│   └── eda/               # EDA module
│       ├── __init__.py
│       ├── data_loader.py    # Data loading utilities
│       ├── visualizations.py # Plotting functions
│       └── main.py           # Main EDA script
│
├── tests/                  # Unit tests
│   ├── __init__.py
│   └── test_data_loader.py
│
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── pyproject.toml        # Modern Python project config
└── run_eda.py            # Entry point script
```

For detailed structure documentation, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## 🚀 Quick Start

> **New to the project?** See **[QUICKSTART.md](QUICKSTART.md)** for a 5-minute setup guide!

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

This will install:

- `pandas` - Data manipulation and analysis
- `matplotlib` - Plotting library
- `seaborn` - Statistical data visualization
- `numpy` - Numerical computing
- `pyyaml` - Configuration file parsing

### Step 2: Run the EDA Script

**Option 1: Using the entry point script (Recommended)**

```bash
python run_eda.py
```

**Option 2: Using the module directly**

```bash
python -m src.eda.main
```

**Option 3: Using the installed command (after `pip install -e .`)**

```bash
run-eda
```

The script will automatically:

1. Load the dataset from `data/raw/bank/bank-full.csv` (configurable in `config/config.yaml`)
2. Perform all analyses
3. Generate 6 visualization files in `reports/figures/`
4. Display statistics in the console

## 📊 Output Files

After running the script, you'll find the following visualization files in `reports/figures/`:

1. **`class_imbalance_analysis.png`**

   - Bar plot and pie chart showing the distribution of target variable
   - Highlights the class imbalance issue

2. **`categorical_conversion_rate.png`**

   - Stacked bar plots for `job`, `marital`, and `education`
   - Shows conversion rates for each category

3. **`numerical_outliers_analysis.png`**

   - Histogram for `age` distribution
   - Boxplot for `balance` to identify outliers

4. **`correlation_heatmap.png`**

   - Pearson correlation matrix for all numerical variables
   - Helps identify relationships between features

5. **`seasonality_analysis.png`**

   - Success rate per month
   - Identifies seasonal trends in the data

6. **`duration_analysis.png`**
   - Boxplot comparing call duration for 'yes' vs 'no' classes
   - Includes data leakage warning

## ⚙️ Configuration

The project uses a YAML configuration file (`config/config.yaml`) for easy customization:

```yaml
data:
  raw: "data/raw/bank/bank-full.csv"
  delimiter: ";"

output:
  figures_dir: "reports/figures"

eda:
  numerical_vars: [age, balance, day, duration, campaign, pdays, previous]
  categorical_vars: [job, marital, education]
  target_var: "y"

plotting:
  style: "whitegrid"
  palette: "husl"
  figure_size: [12, 6]
  font_size: 10
  dpi: 300
```

Modify this file to change data paths, variables to analyze, or plotting styles without editing code.

## 📁 Dataset Structure

The dataset is located at: `data/raw/bank/bank-full.csv`

**Key Features:**

- **Delimiter**: Semicolon (`;`)
- **Target Variable**: `y` (yes/no)
- **No Missing Values**: Dataset is complete

**Variable Categories:**

- **Bank Client Data**: age, job, marital, education, default, balance, housing, loan
- **Last Contact Info**: contact, day, month, duration
- **Campaign Info**: campaign, pdays, previous, poutcome

## 🔍 What the EDA Covers

1. **Class Imbalance Analysis** - Understanding target variable distribution
2. **Categorical Analysis** - Conversion rates by job, marital status, and education
3. **Numerical Analysis** - Distribution and outlier detection
4. **Correlation Analysis** - Relationships between numerical features
5. **Seasonality Analysis** - Monthly trends in subscription rates
6. **Duration Analysis** - Call duration patterns (with data leakage warning)

## ⚠️ Important Notes

### Data Leakage Warning

The `duration` variable should **NOT** be used for model training because:

- It represents the call duration, which is only known **after** the call
- In production, you need to predict **before** making the call
- Including it would give unrealistic performance metrics

**Recommendation**: Exclude `duration` from feature engineering for the classification model.

## 🧪 Testing

Run tests to ensure everything works correctly:

```bash
pytest tests/
```

## 📈 Next Steps

After completing the EDA:

1. **Feature Engineering**

   - Handle categorical variables (encoding)
   - Address class imbalance (SMOTE, undersampling, etc.)
   - Create new features if needed

2. **Model Development**

   - Split data into train/validation/test sets
   - Try multiple algorithms (Logistic Regression, Random Forest, XGBoost, etc.)
   - Tune hyperparameters

3. **Model Evaluation**
   - Use appropriate metrics (AUC-ROC, Precision, Recall, F1-score)
   - Consider class imbalance in evaluation
   - Perform cross-validation

## 🛠️ Development

### Code Organization

- **Modular Design**: Code is organized into reusable modules in `src/eda/`
- **Configuration-Driven**: Settings are in `config/config.yaml`
- **Testable**: Unit tests in `tests/`
- **Documented**: Docstrings and structure documentation

### Adding New Features

1. Add new functions to appropriate modules in `src/eda/`
2. Update `config/config.yaml` if new configuration is needed
3. Add tests in `tests/`
4. Update this README if needed

## 🛠️ Troubleshooting

### Issue: FileNotFoundError

**Solution**: Ensure you're running the script from the project root directory and that the data file exists at the path specified in `config/config.yaml`.

### Issue: ModuleNotFoundError

**Solution**: Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Issue: Import Errors

**Solution**: Install the package in development mode:

```bash
pip install -e .
```

### Issue: Memory Error

**Solution**: If working with the full dataset causes issues, you can modify `config/config.yaml` to use `bank.csv` (10% sample) instead:

```yaml
data:
  raw: "data/raw/bank/bank.csv"
```

## 📚 References

- **Dataset**: [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Citation**: Moro, S., Laureano, R., & Cortez, P. (2011). Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology.

## 📝 License

This project is for educational purposes as part of an AI Final Project.

---

**Happy Analyzing! 🎉**
