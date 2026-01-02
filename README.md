# 📋 Project Overview

This project performs comprehensive **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Baseline Modeling** on the UCI Bank Marketing dataset. The primary goal is to analyze marketing campaign data and predict whether a client will subscribe to a term deposit (variable `y`).

* **Dataset**: UCI Bank Marketing Dataset
* **Task**: Binary Classification (Yes/No subscription prediction)
* **Instances**: 45,211 (`bank-full.csv`)
* **Features**: 16 input variables + 1 target variable

---

## 📁 Project Structure

This project follows a modular Data Science structure, separating data, source code, and artifacts:

```plaintext
AI-FinalProject-MHM-POWER/
│
├── config/                     # Configuration files
│   └── config.yaml             # Main configuration settings
│
├── data/                       # Data directory
│   ├── raw/                    # Raw, immutable data (bank-full.csv, etc.)
│   └── processed/              # Processed data splits
│       ├── train.csv           # Training set
│       ├── val.csv             # Validation set
│       └── test.csv            # Test set
│
├── docs/                       # Documentation
│   └── PROJECT_STRUCTURE.md
│
├── reports/                    # Generated analysis figures
│   └── figures/                # EDA Visualization outputs
│
├── results/                    # Model evaluation artifacts
│   └── charts/                 # Performance plots
│       ├── confusion_matrix_baseline.png
│       └── roc_curve_baseline.png
│
├── src/                        # Source code
│   ├── eda/                    # EDA module
│   │   ├── data_loader.py      # Data loading utilities
│   │   ├── visualizations.py   # Plotting functions
│   │   └── main.py             # Main EDA script
│   │
│   ├── evaluation/             # Model evaluation logic
│   │
│   ├── models/                 # Serialized models
│   │   └── baseline_logreg.pkl # Saved Baseline Logistic Regression
│   │
│   ├── preprocessing/          # Data transformation
│   │   └── main.py             # Main preprocessing pipeline script
│   │
│   ├── training/               # Training pipeline
│   │   └── train_baseline.py   # Baseline model training script
│
├── tests/                      # Unit tests
│   └── test_data_loader.py
│
├── Makefile                    # Make commands for automation
├── pyproject.toml              # Project tool configuration
├── QUICKSTART.md               # Quick setup guide
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── run_eda.py                  # Entry point for EDA
└── setup.py                    # Package setup

```

For detailed structure documentation, see `docs/PROJECT_STRUCTURE.md`.

---

## 🚀 Quick Start

New to the project? See `QUICKSTART.md` for a 5-minute setup guide!

### Prerequisites

* Python 3.7 or higher
* `pip` (Python package installer)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt

```

Or install as a package:

```bash
pip install -e .

```

### Step 2: Run the Pipeline

The project workflow consists of three main stages: EDA, Preprocessing, and Baseline Training.

#### 1. Exploratory Data Analysis (EDA)

Analyze the raw data and generate visualizations.

```bash
python -m src.eda.main
# OR
python run_eda.py

```

#### 2. Data Preprocessing

Run the preprocessing pipeline to clean, transform, and split the data.

```bash
python -m src.preprocessing.main

```

**What happens in this step:**

* **Cleaning**: Handles `unknown` values and drops rows with missing targets.
* **Feature Engineering**: Removes the `duration` column to prevent data leakage.
* **Transformation**:
* **Numerical Features**: Applies `SimpleImputer` (mean strategy) and `StandardScaler` (Z-score normalization).
* **Categorical Features**: Applies `SimpleImputer` (most frequent) and `OneHotEncoder`.


* **Splitting**: Splits data into **Train**, **Validation**, and **Test** sets using stratified sampling to maintain class balance.
* **Output**: Saves `train.csv`, `val.csv`, and `test.csv` to `data/processed/`.

#### 3. Baseline Model Training

Train a Logistic Regression model on the processed data.

```bash
python -m src.training.train_baseline

```

**What happens in this step:**

* **Loading**: Reads the processed CSV files.
* **Modeling**: Trains a **Logistic Regression** model using `class_weight='balanced'` to handle class imbalance.
* **Evaluation**: Calculates key metrics:
* **Accuracy**: Overall correctness.
* **Precision & Recall**: Critical for imbalance (how many actual subscribers did we catch?).
* **F1-Score**: Harmonic mean of precision and recall.
* **ROC-AUC**: Model's ability to distinguish classes.


* **Artifacts**:
* Saves the trained model to `src/models/baseline_logreg.pkl`.
* Generates `Confusion Matrix` and `ROC Curve` plots in `results/charts/`.



---

## 📊 Output Files

### EDA Figures (`reports/figures/`)

* `class_imbalance_analysis.png`: Target variable distribution.
* `categorical_conversion_rate.png`: Conversion rates for job, marital status, and education.
* `numerical_outliers_analysis.png`: Age histogram and balance boxplots.
* `correlation_heatmap.png`: Pearson correlation matrix.
* `seasonality_analysis.png`: Success rate per month.
* `duration_analysis.png`: Call duration patterns.

### Model Results (`results/charts/`)

* `confusion_matrix_baseline.png`: Performance matrix of the baseline model.
* `roc_curve_baseline.png`: Receiver Operating Characteristic curve.

---

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

```

---

## ⚠️ Important Notes

### Data Leakage Warning

The `duration` variable is removed during preprocessing (`src/preprocessing/main.py`) because it represents call duration, which is only known **after** the call is made. Including it would give unrealistic and false performance metrics in a predictive model.

---

## 🧪 Testing

Run tests to ensure everything works correctly:

```bash
pytest tests/

```

---

## 🛠️ Development

### Code Organization

* **EDA**: `src/eda/` - Analysis and visualization.
* **Preprocessing**: `src/preprocessing/` - Data cleaning, transformation pipelines, and splitting logic.
* **Training**: `src/training/` - Model training scripts (e.g., `train_baseline.py`).
* **Models**: `src/models/` - Directory where trained models (like `baseline_logreg.pkl`) are serialized and saved.

### Adding New Features

1. Add new functions to appropriate modules in `src/`.
2. Update `config/config.yaml` if new configuration is needed.
3. Add tests in `tests/`.

---

## 📚 References

* **Dataset**: UCI Machine Learning Repository - Bank Marketing
* **Citation**: Moro, S., Laureano, R., & Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*.

---

## 📝 License

This project is for educational purposes as part of an AI Final Project.
