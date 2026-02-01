# 📋 Project Overview

This project performs comprehensive **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Industrial Machine Learning Modeling** on the UCI Bank Marketing dataset. The primary goal is to predict whether a client will subscribe to a term deposit (variable `y`) while addressing real-world challenges like class imbalance.

* 
**Dataset**: UCI Bank Marketing Dataset 


* 
**Task**: Binary Classification 


* 
**Instances**: 45,211 (`bank-full.csv`) 


* 
**Features**: 16 input variables + 1 target variable 



---

## 📁 Project Structure

This project follows a modular Data Science structure:

```plaintext
AI-FinalProject-MHM-POWER/
│
├── config/                     # Configuration files
│   └── config.yaml             # Main configuration settings
│
├── data/                       # Data directory
├── raw/                    # Raw, immutable data (bank-full.csv)
└── processed/              # Processed data splits (train, val, test)
│
├── results/                    # Model evaluation artifacts
│   └── charts/                 # Performance plots and analysis
│       ├── feature_importance_rf.png      # NEW: RF Feature analysis
│       ├── baseline_vs_rf_comparison.png  # NEW: Model comparison chart
│       ├── confusion_matrix_baseline.png
│       └── roc_curve_baseline.png
│
├── src/                        # Source code
│   ├── eda/                    # EDA module
│   ├── evaluation/             # NEW: Model comparison and evaluation logic
│   │   └── compare_models.py   # Script to compare Baseline vs. RF
│   ├── models/                 # Serialized models (.pkl files)
│   │   ├── baseline_logreg.pkl # Saved Baseline model
│   │   └── random_forest_model.pkl # NEW: Saved Industrial RF model
│   ├── preprocessing/          # Data transformation
│   └── training/               # Training pipelines
│       ├── train_baseline.py   # Phase 1: Logistic Regression
│       └── train_rf.py         # Phase 2: Random Forest
│
├── tests/                      # Unit and Smoke tests
│   └── test_data_loader.py
│   └── __init__.py
│   └── test_smoke.py
├── requirements.txt            # Python dependencies
└── README.md                   # This file

```

---

## 🚀 Pipeline Workflow

### Phase 1: Baseline Foundation

1. 
**EDA**: Analyze raw data and generate 6+ required visualizations.


2. 
**Preprocessing**: Standardize numerical features and encode categorical variables.


3. 
**Baseline Training**: Train a **Logistic Regression** model to establish a reference point.



### Phase 2: Industrial Modeling (Current)

4. **Random Forest Training**:
* Run `python -m src.training.train_rf`.


* 
**Stratified K-Fold CV**: Ensures model stability across data folds.


* 
**Class Weighting**: Uses `class_weight='balanced'` to handle the 88/12 imbalance.




5. 
**Feature Importance**: Identify key drivers of client subscription (e.g., `balance`, `age`).


6. **Model Comparison**:
* Run `python -m src.evaluation.compare_models`.


* Compare Baseline vs. Random Forest across Precision, Recall, and F1-Score.





---

## 📊 Performance Analysis

| Metric | Baseline (LogReg) | Random Forest (Phase 2) |
| --- | --- | --- |
| **Accuracy** | 76.0% | **89.2%** (Target: 90%) |
| **Recall (Class 1)** | **59.9%** (Bolder) | 21.1% (Conservative) 
| **F1-Score** | 0.36 | 0.31 (Needs Improvement) 


**Observation**: While Random Forest significantly improves overall accuracy, it suffers from low recall due to severe class imbalance. This justifies the upcoming transition to **XGBoost + SMOTE**.

---

## 📊 Output Files

### Industrial Model Results (`results/charts/`)

* 
`feature_importance_rf.png`: Bar chart showing the top 15 features influencing the model.


* 
`baseline_vs_rf_comparison.png`: Grouped bar chart comparing Phase 1 and Phase 2 performance.


* 
`confusion_matrix_rf.png`: Visualizing True Positives vs. False Negatives for the RF model.



---

## ⚠️ Data Leakage Warning

The `duration` variable is strictly removed during preprocessing. As per project guidelines, this variable is unknown before a call and its inclusion would lead to unrealistic performance.

---

## 🧪 Testing

Run the automated **Smoke Tests** to verify the training and preprocessing pipelines:

```bash
pytest -v -m smoke tests/test_smoke.py

```

---
## 📚 References

* **Dataset**: UCI Machine Learning Repository - Bank Marketing
* **Citation**: Moro, S., Laureano, R., & Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*.

---

## 📝 License

This project is for educational purposes as part of an AI Final Project.
