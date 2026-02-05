# 📋 Project Overview

This project performs comprehensive **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Industrial Machine Learning Modeling** on the UCI Bank Marketing dataset. The primary goal is to predict whether a client will subscribe to a term deposit (variable `y`) while addressing real-world challenges like class imbalance.

* **Dataset**: UCI Bank Marketing Dataset
* **Task**: Binary Classification
* **Instances**: 45,211 (`bank-full.csv`)
* **Features**: 16 input variables + 1 target variable

---

## 📁 Project Structure

This project follows a modular Data Science structure:

```plaintext
AI-FinalProject-MHM-POWER/
│
├── config/                     # Configuration files
│   └── config.yaml             # Main configuration settings (includes production threshold)
│
├── data/                       # Data directory
│   ├── raw/                    # Raw, immutable data (bank-full.csv)
│   └── processed/              # Processed data splits (train, val, test)
│
├── results/                    # Model evaluation artifacts
│   ├── charts/                 # Performance plots and analysis
│   │   ├── roc_comparison_all.png         # ROC Curve comparing ALL models
│   │   ├── shap_summary_plot.png          # SHAP Global Importance
│   │   ├── shap_force_plot_customer_0.png # SHAP Local Interpretation
│   │   ├── cm_xgboost_weighted_optimized.png # Champion Model Confusion Matrix
│   │   ├── comparison_production_final.png
│   │   └── ... (other confusion matrices)
│   ├── tuning/                 # Threshold tuning artifacts
│   │   └── threshold_tuning_curve.png
│   └── optimization/           # Optuna optimization plots
│
├── src/                        # Source code
│   ├── eda/                    # EDA module
│   │   ├── data_loader.py
│   │   ├── visualizations.py
│   │   └── main.py
│   ├── evaluation/             # Model comparison and evaluation logic
│   │   ├── compare_models.py   # Compares all trained models
│   │   ├── evaluate_final_model.py # Evaluates Champion Model (ROC/CM)
│   │   ├── explainability_shap.py  # Generates SHAP explanations
│   │   └── tune_threshold.py   # Auto-tunes decision threshold
│   ├── models/                 # Serialized models (.pkl files)
│   │   ├── baseline_logreg.pkl
│   │   ├── random_forest_model.pkl
│   │   ├── xgboost_model_smote.pkl
│   │   ├── xgboost_optimized.pkl
│   │   ├── xgboost_weighted.pkl
│   │   └── xgboost_weighted_optimized.pkl  # Champion Model
│   ├── preprocessing/          # Data transformation
│   │   └── main.py             # Main preprocessing script
│   └── training/               # Training pipelines
│       ├── train_baseline.py   # Phase 1: Logistic Regression
│       ├── train_rf.py         # Phase 2: Random Forest
│       ├── train_xgboost.py            # Phase 2: XGBoost + SMOTE
│       ├── optimize_xgboost.py         # Phase 2: Optimization (SMOTE)
│       ├── train_weighted_xgboost.py   # Phase 2: Weighted XGBoost
│       └── optimize_weighted_xgboost.py # Phase 2: Optimization (Weighted)
│
├── tests/                      # Unit and Smoke tests
│   ├── test_data_loader.py
│   └── test_smoke.py
├── requirements.txt            # Python dependencies
└── README.md                   # This file

```

---

## 🚀 Pipeline Workflow

### Phase 1: Baseline Foundation

1. **EDA**: Analyze raw data and generate 6+ required visualizations.
2. **Preprocessing**: Standardize numerical features and encode categorical variables.
3. **Baseline Training**: Train a **Logistic Regression** model to establish a reference point.

### Phase 2: Industrial Modeling & Deployment

This phase represents the iterative journey to find the best performing model for an imbalanced dataset, culminating in deployment.

4. **Random Forest Training**:

* **Goal**: Establish a strong tree-based baseline.
* **Technique**: Uses `class_weight='balanced'` to handle the 88/12 imbalance.
* **Result**: High accuracy but low recall; the model struggled to find minority class instances.

5. **XGBoost with SMOTE (Manual)**:

* **Goal**: Improve Recall by synthesizing new data.
* **Technique**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic examples of subscribers before training XGBoost.
* **Outcome**: Improved Recall compared to Random Forest, but Precision dropped due to the noise introduced by synthetic data.

6. **Optimized XGBoost with SMOTE**:

* **Goal**: Refine the SMOTE-based model.
* **Technique**: Used **Optuna** to search for the best hyperparameters (learning rate, depth) specifically for the SMOTE-augmented dataset.
* **Outcome**: Slight improvement in F1-Score (0.41), but the "synthetic" nature of the data still limited performance.

7. **Weighted XGBoost (The Breakthrough)**:

* **Goal**: Train on pure data without synthetic noise.
* **Technique**: Removed SMOTE and utilized XGBoost's native `scale_pos_weight` parameter to mathematically penalize mistakes on the positive class.
* **Outcome**: Significant jump in Recall (to ~58%) and ROC-AUC, proving that preserving the original data distribution was superior to SMOTE for this specific dataset.

8. **Champion Model Optimization (Weighted + Optuna)**:

* **Technique**: Ran Bayesian Optimization on the Weighted XGBoost model.
* **Result**: Produced the `xgboost_weighted_optimized.pkl` model, achieving the highest ROC-AUC of **0.789**.

9. **Threshold Tuning**:

* **Technique**: Adjusted the decision boundary from the default `0.5` to an optimized **0.5611**.
* **Impact**: Maximized the F1-Score for the "Yes" class, balancing the trade-off between missing customers and annoying them with false calls.

---

## 💻 Essential Commands

Run these commands from the project root (`AI-FinalProject-MHM-POWER/`) to reproduce the results.

### 1. Data Preparation

| Task | Command | Description |
| --- | --- | --- |
| **Preprocess Data** | `python -m src.preprocessing.main` | Cleans, splits, and saves data to `data/processed/` |

### 2. Training & Optimization

| Model Type | Command | Description |
| --- | --- | --- |
| **Baseline** | `python -m src.training.train_baseline` | Trains Logistic Regression |
| **Random Forest** | `python -m src.training.train_rf` | Trains Random Forest (Balanced) |
| **XGB + SMOTE** | `python -m src.training.train_xgboost` | Trains XGBoost with SMOTE |
| **Optimize (SMOTE)** | `python -m src.training.optimize_xgboost` | Optimizes XGBoost (SMOTE) params |
| **Weighted XGB** | `python -m src.training.train_weighted_xgboost` | Trains Weighted XGBoost (No SMOTE) |
| **Champion Optimization** | `python -m src.training.optimize_weighted_xgboost` | **(Best)** Optimizes Weighted XGBoost |

### 3. Evaluation & Explainability (New)

| Task | Command | Description |
| --- | --- | --- |
| **Tune Threshold** | `python -m src.evaluation.tune_threshold` | Finds best threshold & updates `config.yaml` |
| **Compare Models** | `python -m src.evaluation.compare_models` | Generates ROC/Metrics for all models |
| **Final Evaluation** | `python -m src.evaluation.evaluate_final_model` | **ROC & Confusion Matrix** for Champion Model |
| **SHAP Analysis** | `python -m src.evaluation.explainability_shap` | Generates **SHAP** plots for interpretability |
| **Smoke Tests** | `pytest -v -m smoke` | Verifies pipeline integrity |

---

## 📊 Performance Analysis

| Metric | Baseline | RF | XGB (SMOTE) | XGB (Opt+SMOTE) | XGB (Weighted) | **Production (Tuned)** |
| --- | --- | --- | --- | --- | --- | --- |
| **Accuracy** | 76.0% | 88.3% | 89.2% | 87.0% | 82.6% | **86.0%** |
| **Recall** | **59.9%** | 30.3% | 29.7% | 39.1% | 58.1% | **52.0%** (Balanced) |
| **Precision** | 26.6% | 50.6% | 58.4% | 44.0% | 35.4% | **43.0%** |
| **F1-Score** | 0.36 | 0.37 | 0.39 | 0.41 | 0.44 | **0.47** (Best) |
| **ROC-AUC** | 0.749 | 0.755 | 0.773 | 0.750 | 0.787 | **0.789** |

**Observation**:

* **SMOTE Approach**: Steps 5 & 6 showed that while SMOTE improved upon Random Forest, it hit a performance ceiling (F1 ~0.41).
* **Weighted Approach**: Steps 7 & 8 proved that using `scale_pos_weight` was the superior strategy for this dataset, yielding a higher ROC-AUC.
* **Production Model**: By tuning the threshold of the Weighted model to **0.5611**, we achieved the peak F1-Score of **0.47**, striking the optimal balance for the business case.

---

## 📊 Output Files

### Industrial Model Results (`results/charts/` & `results/tuning/`)

* `roc_comparison_all.png`: **Critical**: Comparison of ROC curves for all developed models.
* `shap_summary_plot.png`: **Explainability**: Shows which features (e.g., Balance, Campaign) drive predictions.
* `shap_force_plot_*.png`: Local explanation for specific customer predictions.
* `cm_xgboost_weighted_optimized.png`: Confusion Matrix of the final Champion model.
* `comparison_production_final.png`: Bar chart proving the Production model outperforms all previous versions.
* `feature_importance_rf.png`: Bar chart showing the top 15 features influencing the RF model.

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

```
