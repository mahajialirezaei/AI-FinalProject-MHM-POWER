import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_model(model_path, X, y, model_name="Model"):
    """
    Evaluates a model if the file exists. 
    Returns None if file is missing (to prevent crashes).
    """
    if not model_path.exists():
        print(f"[WARN] {model_name} not found at {model_path}. Skipping.")
        return None

    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        
        # Handle models that might not have predict_proba (just in case)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = [0] * len(y) # Dummy values if probability not available

        return {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred),
            "F1-Score": f1_score(y, y_pred),
            "ROC-AUC": roc_auc_score(y, y_prob)
        }
    except Exception as e:
        print(f"[ERROR] Failed to evaluate {model_name}: {e}")
        return None


def run_comparison():
    # 1. Load Data
    try:
        val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
        X_val = val_df.drop(columns=['target'])
        y_val = val_df['target']
    except FileNotFoundError:
        print("Error: Validation data not found.")
        return

    # 2. Collect Metrics (Dictionary to hold all results)
    all_metrics = {}

    # --- Original Models (Preserved) ---
    baseline_metrics = evaluate_model(MODELS_DIR / "baseline_logreg.pkl", X_val, y_val, "Baseline")
    if baseline_metrics:
        all_metrics["Baseline (LogReg)"] = baseline_metrics

    rf_metrics = evaluate_model(MODELS_DIR / "random_forest_model_smote.pkl", X_val, y_val, "Random Forest")
    if rf_metrics:
        all_metrics["Random Forest"] = rf_metrics

    # --- New Models (Added) ---
    xgb_metrics = evaluate_model(MODELS_DIR / "xgboost_model_smote.pkl", X_val, y_val, "XGBoost (Manual)")
    if xgb_metrics:
        all_metrics["XGBoost (Manual)"] = xgb_metrics

    opt_metrics = evaluate_model(MODELS_DIR / "xgboost_optimized.pkl", X_val, y_val, "XGBoost (Optuna)")
    if opt_metrics:
        all_metrics["XGBoost (Optimized)"] = opt_metrics

    # 3. Create Comparison DataFrame
    if not all_metrics:
        print("No models found to compare.")
        return

    # Helper to construct DataFrame dynamically based on what was found
    # We use the keys of the first available metric as the "Metric" column
    first_metric = list(all_metrics.values())[0]
    data = {"Metric": list(first_metric.keys())}
    
    for model_name, metrics in all_metrics.items():
        data[model_name] = list(metrics.values())

    comparison_df = pd.DataFrame(data)

    print("\n--- Comprehensive Model Comparison ---")
    print(comparison_df)

    # 4. Generate Plot
    df_melted = comparison_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Model Performance Comparison: Baseline vs RF vs XGBoost")
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # 5. Save to a NEW file
    save_path = RESULTS_DIR / "comparison_all_models.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nNew comparison plot saved to {save_path}")
    


if __name__ == "__main__":
    run_comparison()