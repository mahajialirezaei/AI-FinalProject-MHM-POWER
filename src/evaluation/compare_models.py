import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_config():
    """Load project configuration to get the best threshold."""
    if not CONFIG_PATH.exists():
        print("[WARN] Config file not found. Using default threshold 0.5")
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def evaluate_model(model_path, X, y, model_name="Model", threshold=None):
    """
    Evaluates a model. If threshold is provided, uses it for predictions.
    """
    if not model_path.exists():
        print(f"[WARN] {model_name} not found at {model_path.name}. Skipping.")
        return None

    try:
        model = joblib.load(model_path)
        
        # Get probabilities (needed for AUC and Custom Threshold)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = np.zeros(len(y))
            if threshold is not None:
                print(f"[WARN] {model_name} does not support probabilities. Ignoring threshold.")
                threshold = None

        # Generate Predictions
        if threshold is not None:
            # Apply Custom Threshold
            y_pred = (y_prob >= threshold).astype(int)
        else:
            # Default Model Behavior (usually 0.5)
            y_pred = model.predict(X)

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
    # 1. Load Validation Data
    try:
        val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
        X_val = val_df.drop(columns=['target'])
        y_val = val_df['target']
    except FileNotFoundError:
        print("Error: Validation data not found.")
        return

    # 2. Get Best Threshold from Config
    config = load_config()
    best_threshold = config.get("model", {}).get("threshold", 0.5)
    print(f"[INFO] Using Production Threshold from Config: {best_threshold}")

    # 3. Collect Metrics
    all_metrics = {}

    # --- A. Baseline & Phase 2 Models (Standard 0.5 Threshold) ---
    baseline_metrics = evaluate_model(MODELS_DIR / "baseline_logreg.pkl", X_val, y_val, "Baseline")
    if baseline_metrics: all_metrics["Baseline"] = baseline_metrics

    rf_metrics = evaluate_model(MODELS_DIR / "random_forest_model_smote.pkl", X_val, y_val, "Random Forest")
    if rf_metrics: all_metrics["Random Forest"] = rf_metrics

    xgb_metrics = evaluate_model(MODELS_DIR / "xgboost_model_smote.pkl", X_val, y_val, "XGBoost (Manual)")
    if xgb_metrics: all_metrics["XGBoost (Manual)"] = xgb_metrics

    opt_metrics = evaluate_model(MODELS_DIR / "xgboost_optimized.pkl", X_val, y_val, "XGBoost (Optimized)")
    if opt_metrics: all_metrics["XGBoost (Optimized)"] = opt_metrics
    
    weighted_metrics = evaluate_model(MODELS_DIR / "xgboost_weighted_optimized.pkl", X_val, y_val, "XGBoost (Weighted)")
    if weighted_metrics: all_metrics["XGBoost (Weighted)"] = weighted_metrics

    # --- B. Production Model (Custom Threshold) ---
    # We evaluate the Weighted Optimized model AGAIN, but with the specific threshold
    prod_metrics = evaluate_model(
        MODELS_DIR / "xgboost_weighted_optimized.pkl", 
        X_val, 
        y_val, 
        f"Production (Thresh={best_threshold})", 
        threshold=best_threshold
    )
    if prod_metrics:
        all_metrics["Production (Best)"] = prod_metrics


    # 4. Generate DataFrame and Plot
    if not all_metrics:
        print("No models found.")
        return

    # Dynamic DataFrame Construction
    first_metric = list(all_metrics.values())[0]
    data = {"Metric": list(first_metric.keys())}
    for model_name, metrics in all_metrics.items():
        data[model_name] = list(metrics.values())

    comparison_df = pd.DataFrame(data)

    print("\n--- Final Model Comparison Table ---")
    print(comparison_df.round(4))

    # Plot
    df_melted = comparison_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
    
    plt.figure(figsize=(14, 7))
    
    # 1. Capture the axes object 'ax'
    ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis")
    
    plt.title(f"Impact of Optimization & Threshold Tuning (Best Thresh={best_threshold})", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.15) # Increased slightly to make room for text
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.grid(axis='y', alpha=0.3)
    
    # 2. Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
    
    plt.tight_layout()

    save_path = RESULTS_DIR / "comparison_production_final.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[SUCCESS] Final comparison plot saved to {save_path}")


if __name__ == "__main__":
    run_comparison()