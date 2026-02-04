import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix, classification_report

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "tuning"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_resources():
    print(f"[INFO] Loading data and model...")
    try:
        val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
        X_val = val_df.drop(columns=['target'])
        y_val = val_df['target']
    except FileNotFoundError:
        raise FileNotFoundError("Validation data not found.")

    # Load the optimized pipeline
    model_path = MODELS_DIR / "xgboost_optimized.pkl"
    if not model_path.exists():
        print("[WARN] Optimized model not found. Using Manual XGBoost.")
        model_path = MODELS_DIR / "xgboost_model_smote.pkl"
    
    pipeline = joblib.load(model_path)
    return X_val, y_val, pipeline

def find_optimal_threshold(y_true, y_prob):
    """
    Finds the threshold that maximizes F1-Score.
    """
    print("\n[INFO] Scanning thresholds for best F1-Score...")
    
    # Get precision, recall, and thresholds from the curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate F1 for every single threshold
    # Note: precisions and recalls have 1 extra element (0/1), so we slice them
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find index of best F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"      Best Threshold Found: {best_threshold:.4f}")
    print(f"      Max F1-Score:         {best_f1:.4f}")
    
    return best_threshold, precisions, recalls, thresholds, f1_scores

def plot_tuning_results(precisions, recalls, thresholds, f1_scores, best_threshold):
    """Generates Precision-Recall vs Threshold chart."""
    plt.figure(figsize=(10, 6))
    
    # Plot F1, Precision, and Recall curves
    # Note: thresholds is 1 shorter than precision/recall arrays
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision', alpha=0.5)
    plt.plot(thresholds, recalls[:-1], 'g--', label='Recall', alpha=0.5)
    plt.plot(thresholds, f1_scores[:-1], 'r-', label='F1 Score', linewidth=2)
    
    # Mark the best threshold
    plt.axvline(best_threshold, color='black', linestyle=':', label=f'Best Threshold ({best_threshold:.2f})')
    
    plt.title("Precision-Recall-F1 vs. Decision Threshold", fontsize=14)
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = RESULTS_DIR / "threshold_tuning_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Plot saved to {save_path}")

def evaluate_new_threshold(pipeline, X_val, y_val, threshold):
    """Prints the classification report with the NEW threshold."""
    print("\n" + "="*60)
    print(f"PERFORMANCE AT NEW THRESHOLD: {threshold:.4f}")
    print("="*60)
    
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    
    # Apply custom threshold
    y_pred_new = (y_prob >= threshold).astype(int)
    
    print(classification_report(y_val, y_pred_new))
    
    cm = confusion_matrix(y_val, y_pred_new)
    print("Confusion Matrix:")
    print(cm)
    
    # Save a heatmap for the new matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Threshold={threshold:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(RESULTS_DIR / "confusion_matrix_tuned.png")
    plt.close()

if __name__ == "__main__":
    X_val, y_val, pipeline = load_resources()
    
    # 1. Get Probabilities
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    
    # 2. Find Best Threshold
    best_thresh, precisions, recalls, thresholds, f1_scores = find_optimal_threshold(y_val, y_prob)
    
    # 3. Plot Curve
    plot_tuning_results(precisions, recalls, thresholds, f1_scores, best_thresh)
    
    # 4. Evaluate Final Result
    evaluate_new_threshold(pipeline, X_val, y_val, best_thresh)