import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"


def evaluate_model(model_path, X, y):
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1-Score": f1_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, y_prob)
    }


def run_comparison():
    val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
    X_val = val_df.drop(columns=['target'])
    y_val = val_df['target']

    baseline_metrics = evaluate_model(MODELS_DIR / "baseline_logreg.pkl", X_val, y_val)
    rf_metrics = evaluate_model(MODELS_DIR / "random_forest_model.pkl", X_val, y_val)

    comparison_df = pd.DataFrame({
        "Metric": list(baseline_metrics.keys()),
        "Baseline (LogReg)": list(baseline_metrics.values()),
        "Random Forest": list(rf_metrics.values())
    })

    print("\n--- Model Comparison Table ---")
    print(comparison_df)

    df_melted = comparison_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model")
    plt.title("Comparison: Baseline vs Random Forest")
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')

    save_path = RESULTS_DIR / "baseline_vs_rf_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"\nComparison plot saved to {save_path}")


if __name__ == "__main__":
    run_comparison()