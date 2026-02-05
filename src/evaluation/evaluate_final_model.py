import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "src" / "models"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHARTS_DIR = PROJECT_ROOT / "results" / "charts"


def load_test_data():
    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    return X_test, y_test


def evaluate_models():
    X_test, y_test = load_test_data()
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]

    plt.figure(figsize=(10, 8))

    results = []

    for model_file in model_files:
        model_name = model_file.replace('.pkl', '')
        print(f"Evaluating: {model_name}...")

        model = joblib.load(MODELS_DIR / model_file)

        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.savefig(CHARTS_DIR / f"cm_{model_name}.png")
        plt.close()

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

        results.append({
            "Model": model_name,
            "F1-Score": f1_score(y_test, y_pred),
            "AUC": roc_auc
        })

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison - All Models')
    plt.legend(loc='lower right')
    plt.savefig(CHARTS_DIR / "roc_comparison_all.png")
    plt.close()

    print("\n" + "=" * 40)
    print(pd.DataFrame(results).sort_values(by="AUC", ascending=False))
    print("=" * 40)


if __name__ == "__main__":
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluate_models()