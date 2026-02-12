import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_weighted_optimized.pkl"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"


def run_shap_analysis():
    print("Loading champion model and test data...")
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")

    X_test = test_df.drop(columns=["target"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot - Feature Impact on Prediction")
    plt.savefig(RESULTS_DIR / "shap_summary_plot.png", bbox_inches="tight")
    plt.close()
    print(f"Summary plot saved to {RESULTS_DIR / 'shap_summary_plot.png'}")

    idx = 0
    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx, :],
        X_test.iloc[idx, :],
        matplotlib=True,
        show=False,
    )
    plt.savefig(RESULTS_DIR / f"shap_force_plot_customer_{idx}.png", bbox_inches="tight")
    plt.close()
    print(f"Individual force plot saved for customer {idx}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_shap_analysis()
