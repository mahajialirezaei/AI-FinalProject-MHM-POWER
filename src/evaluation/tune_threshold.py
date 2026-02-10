import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from ruamel.yaml import YAML  # <--- NEW IMPORT

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "tuning"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_resources():
    print("[INFO] Loading data and champion model...")
    try:
        val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
        X_val = val_df.drop(columns=["target"])
        y_val = val_df["target"]
    except FileNotFoundError:
        raise FileNotFoundError("Validation data not found.")

    # Load the Weighted Optimized Champion
    model_path = MODELS_DIR / "xgboost_weighted_optimized.pkl"

    if not model_path.exists():
        print("[WARN] Optimized model not found. Falling back to Weighted...")
        model_path = MODELS_DIR / "xgboost_weighted.pkl"

    if not model_path.exists():
        raise FileNotFoundError("No weighted models found!")

    model = joblib.load(model_path)
    print(f"       Loaded: {model_path.name}")

    return X_val, y_val, model, model_path.relative_to(PROJECT_ROOT)


def find_optimal_threshold(y_true, y_prob):
    print("\n[INFO] Scanning thresholds for best F1-Score...")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"      Best Threshold Found: {best_threshold:.4f}")
    print(f"      Max F1-Score:         {best_f1:.4f}")

    return best_threshold, precisions, recalls, thresholds, f1_scores


def save_threshold_to_config(threshold, model_rel_path):
    """
    Updates config.yaml using ruamel.yaml to PRESERVE comments and layout.
    """
    print(f"\n[INFO] Updating {CONFIG_PATH}...")

    # Initialize the Round-Trip YAML editor
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    # 1. Load existing config (Comment-Safe)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = yaml.load(f) or {}
    else:
        config = {}

    # 2. Update 'model' section surgically
    if "model" not in config:
        config["model"] = {}

    config["model"]["threshold"] = float(threshold)
    config["model"]["path"] = str(model_rel_path)

    # 3. Save back to file
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

    print("[SUCCESS] Updated config.yaml (Comments Preserved).")


def evaluate_new_threshold(model, X_val, y_val, threshold):
    print("\n" + "=" * 60)
    print(f"PERFORMANCE AT NEW THRESHOLD: {threshold:.4f}")
    print("=" * 60)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred_new = (y_prob >= threshold).astype(int)

    print(classification_report(y_val, y_pred_new))

    cm = confusion_matrix(y_val, y_pred_new)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"CM (Threshold={threshold:.2f})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix_optimized_threshold.png")
    plt.close()


if __name__ == "__main__":
    X_val, y_val, model, model_rel_path = load_resources()

    y_prob = model.predict_proba(X_val)[:, 1]

    best_thresh, precisions, recalls, thresholds, f1_scores = find_optimal_threshold(y_val, y_prob)

    # Save with comment preservation
    save_threshold_to_config(best_thresh, model_rel_path)

    evaluate_new_threshold(model, X_val, y_val, best_thresh)
