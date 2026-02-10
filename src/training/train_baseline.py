import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from pathlib import Path
from src.training.wandb_utils import (
    init_wandb,
    log_metrics,
    log_config,
    log_artifact,
    log_image,
    log_confusion_matrix,
    finish_wandb,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data(data_dir):
    """
    Load processed train and validation datasets.
    Assumes 'target' column is the label.
    """
    print(f"Loading data from {data_dir}...")
    try:
        train_df = pd.read_csv(data_dir / "train.csv")
        val_df = pd.read_csv(data_dir / "val.csv")

        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]

        X_val = val_df.drop(columns=["target"])
        y_val = val_df["target"]

        print(f"Train set shape: {X_train.shape}")
        print(f"Val set shape: {X_val.shape}")
        return X_train, y_train, X_val, y_val

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Processed data files not found in {data_dir}. Run preprocessing first."
        )


def train_baseline_model(X_train, y_train):
    """
    Trains a Logistic Regression model as the baseline.
    """
    print("\nStarting Baseline Model Training (Logistic Regression)...")

    model_params = {
        "class_weight": "balanced",
        "random_state": 42,
        "max_iter": 1000,
        "solver": "lbfgs",
    }

    # Log hyperparameters to WandB
    log_config({"model_type": "LogisticRegression", **model_params})

    model = LogisticRegression(
        class_weight="balanced", random_state=42, max_iter=1000, solver="lbfgs", verbose=1
    )

    model.fit(X_train, y_train)
    print("Training completed.")
    return model


def evaluate_model(model, X_val, y_val, phase_name="Baseline"):
    """
    Evaluates the model and calculates required metrics.
    """
    print(f"\nEvaluating {phase_name} Model...")

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred),  # بسیار مهم برای تشخیص مشتریان راغب
        "F1 Score": f1_score(y_val, y_pred),
        "ROC AUC": roc_auc_score(y_val, y_prob),
    }

    print("-" * 30)
    print(f"Metrics for {phase_name}:")
    for k, v in metrics.items():
        print(f"{k:15}: {v:.4f}")
    print("-" * 30)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Log metrics to WandB
    log_metrics(
        {
            "val_accuracy": metrics["Accuracy"],
            "val_precision": metrics["Precision"],
            "val_recall": metrics["Recall"],
            "val_f1_score": metrics["F1 Score"],
            "val_roc_auc": metrics["ROC AUC"],
        }
    )

    # Log confusion matrix to WandB
    log_confusion_matrix(y_val, y_pred)

    return metrics, y_pred, y_prob


def plot_results(y_val, y_pred, y_prob, save_dir):
    """
    Generates and saves Confusion Matrix and ROC Curve.
    """
    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix - Baseline Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = save_dir / "confusion_matrix_baseline.png"
    plt.savefig(cm_path)
    plt.close()

    # Log confusion matrix image to WandB
    log_image(str(cm_path), "confusion_matrix")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc_score(y_val, y_prob):.2f})")
    plt.plot([0, 1], [0, 1], "k--")  # خط تصادفی
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Baseline Model")
    plt.legend()
    roc_path = save_dir / "roc_curve_baseline.png"
    plt.savefig(roc_path)
    plt.close()

    # Log ROC curve image to WandB
    log_image(str(roc_path), "roc_curve")

    print(f"Plots saved to {save_dir}")


def save_checkpoint(model, save_dir, filename="baseline_logreg.pkl"):
    """
    Saves the trained model to disk.
    """
    filepath = save_dir / filename
    joblib.dump(model, filepath)
    print(f"Model checkpoint saved to {filepath}")

    # Log model artifact to WandB
    log_artifact(str(filepath), "baseline_logreg", "model")


if __name__ == "__main__":
    # Initialize WandB
    init_wandb(
        run_name="baseline-logistic-regression", tags=["baseline", "logistic-regression", "phase-1"]
    )

    try:
        # 1. Load Data
        X_train, y_train, X_val, y_val = load_processed_data(DATA_PROCESSED_DIR)

        # 2. Train Model
        model = train_baseline_model(X_train, y_train)

        # 3. Evaluate Model
        metrics, y_pred, y_prob = evaluate_model(model, X_val, y_val)

        # 4. Save Plots
        plot_results(y_val, y_pred, y_prob, CHARTS_DIR)

        # 5. Save Model Checkpoint
        save_checkpoint(model, MODELS_DIR)

        print("\nPipeline finished successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        # Finish WandB run
        finish_wandb()
