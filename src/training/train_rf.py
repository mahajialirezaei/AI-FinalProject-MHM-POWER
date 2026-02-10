import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"]

    return X_train, y_train, X_val, y_val


def train_rf_with_cv(X_train, y_train):
    print("Initializing SMOTE and Random Forest Pipeline...")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    pipeline = ImbPipeline([("smote", SMOTE(random_state=42)), ("rf", rf_model)])

    # Log hyperparameters to WandB
    log_config(
        {
            "model_type": "RandomForest",
            "n_estimators": 100,
            "random_state": 42,
            "smote_enabled": True,
            "cv_folds": 5,
        }
    )

    print("Running 5-Fold Stratified Cross-Validation with SMOTE...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1_weighted")

    print(f"CV F1-Scores: {cv_scores}")
    print(f"Mean CV F1-Score: {np.mean(cv_scores):.4f}")

    # Log CV scores to WandB
    for i, score in enumerate(cv_scores):
        log_metrics({f"cv_fold_{i+1}_f1": score})
    log_metrics({"mean_cv_f1": np.mean(cv_scores), "std_cv_f1": np.std(cv_scores)})

    print("Fitting final model on all training data with SMOTE...")
    pipeline.fit(X_train, y_train)

    return pipeline.named_steps["rf"]


def evaluate_and_save(model, X_val, y_val, feature_names):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print("\nValidation Set Performance (After SMOTE):")
    report = classification_report(y_val, y_pred, output_dict=True)
    print(classification_report(y_val, y_pred))

    roc_auc = roc_auc_score(y_val, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Log metrics to WandB
    log_metrics(
        {
            "val_accuracy": report["accuracy"],
            "val_precision": report["weighted avg"]["precision"],
            "val_recall": report["weighted avg"]["recall"],
            "val_f1": report["weighted avg"]["f1-score"],
            "val_roc_auc": roc_auc,
        }
    )

    # Log confusion matrix
    log_confusion_matrix(y_val, y_pred)

    save_path = MODELS_DIR / "random_forest_model_smote.pkl"
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

    # Log model as artifact
    log_artifact(str(save_path), "random_forest_model", "model")


def plot_feature_importance(model, feature_names, save_dir):
    print("\nExtracting Feature Importance...")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fi_df = pd.DataFrame(
        {"Feature": [feature_names[i] for i in indices], "Importance": importances[indices]}
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=fi_df.head(15), palette="viridis")
    plt.title("Top 15 Important Features - Random Forest (SMOTE)")
    plt.xlabel("Mean Decrease in Impurity")
    plt.tight_layout()

    save_path = save_dir / "feature_importance_rf_smote.png"
    plt.savefig(save_path)

    # Log feature importance plot to WandB
    log_image(str(save_path), "feature_importance")

    plt.close()
    print(f"Feature importance plot saved to {save_path}")


if __name__ == "__main__":
    # Initialize WandB
    init_wandb(
        run_name="random-forest-smote", tags=["random-forest", "smote", "baseline", "phase-2"]
    )

    try:
        X_train, y_train, X_val, y_val = load_data()

        rf_model = train_rf_with_cv(X_train, y_train)

        evaluate_and_save(rf_model, X_val, y_val, X_train.columns)

        plot_feature_importance(rf_model, X_train.columns, RESULTS_DIR)

        print("\nStep 3 (with SMOTE) completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        # Finish WandB run
        finish_wandb()
