import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import wandb
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_val = val_df.drop(columns=['target'])
    y_val = val_df['target']

    return X_train, y_train, X_val, y_val


def train_rf_with_cv(X_train, y_train):
    print("Initializing SMOTE and Random Forest Pipeline...")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', rf_model)
    ])

    # Log hyperparameters to WandB
    wandb.config.update({
        "model_type": "RandomForest",
        "n_estimators": 100,
        "random_state": 42,
        "smote_enabled": True,
        "cv_folds": 5
    })

    print("Running 5-Fold Stratified Cross-Validation with SMOTE...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1_weighted')

    print(f"CV F1-Scores: {cv_scores}")
    print(f"Mean CV F1-Score: {np.mean(cv_scores):.4f}")

    # Log CV scores to WandB
    for i, score in enumerate(cv_scores):
        wandb.log({f"cv_fold_{i+1}_f1": score})
    wandb.log({
        "mean_cv_f1": np.mean(cv_scores),
        "std_cv_f1": np.std(cv_scores)
    })

    print("Fitting final model on all training data with SMOTE...")
    pipeline.fit(X_train, y_train)

    return pipeline.named_steps['rf']


def evaluate_and_save(model, X_val, y_val, feature_names):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print("\nValidation Set Performance (After SMOTE):")
    report = classification_report(y_val, y_pred, output_dict=True)
    print(classification_report(y_val, y_pred))

    roc_auc = roc_auc_score(y_val, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Log metrics to WandB
    wandb.log({
        "val_accuracy": report['accuracy'],
        "val_precision": report['weighted avg']['precision'],
        "val_recall": report['weighted avg']['recall'],
        "val_f1": report['weighted avg']['f1-score'],
        "val_roc_auc": roc_auc
    })

    # Log confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_val,
            preds=y_pred,
            class_names=["Class 0", "Class 1"]
        )
    })

    joblib.dump(model, MODELS_DIR / "random_forest_model_smote.pkl")
    print(f"Model saved to {MODELS_DIR / 'random_forest_model_smote.pkl'}")

    # Log model as artifact
    artifact = wandb.Artifact('random_forest_model', type='model')
    artifact.add_file(str(MODELS_DIR / "random_forest_model_smote.pkl"))
    wandb.log_artifact(artifact)


def plot_feature_importance(model, feature_names, save_dir):
    print("\nExtracting Feature Importance...")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fi_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=fi_df.head(15), palette='viridis')
    plt.title('Top 15 Important Features - Random Forest (SMOTE)')
    plt.xlabel('Mean Decrease in Impurity')
    plt.tight_layout()

    save_path = save_dir / "feature_importance_rf_smote.png"
    plt.savefig(save_path)

    # Log feature importance plot to WandB
    wandb.log({"feature_importance": wandb.Image(str(save_path))})

    plt.close()
    print(f"Feature importance plot saved to {save_path}")


if __name__ == "__main__":
    # Initialize WandB
    wandb.init(
        project="ai-finalproject-mhm-power",
        name="random-forest-smote",
        tags=["random-forest", "smote", "baseline"]
    )

    X_train, y_train, X_val, y_val = load_data()

    rf_model = train_rf_with_cv(X_train, y_train)

    evaluate_and_save(rf_model, X_val, y_val, X_train.columns)

    plot_feature_importance(rf_model, X_train.columns, RESULTS_DIR)

    print("\nStep 3 (with SMOTE) completed successfully.")

    # Finish WandB run
    wandb.finish()