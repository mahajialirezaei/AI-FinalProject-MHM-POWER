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
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the processed datasets[cite: 961]."""
    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_val = val_df.drop(columns=['target'])
    y_val = val_df['target']

    return X_train, y_train, X_val, y_val


def train_rf_with_cv(X_train, y_train):
    print("Initializing Random Forest Classifier...")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    print("Running 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='f1_weighted')

    print(f"CV F1-Scores: {cv_scores}")
    print(f"Mean CV F1-Score: {np.mean(cv_scores):.4f}")

    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_and_save(model, X_val, y_val, feature_names):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print("\nValidation Set Performance:")
    print(classification_report(y_val, y_pred))

    joblib.dump(model, MODELS_DIR / "random_forest_model.pkl")
    print(f"Model saved to {MODELS_DIR / 'random_forest_model.pkl'}")


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
    plt.title('Top 15 Important Features - Random Forest')
    plt.xlabel('Mean Decrease in Impurity')
    plt.tight_layout()

    save_path = save_dir / "feature_importance_rf.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance plot saved to {save_path}")

    print("Top 5 Features:")
    print(fi_df.head(5))


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_data()

    rf_model = train_rf_with_cv(X_train, y_train)

    evaluate_and_save(rf_model, X_val, y_val, X_train.columns)

    plot_feature_importance(rf_model, X_train.columns, RESULTS_DIR)

    print("\nStep 3 completed successfully.")