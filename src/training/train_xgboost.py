import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    confusion_matrix, precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.training.wandb_utils import (
    init_wandb, log_metrics, log_config, log_artifact,
    log_image, log_confusion_matrix, finish_wandb
)

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """
    Loads processed training and validation data.
    Consistent with train_rf.py data loading pattern.
    """
    print(f"[INFO] Loading data from {DATA_PROCESSED_DIR}...")
    try:
        train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
        val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")

        X_train = train_df.drop(columns=['target'])
        y_train = train_df['target']
        
        X_val = val_df.drop(columns=['target'])
        y_val = val_df['target']

        print(f"      Train shape: {X_train.shape}")
        print(f"      Val shape:   {X_val.shape}")
        return X_train, y_train, X_val, y_val
    except FileNotFoundError:
        raise FileNotFoundError(f"Data not found in {DATA_PROCESSED_DIR}. Run preprocessing first.")

def train_xgboost_pipeline(X_train, y_train):
    """
    Trains an XGBoost model within a SMOTE pipeline.
    Uses Stratified K-Fold CV to ensure model stability.
    """
    print("\n[INFO] Initializing XGBoost + SMOTE Pipeline...")

    # Professional Hyperparameters
    # These provide a strong starting point for imbalanced binary classification
    xgb_params = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
        # Note: scale_pos_weight is omitted intentionally because we are using SMOTE.
        # Using both simultaneously can over-correct.
    }

    # Log hyperparameters to WandB
    log_config({
        "model_type": "XGBoost",
        "n_estimators": xgb_params['n_estimators'],
        "learning_rate": xgb_params['learning_rate'],
        "max_depth": xgb_params['max_depth'],
        "subsample": xgb_params['subsample'],
        "colsample_bytree": xgb_params['colsample_bytree'],
        "objective": xgb_params['objective'],
        "smote_enabled": True,
        "cv_folds": 5
    })

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('xgb', xgb.XGBClassifier(**xgb_params))
    ])

    print("[INFO] Running 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # We prioritize F1-Score for imbalanced datasets
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1')

    print(f"      CV F1-Scores: {cv_scores}")
    print(f"      Mean CV F1:   {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Log CV scores to WandB
    for i, score in enumerate(cv_scores):
        log_metrics({f"cv_fold_{i+1}_f1": score})
    log_metrics({
        "mean_cv_f1": np.mean(cv_scores),
        "std_cv_f1": np.std(cv_scores)
    })

    print("[INFO] Retraining pipeline on full training set...")
    pipeline.fit(X_train, y_train)

    return pipeline

def evaluate_model(pipeline, X_val, y_val):
    """
    Evaluates the model on the hold-out validation set.
    """
    print("\n[INFO] Evaluating on Validation Set...")

    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    # Print Classification Report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_val, y_pred, output_dict=True)
    print(classification_report(y_val, y_pred))

    # Calculate additional metrics
    roc = roc_auc_score(y_val, y_prob)
    f1 = f1_score(y_val, y_pred)

    print(f"ROC-AUC Score: {roc:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print("="*60)

    # Log metrics to WandB
    log_metrics({
        "val_accuracy": report['accuracy'],
        "val_precision": report['weighted avg']['precision'],
        "val_recall": report['weighted avg']['recall'],
        "val_f1": report['weighted avg']['f1-score'],
        "val_roc_auc": roc,
        "val_f1_binary": f1
    })

    # Log confusion matrix to WandB
    log_confusion_matrix(y_val, y_pred)

    return y_pred, y_prob

def plot_feature_importance(pipeline, feature_names, save_dir):
    """
    Extracts and plots feature importance from the XGBoost step of the pipeline.
    """
    print("\n[INFO] Generating Feature Importance Plot...")

    # Access the XGBoost model step
    model = pipeline.named_steps['xgb']

    # Get importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create DataFrame for plotting
    fi_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=fi_df.head(20), palette='magma')
    plt.title('Top 20 Features - XGBoost (SMOTE)', fontsize=14, fontweight='bold')
    plt.xlabel('Gain (Feature Importance)', fontsize=12)
    plt.tight_layout()

    save_path = save_dir / "feature_importance_xgboost.png"
    plt.savefig(save_path, dpi=300)

    # Log feature importance plot to WandB
    log_image(str(save_path), "feature_importance")

    plt.close()
    print(f"      Saved to: {save_path}")

def save_artifacts(pipeline, save_dir):
    """
    Saves the trained pipeline.
    """
    save_path = save_dir / "xgboost_model_smote.pkl"
    joblib.dump(pipeline, save_path)
    print(f"\n[INFO] Model saved to {save_path}")

    # Log model as artifact to WandB
    log_artifact(str(save_path), "xgboost_model", "model")

if __name__ == "__main__":
    # Initialize WandB
    init_wandb(
        run_name="xgboost-smote",
        tags=["xgboost", "smote", "baseline", "phase-2"]
    )
    
    try:
        # 1. Load Data
        X_train, y_train, X_val, y_val = load_data()

        # 2. Train (Pipeline with SMOTE + XGBoost)
        pipeline = train_xgboost_pipeline(X_train, y_train)

        # 3. Evaluate
        evaluate_model(pipeline, X_val, y_val)

        # 4. Feature Importance
        plot_feature_importance(pipeline, X_train.columns, RESULTS_DIR)

        # 5. Save Model
        save_artifacts(pipeline, MODELS_DIR)

        print("\n[SUCCESS] XGBoost training pipeline completed.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        # Finish WandB run
        finish_wandb()