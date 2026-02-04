import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns  # Added for plotting
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load processed data for optimization."""
    print(f"[INFO] Loading data from {DATA_PROCESSED_DIR}...")
    try:
        train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
        val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")

        X_train = train_df.drop(columns=['target'])
        y_train = train_df['target']
        X_val = val_df.drop(columns=['target'])
        y_val = val_df['target']

        return X_train, y_train, X_val, y_val
    except FileNotFoundError:
        raise FileNotFoundError(f"Data not found. Run preprocessing first.")

def objective(trial, X, y):
    """
    Optuna Objective Function.
    This function is called repeatedly with different parameter combinations.
    """
    
    # 1. Define the Search Space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        # Fixed parameters
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }

    # 2. Construct Pipeline with SMOTE
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('xgb', xgb.XGBClassifier(**param))
    ])

    # 3. Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Optimize for F1 Score
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
    
    return scores.mean()

def run_optimization(n_trials=50):
    """Run the Optuna optimization study."""
    X_train, y_train, X_val, y_val = load_data()

    print(f"\n[INFO] Starting Optuna Optimization with {n_trials} trials...")
    print("       Target Metric: F1-Score (Maximize)")
    
    # Create Study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best F1 Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study, X_train, y_train, X_val, y_val

def train_best_model(study, X_train, y_train, X_val, y_val):
    """Train the final model using the best parameters found."""
    print("\n[INFO] Training Final Model with Best Parameters...")
    
    best_params = study.best_params
    # Add fixed params back in
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss', 
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1
    })

    final_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', xgb.XGBClassifier(**best_params))
    ])

    final_pipeline.fit(X_train, y_train)
    
    # Evaluation
    y_pred = final_pipeline.predict(X_val)
    y_prob = final_pipeline.predict_proba(X_val)[:, 1]
    
    print("\nFinal Validation Report (Optimized Model):")
    print(classification_report(y_val, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")
    
    # Save Model
    save_path = MODELS_DIR / "xgboost_optimized.pkl"
    joblib.dump(final_pipeline, save_path)
    print(f"\n[SUCCESS] Optimized model saved to {save_path}")
    
    return final_pipeline

def plot_feature_importance(pipeline, feature_names):
    """
    Extracts and plots feature importance from the Optimized XGBoost model.
    """
    print("\n[INFO] Generating Feature Importance Plot (Optimized)...")
    
    # Access the XGBoost model step from the pipeline
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
    plt.title('Top 20 Features - XGBoost (Optimized)', fontsize=14, fontweight='bold')
    plt.xlabel('Gain (Feature Importance)', fontsize=12)
    plt.tight_layout()

    save_path = RESULTS_DIR / "feature_importance_xgboost_optimized.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"       Saved to: {save_path}")

def plot_optimization_history(study):
    """Generate Optuna visualization plots."""
    try:
        print("\n[INFO] Generating optimization plots...")
        
        # Plot optimization history
        fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "optuna_history.png")
        plt.close()
        
        # Plot parameter importance
        fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "optuna_param_importance.png")
        plt.close()
        
        print(f"       Plots saved to {RESULTS_DIR}")
    except Exception as e:
        print(f"[WARN] Could not generate plots: {e}")

if __name__ == "__main__":
    # 1. Run Optimization
    # NOTE: Set n_trials higher (e.g., 50) for real results
    study, X_train, y_train, X_val, y_val = run_optimization(n_trials=30)
    
    # 2. Train Final Model
    final_pipeline = train_best_model(study, X_train, y_train, X_val, y_val)
    
    # 3. Plot Feature Importance (NEW STEP)
    plot_feature_importance(final_pipeline, X_train.columns)
    
    # 4. Visualize Optimization History
    plot_optimization_history(study)