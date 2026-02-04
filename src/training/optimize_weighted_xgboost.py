import optuna
# CRITICAL IMPORT: This enables the matplotlib plotting backend
import optuna.visualization.matplotlib as optuna_plt 
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "optimization"

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

def calculate_scale_pos_weight(y):
    """Calculates the weight for the positive class."""
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    weight = num_neg / num_pos
    return weight

def objective(trial, X, y, scale_pos_weight):
    """
    Optuna Objective Function.
    Optimizes XGBoost hyperparameters WITH scale_pos_weight (No SMOTE).
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
        # Crucial: Apply the calculated weight here
        'scale_pos_weight': scale_pos_weight,
        # Fixed parameters
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }

    # 2. Define Model (No Pipeline needed as we aren't using SMOTE)
    model = xgb.XGBClassifier(**param)

    # 3. Stratified Cross-Validation
    # We use 3 folds for speed during optimization
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Optimize for F1 Score
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    
    return scores.mean()

def run_optimization(n_trials=50):
    """Run the Optuna optimization study."""
    # Silence Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_train, X_val, y_val = load_data()

    # Calculate weight once
    weight = calculate_scale_pos_weight(y_train)
    print(f"[INFO] Calculated scale_pos_weight: {weight:.2f}")

    print(f"\n[INFO] Starting Optuna Optimization (Weighted) with {n_trials} trials...")
    
    # Create Study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, weight), n_trials=n_trials)

    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best F1 Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study, X_train, y_train, X_val, y_val, weight

def train_best_model(study, X_train, y_train, X_val, y_val, weight):
    """Train the final model using the best parameters found."""
    print("\n[INFO] Training Final Weighted Model with Best Parameters...")
    
    best_params = study.best_params
    # Add fixed params back in
    best_params.update({
        'scale_pos_weight': weight,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss', 
        'random_state': 42,
        'n_jobs': -1
    })

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = final_model.predict(X_val)
    y_prob = final_model.predict_proba(X_val)[:, 1]
    
    print("\nFinal Validation Report (Weighted Optimized Model):")
    print(classification_report(y_val, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")
    
    # Save Model
    save_path = MODELS_DIR / "xgboost_weighted_optimized.pkl"
    joblib.dump(final_model, save_path)
    print(f"\n[SUCCESS] Optimized model saved to {save_path}")
    
    return final_model

def plot_feature_importance(model, feature_names):
    """Extracts and plots feature importance."""
    print("\n[INFO] Generating Feature Importance Plot...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fi_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=fi_df.head(20), palette='magma')
    plt.title('Top 20 Features - XGBoost (Weighted & Optimized)', fontsize=14, fontweight='bold')
    plt.xlabel('Gain (Feature Importance)', fontsize=12)
    plt.tight_layout()

    save_path = RESULTS_DIR / "feature_importance_xgboost_weighted_opt.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"       Saved to: {save_path}")

def plot_optuna_charts(study):
    """
    Plots Optimization History and Parameter Importance.
    Includes error handling to prevent crashes if something fails.
    """
    print("\n[INFO] Generating Optuna History Charts...")

    # 1. Optimization History
    try:
        plt.figure(figsize=(10, 6))
        optuna_plt.plot_optimization_history(study)
        plt.title("Optimization History (Weighted)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        hist_path = RESULTS_DIR / "optuna_weighted_history.png"
        plt.savefig(hist_path, dpi=300)
        plt.close()
        print(f"       [SUCCESS] History saved to {hist_path}")
    except Exception as e:
        print(f"       [WARN] Optimization History plot failed: {e}")

    # 2. Parameter Importance
    try:
        # Check if we have enough trials for importance
        if len(study.trials) > 1:
            plt.figure(figsize=(10, 8))
            optuna_plt.plot_param_importances(study)
            #plt.title("Hyperparameter Importance", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            imp_path = RESULTS_DIR / "optuna_weighted_param_importance.png"
            plt.savefig(imp_path, dpi=300)
            plt.close()
            print(f"       [SUCCESS] Importance saved to {imp_path}")
        else:
            print("       [WARN] Skipping importance plot (need >1 trial)")
    except Exception as e:
        print(f"       [WARN] Parameter Importance plot failed: {e}")


if __name__ == "__main__":
    # 1. Run Optimization
    # NOTE: n_trials must be > 1 for parameter importance to work
    study, X_train, y_train, X_val, y_val, weight = run_optimization(n_trials=60)
    
    # 2. Train Final Model
    final_model = train_best_model(study, X_train, y_train, X_val, y_val, weight)
    
    # 3. Plot Feature Importance
    plot_feature_importance(final_model, X_train.columns)

    # 4. Plot Optuna History (Re-added as requested)
    plot_optuna_charts(study)