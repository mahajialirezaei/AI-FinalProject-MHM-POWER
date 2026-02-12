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
from src.training.wandb_utils import (
    init_wandb,
    log_metrics,
    log_config,
    log_artifact,
    log_image,
    log_confusion_matrix,
    finish_wandb,
)

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

        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]
        X_val = val_df.drop(columns=["target"])
        y_val = val_df["target"]

        return X_train, y_train, X_val, y_val
    except FileNotFoundError:
        raise FileNotFoundError("Data not found. Run preprocessing first.")


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
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        # Crucial: Apply the calculated weight here
        "scale_pos_weight": scale_pos_weight,
        # Fixed parameters
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    # 2. Define Model (No Pipeline needed as we aren't using SMOTE)
    model = xgb.XGBClassifier(**param)

    # 3. Stratified Cross-Validation
    # We use 3 folds for speed during optimization
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Optimize for F1 Score
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

    return scores.mean()


def run_optimization(n_trials=50):
    """Run the Optuna optimization study."""
    # Silence Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_train, X_val, y_val = load_data()

    # Calculate weight once
    weight = calculate_scale_pos_weight(y_train)
    print(f"[INFO] Calculated scale_pos_weight: {weight:.2f}")  # noqa: F541

    print(f"\n[INFO] Starting Optuna Optimization (Weighted) with {n_trials} trials...")

    # Log optimization config to WandB
    log_config(
        {
            "optimization": "optuna",
            "n_trials": n_trials,
            "method": "weighted",
            "scale_pos_weight": weight,
            "cv_folds": 3,
            "direction": "maximize",
            "metric": "f1",
        }
    )

    # Create Study
    study = optuna.create_study(direction="maximize")

    # Callback to log each trial to WandB
    def callback(study, trial):
        log_metrics({"trial_f1": trial.value, "trial_number": trial.number})

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, weight),
        n_trials=n_trials,
        callbacks=[callback],
    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best F1 Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Log best results to WandB
    log_metrics({"best_f1_score": study.best_value})
    log_config({"best_params": study.best_params})

    return study, X_train, y_train, X_val, y_val, weight


def train_best_model(study, X_train, y_train, X_val, y_val, weight):
    """Train the final model using the best parameters found."""
    print("\n[INFO] Training Final Weighted Model with Best Parameters...")

    best_params = study.best_params
    # Add fixed params back in
    best_params.update(
        {
            "scale_pos_weight": weight,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # Evaluation
    y_pred = final_model.predict(X_val)
    y_prob = final_model.predict_proba(X_val)[:, 1]

    print("\nFinal Validation Report (Weighted Optimized Model):")
    report = classification_report(y_val, y_pred, output_dict=True)
    print(classification_report(y_val, y_pred))
    auc = roc_auc_score(y_val, y_prob)
    f1 = f1_score(y_val, y_pred)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Log metrics to WandB
    log_metrics(
        {
            "final_val_accuracy": report["accuracy"],
            "final_val_precision": report["weighted avg"]["precision"],
            "final_val_recall": report["weighted avg"]["recall"],
            "final_val_f1": report["weighted avg"]["f1-score"],
            "final_val_roc_auc": auc,
            "final_val_f1_binary": f1,
        }
    )

    # Log confusion matrix to WandB
    log_confusion_matrix(y_val, y_pred)

    # Save Model
    save_path = MODELS_DIR / "xgboost_weighted_optimized.pkl"
    joblib.dump(final_model, save_path)
    print(f"\n[SUCCESS] Optimized model saved to {save_path}")

    # Log model artifact to WandB
    log_artifact(str(save_path), "xgboost_weighted_optimized", "model")

    return final_model


def plot_feature_importance(model, feature_names):
    """Extracts and plots feature importance."""
    print("\n[INFO] Generating Feature Importance Plot...")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fi_df = pd.DataFrame(
        {"Feature": [feature_names[i] for i in indices], "Importance": importances[indices]}
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=fi_df.head(20), palette="magma")
    plt.title("Top 20 Features - XGBoost (Weighted & Optimized)", fontsize=14, fontweight="bold")
    plt.xlabel("Gain (Feature Importance)", fontsize=12)
    plt.tight_layout()

    save_path = RESULTS_DIR / "feature_importance_xgboost_weighted_opt.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"       Saved to: {save_path}")

    # Log feature importance plot to WandB
    log_image(str(save_path), "feature_importance")


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
        plt.title("Optimization History (Weighted)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        hist_path = RESULTS_DIR / "optuna_weighted_history.png"
        plt.savefig(hist_path, dpi=300)
        plt.close()
        print(f"       [SUCCESS] History saved to {hist_path}")

        # Log optimization history to WandB
        log_image(str(hist_path), "optimization_history")
    except Exception as e:
        print(f"       [WARN] Optimization History plot failed: {e}")

    # 2. Parameter Importance
    try:
        # Check if we have enough trials for importance
        if len(study.trials) > 1:
            plt.figure(figsize=(10, 8))
            optuna_plt.plot_param_importances(study)
            # plt.title("Hyperparameter Importance", fontsize=14, fontweight='bold')
            plt.tight_layout()

            imp_path = RESULTS_DIR / "optuna_weighted_param_importance.png"
            plt.savefig(imp_path, dpi=300)
            plt.close()
            print(f"       [SUCCESS] Importance saved to {imp_path}")

            # Log parameter importance to WandB
            log_image(str(imp_path), "parameter_importance")
        else:
            print("       [WARN] Skipping importance plot (need >1 trial)")
    except Exception as e:
        print(f"       [WARN] Parameter Importance plot failed: {e}")


if __name__ == "__main__":
    # Initialize WandB
    init_wandb(
        run_name="xgboost-weighted-optimized-champion",
        tags=["xgboost", "weighted", "optimized", "optuna", "champion", "phase-2"],
    )

    try:
        # 1. Run Optimization
        # NOTE: n_trials must be > 1 for parameter importance to work
        study, X_train, y_train, X_val, y_val, weight = run_optimization(n_trials=60)

        # 2. Train Final Model
        final_model = train_best_model(study, X_train, y_train, X_val, y_val, weight)

        # 3. Plot Feature Importance
        plot_feature_importance(final_model, X_train.columns)

        # 4. Plot Optuna History (Re-added as requested)
        plot_optuna_charts(study)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        # Finish WandB run
        finish_wandb()
