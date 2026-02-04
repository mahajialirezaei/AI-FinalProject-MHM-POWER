import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "weighted_training"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
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
        raise FileNotFoundError("Data not found. Run preprocessing first.")

def train_weighted_model(X_train, y_train):
    """
    Trains XGBoost using scale_pos_weight instead of SMOTE.
    """
    # 1. Calculate the Weight
    # Formula: count(negative) / count(positive)
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    weight = num_neg / num_pos
    
    print(f"\n[INFO] Class Imbalance Detected:")
    print(f"       Negative samples: {num_neg}")
    print(f"       Positive samples: {num_pos}")
    print(f"       Calculated scale_pos_weight: {weight:.2f}")

    # 2. Define Model with Weight
    # We use the same 'sensible defaults' as before, but added the weight
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,            # Slightly reduced depth to prevent overfitting on weight
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=weight, # <--- THE KEY CHANGE
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )

    # 3. Stratified CV to verify stability
    print("\n[INFO] Running Stratified CV (Weighted)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=skf, scoring='f1')
    
    print(f"       CV F1-Scores: {cv_scores}")
    print(f"       Mean CV F1:   {np.mean(cv_scores):.4f}")

    # 4. Final Training
    print("[INFO] Training final weighted model...")
    xgb_model.fit(X_train, y_train)
    
    return xgb_model

def evaluate_model(model, X_val, y_val):
    print("\n[INFO] Evaluating on Validation Set...")
    
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (WEIGHTED MODEL)")
    print("="*60)
    print(classification_report(y_val, y_pred))
    
    auc = roc_auc_score(y_val, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    return y_pred

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_data()
    
    # Train
    model = train_weighted_model(X_train, y_train)
    
    # Evaluate
    evaluate_model(model, X_val, y_val)
    
    # Save
    save_path = MODELS_DIR / "xgboost_weighted.pkl"
    joblib.dump(model, save_path)
    print(f"\n[SUCCESS] Weighted model saved to {save_path}")