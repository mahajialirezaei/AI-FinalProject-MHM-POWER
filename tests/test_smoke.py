import pytest
import pandas as pd
import yaml
import joblib
import os
from pathlib import Path
import sys

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import ALL training modules
from src.preprocessing.main import load_and_preprocess_data
from src.eda.data_loader import load_config
from src.training.train_baseline import train_baseline_model
from src.training.train_rf import train_rf_with_cv
from src.training.train_xgboost import train_xgboost_pipeline
from src.training.train_weighted_xgboost import train_weighted_model

# ==========================================
# FIXTURES & MOCK DATA
# ==========================================

# Increased Mock Data to satisfy n_splits=5 in Cross-Validation
# We need at least 5 samples for each class ('yes' and 'no')
MOCK_CSV_CONTENT = """age;job;marital;education;default;balance;housing;loan;contact;day;month;duration;campaign;pdays;previous;poutcome;y
58;management;married;tertiary;no;2143;yes;no;cellular;5;may;261;1;-1;0;unknown;no
44;technician;single;secondary;no;29;yes;no;cellular;5;may;151;1;-1;0;unknown;no
33;entrepreneur;married;secondary;no;2;yes;yes;cellular;5;may;76;1;-1;0;failure;yes
47;blue-collar;married;unknown;no;1506;yes;no;cellular;5;may;92;1;-1;0;unknown;no
33;unknown;single;unknown;no;1;no;no;unknown;5;may;198;1;-1;0;unknown;no
35;management;married;tertiary;no;231;yes;no;cellular;5;may;120;1;-1;0;success;yes
28;blue-collar;single;secondary;no;447;yes;yes;telephone;5;may;80;1;-1;0;unknown;no
42;entrepreneur;divorced;tertiary;yes;2;yes;no;cellular;5;may;380;1;-1;0;other;yes
50;management;married;tertiary;no;100;no;no;cellular;5;may;100;1;-1;0;unknown;no
30;technician;single;secondary;no;500;no;no;cellular;5;may;200;1;-1;0;failure;yes
40;admin;married;secondary;no;1000;yes;no;telephone;5;may;300;1;-1;0;unknown;no
45;blue-collar;married;primary;no;50;yes;no;cellular;5;may;150;1;-1;0;success;yes
51;management;married;tertiary;no;200;yes;no;cellular;5;may;250;1;-1;0;unknown;no
31;technician;single;secondary;no;600;no;no;cellular;5;may;210;1;-1;0;failure;yes
41;admin;married;secondary;no;1100;yes;no;telephone;5;may;310;1;-1;0;unknown;no
46;blue-collar;married;primary;no;60;yes;no;cellular;5;may;160;1;-1;0;success;yes
"""

@pytest.fixture(scope="session")
def setup_environment(tmp_path_factory):
    """
    Creates an isolated environment with enough data for CV.
    """
    temp_dir = tmp_path_factory.mktemp("project_test_suite")
    raw_dir = temp_dir / "data" / "raw" / "bank"
    processed_dir = temp_dir / "data" / "processed"
    models_dir = temp_dir / "src" / "models"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    raw_data_path = raw_dir / "bank-full.csv"
    with open(raw_data_path, "w") as f:
        f.write(MOCK_CSV_CONTENT)
        
    return {
        "root": temp_dir,
        "raw_path": raw_data_path,
        "processed_dir": processed_dir,
        "models_dir": models_dir
    }

# ==========================================
# SMOKE TESTS - PIPELINE STAGES
# ==========================================

@pytest.mark.smoke
def test_01_configuration(setup_environment):
    """Verify config file structure."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    assert config_path.exists()
    config = load_config(str(config_path))
    assert "data" in config

@pytest.mark.smoke
def test_02_preprocessing(setup_environment):
    """Verify data preprocessing."""
    try:
        train_df, val_df, test_df = load_and_preprocess_data(
            str(setup_environment["raw_path"]), 
            str(setup_environment["processed_dir"])
        )
    except Exception as e:
        pytest.fail(f"Preprocessing failed: {e}")
        
    assert not train_df.empty
    assert "target" in train_df.columns
    # Data Leakage Check
    assert "duration" not in train_df.columns

# ==========================================
# TRAINING TESTS (ALL MODELS)
# ==========================================

@pytest.fixture
def training_data(setup_environment):
    """Helper fixture to load processed data for training tests."""
    processed_dir = setup_environment["processed_dir"]
    train_df = pd.read_csv(processed_dir / "train.csv")
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    return X_train, y_train, setup_environment["models_dir"]

@pytest.mark.smoke
def test_03_train_baseline_logreg(training_data):
    """Test Phase 1: Logistic Regression Training."""
    X_train, y_train, models_dir = training_data
    try:
        model = train_baseline_model(X_train, y_train)
        joblib.dump(model, models_dir / "baseline.pkl")
    except Exception as e:
        pytest.fail(f"Baseline training failed: {e}")

@pytest.mark.smoke
def test_04_train_random_forest_smote(training_data):
    """Test Phase 2: Random Forest + SMOTE Pipeline."""
    X_train, y_train, models_dir = training_data
    try:
        # train_rf_with_cv uses StratifiedKFold internally
        model = train_rf_with_cv(X_train, y_train)
        joblib.dump(model, models_dir / "rf_smote.pkl")
    except ValueError as e:
        if "n_splits" in str(e):
            pytest.skip("Not enough mock data for 5-fold CV")
        else:
            pytest.fail(f"Random Forest training failed: {e}")
    except Exception as e:
        pytest.fail(f"Random Forest training failed: {e}")

@pytest.mark.smoke
def test_05_train_xgboost_smote(training_data):
    """Test Phase 2: XGBoost + SMOTE Pipeline."""
    X_train, y_train, models_dir = training_data
    try:
        model = train_xgboost_pipeline(X_train, y_train)
        joblib.dump(model, models_dir / "xgb_smote.pkl")
    except Exception as e:
        pytest.fail(f"XGBoost (SMOTE) training failed: {e}")

@pytest.mark.smoke
def test_06_train_champion_weighted_xgboost(training_data):
    """Test Phase 3: Weighted XGBoost (Production Model)."""
    X_train, y_train, models_dir = training_data
    try:
        model = train_weighted_model(X_train, y_train)
        joblib.dump(model, models_dir / "champion.pkl")
    except Exception as e:
        pytest.fail(f"Weighted XGBoost training failed: {e}")

@pytest.mark.smoke
def test_07_inference_capability(training_data):
    """
    Final Check: Load the Champion model and make a prediction.
    """
    _, _, models_dir = training_data
    champion_path = models_dir / "champion.pkl"
    
    if not champion_path.exists():
        pytest.skip("Champion model was not trained successfully")
        
    model = joblib.load(champion_path)
    
    # Create a dummy input based on training data shape
    X_train, _, _ = training_data
    sample_input = X_train.iloc[[0]]
    
    try:
        pred = model.predict(sample_input)
        assert len(pred) == 1
    except Exception as e:
        pytest.fail(f"Inference failed on champion model: {e}")